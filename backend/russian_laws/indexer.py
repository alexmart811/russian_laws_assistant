import pandas as pd
from omegaconf import DictConfig
from qdrant_client.models import PointStruct
from russian_laws.embeddings import EmbeddingModel
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder
from tqdm import tqdm


class ArticleIndexer:
    """Индексация статей в Qdrant с поддержкой чанкирования."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        qdrant_manager: QdrantManager,
        config: DictConfig,
        sparse_encoder: SparseEncoder | None = None,
    ):
        self.embedding_model = embedding_model
        self.qdrant_manager = qdrant_manager
        self.config = config
        self.sparse_encoder = sparse_encoder
        self.hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if not text or not text.strip():
            return []
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            chunk_text_str = " ".join(words[start : start + chunk_size])
            if chunk_text_str.strip():
                chunks.append(chunk_text_str)
            start += chunk_size - chunk_overlap
            if start >= len(words):
                break
        return chunks if chunks else [text]

    @staticmethod
    def _parse_paragraphs(text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if ";\n" in para:
                paragraphs.extend(p.strip() for p in para.split(";\n") if p.strip())
            else:
                paragraphs.append(para)
        return paragraphs

    @staticmethod
    def _count_words(text: str) -> int:
        return len(text.split())

    def _create_parent_child_chunks(self, text: str, article_id: int) -> list[dict]:
        config = self.config.embedding.chunking

        min_child_size = config.child.get("min_size", 30)
        max_child_size = config.child.get("max_size", 150)
        children_per_parent = config.parent.get("children_per_parent", 3)
        max_parent_size = config.parent.get("max_size", 500)
        parent_overlap = config.parent.get("overlap", False)

        fallback = [
            {
                "parent_id": f"{article_id}_p0",
                "parent_text": text,
                "children": [{"child_idx": 0, "child_text": text}],
            }
        ]

        paragraphs = self._parse_paragraphs(text)
        if not paragraphs:
            return fallback

        valid_children = []
        for i, para in enumerate(paragraphs):
            word_count = self._count_words(para)
            if min_child_size <= word_count <= max_child_size:
                valid_children.append({"child_idx": i, "child_text": para})
            elif word_count > max_child_size:
                words = para.split()
                for j in range(0, len(words), max_child_size):
                    chunk_text = " ".join(words[j : j + max_child_size])
                    if self._count_words(chunk_text) >= min_child_size:
                        valid_children.append(
                            {"child_idx": len(valid_children), "child_text": chunk_text}
                        )

        if not valid_children:
            return fallback

        parents = []
        step = children_per_parent if not parent_overlap else children_per_parent - 1

        for i in range(0, len(valid_children), step):
            child_group = valid_children[i : i + children_per_parent]
            if not child_group:
                continue

            parent_text = "\n\n".join(c["child_text"] for c in child_group)
            if self._count_words(parent_text) > max_parent_size:
                child_group = child_group[:2]
                parent_text = "\n\n".join(c["child_text"] for c in child_group)

            parents.append({
                "parent_id": f"{article_id}_p{len(parents)}",
                "parent_text": parent_text,
                "children": child_group,
            })

        return parents if parents else fallback

    def _make_vector(self, text: str) -> dict | list:
        embedding = self.embedding_model.encode_passages([text])[0].tolist()
        if self.hybrid_enabled and self.sparse_encoder:
            sparse_vec = self.sparse_encoder.encode(text)
            return {
                "dense": embedding,
                "sparse": {
                    "indices": [idx for idx, _ in sparse_vec],
                    "values": [val for _, val in sparse_vec],
                },
            }
        return embedding

    def index_articles(
        self, articles_df: pd.DataFrame, batch_size: int = 32, recreate: bool = True
    ) -> int:
        chunking_enabled = self.config.embedding.chunking.enabled
        strategy = self.config.embedding.chunking.get("strategy", "fixed")

        print(f"\nИндексация {len(articles_df)} статей...")
        if chunking_enabled:
            print(f"Чанкирование: {strategy}")
        else:
            print("Чанкирование: отключено")

        self.qdrant_manager.create_collection(recreate=recreate)

        points = []
        point_id = 0
        total_chunks = 0

        for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Индексация"):
            text = f"{row['article_title']} {row['article_text']}"
            article_id = int(row["article_id"])

            if chunking_enabled and strategy == "parent_child":
                hierarchy = self._create_parent_child_chunks(row["article_text"], article_id)

                for parent in hierarchy:
                    for child in parent["children"]:
                        child_text = child["child_text"]
                        point = PointStruct(
                            id=point_id,
                            vector=self._make_vector(child_text),
                            payload={
                                "article_id": article_id,
                                "article_num": str(row["article_num"]),
                                "article_title": row["article_title"],
                                "article_text": row["article_text"],
                                "codex": row["codex"],
                                "parent_id": parent["parent_id"],
                                "child_idx": child["child_idx"],
                                "child_text": child_text,
                                "parent_text": parent["parent_text"],
                            },
                        )
                        points.append(point)
                        point_id += 1
                        total_chunks += 1

                        if len(points) >= batch_size:
                            self.qdrant_manager.upsert_points(points)
                            points = []

            elif chunking_enabled:
                chunk_size = self.config.embedding.chunking.chunk_size
                chunk_overlap = self.config.embedding.chunking.chunk_overlap
                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                total_chunks += len(chunks)

                for chunk_idx, chunk in enumerate(chunks):
                    point = PointStruct(
                        id=point_id,
                        vector=self._make_vector(chunk),
                        payload={
                            "article_id": article_id,
                            "chunk_idx": chunk_idx,
                            "total_chunks": len(chunks),
                            "article_num": str(row["article_num"]),
                            "article_title": row["article_title"],
                            "article_text": row["article_text"],
                            "codex": row["codex"],
                        },
                    )
                    points.append(point)
                    point_id += 1

                    if len(points) >= batch_size:
                        self.qdrant_manager.upsert_points(points)
                        points = []
            else:
                point = PointStruct(
                    id=article_id,
                    vector=self._make_vector(text),
                    payload={
                        "article_id": article_id,
                        "article_num": str(row["article_num"]),
                        "article_title": row["article_title"],
                        "article_text": row["article_text"],
                        "codex": row["codex"],
                    },
                )
                points.append(point)
                total_chunks += 1

                if len(points) >= batch_size:
                    self.qdrant_manager.upsert_points(points)
                    points = []

        if points:
            self.qdrant_manager.upsert_points(points)

        if chunking_enabled:
            print(f"✓ Создано {total_chunks} чанков из {len(articles_df)} статей")
        else:
            print(f"✓ Проиндексировано {len(articles_df)} статей")

        return total_chunks

"""Модуль для индексации статей в Qdrant."""

import pandas as pd
from omegaconf import DictConfig
from qdrant_client.models import PointStruct
from russian_laws.embeddings import EmbeddingModel
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder
from tqdm import tqdm


class ArticleIndexer:
    """Индексатор статей в Qdrant с поддержкой чанкирования."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        qdrant_manager: QdrantManager,
        config: DictConfig,
        sparse_encoder: SparseEncoder | None = None,
    ):
        """Инициализация индексатора.

        Args:
            embedding_model: Модель для генерации эмбеддингов
            qdrant_manager: Менеджер для работы с Qdrant
            config: Конфигурация с параметрами чанкирования
            sparse_encoder: Опциональный sparse encoder для гибридного поиска
        """
        self.embedding_model = embedding_model
        self.qdrant_manager = qdrant_manager
        self.config = config
        self.sparse_encoder = sparse_encoder
        self.hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Разбивает текст на чанки с перекрытием (фиксированное чанкирование).

        Args:
            text: Текст для разбиения
            chunk_size: Размер чанка в словах
            chunk_overlap: Перекрытие между чанками в словах

        Returns:
            Список текстовых чанков
        """
        if not text or not text.strip():
            return []

        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)
            if chunk_text_str.strip():
                chunks.append(chunk_text_str)

            # Переход к следующему чанку с перекрытием
            start += chunk_size - chunk_overlap

            # Если осталось меньше слов, чем размер перекрытия, останавливаемся
            if start >= len(words):
                break

        return chunks if chunks else [text]

    @staticmethod
    def _parse_paragraphs(text: str) -> list[str]:
        """Разбивает текст на абзацы для юридических документов.

        Args:
            text: Текст статьи

        Returns:
            Список абзацев
        """
        if not text or not text.strip():
            return []

        # Юридические тексты используют разделители:
        # 1. Двойной перенос строки
        # 2. Точка с запятой + перенос (для списков)
        paragraphs = []

        # Сначала разбиваем по двойным переносам
        raw_paragraphs = text.split("\n\n")

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # Дополнительно разбиваем длинные параграфы по точке с запятой
            if ";\n" in para:
                sub_paras = [p.strip() for p in para.split(";\n") if p.strip()]
                paragraphs.extend(sub_paras)
            else:
                paragraphs.append(para)

        return [p for p in paragraphs if p]

    @staticmethod
    def _count_words(text: str) -> int:
        """Подсчитывает количество слов в тексте.

        Args:
            text: Текст

        Returns:
            Количество слов
        """
        return len(text.split())

    def _create_parent_child_chunks(self, text: str, article_id: int) -> list[dict]:
        """Создает иерархию parent-child чанков.

        Args:
            text: Полный текст статьи
            article_id: ID статьи

        Returns:
            Список parent чанков с вложенными child чанками
        """
        config = self.config.embedding.chunking

        min_child_size = config.child.get("min_size", 30)
        max_child_size = config.child.get("max_size", 150)
        children_per_parent = config.parent.get("children_per_parent", 3)
        max_parent_size = config.parent.get("max_size", 500)
        parent_overlap = config.parent.get("overlap", False)

        # 1. Разбиваем на абзацы
        paragraphs = self._parse_paragraphs(text)

        if not paragraphs:
            # Если не удалось разбить, возвращаем весь текст как один parent-child
            return [
                {
                    "parent_id": f"{article_id}_p0",
                    "parent_text": text,
                    "children": [{"child_idx": 0, "child_text": text}],
                }
            ]

        # 2. Фильтруем child чанки по размеру
        valid_children = []
        for i, para in enumerate(paragraphs):
            word_count = self._count_words(para)
            if min_child_size <= word_count <= max_child_size:
                valid_children.append({"child_idx": i, "child_text": para})
            elif word_count > max_child_size:
                # Слишком большой параграф - разбиваем дальше
                words = para.split()
                for j in range(0, len(words), max_child_size):
                    chunk_words = words[j : j + max_child_size]
                    chunk_text = " ".join(chunk_words)
                    if self._count_words(chunk_text) >= min_child_size:
                        valid_children.append(
                            {"child_idx": len(valid_children), "child_text": chunk_text}
                        )

        if not valid_children:
            # Все параграфы слишком короткие - берем весь текст
            return [
                {
                    "parent_id": f"{article_id}_p0",
                    "parent_text": text,
                    "children": [{"child_idx": 0, "child_text": text}],
                }
            ]

        # 3. Группируем children в parents
        parents = []
        step = children_per_parent if not parent_overlap else children_per_parent - 1

        for i in range(0, len(valid_children), step):
            child_group = valid_children[i : i + children_per_parent]

            if not child_group:
                continue

            parent_text = "\n\n".join([c["child_text"] for c in child_group])

            # Проверяем размер parent
            if self._count_words(parent_text) > max_parent_size:
                # Если слишком большой, уменьшаем количество children
                child_group = child_group[:2]
                parent_text = "\n\n".join([c["child_text"] for c in child_group])

            parent = {
                "parent_id": f"{article_id}_p{len(parents)}",
                "parent_text": parent_text,
                "children": child_group,
            }
            parents.append(parent)

        return (
            parents
            if parents
            else [
                {
                    "parent_id": f"{article_id}_p0",
                    "parent_text": text,
                    "children": [{"child_idx": 0, "child_text": text}],
                }
            ]
        )

    def index_articles(
        self, articles_df: pd.DataFrame, batch_size: int = 32, recreate: bool = True
    ) -> int:
        """Индексирует статьи в Qdrant с поддержкой чанкирования.

        Args:
            articles_df: DataFrame со статьями
            batch_size: Размер батча для индексации
            recreate: Пересоздать коллекцию

        Returns:
            Количество созданных точек (чанков)
        """
        chunking_enabled = self.config.embedding.chunking.enabled
        strategy = self.config.embedding.chunking.get("strategy", "fixed")

        print(f"\nИндексация {len(articles_df)} статей в Qdrant...")
        if chunking_enabled:
            print(f"Чанкирование: включено (стратегия: {strategy})")
            if strategy == "parent_child":
                print(
                    f"  Child: {self.config.embedding.chunking.child.min_size}-"
                    f"{self.config.embedding.chunking.child.max_size} слов"
                )
                print(
                    f"  Parent: {self.config.embedding.chunking.parent.children_per_parent} children"
                )
            else:
                print(
                    f"  Размер чанка: {self.config.embedding.chunking.chunk_size}, "
                    f"перекрытие: {self.config.embedding.chunking.chunk_overlap}"
                )
        else:
            print("Чанкирование: отключено")

        # Создаем коллекцию
        self.qdrant_manager.create_collection(recreate=recreate)

        # Индексируем статьи батчами
        points = []
        point_id = 0
        total_chunks = 0

        for _, row in tqdm(
            articles_df.iterrows(), total=len(articles_df), desc="Индексация"
        ):
            # Создаем текст для индексации (название + текст статьи)
            text = f"{row['article_title']} {row['article_text']}"
            article_id = int(row["article_id"])

            if chunking_enabled and strategy == "parent_child":
                # PARENT-CHILD СТРАТЕГИЯ
                hierarchy = self._create_parent_child_chunks(
                    row["article_text"], article_id
                )

                for parent in hierarchy:
                    for child in parent["children"]:
                        # Вектор генерируем для CHILD текста (маленький)
                        child_text = child["child_text"]
                        embedding = self.embedding_model.encode([child_text])[
                            0
                        ].tolist()

                        # Генерируем sparse вектор
                        if self.hybrid_enabled and self.sparse_encoder:
                            sparse_vec = self.sparse_encoder.encode(child_text)
                            vector_data = {
                                "dense": embedding,
                                "sparse": {
                                    "indices": [idx for idx, _ in sparse_vec],
                                    "values": [val for _, val in sparse_vec],
                                },
                            }
                        else:
                            vector_data = embedding

                        point = PointStruct(
                            id=point_id,
                            vector=vector_data,
                            payload={
                                "article_id": article_id,
                                "article_num": str(row["article_num"]),
                                "article_title": row["article_title"],
                                "article_text": row["article_text"],
                                "codex": row["codex"],
                                # Parent-child поля
                                "parent_id": parent["parent_id"],
                                "child_idx": child["child_idx"],
                                "child_text": child_text,
                                "parent_text": parent["parent_text"],
                            },
                        )
                        points.append(point)
                        point_id += 1
                        total_chunks += 1

                        # Сохраняем батчами
                        if len(points) >= batch_size:
                            self.qdrant_manager.upsert_points(points)
                            points = []

            elif chunking_enabled:
                # FIXED СТРАТЕГИЯ (старый способ)
                chunk_size = self.config.embedding.chunking.chunk_size
                chunk_overlap = self.config.embedding.chunking.chunk_overlap
                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                total_chunks += len(chunks)

                # Создаем точку для каждого чанка
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.embedding_model.encode([chunk])[0].tolist()

                    # Генерируем sparse вектор если гибридный режим включен
                    if self.hybrid_enabled and self.sparse_encoder:
                        sparse_vec = self.sparse_encoder.encode(chunk)
                        vector_data = {
                            "dense": embedding,
                            "sparse": {
                                "indices": [idx for idx, _ in sparse_vec],
                                "values": [val for _, val in sparse_vec],
                            },
                        }
                    else:
                        vector_data = embedding

                    point = PointStruct(
                        id=point_id,
                        vector=vector_data,
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

                    # Сохраняем батчами
                    if len(points) >= batch_size:
                        self.qdrant_manager.upsert_points(points)
                        points = []
            else:
                # БЕЗ ЧАНКИРОВАНИЯ
                embedding = self.embedding_model.encode([text])[0].tolist()

                # Генерируем sparse вектор если гибридный режим включен
                if self.hybrid_enabled and self.sparse_encoder:
                    sparse_vec = self.sparse_encoder.encode(text)
                    vector_data = {
                        "dense": embedding,
                        "sparse": {
                            "indices": [idx for idx, _ in sparse_vec],
                            "values": [val for _, val in sparse_vec],
                        },
                    }
                else:
                    vector_data = embedding

                point = PointStruct(
                    id=article_id,
                    vector=vector_data,
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

                # Сохраняем батчами
                if len(points) >= batch_size:
                    self.qdrant_manager.upsert_points(points)
                    points = []

        # Сохраняем оставшиеся точки
        if points:
            self.qdrant_manager.upsert_points(points)

        if chunking_enabled:
            print(
                f"✓ Индексация завершена. Создано {total_chunks} чанков из {len(articles_df)} статей"
            )
        else:
            print(f"✓ Индексация завершена. Проиндексировано {len(articles_df)} статей")

        return total_chunks

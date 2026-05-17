"""Адаптер Qdrant, реализующий протокол VectorStore."""

from omegaconf import DictConfig
from qdrant_client.models import PointStruct

from russian_laws.qdrant_manager import QdrantManager
from russian_laws.vector_stores.base import Document, SearchHit


class QdrantVectorStore:
    """Тонкая обёртка над `QdrantManager`, нормализующая входы/выходы."""

    backend_name = "qdrant"

    def __init__(self, config: DictConfig):
        self.config = config
        self.manager = QdrantManager(config)
        self.hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)

    def create_collection(self, recreate: bool = False) -> None:
        self.manager.create_collection(recreate=recreate)

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        points: list[PointStruct] = []
        for doc in documents:
            if self.hybrid_enabled and doc.sparse is not None:
                vector_data: dict | list[float] = {
                    "dense": doc.dense,
                    "sparse": {
                        "indices": [idx for idx, _ in doc.sparse],
                        "values": [val for _, val in doc.sparse],
                    },
                }
            else:
                vector_data = doc.dense

            points.append(
                PointStruct(id=doc.id, vector=vector_data, payload=doc.payload)
            )

        self.manager.upsert_points(points)

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        points = self.manager.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            SearchHit(id=p.id, score=float(p.score or 0.0), payload=p.payload or {})
            for p in points
        ]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        points = self.manager.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            SearchHit(id=p.id, score=float(p.score or 0.0), payload=p.payload or {})
            for p in points
        ]

    def close(self) -> None:
        self.manager.close()

"""Адаптер Weaviate v4.

hybrid_search сводится к dense near_vector — Weaviate требует текст запроса
для встроенного BM25, а интерфейс принимает только векторы.
"""

import uuid
from typing import Any

from omegaconf import DictConfig

from russian_laws.vector_stores.base import Document, SearchHit


class WeaviateVectorStore:
    backend_name = "weaviate"

    def __init__(self, config: DictConfig):
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Timeout

        wcfg = config.vector_store
        self.config = config
        self.collection_name = wcfg.collection_name

        self.client = weaviate.connect_to_custom(
            http_host=wcfg.http_host,
            http_port=int(wcfg.http_port),
            http_secure=bool(wcfg.get("http_secure", False)),
            grpc_host=wcfg.get("grpc_host", wcfg.http_host),
            grpc_port=int(wcfg.get("grpc_port", 50051)),
            grpc_secure=bool(wcfg.get("grpc_secure", False)),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            ),
        )

    def _collection(self):
        return self.client.collections.get(self.collection_name)

    def create_collection(self, recreate: bool = False) -> None:
        from weaviate.classes.config import (
            Configure,
            DataType,
            Property,
            VectorDistances,
        )

        if self.client.collections.exists(self.collection_name):
            if recreate:
                self.client.collections.delete(self.collection_name)
            else:
                return

        distance_map = {
            "Cosine": VectorDistances.COSINE,
            "Euclid": VectorDistances.L2_SQUARED,
            "Dot": VectorDistances.DOT,
        }
        distance = distance_map.get(
            self.config.qdrant.get("distance", "Cosine"), VectorDistances.COSINE
        )

        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=distance,
                ef_construction=int(self.config.qdrant.hnsw_config.get("ef_construct", 100)),
                max_connections=int(self.config.qdrant.hnsw_config.get("m", 16)),
            ),
            properties=[
                Property(name="article_id", data_type=DataType.INT),
                Property(name="article_num", data_type=DataType.TEXT),
                Property(name="article_title", data_type=DataType.TEXT),
                Property(name="article_text", data_type=DataType.TEXT),
                Property(name="codex", data_type=DataType.TEXT),
                Property(name="parent_text", data_type=DataType.TEXT),
                Property(name="child_text", data_type=DataType.TEXT),
                Property(name="parent_id", data_type=DataType.INT),
                Property(name="child_idx", data_type=DataType.INT),
            ],
        )

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        coll = self._collection()
        with coll.batch.dynamic() as batch:
            for doc in documents:
                props = {k: v for k, v in doc.payload.items() if v is not None}
                batch.add_object(properties=props, vector=doc.dense, uuid=_to_uuid(doc.id))

    def _hits_from_response(self, response: Any) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for obj in response.objects:
            score = 0.0
            meta = getattr(obj, "metadata", None)
            if meta is not None:
                raw = getattr(meta, "score", None)
                if raw is None:
                    raw = getattr(meta, "distance", 0.0)
                score = float(raw or 0.0)
            payload = dict(obj.properties or {})
            article_id = payload.get("article_id")
            hit_id = int(article_id) if article_id is not None else str(obj.uuid)
            hits.append(SearchHit(id=hit_id, score=score, payload=payload))
        return hits

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,  # noqa: ARG002
    ) -> list[SearchHit]:
        from weaviate.classes.query import MetadataQuery

        effective_limit = (
            limit if limit is not None else int(self.config.qdrant.search.limit)
        )
        response = self._collection().query.near_vector(
            near_vector=query_vector,
            limit=effective_limit,
            return_metadata=MetadataQuery(distance=True, score=True),
        )
        return self._hits_from_response(response)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],  # noqa: ARG002
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        return self.search(dense_vector, limit=limit, score_threshold=score_threshold)

    def close(self) -> None:
        self.client.close()


def _to_uuid(raw_id: int | str) -> str:
    """UUIDv5 из произвольного id (Weaviate требует UUID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_id)))

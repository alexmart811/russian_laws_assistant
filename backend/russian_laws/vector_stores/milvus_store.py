"""Адаптер Milvus 2.4+, реализующий протокол VectorStore."""

from typing import Any

from omegaconf import DictConfig

from russian_laws.vector_stores.base import Document, SearchHit


_PAYLOAD_INT_FIELDS = ("article_id", "child_idx")
_PAYLOAD_STR_FIELDS = (
    "article_num",
    "article_title",
    "article_text",
    "codex",
    "parent_id",
    "parent_text",
    "child_text",
)
_PAYLOAD_FIELDS = _PAYLOAD_INT_FIELDS + _PAYLOAD_STR_FIELDS

# Milvus VARCHAR ограничен 65535 байтами (UTF-8, кириллица — 2 байта на символ)
_MAX_VARCHAR_BYTES = 65000


def _clip_varchar(value: str) -> str:
    """Обрезает строку до безопасного размера в байтах, не ломая UTF-8."""
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_VARCHAR_BYTES:
        return value
    return encoded[:_MAX_VARCHAR_BYTES].decode("utf-8", errors="ignore")


class MilvusVectorStore:
    backend_name = "milvus"

    def __init__(self, config: DictConfig):
        from pymilvus import MilvusClient

        mcfg = config.vector_store
        self.config = config
        self.collection_name = mcfg.collection_name
        self.dim = int(config.qdrant.vector_size)
        self.hybrid_enabled = bool(
            config.qdrant.get("hybrid", {}).get("enabled", False)
        )
        self.rrf_k = int(mcfg.get("rrf_k", 60))

        uri = mcfg.get("uri") or f"http://{mcfg.host}:{mcfg.port}"
        token = mcfg.get("token") or ""
        self.client = MilvusClient(uri=uri, token=token, timeout=120)

    def create_collection(self, recreate: bool = False) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.collection_name):
            if recreate:
                self.client.drop_collection(self.collection_name)
            else:
                return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        if self.hybrid_enabled:
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        for name in _PAYLOAD_FIELDS:
            if name in _PAYLOAD_INT_FIELDS:
                schema.add_field(name, DataType.INT64, nullable=True)
            else:
                schema.add_field(name, DataType.VARCHAR, max_length=65535, nullable=True)

        index_params = self.client.prepare_index_params()
        distance = self.config.qdrant.get("distance", "Cosine")
        metric_map = {"Cosine": "COSINE", "Euclid": "L2", "Dot": "IP"}
        metric = metric_map.get(distance, "COSINE")
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type=metric,
            params={
                "M": int(self.config.qdrant.hnsw_config.get("m", 16)),
                "efConstruction": int(self.config.qdrant.hnsw_config.get("ef_construct", 100)),
            },
        )
        if self.hybrid_enabled:
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        rows: list[dict[str, Any]] = []
        for doc in documents:
            row: dict[str, Any] = {"id": int(doc.id), "dense": doc.dense}
            if self.hybrid_enabled and doc.sparse is not None:
                row["sparse"] = {idx: float(val) for idx, val in doc.sparse}
            # pymilvus требует все объявленные поля — пишем типобезопасные дефолты
            for k in _PAYLOAD_INT_FIELDS:
                val = doc.payload.get(k)
                row[k] = int(val) if val is not None else 0
            for k in _PAYLOAD_STR_FIELDS:
                val = doc.payload.get(k)
                row[k] = _clip_varchar(str(val)) if val is not None else ""
            rows.append(row)

        self.client.upsert(collection_name=self.collection_name, data=rows)

    def _hits_from_results(self, results: list[Any]) -> list[SearchHit]:
        if not results:
            return []
        hits: list[SearchHit] = []
        for hit in results[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            payload = {k: entity.get(k) for k in _PAYLOAD_FIELDS if k in entity}
            hit_id = entity.get("article_id") or hit.get("id")
            score = float(hit.get("distance", 0.0))
            hits.append(SearchHit(id=int(hit_id), score=score, payload=payload))
        return hits

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,  # noqa: ARG002
    ) -> list[SearchHit]:
        effective_limit = (
            limit if limit is not None else int(self.config.qdrant.search.limit)
        )
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense",
            limit=effective_limit,
            output_fields=list(_PAYLOAD_FIELDS),
        )
        return self._hits_from_results(results)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],
        limit: int | None = None,
        score_threshold: float | None = None,  # noqa: ARG002
    ) -> list[SearchHit]:
        from pymilvus import AnnSearchRequest, RRFRanker

        if not self.hybrid_enabled:
            return self.search(dense_vector, limit=limit)

        effective_limit = (
            limit if limit is not None else int(self.config.qdrant.search.limit)
        )

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=effective_limit * 3,
        )
        sparse_req = AnnSearchRequest(
            data=[{idx: float(val) for idx, val in sparse_vector}],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=effective_limit * 3,
        )
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=self.rrf_k),
            limit=effective_limit,
            output_fields=list(_PAYLOAD_FIELDS),
        )
        return self._hits_from_results(results)

    def close(self) -> None:
        self.client.close()

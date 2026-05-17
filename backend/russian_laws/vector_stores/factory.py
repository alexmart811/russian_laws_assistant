"""Фабрика векторных хранилищ — выбирает backend по конфигу."""

from omegaconf import DictConfig

from russian_laws.vector_stores.base import VectorStore


def make_vector_store(config: DictConfig) -> VectorStore:
    """Создаёт VectorStore на основании `config.vector_store.backend`.

    Если секция `vector_store` отсутствует — поведение по умолчанию: Qdrant.
    """
    backend = "qdrant"
    vs_cfg = config.get("vector_store") if hasattr(config, "get") else None
    if vs_cfg is not None:
        backend = vs_cfg.get("backend", "qdrant")

    if backend == "qdrant":
        from russian_laws.vector_stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(config)
    if backend == "weaviate":
        from russian_laws.vector_stores.weaviate_store import WeaviateVectorStore

        return WeaviateVectorStore(config)
    if backend == "milvus":
        from russian_laws.vector_stores.milvus_store import MilvusVectorStore

        return MilvusVectorStore(config)

    raise ValueError(f"Unknown vector_store.backend: {backend}")

"""Абстракция векторного хранилища для сравнения latency между Qdrant/Weaviate/Milvus."""

from russian_laws.vector_stores.base import Document, SearchHit, VectorStore
from russian_laws.vector_stores.factory import make_vector_store

__all__ = ["Document", "SearchHit", "VectorStore", "make_vector_store"]

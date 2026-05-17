from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchHit:
    id: int | str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """sparse — пары (token_id, weight) для гибридного поиска."""
    id: int | str
    dense: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    sparse: list[tuple[int, float]] | None = None


@runtime_checkable
class VectorStore(Protocol):
    def create_collection(self, recreate: bool = False) -> None: ...

    def upsert_documents(self, documents: list[Document]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]: ...

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]: ...

    def close(self) -> None: ...

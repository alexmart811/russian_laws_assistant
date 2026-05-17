from typing import Any

from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.collection_name = config.qdrant.collection_name
        self.vector_size = config.qdrant.vector_size

        self.client = QdrantClient(
            url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=120
        )

        self.distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

    def create_collection(self, recreate: bool = False) -> None:
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists and recreate:
            print(f"Удаление существующей коллекции '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            print(f"Создание коллекции '{self.collection_name}'...")
            distance = self.distance_map.get(
                self.config.qdrant.distance, Distance.COSINE
            )
            hybrid_enabled = self.config.qdrant.get("hybrid", {}).get("enabled", False)

            if hybrid_enabled:
                sparse_modifier = self.config.qdrant.hybrid.get(
                    "sparse_modifier", "idf"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_size,
                            distance=distance,
                            on_disk=self.config.qdrant.on_disk,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=sparse_modifier,
                        )
                    },
                    hnsw_config=self.config.qdrant.hnsw_config,
                )
                print(f"Коллекция '{self.collection_name}' создана (hybrid: dense + sparse)")
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=distance,
                        on_disk=self.config.qdrant.on_disk,
                    ),
                    hnsw_config=self.config.qdrant.hnsw_config,
                )
                print(f"Коллекция '{self.collection_name}' создана (только dense)")
        else:
            print(f"Коллекция '{self.collection_name}' уже существует")

    def get_collection_info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_cfg = info.config.params.vectors
            vector_size = getattr(vectors_cfg, "size", None)
            return {
                "name": self.collection_name,
                "vector_size": vector_size,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            return {"error": str(e)}

    def upsert_points(self, points: list[PointStruct]) -> None:
        if not points:
            print("Нет точек для добавления")
            return
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Добавлено {len(points)} точек в коллекцию")

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_dict: Filter | None = None,
    ) -> list[Any]:
        effective_limit: int = (
            limit if limit is not None else int(self.config.qdrant.search.limit)
        )
        effective_threshold: float = (
            score_threshold
            if score_threshold is not None
            else float(self.config.qdrant.search.score_threshold)
        )
        hybrid_enabled = self.config.qdrant.get("hybrid", {}).get("enabled", False)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense" if hybrid_enabled else None,
            limit=effective_limit,
            score_threshold=effective_threshold,
            query_filter=filter_dict,
        )
        return results.points

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_dict: Filter | None = None,
    ) -> list[Any]:
        effective_limit: int = (
            limit if limit is not None else int(self.config.qdrant.search.limit)
        )
        effective_threshold: float = (
            score_threshold
            if score_threshold is not None
            else float(self.config.qdrant.search.score_threshold)
        )

        fusion_strategy = self.config.qdrant.hybrid.get(
            "fusion_strategy", "weighted_sum"
        )

        sparse_indices = [idx for idx, _ in sparse_vector]
        sparse_values = [val for _, val in sparse_vector]
        sparse_query_obj = SparseVector(indices=sparse_indices, values=sparse_values)

        if fusion_strategy == "union_rerank":
            k_per_method: int = int(
                self.config.qdrant.hybrid.get("candidates_per_method", effective_limit)
            )

            dense_results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=k_per_method,
                score_threshold=0.0,
                query_filter=filter_dict,
            )
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query_obj,
                using="sparse",
                limit=k_per_method,
                score_threshold=0.0,
                query_filter=filter_dict,
            )

            points_map: dict[Any, Any] = {}
            for point in dense_results.points:
                points_map[point.id] = point
                point.score = float(point.score) if hasattr(point, "score") else 0.0
            for point in sparse_results.points:
                if point.id not in points_map:
                    point.score = float(point.score) if hasattr(point, "score") else 0.0
                    points_map[point.id] = point

            return list(points_map.values())

        else:
            alpha = self.config.qdrant.hybrid.get("alpha", 0.5)

            dense_results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=effective_limit * 3,
                score_threshold=0.0,
                query_filter=filter_dict,
            )
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query_obj,
                using="sparse",
                limit=effective_limit * 3,
                score_threshold=0.0,
                query_filter=filter_dict,
            )

            dense_scores_map: dict[Any, float] = {
                p.id: float(p.score) if hasattr(p, "score") else 0.0
                for p in dense_results.points
            }
            sparse_scores_map: dict[Any, float] = {
                p.id: float(p.score) if hasattr(p, "score") else 0.0
                for p in sparse_results.points
            }

            sparse_scores_normalized = dict(
                zip(sparse_scores_map.keys(), self._normalize_scores(list(sparse_scores_map.values())))
            ) if sparse_scores_map else {}

            dense_scores_normalized = dict(
                zip(dense_scores_map.keys(), self._normalize_scores(list(dense_scores_map.values())))
            ) if dense_scores_map else {}

            all_point_ids = set(dense_scores_map.keys()) | set(sparse_scores_map.keys())
            combined_scores: dict[Any, float] = {
                pid: alpha * dense_scores_normalized.get(pid, 0.0)
                     + (1 - alpha) * sparse_scores_normalized.get(pid, 0.0)
                for pid in all_point_ids
            }

            points_map_ws: dict[Any, Any] = {p.id: p for p in dense_results.points}
            points_map_ws.update({p.id: p for p in sparse_results.points})

            final_results = []
            for score, point in sorted(
                [(score, points_map_ws[pid]) for pid, score in combined_scores.items()],
                key=lambda x: x[0],
                reverse=True,
            ):
                if score < effective_threshold:
                    break
                point.score = score
                final_results.append(point)
                if len(final_results) >= effective_limit:
                    break

            return final_results

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        print(f"Коллекция '{self.collection_name}' удалена")

    def close(self) -> None:
        self.client.close()

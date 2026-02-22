"""Модуль для работы с Qdrant векторной базой данных."""

from typing import Any

from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantManager:
    """Менеджер для работы с Qdrant."""

    def __init__(self, config: DictConfig):
        """Инициализация клиента Qdrant.

        Args:
            config: Конфигурация Hydra с параметрами подключения
        """
        self.config = config
        self.collection_name = config.qdrant.collection_name
        self.vector_size = config.qdrant.vector_size

        # Создаем клиента
        self.client = QdrantClient(
            url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=120
        )

        # Маппинг типов расстояний
        self.distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

    def create_collection(self, recreate: bool = False) -> None:
        """Создает коллекцию в Qdrant с поддержкой гибридного поиска.

        Args:
            recreate: Пересоздать коллекцию, если она существует
        """
        # Проверяем существование коллекции
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
                # Создаем коллекцию с поддержкой dense + sparse векторов
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
                print(
                    f"Коллекция '{self.collection_name}' создана (hybrid: dense + sparse)"
                )
            else:
                # Создаем коллекцию только с dense векторами (legacy)
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
        """Получает информацию о коллекции.

        Returns:
            Словарь с информацией о коллекции
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            return {"error": str(e)}

    def upsert_points(self, points: list[PointStruct]) -> None:
        """Добавляет или обновляет точки в коллекции.

        Args:
            points: Список точек для добавления
        """
        if not points:
            print("Нет точек для добавления")
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Добавлено {len(points)} точек в коллекцию")

    def search(
        self,
        query_vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_dict: dict | None = None,
    ) -> list[Any]:
        """Выполняет поиск по dense векторам (legacy).

        Args:
            query_vector: Вектор запроса
            limit: Количество результатов
            score_threshold: Порог схожести
            filter_dict: Фильтры для поиска

        Returns:
            Список найденных точек с метаданными
        """
        limit = limit or self.config.qdrant.search.limit
        score_threshold = score_threshold or self.config.qdrant.search.score_threshold

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_dict,
        )

        return results.points

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Нормализует скоры в диапазон [0, 1] с помощью min-max нормализации.

        Args:
            scores: Список скоров

        Returns:
            Нормализованные скоры
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Если все скоры одинаковые
        if max_score == min_score:
            return [1.0] * len(scores)

        # Min-max нормализация
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[tuple[int, float]],
        limit: int | None = None,
        score_threshold: float | None = None,
        filter_dict: dict | None = None,
    ) -> list[Any]:
        """Выполняет гибридный поиск (dense + sparse) с нормализацией скоров.

        Args:
            dense_vector: Dense вектор запроса
            sparse_vector: Sparse вектор запроса (список пар (token_id, weight))
            limit: Количество результатов
            score_threshold: Порог схожести
            filter_dict: Фильтры для поиска

        Returns:
            Список найденных точек с метаданными
        """
        limit = limit or self.config.qdrant.search.limit
        score_threshold = score_threshold or self.config.qdrant.search.score_threshold
        alpha = self.config.qdrant.hybrid.get("alpha", 0.5)

        # Преобразуем sparse вектор в формат Qdrant
        sparse_indices = [idx for idx, _ in sparse_vector]
        sparse_values = [val for _, val in sparse_vector]

        # 1. Dense поиск
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=limit * 3,
            score_threshold=0.0,
            query_filter=filter_dict,
        )

        # 2. Sparse поиск
        sparse_query_obj = SparseVector(indices=sparse_indices, values=sparse_values)
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_query_obj,
            using="sparse",
            limit=limit * 3,
            score_threshold=0.0,
            query_filter=filter_dict,
        )

        # 3. Собираем все скоры для нормализации
        dense_scores_map: dict[int, float] = {}
        sparse_scores_map: dict[int, float] = {}

        for point in dense_results.points:
            dense_scores_map[point.id] = (
                float(point.score) if hasattr(point, "score") else 0.0
            )

        for point in sparse_results.points:
            sparse_scores_map[point.id] = (
                float(point.score) if hasattr(point, "score") else 0.0
            )

        # 4. Нормализуем sparse скоры в диапазон [0, 1]
        sparse_scores_list = list(sparse_scores_map.values())
        if sparse_scores_list:
            normalized_sparse_scores = self._normalize_scores(sparse_scores_list)
            sparse_scores_normalized = dict(
                zip(sparse_scores_map.keys(), normalized_sparse_scores)
            )
        else:
            sparse_scores_normalized = {}

        # Dense скоры уже нормализованы (косинусное сходство в [0, 1])
        # Но для консистентности можем их тоже нормализовать
        dense_scores_list = list(dense_scores_map.values())
        if dense_scores_list:
            normalized_dense_scores = self._normalize_scores(dense_scores_list)
            dense_scores_normalized = dict(
                zip(dense_scores_map.keys(), normalized_dense_scores)
            )
        else:
            dense_scores_normalized = {}

        # 5. Комбинируем нормализованные скоры
        all_point_ids = set(dense_scores_map.keys()) | set(sparse_scores_map.keys())
        combined_scores: dict[int, float] = {}

        for point_id in all_point_ids:
            dense_norm = dense_scores_normalized.get(point_id, 0.0)
            sparse_norm = sparse_scores_normalized.get(point_id, 0.0)

            combined_score = alpha * dense_norm + (1 - alpha) * sparse_norm
            combined_scores[point_id] = combined_score

        # 6. Получаем точки и сортируем
        points_map = {p.id: p for p in dense_results.points}
        points_map.update({p.id: p for p in sparse_results.points})

        sorted_results = sorted(
            [(score, points_map[pid]) for pid, score in combined_scores.items()],
            key=lambda x: x[0],
            reverse=True,
        )

        # 7. Применяем порог и лимит
        final_results = []
        for score, point in sorted_results:
            if score >= score_threshold:
                point.score = score
                final_results.append(point)
                if len(final_results) >= limit:
                    break

        return final_results

    def delete_collection(self) -> None:
        """Удаляет коллекцию."""
        self.client.delete_collection(self.collection_name)
        print(f"Коллекция '{self.collection_name}' удалена")

    def close(self) -> None:
        """Закрывает соединение с Qdrant."""
        self.client.close()

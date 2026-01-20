"""Модуль для работы с Qdrant векторной базой данных."""

from typing import Any

from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


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
        """Создает коллекцию в Qdrant.

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

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=distance,
                    on_disk=self.config.qdrant.on_disk,
                ),
                hnsw_config=self.config.qdrant.hnsw_config,
            )
            print(f"Коллекция '{self.collection_name}' создана")
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
        """Выполняет поиск похожих векторов.

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
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_dict,
        )

        return results.points

    def delete_collection(self) -> None:
        """Удаляет коллекцию."""
        self.client.delete_collection(self.collection_name)
        print(f"Коллекция '{self.collection_name}' удалена")

    def close(self) -> None:
        """Закрывает соединение с Qdrant."""
        self.client.close()

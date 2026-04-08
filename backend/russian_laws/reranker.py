"""Модуль для переранжирования результатов поиска."""

from omegaconf import DictConfig
from sentence_transformers import CrossEncoder


class Reranker:
    """Reranker для улучшения качества retrieval."""

    def __init__(self, config: DictConfig):
        """Инициализация reranker.

        Args:
            config: Конфигурация Hydra с параметрами reranker
        """
        self.config = config
        self.model_name = config.reranker.model_name
        self.device = config.reranker.get("device", "cuda")
        self.top_k = config.reranker.get("top_k", 5)

        print(f"Загрузка reranker: {self.model_name}...")
        self.model = CrossEncoder(self.model_name, device=self.device)
        print(f"✓ Reranker загружен на {self.device}")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[int]:
        """Переранжирует документы по релевантности к запросу.

        Args:
            query: Запрос пользователя
            documents: Список текстов документов для переранжирования
            top_k: Количество документов для возврата (если None, берется из конфига)

        Returns:
            Список индексов документов, отсортированных по релевантности (от лучшего к худшему)
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Формируем пары (query, document) для reranker
        pairs = [[query, doc] for doc in documents]

        # Получаем scores от reranker
        scores = self.model.predict(pairs)

        # Сортируем индексы по убыванию score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        # Возвращаем top_k индексов
        return ranked_indices[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Переранжирует документы и возвращает индексы со scores.

        Args:
            query: Запрос пользователя
            documents: Список текстов документов
            top_k: Количество документов для возврата

        Returns:
            Список кортежей (индекс, score) отсортированный по score
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Формируем пары (query, document)
        pairs = [[query, doc] for doc in documents]

        # Получаем scores
        scores = self.model.predict(pairs)

        # Сортируем и возвращаем с scores
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

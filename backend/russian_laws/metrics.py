"""Метрики для оценки качества retrieval моделей."""

import numpy as np


def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    """Вычисляет Recall@k.

    Recall@k показывает, какая доля релевантных документов была найдена
    в топ-k результатах.

    Args:
        relevant_ids: Список ID релевантных документов
        retrieved_ids: Список ID найденных документов (отсортированы по релевантности)
        k: Количество топовых результатов для рассмотрения

    Returns:
        Recall@k значение от 0 до 1
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    relevant_ids: list[int], retrieved_ids: list[int], k: int | None = None
) -> float:
    """Вычисляет Mean Reciprocal Rank (MRR).

    MRR - это обратный ранг первого релевантного документа в результатах поиска.
    Значение 1.0 означает, что первый результат релевантен.

    Args:
        relevant_ids: Список ID релевантных документов
        retrieved_ids: Список ID найденных документов (отсортированы по релевантности)
        k: Максимальное количество результатов для рассмотрения (опционально)

    Returns:
        MRR значение от 0 до 1
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Ограничиваем поиск первыми k результатами, если k задан
    search_list = retrieved_ids[:k] if k else retrieved_ids

    for rank, doc_id in enumerate(search_list, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    """Вычисляет Normalized Discounted Cumulative Gain (nDCG@k).

    nDCG@k оценивает качество ранжирования, учитывая позицию релевантных документов.
    Документы на более высоких позициях получают больший вес.

    Args:
        relevant_ids: Список ID релевантных документов
        retrieved_ids: Список ID найденных документов (отсортированы по релевантности)
        k: Количество топовых результатов для рассмотрения

    Returns:
        nDCG@k значение от 0 до 1
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            # Релевантность = 1 если документ релевантен, иначе 0
            relevance = 1.0
            # Дисконтирование по логарифмической шкале
            dcg += relevance / np.log2(i + 1)

    idcg = 0.0
    for i in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class RetrievalMetrics:
    """Класс для вычисления и агрегации метрик retrieval."""

    def __init__(self, k_values: list[int] | None = None):
        """Инициализация метрик.

        Args:
            k_values: Список значений k для вычисления метрик (по умолчанию [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.reset()

    def reset(self) -> None:
        """Сбрасывает накопленные метрики."""
        self.recall_scores = {k: [] for k in self.k_values}
        self.ndcg_scores = {k: [] for k in self.k_values}
        self.mrr_scores = {k: [] for k in self.k_values}

    def update(self, relevant_ids: list[int], retrieved_ids: list[int]) -> None:
        """Обновляет метрики для одного запроса.

        Args:
            relevant_ids: Список ID релевантных документов
            retrieved_ids: Список ID найденных документов
        """
        for k in self.k_values:
            # Recall@k
            recall = recall_at_k(relevant_ids, retrieved_ids, k)
            self.recall_scores[k].append(recall)

            # nDCG@k
            ndcg = ndcg_at_k(relevant_ids, retrieved_ids, k)
            self.ndcg_scores[k].append(ndcg)

            # MRR@k
            mrr = mean_reciprocal_rank(relevant_ids, retrieved_ids, k)
            self.mrr_scores[k].append(mrr)

    def compute(self) -> dict[str, float]:
        """Вычисляет средние значения метрик.

        Returns:
            Словарь с усредненными метриками
        """
        results = {}

        for k in self.k_values:
            # Средний Recall@k
            if self.recall_scores[k]:
                results[f"recall@{k}"] = np.mean(self.recall_scores[k])

            # Средний nDCG@k
            if self.ndcg_scores[k]:
                results[f"ndcg@{k}"] = np.mean(self.ndcg_scores[k])

            # Средний MRR@k
            if self.mrr_scores[k]:
                results[f"mrr@{k}"] = np.mean(self.mrr_scores[k])

        return results

    def compute_detailed(self) -> dict[str, dict]:
        """Вычисляет детальную статистику метрик.

        Returns:
            Словарь с детальной статистикой (mean, std, min, max)
        """
        results = {}

        for k in self.k_values:
            # Recall@k
            if self.recall_scores[k]:
                results[f"recall@{k}"] = {
                    "mean": np.mean(self.recall_scores[k]),
                    "std": np.std(self.recall_scores[k]),
                    "min": np.min(self.recall_scores[k]),
                    "max": np.max(self.recall_scores[k]),
                }

            # nDCG@k
            if self.ndcg_scores[k]:
                results[f"ndcg@{k}"] = {
                    "mean": np.mean(self.ndcg_scores[k]),
                    "std": np.std(self.ndcg_scores[k]),
                    "min": np.min(self.ndcg_scores[k]),
                    "max": np.max(self.ndcg_scores[k]),
                }

            # MRR@k
            if self.mrr_scores[k]:
                results[f"mrr@{k}"] = {
                    "mean": np.mean(self.mrr_scores[k]),
                    "std": np.std(self.mrr_scores[k]),
                    "min": np.min(self.mrr_scores[k]),
                    "max": np.max(self.mrr_scores[k]),
                }

        return results

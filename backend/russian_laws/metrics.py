"""Метрики для оценки качества retrieval моделей и RAG системы."""

import numpy as np
import openai
from omegaconf import DictConfig


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


class RAGMetrics:
    """Собственные метрики для оценки RAG системы (без RAGAS).

    Метрики:
    - Faithfulness: насколько ответ основан на контексте (LLM-as-judge)
    - Answer Relevance: семантическая близость ответа к вопросу (embeddings)
    """

    FAITHFULNESS_PROMPT = """Ты — эксперт по оценке качества ответов.

Контекст (документы, на основе которых был сгенерирован ответ):
{context}

---

Вопрос пользователя: {question}

Ответ системы: {answer}

---

Оцени, насколько ответ ПОЛЕЗЕН и ОСНОВАН на предоставленном контексте:
- 1.0 = Ответ полностью отвечает на вопрос и основан на контексте
- 0.5 = Ответ частично отвечает на вопрос или содержит незначительные домыслы
- 0.0 = Ответ НЕ отвечает на вопрос (включая "информация отсутствует", "не найдено" и т.п.) ИЛИ содержит галлюцинации

Ответь ТОЛЬКО одним числом: 0.0, 0.5 или 1.0"""

    def __init__(self, config: DictConfig):
        """Инициализация RAG метрик.

        Args:
            config: Конфигурация с параметрами LLM и embeddings
        """
        self.config = config
        self.samples: list[dict] = []

        # Клиент для LLM (faithfulness)
        self.llm_client = openai.OpenAI(
            api_key=config.ragas.llm.api_key,
            base_url=config.ragas.llm.base_url,
        )
        self.llm_model = config.ragas.llm.model

        # Клиент для embeddings (answer relevance)
        self.embedding_client = openai.OpenAI(
            api_key=config.ragas.llm.api_key,
            base_url=config.ragas.llm.base_url,
        )
        self.embedding_model = config.ragas.get(
            "embedding_model", "openai/text-embedding-3-small"
        )

    def update(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> None:
        """Добавляет один сэмпл для оценки."""
        self.samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Получает embeddings для списка текстов (батч)."""
        if not texts:
            return []
        response = self.embedding_client.embeddings.create(
            input=texts,
            model=self.embedding_model,
        )
        return [item.embedding for item in response.data]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _compute_faithfulness_single(self, sample: dict) -> float:
        """Вычисляет faithfulness для одного сэмпла через LLM."""
        question = sample["question"]
        answer = sample["answer"]
        contexts = sample["contexts"]

        context_text = "\n\n---\n\n".join(contexts[:5])  # Ограничиваем до 5 контекстов

        prompt = self.FAITHFULNESS_PROMPT.format(
            context=context_text,
            question=question,
            answer=answer,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            content = response.choices[0].message.content
            if content is None:
                return 0.5
            result_text = content.strip()

            # Парсим число из ответа
            for val in ["1.0", "0.5", "0.0", "1", "0"]:
                if val in result_text:
                    return float(val)

            return 0.5

        except Exception as e:
            print(f"[RAG Metrics] Ошибка faithfulness: {e}")
            return 0.0

    def compute(self) -> dict[str, float]:
        """Вычисляет метрики для всех сэмплов (параллельно)."""
        if not self.samples:
            return {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        print(f"\n{'=' * 60}")
        print(f"ВЫЧИСЛЕНИЕ RAG МЕТРИК ({len(self.samples)} сэмплов)")
        print(f"{'=' * 60}")

        # 1. Батчинг embeddings — один запрос вместо N*2
        print("[RAG Metrics] Получение embeddings (батч)...")
        all_texts = []
        for s in self.samples:
            all_texts.append(s["question"])
            all_texts.append(s["answer"])

        try:
            all_embeddings = self._get_embeddings_batch(all_texts)

            # Вычисляем answer relevance из батча
            relevance_scores = []
            for i in range(len(self.samples)):
                q_emb = all_embeddings[i * 2]
                a_emb = all_embeddings[i * 2 + 1]
                rel_score = self._cosine_similarity(q_emb, a_emb)
                relevance_scores.append(rel_score)
        except Exception as e:
            print(f"[RAG Metrics] Ошибка batch embeddings: {e}")
            relevance_scores = [0.0] * len(self.samples)

        # 2. Параллельные LLM вызовы для faithfulness
        print("[RAG Metrics] Вычисление faithfulness (параллельно)...")
        faithfulness_scores = [0.0] * len(self.samples)

        max_workers = min(10, len(self.samples))  # Не более 10 параллельных запросов

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._compute_faithfulness_single, sample): idx
                for idx, sample in enumerate(self.samples)
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(self.samples),
                desc="Faithfulness",
            ):
                idx = future_to_idx[future]
                try:
                    faithfulness_scores[idx] = future.result()
                except Exception as e:
                    print(f"[RAG Metrics] Ошибка в потоке {idx}: {e}")
                    faithfulness_scores[idx] = 0.0

        # Средние значения
        avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

        print(f"\n[RAG Metrics] Faithfulness scores: {faithfulness_scores}")
        print(
            f"[RAG Metrics] Answer Relevance scores: {[f'{s:.3f}' for s in relevance_scores]}"
        )
        print(f"[RAG Metrics] Avg Faithfulness: {avg_faithfulness:.4f}")
        print(f"[RAG Metrics] Avg Answer Relevance: {avg_relevance:.4f}")

        return {
            "faithfulness": float(avg_faithfulness),
            "answer_relevance": float(avg_relevance),
        }

    def reset(self) -> None:
        """Сбрасывает накопленные сэмплы."""
        self.samples = []


# Алиас для обратной совместимости
RAGASMetrics = RAGMetrics

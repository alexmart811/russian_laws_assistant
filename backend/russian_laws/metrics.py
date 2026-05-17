import re
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np
import openai
from omegaconf import DictConfig

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _prf(matches: int, pred_count: int, ref_count: int) -> tuple[float, float, float]:
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0, 0.0
    precision = matches / pred_count
    recall = matches / ref_count
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    """ROUGE-N (precision, recall, F1) на уровне n-грамм."""
    pred_ngrams = _ngrams(_tokenize(prediction), n)
    ref_ngrams = _ngrams(_tokenize(reference), n)

    if not pred_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_ngrams)
    ref_counts = Counter(ref_ngrams)
    matches = sum((pred_counts & ref_counts).values())
    p, r, f1 = _prf(matches, sum(pred_counts.values()), sum(ref_counts.values()))
    return {"precision": p, "recall": r, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """LCS через rolling-array DP — O(len(a)*len(b)) время, O(min) память."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len(b)]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-L на основе LCS."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    p, r, f1 = _prf(lcs, len(pred_tokens), len(ref_tokens))
    return {"precision": p, "recall": r, "f1": f1}


class RougeMetrics:
    """Агрегатор ROUGE-1, ROUGE-2, ROUGE-L."""

    METRICS = ("rouge1", "rouge2", "rougeL")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.scores: dict[str, dict[str, list[float]]] = {
            m: {"precision": [], "recall": [], "f1": []} for m in self.METRICS
        }

    def update(self, prediction: str, reference: str) -> None:
        r1 = rouge_n(prediction, reference, 1)
        r2 = rouge_n(prediction, reference, 2)
        rl = rouge_l(prediction, reference)
        for name, vals in (("rouge1", r1), ("rouge2", r2), ("rougeL", rl)):
            for key, value in vals.items():
                self.scores[name][key].append(value)

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for name in self.METRICS:
            for key in ("precision", "recall", "f1"):
                values = self.scores[name][key]
                if values:
                    results[f"{name}_{key}"] = float(np.mean(values))
        return results


def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(
    relevant_ids: list[int], retrieved_ids: list[int], k: int | None = None
) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    search_list = retrieved_ids[:k] if k else retrieved_ids
    for rank, doc_id in enumerate(search_list, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)

    dcg = sum(
        1.0 / np.log2(i + 1)
        for i, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_set
    )
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg else 0.0


class RetrievalMetrics:
    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 3, 5, 10]
        self.reset()

    def reset(self) -> None:
        self.recall_scores = {k: [] for k in self.k_values}
        self.ndcg_scores = {k: [] for k in self.k_values}
        self.mrr_scores = {k: [] for k in self.k_values}

    def update(self, relevant_ids: list[int], retrieved_ids: list[int]) -> None:
        for k in self.k_values:
            self.recall_scores[k].append(recall_at_k(relevant_ids, retrieved_ids, k))
            self.ndcg_scores[k].append(ndcg_at_k(relevant_ids, retrieved_ids, k))
            self.mrr_scores[k].append(mean_reciprocal_rank(relevant_ids, retrieved_ids, k))

    def compute(self) -> dict[str, float]:
        results = {}
        for k in self.k_values:
            if self.recall_scores[k]:
                results[f"recall@{k}"] = np.mean(self.recall_scores[k])
            if self.ndcg_scores[k]:
                results[f"ndcg@{k}"] = np.mean(self.ndcg_scores[k])
            if self.mrr_scores[k]:
                results[f"mrr@{k}"] = np.mean(self.mrr_scores[k])
        return results

    def compute_detailed(self) -> dict[str, dict]:
        results = {}
        for k in self.k_values:
            for prefix, scores in (
                (f"recall@{k}", self.recall_scores[k]),
                (f"ndcg@{k}", self.ndcg_scores[k]),
                (f"mrr@{k}", self.mrr_scores[k]),
            ):
                if scores:
                    results[prefix] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                    }
        return results


class RAGMetrics:
    """Оценка RAG: faithfulness (LLM-as-judge) и answer relevance (embeddings)."""

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
        self.config = config
        self.samples: list[dict] = []

        self.llm_client = openai.OpenAI(
            api_key=config.ragas.llm.api_key,
            base_url=config.ragas.llm.base_url,
        )
        self.llm_model = config.ragas.llm.model

        self.embedding_client = openai.OpenAI(
            api_key=config.ragas.llm.api_key,
            base_url=config.ragas.llm.base_url,
        )
        self.embedding_model = config.ragas.get(
            "embedding_model", "openai/text-embedding-3-small"
        )

    def update(self, question: str, answer: str, contexts: list[str]) -> None:
        self.samples.append({"question": question, "answer": answer, "contexts": contexts})

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.embedding_client.embeddings.create(
            input=texts, model=self.embedding_model
        )
        return [item.embedding for item in response.data]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _compute_faithfulness_single(self, sample: dict) -> float:
        context_text = "\n\n---\n\n".join(sample["contexts"][:5])
        prompt = self.FAITHFULNESS_PROMPT.format(
            context=context_text,
            question=sample["question"],
            answer=sample["answer"],
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
            for val in ["1.0", "0.5", "0.0", "1", "0"]:
                if val in content.strip():
                    return float(val)
            return 0.5
        except Exception as e:
            print(f"[RAG Metrics] Ошибка faithfulness: {e}")
            return 0.0

    def compute(self) -> dict[str, float]:
        if not self.samples:
            return {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        print(f"\n{'=' * 60}")
        print(f"ВЫЧИСЛЕНИЕ RAG МЕТРИК ({len(self.samples)} сэмплов)")
        print(f"{'=' * 60}")

        print("[RAG Metrics] Получение embeddings (батч)...")
        all_texts = [t for s in self.samples for t in (s["question"], s["answer"])]
        try:
            all_embeddings = self._get_embeddings_batch(all_texts)
            relevance_scores = [
                self._cosine_similarity(all_embeddings[i * 2], all_embeddings[i * 2 + 1])
                for i in range(len(self.samples))
            ]
        except Exception as e:
            print(f"[RAG Metrics] Ошибка batch embeddings: {e}")
            relevance_scores = [0.0] * len(self.samples)

        print("[RAG Metrics] Вычисление faithfulness (параллельно)...")
        faithfulness_scores = [0.0] * len(self.samples)
        with ThreadPoolExecutor(max_workers=min(10, len(self.samples))) as executor:
            future_to_idx = {
                executor.submit(self._compute_faithfulness_single, sample): idx
                for idx, sample in enumerate(self.samples)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(self.samples), desc="Faithfulness"):
                idx = future_to_idx[future]
                try:
                    faithfulness_scores[idx] = future.result()
                except Exception as e:
                    print(f"[RAG Metrics] Ошибка в потоке {idx}: {e}")

        avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

        print(f"[RAG Metrics] Avg Faithfulness: {avg_faithfulness:.4f}")
        print(f"[RAG Metrics] Avg Answer Relevance: {avg_relevance:.4f}")

        return {
            "faithfulness": float(avg_faithfulness),
            "answer_relevance": float(avg_relevance),
        }

    def reset(self) -> None:
        self.samples = []


class LatencyMetrics:
    """Накапливает замеры времени и считает mean, median, p95, p99, min, max."""

    def __init__(self, name: str = "llm"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.latencies: list[float] = []

    def update(self, seconds: float) -> None:
        self.latencies.append(float(seconds))

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.update(time.perf_counter() - start)

    def compute(self) -> dict[str, float]:
        if not self.latencies:
            return {}
        arr = np.asarray(self.latencies, dtype=np.float64)
        prefix = f"latency_{self.name}"
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_p95": float(np.percentile(arr, 95)),
            f"{prefix}_p99": float(np.percentile(arr, 99)),
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_total": float(np.sum(arr)),
            f"{prefix}_count": int(arr.size),
        }


RAGASMetrics = RAGMetrics

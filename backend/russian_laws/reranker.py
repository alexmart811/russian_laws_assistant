from omegaconf import DictConfig
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, config: DictConfig):
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
        """Возвращает индексы документов, отсортированных по релевантности."""
        if not documents:
            return []

        top_k = top_k or self.top_k
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Возвращает список (индекс, score), отсортированный по убыванию."""
        if not documents:
            return []

        top_k = top_k or self.top_k
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

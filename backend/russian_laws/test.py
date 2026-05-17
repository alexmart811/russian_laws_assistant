from pathlib import Path

import fire
import pandas as pd
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from russian_laws.embeddings import EmbeddingModel
from russian_laws.generator import LLMGenerator
from russian_laws.indexer import ArticleIndexer
from russian_laws.metrics import (
    LatencyMetrics,
    RAGASMetrics,
    RetrievalMetrics,
    RougeMetrics,
)
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder
from russian_laws.vector_stores import VectorStore, make_vector_store
from torch.utils.data import DataLoader, Dataset


class QueryDataset(Dataset):
    def __init__(self, queries_df: pd.DataFrame):
        self.queries_df = queries_df

    def __len__(self) -> int:
        return len(self.queries_df)

    def __getitem__(self, idx: int) -> dict:
        # relevant_ids передаётся как pipe-separated строка — DataLoader
        # не поддерживает списки переменной длины в батче
        row = self.queries_df.iloc[idx]
        if "relevant_ids" in self.queries_df.columns and pd.notna(row.get("relevant_ids")):
            relevant_ids = str(row["relevant_ids"])
            article_id = int(relevant_ids.split("|")[0])
        else:
            article_id = int(row["article_id"])
            relevant_ids = str(article_id)
        return {
            "query_text": row["query_text"],
            "article_id": article_id,
            "relevant_ids": relevant_ids,
        }


class RetrievalTester(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
        model_name: str,
        k_values: list[int] | None = None,
    ):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.k_values = k_values or [1, 3, 5, 10, 20]

        vs_cfg_init = config.get("vector_store") if hasattr(config, "get") else None
        backend_init = (
            vs_cfg_init.get("backend", "qdrant") if vs_cfg_init is not None else "qdrant"
        )

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "k_values": self.k_values,
                "qdrant_collection": config.qdrant.collection_name,
                "vector_size": config.qdrant.vector_size,
                "vector_backend": backend_init,
            }
        )

        self.embedding_model = None
        self.sparse_encoder = None
        self.qdrant_manager = None
        self.vector_store: VectorStore | None = None
        self.reranker = None
        self.llm_generator = None
        self.metrics = RetrievalMetrics(k_values=self.k_values)
        self.ragas_metrics = None
        self.rouge_metrics = RougeMetrics()
        self.article_texts: dict[int, str] = {}
        self.llm_latency = LatencyMetrics(name="llm")
        self.retrieval_latency = LatencyMetrics(name="retrieval")
        self.hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
        self.reranker_enabled = config.reranker.get("enabled", False)
        self.ragas_enabled = config.ragas.get("enabled", False)
        vs_cfg = config.get("vector_store") if hasattr(config, "get") else None
        self.vector_backend = (
            vs_cfg.get("backend", "qdrant") if vs_cfg is not None else "qdrant"
        )

    def setup(self, stage: str | None = None) -> None:
        if stage != "test" and stage is not None:
            return

        model_config = OmegaConf.create({"embedding": dict(self.config.embedding)})
        model_config.embedding.model_name = self.model_name

        print(f"Загрузка модели: {self.model_name}")
        self.embedding_model = EmbeddingModel(model_config)
        self.config.qdrant.vector_size = self.embedding_model.get_embedding_dim()

        if self.hybrid_enabled:
            print("Инициализация sparse encoder...")
            self.sparse_encoder = SparseEncoder(self.config)

        self.qdrant_manager = QdrantManager(self.config)
        self.vector_store = make_vector_store(self.config)
        print(f"✓ Vector store backend: {self.vector_backend}")

        if self.reranker_enabled:
            from russian_laws.reranker import Reranker
            print("\nИнициализация reranker...")
            self.reranker = Reranker(self.config)

        if self.ragas_enabled:
            print("\nИнициализация LLM для генерации ответов...")
            self.llm_generator = LLMGenerator(self.config)
            print("\nИнициализация RAG метрик...")
            self.ragas_metrics = RAGASMetrics(config=self.config)
            print(f"✓ RAG метрики (judge: {self.config.ragas.llm.model})")

        print("Компоненты инициализированы")

    def index_articles(self, articles_df: pd.DataFrame, batch_size: int = 32) -> None:
        indexer = ArticleIndexer(
            embedding_model=self.embedding_model,
            qdrant_manager=self.qdrant_manager,
            config=self.config,
            sparse_encoder=self.sparse_encoder,
        )
        indexer.index_articles(articles_df, batch_size=batch_size)

        if "article_id" in articles_df.columns and "article_text" in articles_df.columns:
            self.article_texts = {
                int(row["article_id"]): str(row["article_text"])
                for _, row in articles_df[["article_id", "article_text"]].iterrows()
            }

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        query_text = batch["query_text"][0]
        relevant_id = int(batch["article_id"][0])
        relevant_ids_str = batch.get("relevant_ids", [str(relevant_id)])[0]
        relevant_ids = [int(x) for x in str(relevant_ids_str).split("|") if x]

        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        max_k = max(self.k_values)
        search_limit = (
            int(self.config.reranker.get("candidates_limit", 50))
            if self.reranker_enabled
            else max_k
        )

        assert self.vector_store is not None
        if self.hybrid_enabled and self.sparse_encoder is not None:
            sparse_query = self.sparse_encoder.encode(query_text)
            with self.retrieval_latency.measure():
                results = self.vector_store.hybrid_search(
                    dense_vector=query_embedding,
                    sparse_vector=sparse_query,
                    limit=search_limit,
                    score_threshold=0.0,
                )
        else:
            with self.retrieval_latency.measure():
                results = self.vector_store.search(
                    query_vector=query_embedding,
                    limit=search_limit,
                    score_threshold=0.0,
                )

        if self.reranker_enabled and self.reranker is not None and results:
            docs = [
                (p.payload or {}).get("parent_text")
                or (p.payload or {}).get("article_text", "")
                for p in results
            ]
            ranked_indices = self.reranker.rerank(
                query=query_text, documents=docs, top_k=max_k
            )
            results = [results[idx] for idx in ranked_indices]

        retrieved_ids = []
        seen_article_ids: set[int] = set()
        for point in results:
            article_id = point.payload.get("article_id")
            if article_id is not None and article_id not in seen_article_ids:
                retrieved_ids.append(article_id)
                seen_article_ids.add(article_id)

        self.metrics.update(relevant_ids=relevant_ids, retrieved_ids=retrieved_ids)

        if self.ragas_enabled and self.llm_generator and self.ragas_metrics:
            contexts = []
            for point in results:
                payload = point.payload or {}
                text = payload.get("parent_text") or payload.get("article_text", "")
                if not text:
                    continue
                codex = str(payload.get("codex") or "").strip()
                num = str(payload.get("article_num") or "").strip()
                title = str(payload.get("article_title") or "").strip()
                header_parts = []
                if codex:
                    header_parts.append(codex)
                if num and num.lower() != "nan":
                    header_parts.append(f"ст. {num}")
                if title:
                    header_parts.append(title)
                header = " | ".join(header_parts)
                contexts.append(f"[{header}]\n{text}" if header else text)

            if contexts:
                try:
                    context_text = "\n\n".join(contexts)
                    with self.llm_latency.measure():
                        result = self.llm_generator.generate(query_text, context_text)
                    answer_text = result if isinstance(result, str) else result[0]

                    print("\n[RAGAS INPUT DEBUG]")
                    print(f"  Question: {query_text[:100]}...")
                    print(
                        f"  Answer: {answer_text[:200]}..."
                        if len(answer_text) > 200
                        else f"  Answer: {answer_text}"
                    )
                    print(
                        f"  Contexts ({len(contexts)}): {contexts[0][:150]}..."
                        if contexts
                        else "  Contexts: []"
                    )

                    self.ragas_metrics.update(
                        question=query_text, answer=answer_text, contexts=contexts
                    )

                    reference_text = self.article_texts.get(relevant_id)
                    if reference_text:
                        self.rouge_metrics.update(
                            prediction=answer_text, reference=reference_text
                        )
                except Exception as e:
                    print(f"Ошибка генерации ответа для RAGAS: {e}")

        return {
            "query_text": query_text,
            "relevant_id": relevant_id,
            "retrieved_ids": retrieved_ids[:10],
        }

    def on_test_epoch_end(self) -> None:
        metrics = self.metrics.compute()

        if self.ragas_enabled and self.ragas_metrics:
            print("\n" + "=" * 80)
            print("ВЫЧИСЛЕНИЕ RAGAS МЕТРИК")
            print("=" * 80)
            metrics.update(self.ragas_metrics.compute())

        rouge_results = self.rouge_metrics.compute()
        if rouge_results:
            metrics.update(rouge_results)

        metrics.update(self.retrieval_latency.compute())
        metrics.update(self.llm_latency.compute())

        for metric_name, value in metrics.items():
            self.log(metric_name, value, prog_bar=True)

        print("\n" + "=" * 80)
        print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (vector_backend = {self.vector_backend})")
        print("=" * 80)
        for metric_name, value in metrics.items():
            print(f"{metric_name:25s}: {value:.4f}")
        print("=" * 80)


def run_test(
    test_data: str = "data/processed/test_queries.csv",
    articles_data: str = "data/processed/articles_indexed.csv",
    model_name: str | None = None,
    index_articles: bool = True,
    experiment_name: str = "russian_laws_retrieval",
    run_name: str = "baseline_test",
    config_overrides: str = "",
    limit: int | None = None,
) -> dict:
    overrides = [o.strip() for o in config_overrides.split(",") if o.strip()]

    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    model_name = model_name or cfg.embedding.model_name

    print("Конфигурация загружена")
    print(f"Модель эмбеддингов: {model_name}")

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri="mlruns",
    )
    mlflow_logger.log_hyperparams(
        {
            "model_name": model_name,
            "test_data": test_data,
            "articles_data": articles_data,
        }
    )

    tester = RetrievalTester(config=cfg, model_name=model_name)
    tester.setup(stage="test")

    print(f"\nЗагрузка тестовых данных из {test_data}")
    test_df = pd.read_csv(test_data)

    # rrncb-формат: разворачиваем document → article_ids по source_file
    if {"question", "document"}.issubset(test_df.columns) and "query_text" not in test_df.columns:
        print("Обнаружен rrncb-формат — разворачиваю document → article_ids по source_file")
        articles_for_map = pd.read_csv(articles_data)
        src_to_ids = (
            articles_for_map.dropna(subset=["source_file"])
            .groupby("source_file")["article_id"]
            .apply(lambda s: [int(x) for x in s.tolist()])
            .to_dict()
        )
        rows = []
        skipped = 0
        for _, r in test_df.iterrows():
            ids = src_to_ids.get(str(r["document"]), [])
            if not ids:
                skipped += 1
                continue
            rows.append(
                {
                    "query_text": str(r["question"]),
                    "article_id": ids[0],
                    "relevant_ids": "|".join(str(i) for i in ids),
                }
            )
        if skipped:
            print(f"  пропущено {skipped} запросов без совпадений по source_file")
        test_df = pd.DataFrame(rows)

    if limit is not None:
        test_df = test_df.head(limit)
        print(f"Загружено {len(test_df)} тестовых запросов (limit={limit})")
    else:
        print(f"Загружено {len(test_df)} тестовых запросов")

    articles_df_full = pd.read_csv(articles_data)
    if {"article_id", "article_text"}.issubset(articles_df_full.columns):
        tester.article_texts = {
            int(row["article_id"]): str(row["article_text"])
            for _, row in articles_df_full[["article_id", "article_text"]].dropna().iterrows()
        }

    if index_articles:
        print(f"\nЗагрузка статей из {articles_data}")
        articles_df = articles_df_full

        if limit is not None:
            relevant_ids = list(test_df["article_id"].tolist())
            relevant_articles = articles_df[articles_df["article_id"].isin(relevant_ids)]
            other_articles = articles_df[~articles_df["article_id"].isin(relevant_ids)].head(limit * 10)
            articles_df = pd.concat([relevant_articles, other_articles], ignore_index=True)
            print(f"Загружено {len(articles_df)} статей (режим отладки)")
        else:
            print(f"Загружено {len(articles_df)} статей")

        tester.index_articles(articles_df)

    test_dataset = QueryDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    trainer = pl.Trainer(
        logger=mlflow_logger,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nНачало тестирования...")
    trainer.test(tester, dataloaders=test_loader)

    final_metrics = tester.metrics.compute()

    tester.qdrant_manager.close()
    if tester.vector_store is not None and tester.vector_backend != "qdrant":
        tester.vector_store.close()

    print("\nТестирование завершено")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow run: {run_name}")

    return final_metrics


def main():
    return {"run_test": run_test}


if __name__ == "__main__":
    fire.Fire(main())

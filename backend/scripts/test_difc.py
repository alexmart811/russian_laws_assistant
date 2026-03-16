"""Скрипт для тестирования поиска и генерации ответов на DIFC документах."""

import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from russian_laws.embeddings import EmbeddingModel
from russian_laws.generator import LLMGenerator
from russian_laws.metrics import RetrievalMetrics
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder

load_dotenv()


class DifcTester:
    """Тестирование на DIFC документах."""

    def __init__(self, config_path: str = "conf"):
        """Инициализация тестера.

        Args:
            config_path: Путь к директории с конфигами
        """
        config_dir = Path(config_path).absolute()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            self.config = compose(config_name="config")

        # Переопределяем коллекцию на DIFC
        self.config.qdrant.collection_name = "difc_docs_hybrid"

        self.embedding_model = EmbeddingModel(self.config)
        self.qdrant_manager = QdrantManager(self.config)

        # Sparse encoder для гибридного поиска
        hybrid_enabled = self.config.qdrant.get("hybrid", {}).get("enabled", False)
        self.sparse_encoder = SparseEncoder(self.config) if hybrid_enabled else None

        self.llm_generator = LLMGenerator(self.config)
        self.retrieval_metrics = RetrievalMetrics()

        # Загружаем вопросы
        questions_path = Path("data/raw/docs_corpus/questions.json")
        with open(questions_path, encoding="utf-8") as f:
            self.questions = json.load(f)

        print(f"Загружено {len(self.questions)} вопросов")

    def run_test(self, limit: int = 5) -> dict:
        """Запускает тестирование.

        Args:
            limit: Количество документов для поиска

        Returns:
            Словарь с метриками
        """
        print(f"\n{'='*80}")
        print(f"Начало тестирования на {len(self.questions)} вопросах")
        print(f"Коллекция: {self.config.qdrant.collection_name}")
        print(f"Limit: {limit}")
        print(f"{'='*80}\n")

        for i, item in enumerate(self.questions, 1):
            question = item["question"]
            answer_type = item.get("answer_type", "free_text")

            print(f"\n[{i}/{len(self.questions)}] {question}")
            print(f"Тип ответа: {answer_type}")

            # Поиск релевантных документов
            query_vector = self.embedding_model.encode([question])[0].cpu().tolist()

            hybrid_enabled = self.config.qdrant.get("hybrid", {}).get("enabled", False)
            if hybrid_enabled and self.sparse_encoder is not None:
                sparse_query = self.sparse_encoder.encode(question)
                results = self.qdrant_manager.hybrid_search(
                    dense_vector=query_vector,
                    sparse_vector=sparse_query,
                    limit=limit,
                )
            else:
                results = self.qdrant_manager.search(
                    query_vector=query_vector,
                    limit=limit,
                )

            # Собираем контексты для LLM (используем parent_text если есть)
            contexts = []
            for point in results:
                payload = point.payload or {}
                context_text = payload.get("parent_text") or payload.get("text", "")
                contexts.append(context_text)

            # Генерируем ответ
            context_combined = "\n\n---\n\n".join(contexts)
            answer = self.llm_generator.generate(
                query=question,
                context=context_combined,
                answer_type=answer_type,
            )

            print(f"Ответ: {answer}")
            print(f"Найдено документов: {len(results)}")

            # Обновляем метрики ретривала
            # (для DIFC нет ground truth, поэтому метрики будут базовыми)
            self.retrieval_metrics.update(
                predictions=[],  # Нет ground truth
                ground_truth=[],
            )

        # Расчет метрик
        metrics = self.retrieval_metrics.compute()

        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print(f"{'='*80}")
        print(f"Всего вопросов: {len(self.questions)}")
        print(f"\nМетрики:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
        print(f"{'='*80}\n")

        return metrics


def main():
    """Запуск через Fire CLI."""
    fire.Fire(DifcTester)


if __name__ == "__main__":
    main()

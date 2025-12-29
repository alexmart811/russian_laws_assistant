"""Модуль для тестирования retrieval модели с использованием PyTorch Lightning."""

import subprocess
from pathlib import Path

import fire
import pandas as pd
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from qdrant_client.models import PointStruct
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from russian_laws.embeddings import EmbeddingModel
from russian_laws.metrics import RetrievalMetrics
from russian_laws.qdrant_manager import QdrantManager


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Разбивает текст на чанки с перекрытием.

    Args:
        text: Текст для разбиения
        chunk_size: Размер чанка в словах
        chunk_overlap: Перекрытие между чанками в словах

    Returns:
        Список текстовых чанков
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)

        # Переход к следующему чанку с перекрытием
        start += chunk_size - chunk_overlap

        # Если осталось меньше слов, чем размер перекрытия, останавливаемся
        if start >= len(words):
            break

    return chunks if chunks else [text]


class QueryDataset(Dataset):
    """Dataset для тестовых запросов."""

    def __init__(self, queries_df: pd.DataFrame):
        """Инициализация датасета.

        Args:
            queries_df: DataFrame с тестовыми запросами
        """
        self.queries_df = queries_df

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.queries_df)

    def __getitem__(self, idx: int) -> dict:
        """Возвращает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Словарь с query_text и article_id
        """
        row = self.queries_df.iloc[idx]
        return {
            "query_text": row["query_text"],
            "article_id": int(row["article_id"]),
        }


class RetrievalTester(pl.LightningModule):
    """Lightning модуль для тестирования retrieval модели."""

    def __init__(
        self,
        config: DictConfig,
        model_name: str,
        k_values: list[int] | None = None,
    ):
        """Инициализация тестера.

        Args:
            config: Конфигурация Hydra
            model_name: Название модели эмбеддингов
            k_values: Значения k для метрик
        """
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.k_values = k_values or [1, 3, 5, 10, 20]

        # Сохраняем гиперпараметры
        self.save_hyperparameters(
            {
                "model_name": model_name,
                "k_values": self.k_values,
                "qdrant_collection": config.qdrant.collection_name,
                "vector_size": config.qdrant.vector_size,
            }
        )

        # Инициализация компонентов
        self.embedding_model = None
        self.qdrant_manager = None
        self.metrics = RetrievalMetrics(k_values=self.k_values)

    def setup(self, stage: str | None = None) -> None:
        """Настройка компонентов перед тестированием.

        Args:
            stage: Стадия выполнения (fit, test, predict)
        """
        if stage == "test" or stage is None:
            # Создаем конфиг для модели эмбеддингов на основе основного конфига
            model_config = OmegaConf.create({"embedding": dict(self.config.embedding)})
            # Переопределяем model_name, если он был передан
            model_config.embedding.model_name = self.model_name

            print(f"Загрузка модели: {self.model_name}")
            print(f"Устройство: {model_config.embedding.device}")
            self.embedding_model = EmbeddingModel(model_config)

            # Обновляем vector_size в конфиге для Qdrant
            vector_size = self.embedding_model.get_embedding_dim()
            self.config.qdrant.vector_size = vector_size

            # Инициализация Qdrant менеджера
            self.qdrant_manager = QdrantManager(self.config)

            print("Компоненты инициализированы")

    def index_articles(self, articles_df: pd.DataFrame, batch_size: int = 32) -> None:
        """Индексирует статьи в Qdrant с поддержкой чанкирования.

        Args:
            articles_df: DataFrame со статьями
            batch_size: Размер батча для индексации
        """
        # Проверяем, включено ли чанкирование
        chunking_enabled = self.config.embedding.chunking.enabled
        chunk_size = self.config.embedding.chunking.chunk_size
        chunk_overlap = self.config.embedding.chunking.chunk_overlap

        print(f"\nИндексация {len(articles_df)} статей в Qdrant...")
        if chunking_enabled:
            print(
                f"Чанкирование: включено (размер={chunk_size}, перекрытие={chunk_overlap})"
            )
        else:
            print("Чанкирование: отключено")

        # Создаем коллекцию (пересоздаем, если существует)
        self.qdrant_manager.create_collection(recreate=True)

        # Индексируем статьи батчами
        points = []
        point_id = 0
        total_chunks = 0

        for _, row in tqdm(
            articles_df.iterrows(), total=len(articles_df), desc="Индексация"
        ):
            # Создаем текст для индексации (название + текст статьи)
            text = f"{row['article_title']} {row['article_text']}"

            if chunking_enabled:
                # Разбиваем на чанки
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                total_chunks += len(chunks)

                # Создаем точку для каждого чанка
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.embedding_model.encode([chunk])[0].tolist()

                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "article_id": int(row["article_id"]),
                            "chunk_idx": chunk_idx,
                            "total_chunks": len(chunks),
                            "article_num": str(row["article_num"]),
                            "article_title": row["article_title"],
                            "codex": row["codex"],
                        },
                    )
                    points.append(point)
                    point_id += 1

                    # Сохраняем батчами
                    if len(points) >= batch_size:
                        self.qdrant_manager.upsert_points(points)
                        points = []
            else:
                embedding = self.embedding_model.encode([text])[0].tolist()

                point = PointStruct(
                    id=int(row["article_id"]),
                    vector=embedding,
                    payload={
                        "article_id": int(row["article_id"]),
                        "article_num": str(row["article_num"]),
                        "article_title": row["article_title"],
                        "codex": row["codex"],
                    },
                )
                points.append(point)

                # Сохраняем батчами
                if len(points) >= batch_size:
                    self.qdrant_manager.upsert_points(points)
                    points = []

        # Сохраняем оставшиеся точки
        if points:
            self.qdrant_manager.upsert_points(points)

        if chunking_enabled:
            print(
                f"✓ Индексация завершена. Создано {total_chunks} чанков из {len(articles_df)} статей"
            )
        else:
            print(f"✓ Индексация завершена. Проиндексировано {len(articles_df)} статей")

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """Один шаг тестирования.

        Args:
            batch: Батч с данными
            batch_idx: Индекс батча

        Returns:
            Словарь с результатами
        """
        query_text = batch["query_text"][0]  # Батч размера 1
        relevant_id = int(batch["article_id"][0])

        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        # Ищем в Qdrant
        max_k = max(self.k_values)
        results = self.qdrant_manager.search(
            query_vector=query_embedding, limit=max_k, score_threshold=0.0
        )

        # Извлекаем ID найденных статей (убираем дубликаты, сохраняя порядок)
        retrieved_ids = []
        seen_article_ids = set()
        for point in results:
            article_id = point.payload.get("article_id")
            if article_id is not None and article_id not in seen_article_ids:
                retrieved_ids.append(article_id)
                seen_article_ids.add(article_id)

        # Обновляем метрики
        self.metrics.update(relevant_ids=[relevant_id], retrieved_ids=retrieved_ids)

        return {
            "query_text": query_text,
            "relevant_id": relevant_id,
            "retrieved_ids": retrieved_ids[:10],
        }

    def on_test_epoch_end(self) -> None:
        """Вызывается в конце тестовой эпохи."""
        # Вычисляем финальные метрики
        metrics = self.metrics.compute()

        # Логируем метрики
        for metric_name, value in metrics.items():
            self.log(metric_name, value, prog_bar=True)

        # Выводим результаты
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 80)
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.4f}")
        print("=" * 80)


def run_test(
    test_data: str = "data/processed/test_queries.csv",
    articles_data: str = "data/processed/articles_indexed.csv",
    model_name: str | None = None,
    index_articles: bool = True,
    experiment_name: str = "russian_laws_retrieval",
    run_name: str = "baseline_test",
) -> dict:
    """Запускает тестирование retrieval модели.

    Args:
        test_data: Путь к тестовым запросам
        articles_data: Путь к статьям
        model_name: Название модели эмбеддингов (если None, берется из конфига)
        index_articles: Индексировать ли статьи перед тестированием
        experiment_name: Название эксперимента MLflow
        run_name: Название run в MLflow

    Returns:
        Словарь с метриками
    """
    print("Загрузка данных через DVC...")
    project_root = Path(__file__).parent.parent
    download_script = project_root / "scripts" / "download_files.sh"

    subprocess.run(
        ["bash", str(download_script)],
        cwd=str(project_root),
        check=True,
    )
    print("Данные загружены")

    # Загружаем конфигурацию
    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

    # Используем модель из конфига, если не передана явно
    model_name = model_name or cfg.embedding.model_name

    print("Конфигурация загружена")
    print(f"Модель эмбеддингов: {model_name}")

    # Создаем MLflow логгер
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri="mlruns",
    )

    # Логируем параметры
    mlflow_logger.log_hyperparams(
        {
            "model_name": model_name,
            "test_data": test_data,
            "articles_data": articles_data,
        }
    )

    # Создаем тестер
    tester = RetrievalTester(config=cfg, model_name=model_name)
    tester.setup(stage="test")

    # Загружаем данные
    print(f"\nЗагрузка тестовых данных из {test_data}")
    test_df = pd.read_csv(test_data)
    print(f"Загружено {len(test_df)} тестовых запросов")

    # Индексируем статьи, если требуется
    if index_articles:
        print(f"\nЗагрузка статей из {articles_data}")
        articles_df = pd.read_csv(articles_data)
        print(f"Загружено {len(articles_df)} статей")

        tester.index_articles(articles_df)

    # Создаем датасет и даталоадер
    test_dataset = QueryDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Создаем трейнер
    trainer = pl.Trainer(
        logger=mlflow_logger,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Запускаем тестирование
    print("\nНачало тестирования...")
    trainer.test(tester, dataloaders=test_loader)

    # Получаем финальные метрики
    final_metrics = tester.metrics.compute()

    # Закрываем соединение
    tester.qdrant_manager.close()

    print("\nТестирование завершено")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow run: {run_name}")

    return final_metrics


def main():
    """Точка входа для Fire CLI."""
    return {"run_test": run_test}


if __name__ == "__main__":
    fire.Fire(main())

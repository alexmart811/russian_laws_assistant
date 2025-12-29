"""Модуль для обучения retrieval моделей с использованием PyTorch Lightning."""

import random
import subprocess
from pathlib import Path

import fire
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from russian_laws.metrics import RetrievalMetrics


class RetrievalDataset(Dataset):
    """Dataset для обучения retriever с негативными примерами."""

    def __init__(
        self,
        queries_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        num_negatives: int = 4,
        strategy: str = "random",
    ):
        """Инициализация датасета.

        Args:
            queries_df: DataFrame с query-positive парами
            articles_df: DataFrame со всеми статьями
            num_negatives: Количество негативных примеров
            strategy: Стратегия семплирования негативов (random, same_codex)
        """
        self.queries_df = queries_df.reset_index(drop=True)
        self.articles_df = articles_df.set_index("article_id")
        self.num_negatives = num_negatives
        self.strategy = strategy

        # Создаем индекс по кодексам для same_codex стратегии
        self.codex_to_articles = {}
        for article_id, row in self.articles_df.iterrows():
            codex = row["codex"]
            if codex not in self.codex_to_articles:
                self.codex_to_articles[codex] = []
            self.codex_to_articles[codex].append(article_id)

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.queries_df)

    def __getitem__(self, idx: int) -> dict:
        """Возвращает элемент датасета.

        Args:
            idx: Индекс элемента

        Returns:
            Словарь с query, positive и negatives
        """
        row = self.queries_df.iloc[idx]
        query_text = row["query_text"]
        positive_id = int(row["article_id"])
        positive_article = self.articles_df.loc[positive_id]

        # Семплируем негативы
        if self.strategy == "same_codex":
            # Негативы из того же кодекса
            codex = positive_article["codex"]
            candidates = [
                aid
                for aid in self.codex_to_articles.get(codex, [])
                if aid != positive_id
            ]
            if len(candidates) < self.num_negatives:
                # Дополняем случайными, если не хватает
                all_articles = [
                    aid for aid in self.articles_df.index if aid != positive_id
                ]
                candidates.extend(
                    random.sample(all_articles, self.num_negatives - len(candidates))
                )
        else:  # random
            # Случайные негативы
            candidates = [aid for aid in self.articles_df.index if aid != positive_id]

        negative_ids = random.sample(
            candidates, min(self.num_negatives, len(candidates))
        )
        negative_articles = [self.articles_df.loc[nid] for nid in negative_ids]

        return {
            "query": query_text,
            "positive": f"{positive_article['article_title']} {positive_article['article_text']}",
            "negatives": [
                f"{art['article_title']} {art['article_text']}"
                for art in negative_articles
            ],
            "positive_id": positive_id,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate функция для DataLoader.

    Args:
        batch: Список элементов датасета

    Returns:
        Батч в виде словаря
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negatives"] for item in batch]
    positive_ids = [item["positive_id"] for item in batch]

    return {
        "query": queries,
        "positive": positives,
        "negatives": negatives,
        "positive_id": positive_ids,
    }


class RetrievalModel(pl.LightningModule):
    """Lightning модуль для обучения retriever модели."""

    def __init__(self, config: DictConfig, model_name: str):
        """Инициализация модели.

        Args:
            config: Конфигурация Hydra
            model_name: Название модели эмбеддингов
        """
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.save_hyperparameters(
            {
                "model_name": model_name,
                "learning_rate": config.train.training.learning_rate,
                "batch_size": config.train.training.batch_size,
                "num_epochs": config.train.training.num_epochs,
            }
        )

        # Загружаем модель и токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Включаем gradient checkpointing для экономии памяти
        gradient_checkpointing = getattr(
            config.train.training, "gradient_checkpointing", False
        )
        if gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        # Параметры обучения
        self.learning_rate = config.train.training.learning_rate
        self.weight_decay = config.train.training.weight_decay
        self.warmup_steps = config.train.training.warmup_steps

        # Метрики для валидации
        self.val_metrics = RetrievalMetrics(k_values=[1, 3, 5, 10, 20])

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling для получения эмбеддинга."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Генерирует эмбеддинги для текстов.

        Args:
            texts: Список текстов

        Returns:
            Тензор с эмбеддингами [batch_size, embedding_dim]
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.embedding.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state

        embeddings = self._mean_pooling(token_embeddings, encoded["attention_mask"])

        # Нормализация
        if self.config.embedding.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, query: list[str], documents: list[str]) -> torch.Tensor:
        """Forward pass для вычисления similarity scores.

        Args:
            query: Список запросов
            documents: Список документов

        Returns:
            Матрица similarity scores [batch_size, num_documents]
        """
        query_embeddings = self.encode(query)
        doc_embeddings = self.encode(documents)

        # Cosine similarity
        scores = torch.matmul(query_embeddings, doc_embeddings.t())
        return scores

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Один шаг обучения.

        Args:
            batch: Батч с данными
            batch_idx: Индекс батча

        Returns:
            Loss значение
        """
        queries = batch["query"]
        positives = batch["positive"]
        negatives_list = batch["negatives"]  # Список списков негативов

        batch_size = len(queries)

        # Проверяем структуру батча
        if not negatives_list or len(negatives_list) == 0:
            # Fallback если нет негативов
            query_embeddings = self.encode(queries)
            pos_embeddings = self.encode(positives)
            loss = F.mse_loss(query_embeddings, pos_embeddings)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            return loss

        # Убеждаемся, что все элементы имеют одинаковое количество негативов
        num_negatives = (
            len(negatives_list[0])
            if len(negatives_list) > 0 and negatives_list[0]
            else 0
        )

        if num_negatives == 0:
            query_embeddings = self.encode(queries)
            pos_embeddings = self.encode(positives)
            loss = F.mse_loss(query_embeddings, pos_embeddings)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            return loss

        # Формируем все документы для батча
        all_documents = []
        for i in range(batch_size):
            all_documents.append(positives[i])
            # Безопасно добавляем негативы, проверяя границы
            if (
                i < len(negatives_list)
                and negatives_list[i]
                and len(negatives_list[i]) >= num_negatives
            ):
                all_documents.extend(negatives_list[i][:num_negatives])
            elif i < len(negatives_list) and negatives_list[i]:
                # Если меньше негативов, дополняем позитивом
                all_documents.extend(negatives_list[i])
                all_documents.extend(
                    [positives[i]] * (num_negatives - len(negatives_list[i]))
                )
            else:
                # Если совсем нет негативов, дублируем позитив
                all_documents.extend([positives[i]] * num_negatives)

        # Вычисляем эмбеддинги
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(all_documents)

        # Reshape для батча
        embedding_dim = doc_embeddings.shape[-1]
        doc_embeddings = doc_embeddings.view(
            batch_size, num_negatives + 1, embedding_dim
        )  # [batch_size, num_docs, embedding_dim]

        # Вычисляем similarity scores
        scores = torch.bmm(
            query_embeddings.unsqueeze(1), doc_embeddings.transpose(1, 2)
        ).squeeze(
            1
        )  # [batch_size, num_docs]

        # InfoNCE Loss: позитив должен иметь максимальный score
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(scores, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Один шаг валидации (не используется, валидация через test.py)."""
        pass

    def on_validation_epoch_end(self) -> None:
        """Вызывается в конце валидационной эпохи."""
        pass

    def configure_optimizers(self):
        """Настройка оптимизатора и scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Linear warmup + cosine decay
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def evaluate_on_test(
    model: RetrievalModel,
    test_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    k_values: list[int],
    max_samples: int = 100,
) -> dict:
    """Оценивает модель на тестовой выборке.

    Args:
        model: Обученная модель
        test_df: Тестовая выборка
        articles_df: DataFrame со статьями
        k_values: Значения k для метрик
        max_samples: Максимальное количество примеров для оценки (для скорости)

    Returns:
        Словарь с метриками
    """
    model.eval()
    metrics = RetrievalMetrics(k_values=k_values)

    # Ограничиваем количество примеров для быстрой оценки
    eval_df = test_df.head(max_samples) if len(test_df) > max_samples else test_df
    print(f"Оценка на {len(eval_df)} примерах из {len(test_df)}")

    # Создаем словарь статей для быстрого доступа
    articles_dict = {
        int(row["article_id"]): f"{row['article_title']} {row['article_text']}"
        for _, row in articles_df.iterrows()
    }

    all_articles = list(articles_dict.values())
    all_ids = list(articles_dict.keys())

    # Предвычисляем эмбеддинги всех статей (один раз)
    print("Предвычисление эмбеддингов статей...")
    batch_size = 32
    all_article_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_articles), batch_size):
            batch_articles = all_articles[i : i + batch_size]
            batch_embeddings = model.encode(batch_articles)
            all_article_embeddings.append(batch_embeddings)
    all_article_embeddings = torch.cat(all_article_embeddings, dim=0)

    # Оцениваем каждый запрос
    print("Оценка запросов...")
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            query_text = row["query_text"]
            positive_id = int(row["article_id"])

            # Генерируем эмбеддинг запроса
            query_embedding = model.encode([query_text])[0]

            # Вычисляем similarity со всеми статьями
            similarities = torch.matmul(
                query_embedding.unsqueeze(0), all_article_embeddings.t()
            ).squeeze()

            # Сортируем по similarity
            sorted_indices = torch.argsort(similarities, descending=True)
            retrieved_ids = [all_ids[idx] for idx in sorted_indices.cpu().numpy()]

            # Обновляем метрики
            metrics.update(relevant_ids=[positive_id], retrieved_ids=retrieved_ids)

    return metrics.compute()


def train_model(
    model_name: str,
    model_save_name: str,
    config: DictConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    experiment_name: str,
) -> str:
    """Обучает одну модель.

    Args:
        model_name: Название модели для обучения
        model_save_name: Имя для сохранения модели
        config: Конфигурация Hydra
        train_df: DataFrame с обучающими данными
        test_df: DataFrame с тестовыми данными
        articles_df: DataFrame со статьями
        experiment_name: Название эксперимента MLflow

    Returns:
        Путь к сохраненной модели
    """
    print(f"\n{'=' * 80}")
    print(f"Обучение модели: {model_name}")
    print(f"{'=' * 80}")

    # Создаем MLflow логгер
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"train_{model_save_name}",
        tracking_uri="mlruns",
    )

    # Создаем датасет и даталоадер
    train_dataset = RetrievalDataset(
        queries_df=train_df,
        articles_df=articles_df,
        num_negatives=config.train.negatives.num_negatives,
        strategy=config.train.negatives.strategy,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Создаем модель
    model = RetrievalModel(config=config, model_name=model_name)

    # Callback для сохранения лучшей модели
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.train.save.dir) / model_save_name,
        filename="best-{epoch:02d}-{step:05d}",
        monitor="train_loss",
        mode="min",
        save_top_k=config.train.save.save_top_k,
        save_last=True,
    )

    # Настройки для экономии памяти
    precision = getattr(config.train.training, "precision", "32-true")
    accumulate_grad_batches = getattr(
        config.train.training, "gradient_accumulation_steps", 1
    )

    # Создаем трейнер
    trainer = pl.Trainer(
        max_epochs=config.train.training.num_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        accelerator=config.embedding.device,
        devices=1,
        precision=precision,  # Mixed precision (16-mixed) для экономии памяти
        accumulate_grad_batches=accumulate_grad_batches,  # Накопление градиентов
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50,
    )

    # Обучаем
    trainer.fit(model, train_loader)

    # Оцениваем на тестовой выборке
    print("\nОценка на тестовой выборке...")
    test_metrics = evaluate_on_test(
        model=model,
        test_df=test_df,
        articles_df=articles_df,
        k_values=[1, 3, 5, 10, 20],
    )

    # Логируем метрики
    for metric_name, value in test_metrics.items():
        mlflow_logger.log_metrics({f"test_{metric_name}": value})
        print(f"{metric_name}: {value:.4f}")

    # Сохраняем финальную модель
    models_dir = Path(config.train.save.dir) / model_save_name
    final_model_path = models_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print("\nОбучение завершено")
    print(f"Финальная модель сохранена: {final_model_path}")

    return str(final_model_path)


def run_training(
    train_data: str = "data/processed/train_queries.csv",
    test_data: str = "data/processed/test_queries.csv",
    articles_data: str = "data/processed/articles_indexed.csv",
    experiment_name: str = "russian_laws_training",
) -> dict:
    """Запускает обучение всех моделей.

    Args:
        train_data: Путь к обучающим данным
        test_data: Путь к тестовым данным
        articles_data: Путь к статьям
        experiment_name: Название эксперимента MLflow

    Returns:
        Словарь с путями к сохраненным моделям
    """
    # Загружаем данные через DVC
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

    print("Конфигурация загружена")

    # Загружаем данные
    print("\nЗагрузка данных...")
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    articles_df = pd.read_csv(articles_data)
    print(f"Загружено {len(train_df)} обучающих пар")
    print(f"Загружено {len(test_df)} тестовых пар")
    print(f"Загружено {len(articles_df)} статей")

    # Создаем директорию для моделей
    models_dir = Path(cfg.train.save.dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Обучаем модели по очереди
    saved_models = {}
    for model_config in cfg.train.models:
        model_name = model_config.name
        model_save_name = model_config.save_name

        try:
            best_model_path = train_model(
                model_name=model_name,
                model_save_name=model_save_name,
                config=cfg,
                train_df=train_df,
                test_df=test_df,
                articles_df=articles_df,
                experiment_name=experiment_name,
            )
            saved_models[model_save_name] = best_model_path
        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'=' * 80}")
    for name, path in saved_models.items():
        print(f"{name}: {path}")

    return saved_models


def main():
    """Точка входа для Fire CLI."""
    return {"run_training": run_training}


if __name__ == "__main__":
    fire.Fire(main())

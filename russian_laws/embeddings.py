"""Модуль для генерации эмбеддингов текстов."""

import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """Класс для генерации эмбеддингов с использованием трансформеров."""

    def __init__(self, config: DictConfig):
        """Инициализация модели эмбеддингов.

        Args:
            config: Конфигурация Hydra с параметрами модели
        """
        self.config = config
        self.model_name = config.embedding.model_name
        self.device = torch.device(config.embedding.device)
        self.batch_size = config.embedding.batch_size
        self.max_length = config.embedding.max_length
        self.normalize = config.embedding.normalize
        self.pooling = config.embedding.pooling

        print(f"Загрузка модели эмбеддингов: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"Модель загружена на устройство: {self.device}")

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling для получения эмбеддинга всего текста.

        Args:
            token_embeddings: Эмбеддинги токенов [batch_size, seq_len, hidden_size]
            attention_mask: Маска внимания [batch_size, seq_len]

        Returns:
            Усредненные эмбеддинги [batch_size, hidden_size]
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _cls_pooling(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """CLS pooling - берем эмбеддинг первого токена [CLS].

        Args:
            token_embeddings: Эмбеддинги токенов [batch_size, seq_len, hidden_size]

        Returns:
            CLS эмбеддинги [batch_size, hidden_size]
        """
        return token_embeddings[:, 0, :]

    def _max_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Max pooling для получения эмбеддинга всего текста.

        Args:
            token_embeddings: Эмбеддинги токенов [batch_size, seq_len, hidden_size]
            attention_mask: Маска внимания [batch_size, seq_len]

        Returns:
            Max pooled эмбеддинги [batch_size, hidden_size]
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9  # Маскируем padding
        return torch.max(token_embeddings, 1)[0]

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Генерирует эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для кодирования

        Returns:
            Тензор с эмбеддингами [batch_size, embedding_dim]
        """
        # Токенизация
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Получаем эмбеддинги
        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state

        # Применяем pooling
        if self.pooling == "mean":
            embeddings = self._mean_pooling(token_embeddings, encoded["attention_mask"])
        elif self.pooling == "cls":
            embeddings = self._cls_pooling(token_embeddings)
        elif self.pooling == "max":
            embeddings = self._max_pooling(token_embeddings, encoded["attention_mask"])
        else:
            raise ValueError(f"Неизвестный тип pooling: {self.pooling}")

        # Нормализация
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Генерирует эмбеддинги для списка текстов с батчингом.

        Args:
            texts: Список текстов для кодирования

        Returns:
            Список эмбеддингов
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings.cpu())

        # Конкатенируем все батчи
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings.tolist()

    def get_embedding_dim(self) -> int:
        """Возвращает размерность эмбеддинга.

        Returns:
            Размерность эмбеддинга
        """
        return self.model.config.hidden_size

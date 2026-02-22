"""Модуль для генерации sparse векторов (BM25) для гибридного поиска."""

import re
from collections import Counter

import nltk
from omegaconf import DictConfig

# Загружаем стоп-слова при импорте модуля
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Загрузка стоп-слов NLTK...")
    nltk.download("stopwords", quiet=True)


class SparseEncoder:
    """Энкодер для генерации BM25-подобных sparse векторов."""

    def __init__(self, config: DictConfig):
        """Инициализация sparse encoder.

        Args:
            config: Конфигурация Hydra с параметрами sparse encoding
        """
        self.config = config
        self.vocabulary: dict[str, int] = {}
        self.token_id_counter = 0
        self.use_stemming = config.sparse.get("use_stemming", False)
        self.min_token_length = config.sparse.get("min_token_length", 2)
        self.language = config.sparse.get("language", "russian")

        # Загружаем стоп-слова из NLTK
        from nltk.corpus import stopwords

        self.stopwords = set(stopwords.words(self.language))

        print(
            f"Sparse encoder инициализирован (язык: {self.language}, "
            f"стоп-слов: {len(self.stopwords)})"
        )

    def _tokenize(self, text: str) -> list[str]:
        """Токенизирует текст на русском языке.

        Args:
            text: Входной текст

        Returns:
            Список токенов
        """
        # Приводим к lowercase
        text = text.lower()

        # Разбиваем на слова (только буквы и цифры)
        tokens = re.findall(r"\b\w+\b", text)

        # Фильтруем стоп-слова и короткие токены
        tokens = [
            token
            for token in tokens
            if token not in self.stopwords and len(token) >= self.min_token_length
        ]

        return tokens

    def _get_token_id(self, token: str) -> int:
        """Получает или создает ID для токена.

        Args:
            token: Токен

        Returns:
            ID токена в словаре
        """
        if token not in self.vocabulary:
            self.vocabulary[token] = self.token_id_counter
            self.token_id_counter += 1
        return self.vocabulary[token]

    def encode(self, text: str) -> list[tuple[int, float]]:
        """Генерирует sparse вектор для текста.

        Args:
            text: Входной текст

        Returns:
            Список пар (token_id, weight) для Qdrant sparse vectors
        """
        tokens = self._tokenize(text)

        if not tokens:
            return []

        # Подсчитываем частоту токенов (TF)
        token_counts = Counter(tokens)

        # Генерируем sparse вектор
        sparse_vector = []
        for token, count in token_counts.items():
            token_id = self._get_token_id(token)
            # Простое TF взвешивание (IDF будет применять Qdrant через modifier)
            weight = float(count)
            sparse_vector.append((token_id, weight))

        return sparse_vector

    def encode_batch(self, texts: list[str]) -> list[list[tuple[int, float]]]:
        """Генерирует sparse векторы для списка текстов.

        Args:
            texts: Список текстов

        Returns:
            Список sparse векторов
        """
        return [self.encode(text) for text in texts]

    def get_vocabulary_size(self) -> int:
        """Возвращает размер словаря.

        Returns:
            Количество уникальных токенов
        """
        return len(self.vocabulary)

    def save_vocabulary(self, path: str) -> None:
        """Сохраняет словарь в файл.

        Args:
            path: Путь для сохранения
        """
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
        print(f"Словарь сохранен: {path} ({len(self.vocabulary)} токенов)")

    def load_vocabulary(self, path: str) -> None:
        """Загружает словарь из файла.

        Args:
            path: Путь к файлу словаря
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)
        self.token_id_counter = max(self.vocabulary.values()) + 1
        print(f"Словарь загружен: {path} ({len(self.vocabulary)} токенов)")

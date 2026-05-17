import json
import re
from collections import Counter
from pathlib import Path

import nltk
from omegaconf import DictConfig

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


class SparseEncoder:
    """BM25-подобный sparse encoder для гибридного поиска."""

    def __init__(self, config: DictConfig):
        self.vocabulary: dict[str, int] = {}
        self.token_id_counter = 0
        self.use_stemming = config.sparse.get("use_stemming", False)
        self.min_token_length = config.sparse.get("min_token_length", 2)
        self.language = config.sparse.get("language", "russian")
        self._vocabulary_path: str | None = config.sparse.get("vocabulary_path")

        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words(self.language))

        if self._vocabulary_path and Path(self._vocabulary_path).exists():
            self.load_vocabulary(self._vocabulary_path)
        else:
            print(
                f"Sparse encoder инициализирован (язык: {self.language}, "
                f"стоп-слов: {len(self.stopwords)})"
            )

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return [
            token
            for token in tokens
            if token not in self.stopwords and len(token) >= self.min_token_length
        ]

    def _get_token_id(self, token: str) -> int:
        if token not in self.vocabulary:
            self.vocabulary[token] = self.token_id_counter
            self.token_id_counter += 1
        return self.vocabulary[token]

    def encode(self, text: str) -> list[tuple[int, float]]:
        """Возвращает sparse вектор как список пар (token_id, weight)."""
        tokens = self._tokenize(text)
        if not tokens:
            return []
        return [
            (self._get_token_id(token), float(count))
            for token, count in Counter(tokens).items()
        ]

    def encode_batch(self, texts: list[str]) -> list[list[tuple[int, float]]]:
        return [self.encode(text) for text in texts]

    def get_vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def save_vocabulary(self, path: str | None = None) -> None:
        path = path or self._vocabulary_path
        if path is None:
            raise ValueError("Не указан путь для сохранения словаря")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
        print(f"Словарь сохранён: {path} ({len(self.vocabulary)} токенов)")

    def load_vocabulary(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.vocabulary = json.load(f)
        self.token_id_counter = (
            max(self.vocabulary.values()) + 1 if self.vocabulary else 0
        )
        print(f"Словарь загружен: {path} ({len(self.vocabulary)} токенов)")

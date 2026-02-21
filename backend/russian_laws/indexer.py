"""Модуль для индексации статей в Qdrant."""

import pandas as pd
from omegaconf import DictConfig
from qdrant_client.models import PointStruct
from russian_laws.embeddings import EmbeddingModel
from russian_laws.qdrant_manager import QdrantManager
from tqdm import tqdm


class ArticleIndexer:
    """Индексатор статей в Qdrant с поддержкой чанкирования."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        qdrant_manager: QdrantManager,
        config: DictConfig,
    ):
        """Инициализация индексатора.

        Args:
            embedding_model: Модель для генерации эмбеддингов
            qdrant_manager: Менеджер для работы с Qdrant
            config: Конфигурация с параметрами чанкирования
        """
        self.embedding_model = embedding_model
        self.qdrant_manager = qdrant_manager
        self.config = config

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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

    def index_articles(
        self, articles_df: pd.DataFrame, batch_size: int = 32, recreate: bool = True
    ) -> int:
        """Индексирует статьи в Qdrant с поддержкой чанкирования.

        Args:
            articles_df: DataFrame со статьями
            batch_size: Размер батча для индексации
            recreate: Пересоздать коллекцию

        Returns:
            Количество созданных точек (чанков)
        """
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

        # Создаем коллекцию
        self.qdrant_manager.create_collection(recreate=recreate)

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
                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
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
                            "article_text": row["article_text"],
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
                        "article_text": row["article_text"],
                        "codex": row["codex"],
                    },
                )
                points.append(point)
                total_chunks += 1

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

        return total_chunks

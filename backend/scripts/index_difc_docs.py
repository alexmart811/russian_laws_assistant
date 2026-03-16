"""Скрипт для индексации DIFC документов в Qdrant.

Создает новую коллекцию и индексирует секции документов.
"""

import sys
from pathlib import Path

import fire
import pandas as pd
from hydra import compose, initialize_config_dir
from tqdm import tqdm

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from russian_laws.embeddings import EmbeddingModel
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder


def index_difc_documents(
    documents_path: str = "data/processed/difc_documents.csv",
    collection_name: str = "difc_docs_hybrid",
    batch_size: int = 32,
    recreate_collection: bool = True,
) -> None:
    """Индексирует DIFC документы в Qdrant.

    Args:
        documents_path: Путь к CSV с обработанными документами
        collection_name: Название коллекции в Qdrant
        batch_size: Размер батча для индексации
        recreate_collection: Пересоздать коллекцию если существует
    """
    print(f"🚀 Начало индексации DIFC документов")
    print(f"{'='*80}\n")

    # Загружаем конфигурацию
    print("📋 Загрузка конфигурации...")
    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

    # Переопределяем название коллекции
    cfg.qdrant.collection_name = collection_name

    print(f"✓ Конфигурация загружена")
    print(f"  Модель эмбеддингов: {cfg.embedding.model_name}")
    print(f"  Hybrid search: {cfg.qdrant.hybrid.enabled}")
    print(f"  Коллекция: {collection_name}\n")

    # Загружаем данные
    print(f"📄 Загрузка документов из {documents_path}...")
    df = pd.read_csv(documents_path)
    print(f"✓ Загружено {len(df)} секций\n")

    # Инициализируем компоненты
    print("🔧 Инициализация компонентов...")

    # Embedding модель
    embedding_model = EmbeddingModel(cfg)
    print(f"✓ Embedding модель загружена на {cfg.embedding.device}")

    # Sparse encoder (если включен hybrid mode)
    sparse_encoder = None
    if cfg.qdrant.hybrid.enabled:
        sparse_encoder = SparseEncoder(cfg)
        print(f"✓ Sparse encoder инициализирован (язык: {cfg.sparse.language})")

    # Qdrant менеджер
    qdrant_manager = QdrantManager(cfg)
    print(f"✓ Подключение к Qdrant: {cfg.qdrant.url}\n")

    # Создаем коллекцию
    if recreate_collection:
        print(f"🗑️  Удаление существующей коллекции {collection_name} (если есть)...")
        try:
            qdrant_manager.client.delete_collection(collection_name)
            print(f"✓ Коллекция удалена")
        except Exception as e:
            print(f"  Коллекция не существовала или ошибка: {e}")

    print(f"\n📦 Создание новой коллекции {collection_name}...")
    qdrant_manager.create_collection()
    print(f"✓ Коллекция создана\n")

    # Подготавливаем данные для индексации
    print("🔄 Начало индексации...")
    print(f"{'─'*80}\n")

    # Индексируем батчами
    total_indexed = 0

    for start_idx in tqdm(range(0, len(df), batch_size), desc="Индексация"):
        batch_df = df.iloc[start_idx : start_idx + batch_size]

        # Подготавливаем тексты для эмбеддинга
        texts = batch_df["section_text"].tolist()

        # Генерируем dense эмбеддинги
        dense_vectors = embedding_model.encode(texts)

        # Генерируем sparse векторы
        sparse_vectors = None
        if sparse_encoder:
            sparse_vectors = sparse_encoder.encode_batch(texts)

        # Подготавливаем точки для upsert
        points = []
        for idx, row in batch_df.iterrows():
            local_idx = idx - start_idx

            # Payload с метаданными
            payload = {
                "doc_id": row["doc_id"],
                "section_id": row["section_id"],
                "section_type": row["section_type"],
                "section_title": row["section_title"],
                "section_number": row.get("section_number", ""),
                "section_text": row["section_text"],
                "case_number": row.get("case_number", ""),
                "case_date": row.get("case_date", ""),
                "doc_title": row.get("doc_title", ""),
            }

            # Векторы
            vectors = {"dense": dense_vectors[local_idx].tolist()}
            if sparse_vectors:
                sparse_vec = sparse_vectors[local_idx]
                # sparse_vec это список кортежей [(index, value), ...]
                indices = [int(idx) for idx, _ in sparse_vec]
                values = [float(val) for _, val in sparse_vec]
                vectors["sparse"] = {
                    "indices": indices,
                    "values": values,
                }

            from qdrant_client.models import PointStruct

            points.append(
                PointStruct(
                    id=int(idx),
                    vector=vectors,
                    payload=payload,
                )
            )

        # Загружаем в Qdrant
        qdrant_manager.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        total_indexed += len(points)

    print(f"\n{'='*80}")
    print(f"✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print(f"{'='*80}")
    print(f"📊 Проиндексировано секций: {total_indexed}")
    print(f"📦 Коллекция: {collection_name}")
    print(f"🔍 Hybrid search: {'Включен' if cfg.qdrant.hybrid.enabled else 'Выключен'}")

    # Проверяем коллекцию
    print(f"\n📈 Информация о коллекции:")
    collection_info = qdrant_manager.client.get_collection(collection_name)
    print(f"  Точек в коллекции: {collection_info.points_count}")
    print(
        f"  Размер векторов (dense): {collection_info.config.params.vectors['dense'].size}"
    )
    if cfg.qdrant.hybrid.enabled:
        print(f"  Sparse векторы: Включены")


if __name__ == "__main__":
    fire.Fire(index_difc_documents)

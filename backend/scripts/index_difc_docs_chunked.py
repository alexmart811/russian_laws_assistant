"""Скрипт для чанкирования и индексации DIFC документов в Qdrant.

Применяет parent-child chunking к крупным текстам документов,
затем индексирует в Qdrant с hybrid search.
"""

import sys
from pathlib import Path

import fire
import pandas as pd
from hydra import compose, initialize_config_dir
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from russian_laws.embeddings import EmbeddingModel
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder


def _count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте."""
    return len(text.split())


def _create_parent_child_chunks(
    text: str,
    doc_id: str,
    config,
) -> list[dict]:
    """Создает parent-child чанки из текста.

    Args:
        text: Исходный текст
        doc_id: ID документа
        config: Конфигурация с параметрами чанкирования

    Returns:
        Список child-чанков с информацией о parent
    """
    # Параметры чанкирования
    child_min_words = config.embedding.chunking.child.get("min_size", 50)
    child_max_words = config.embedding.chunking.child.get("max_size", 200)
    children_per_parent = config.embedding.chunking.parent.get("children_per_parent", 3)
    parent_max_words = config.embedding.chunking.parent.get("max_size", 800)
    parent_overlap = config.embedding.chunking.parent.get("overlap", 50)

    # Разбиваем на абзацы (по двойным переносам строк)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Создаем child чанки из абзацев
    child_chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = _count_words(para)

        if current_words + para_words > child_max_words and current_chunk:
            # Сохраняем текущий чанк
            child_text = "\n\n".join(current_chunk)
            if _count_words(child_text) >= child_min_words:
                child_chunks.append(child_text)
            current_chunk = [para]
            current_words = para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    # Добавляем последний чанк
    if current_chunk:
        child_text = "\n\n".join(current_chunk)
        if _count_words(child_text) >= child_min_words:
            child_chunks.append(child_text)

    # Если не получилось создать child чанки, используем весь текст
    if not child_chunks:
        child_chunks = [text]

    # Группируем child чанки в parent чанки
    result = []
    parent_id = 0

    for i in range(0, len(child_chunks), children_per_parent):
        parent_children = child_chunks[i : i + children_per_parent]

        # Создаем parent текст
        parent_text = "\n\n".join(parent_children)
        parent_words = _count_words(parent_text)

        # Если parent слишком большой, обрезаем
        if parent_words > parent_max_words:
            # Берем первые parent_max_words слов
            words = parent_text.split()
            parent_text = " ".join(words[:parent_max_words])

        parent_id += 1
        parent_uid = f"{doc_id}_p{parent_id}"

        # Создаем записи для каждого child чанка
        for child_idx, child_text in enumerate(parent_children):
            result.append(
                {
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_uid,
                    "child_index": child_idx,
                }
            )

    return result


def index_difc_documents(
    documents_path: str = "data/processed/difc_documents.csv",
    collection_name: str = "difc_docs_hybrid",
    batch_size: int = 16,
    recreate_collection: bool = True,
) -> None:
    """Чанкирует и индексирует DIFC документы в Qdrant.

    Args:
        documents_path: Путь к CSV с обработанными документами
        collection_name: Название коллекции в Qdrant
        batch_size: Размер батча для индексации
        recreate_collection: Пересоздать коллекцию если существует
    """
    print(f"🚀 Начало индексации DIFC документов с чанкированием")
    print(f"{'='*80}\n")

    # Загружаем конфигурацию
    print("📋 Загрузка конфигурации...")
    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

    cfg.qdrant.collection_name = collection_name

    print(f"✓ Конфигурация загружена")
    print(f"  Модель эмбеддингов: {cfg.embedding.model_name}")
    print(f"  Hybrid search: {cfg.qdrant.hybrid.enabled}")
    print(f"  Chunking: parent-child")
    print(f"  Коллекция: {collection_name}\n")

    # Загружаем данные
    print(f"📄 Загрузка документов из {documents_path}...")
    df = pd.read_csv(documents_path)
    print(f"✓ Загружено {len(df)} единиц (документы/статьи)\n")

    # Инициализируем компоненты
    print("🔧 Инициализация компонентов...")

    embedding_model = EmbeddingModel(cfg)
    print(f"✓ Embedding модель загружена на {cfg.embedding.device}")

    sparse_encoder = None
    if cfg.qdrant.hybrid.enabled:
        sparse_encoder = SparseEncoder(cfg)
        print(f"✓ Sparse encoder инициализирован")

    qdrant_manager = QdrantManager(cfg)
    print(f"✓ Подключение к Qdrant: {cfg.qdrant.url}\n")

    # Создаем коллекцию
    if recreate_collection:
        print(f"🗑️  Удаление существующей коллекции {collection_name} (если есть)...")
        try:
            qdrant_manager.client.delete_collection(collection_name)
            print(f"✓ Коллекция удалена")
        except Exception as e:
            print(f"  Коллекция не существовала")

    print(f"\n📦 Создание новой коллекции {collection_name}...")
    qdrant_manager.create_collection()
    print(f"✓ Коллекция создана\n")

    # Чанкируем документы
    print("✂️  Чанкирование документов...")
    all_chunks = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Чанкирование"):
        text = row["text"]
        doc_id = row["unit_id"]

        # Создаем чанки
        chunks = _create_parent_child_chunks(text, doc_id, cfg)

        # Добавляем метаданные
        for chunk in chunks:
            chunk.update(
                {
                    "doc_id": row["doc_id"],
                    "unit_id": row["unit_id"],
                    "unit_type": row["unit_type"],
                    "unit_title": row["unit_title"],
                    "doc_type": row["doc_type"],
                    "case_number": row.get("case_number", ""),
                    "case_date": row.get("case_date", ""),
                    "court": row.get("court", ""),
                    "doc_title": row.get("doc_title", ""),
                }
            )

        all_chunks.extend(chunks)

    print(f"\n✓ Создано {len(all_chunks)} child чанков")
    print(f"  Среднее количество чанков на документ: {len(all_chunks) / len(df):.1f}\n")

    # Индексируем батчами
    print("🔄 Начало индексации...")
    print(f"{'─'*80}\n")

    total_indexed = 0

    for start_idx in tqdm(range(0, len(all_chunks), batch_size), desc="Индексация"):
        batch = all_chunks[start_idx : start_idx + batch_size]

        # Тексты для эмбеддинга (используем child_text)
        texts = [chunk["child_text"] for chunk in batch]

        # Генерируем dense эмбеддинги
        dense_vectors = embedding_model.encode(texts)

        # Генерируем sparse векторы
        sparse_vectors = None
        if sparse_encoder:
            sparse_vectors = sparse_encoder.encode_batch(texts)

        # Подготавливаем точки для upsert
        points = []
        for local_idx, chunk in enumerate(batch):
            point_id = start_idx + local_idx

            # Payload
            payload = {
                "doc_id": chunk["doc_id"],
                "unit_id": chunk["unit_id"],
                "unit_type": chunk["unit_type"],
                "unit_title": chunk["unit_title"],
                "doc_type": chunk["doc_type"],
                "child_text": chunk["child_text"],
                "parent_text": chunk["parent_text"],
                "parent_id": chunk["parent_id"],
                "case_number": chunk["case_number"],
                "case_date": chunk["case_date"],
                "court": chunk["court"],
                "doc_title": chunk["doc_title"],
            }

            # Векторы
            vectors = {"dense": dense_vectors[local_idx].tolist()}
            if sparse_vectors:
                sparse_vec = sparse_vectors[local_idx]
                indices = [int(idx) for idx, _ in sparse_vec]
                values = [float(val) for _, val in sparse_vec]
                vectors["sparse"] = {
                    "indices": indices,
                    "values": values,
                }

            from qdrant_client.models import PointStruct

            points.append(
                PointStruct(
                    id=point_id,
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
    print(f"📊 Исходных единиц: {len(df)}")
    print(f"✂️  Child чанков: {len(all_chunks)}")
    print(f"📦 Проиндексировано в Qdrant: {total_indexed}")
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

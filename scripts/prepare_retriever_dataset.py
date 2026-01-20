"""Скрипт для подготовки датасетов для обучения retriever моделей.

Создает train и test датасеты из articles.csv с query-positive парами.
Негативные примеры будут семплироваться динамически во время обучения.

Выходные файлы:
- train_queries.csv: пары (query, positive_article) для обучения
- test_queries.csv: пары (query, positive_article) для тестирования
- articles_indexed.csv: все статьи с уникальными ID для семплирования негативов

Структура выходных CSV:
- query_id: уникальный ID запроса
- query_text: текст запроса
- article_id: ID позитивной статьи
- article_num: номер статьи
- article_title: название статьи
- article_text: полный текст статьи
- codex: название кодекса
"""

from pathlib import Path

import fire
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_datasets(
    articles_path: str = "data/processed/articles.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.15,
    random_state: int = 42,
    min_query_length: int = 10,
) -> None:
    """Подготавливает train и test датасеты для обучения retriever.

    Args:
        articles_path: Путь к CSV файлу со статьями и запросами
        output_dir: Директория для сохранения результатов
        test_size: Доля данных для тестового набора (0.0-1.0)
        random_state: Seed для воспроизводимости
        min_query_length: Минимальная длина запроса в символах
    """
    print(f"Загрузка данных из {articles_path}...")
    df = pd.read_csv(articles_path)

    print(f"Всего статей: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")

    # Проверяем наличие необходимых колонок
    required_columns = ["Номер статьи", "Текст статьи", "Кодекс", "Запрос"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")

    # Фильтруем строки с валидными запросами
    print("\nФильтрация данных...")
    df_filtered = df.dropna(subset=["Запрос"]).copy()
    print(f"Статей с запросами: {len(df_filtered)}")

    # Убираем слишком короткие запросы
    df_filtered["query_length"] = df_filtered["Запрос"].str.len()
    df_filtered = df_filtered[df_filtered["query_length"] >= min_query_length]
    print(
        f"После фильтрации по длине (>={min_query_length} символов): {len(df_filtered)}"
    )

    # Убираем дубликаты запросов
    initial_count = len(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=["Запрос"], keep="first")
    duplicates_removed = initial_count - len(df_filtered)
    if duplicates_removed > 0:
        print(f"Удалено дубликатов запросов: {duplicates_removed}")

    # Создаем уникальные ID для статей и запросов
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["article_id"] = df_filtered.index
    df_filtered["query_id"] = df_filtered.index

    # Подготавливаем датасет с query-positive парами
    dataset = pd.DataFrame(
        {
            "query_id": df_filtered["query_id"],
            "query_text": df_filtered["Запрос"],
            "article_id": df_filtered["article_id"],
            "article_num": df_filtered["Номер статьи"],
            "article_title": df_filtered["Название статьи"].fillna(""),
            "article_text": df_filtered["Текст статьи"],
            "codex": df_filtered["Кодекс"],
        }
    )

    print(f"\nИтого пар (query, positive_article): {len(dataset)}")

    # Статистика по кодексам
    print("\nРаспределение по кодексам:")
    print(dataset["codex"].value_counts())

    # Разделяем на train и test с стратификацией по кодексам
    print(f"\nРазделение на train ({1 - test_size:.0%}) и test ({test_size:.0%})...")
    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["codex"],
    )

    print(f"Train размер: {len(train_df)}")
    print(f"Test размер: {len(test_df)}")

    # Создаем выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Сохраняем датасеты
    train_path = output_path / "train_queries.csv"
    test_path = output_path / "test_queries.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Train датасет сохранен: {train_path}")
    print(f"✓ Test датасет сохранен: {test_path}")

    # Также сохраняем полный список статей с ID для использования при семплировании негативов
    articles_with_ids = df[
        ["Номер статьи", "Название статьи", "Текст статьи", "Кодекс"]
    ].copy()
    articles_with_ids["article_id"] = articles_with_ids.index
    articles_with_ids.columns = [
        "article_num",
        "article_title",
        "article_text",
        "codex",
        "article_id",
    ]

    # Переставляем колонки
    articles_with_ids = articles_with_ids[
        ["article_id", "article_num", "article_title", "article_text", "codex"]
    ]

    articles_indexed_path = output_path / "articles_indexed.csv"
    articles_with_ids.to_csv(articles_indexed_path, index=False, encoding="utf-8-sig")
    print(f"✓ Индексированные статьи сохранены: {articles_indexed_path}")

    # Выводим примеры
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ИЗ TRAIN ДАТАСЕТА:")
    print("=" * 80)
    for i in range(min(3, len(train_df))):
        row = train_df.iloc[i]
        print(f"\nПример {i + 1}:")
        print(f"  Query ID: {row['query_id']}")
        print(f"  Запрос: {row['query_text']}")
        print(f"  Кодекс: {row['codex']}")
        print(f"  Статья: {row['article_num']}")
        print(f"  Название: {row['article_title'][:100]}...")
        print(f"  Текст статьи: {row['article_text'][:150]}...")

    print("\n" + "=" * 80)
    print("Готово! Датасеты подготовлены для обучения retriever.")
    print("=" * 80)


def main():
    """CLI entry point."""
    fire.Fire(prepare_datasets)


if __name__ == "__main__":
    main()

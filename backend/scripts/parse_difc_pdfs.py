"""Парсер PDF документов DIFC для создания структурированного датасета.

Извлекает полный текст документов, сохраняет их как единицы (documents) или
разбивает на статьи/разделы если они явно выделены.

Не дробит на мелкие фрагменты - чанкирование будет отдельным этапом.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import pandas as pd
import pypdf


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, dict]:
    """Извлекает текст и метаданные из PDF."""
    reader = pypdf.PdfReader(pdf_path)

    # Извлекаем метаданные
    metadata = {}
    if reader.metadata:
        metadata = {
            "title": reader.metadata.get("/Title", ""),
            "creator": reader.metadata.get("/Creator", ""),
            "creation_date": reader.metadata.get("/CreationDate", ""),
        }

    # Извлекаем текст со всех страниц
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"

    return full_text, metadata


def clean_header_footer(text: str) -> str:
    """Убирает повторяющиеся заголовки и футеры на каждой странице."""
    # Убираем короткие строки в начале (типичные заголовки дел)
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Пропускаем очень короткие строки (менее 10 символов)
        if len(line.strip()) > 10:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_case_metadata(text: str) -> dict:
    """Извлекает метаданные о деле."""
    metadata = {
        "case_number": "",
        "case_date": "",
        "parties": [],
        "court": "",
    }

    # Case number
    case_match = re.search(r"Claim No:\s*([A-Z\s\-]+\d+/\d+)", text, re.IGNORECASE)
    if case_match:
        metadata["case_number"] = case_match.group(1).strip()

    # Date
    date_match = re.search(r"([A-Z]+\s+\d{1,2},\s+\d{4})", text)
    if date_match:
        metadata["case_date"] = date_match.group(1)

    # Court
    court_match = re.search(
        r"(COURT OF APPEAL|COURT OF FIRST INSTANCE|SMALL CLAIMS TRIBUNAL)",
        text,
        re.IGNORECASE,
    )
    if court_match:
        metadata["court"] = court_match.group(1)

    # Parties
    claimant_match = re.search(
        r"BETWEEN\s+(.*?)\s+(?:Claimant|Appellant)", text, re.DOTALL | re.IGNORECASE
    )
    if claimant_match:
        metadata["parties"].append(("Claimant", claimant_match.group(1).strip()))

    defendant_match = re.search(
        r"and\s+(.*?)\s+(?:Defendant|Respondent)", text, re.DOTALL | re.IGNORECASE
    )
    if defendant_match:
        metadata["parties"].append(("Defendant", defendant_match.group(1).strip()))

    return metadata


def detect_document_type(text: str) -> str:
    """Определяет тип документа (закон, судебное решение, регламент)."""
    text_lower = text.lower()

    if "law no." in text_lower or "difc law" in text_lower:
        return "law"
    elif "judgment" in text_lower or "order" in text_lower:
        return "court_decision"
    elif "regulation" in text_lower or "rules" in text_lower:
        return "regulation"
    else:
        return "other"


def extract_main_content(text: str, doc_type: str) -> str:
    """Извлекает основной контент, убирая титульники и оглавления."""

    # Убираем заголовок дела (первые строки до "BETWEEN" или "UPON")
    if doc_type == "court_decision":
        # Ищем начало основного текста
        main_start = None
        for pattern in [
            r"UPON",
            r"WHEREAS",
            r"IT IS HEREBY",
            r"PART \d+",
            r"Article \d+",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                main_start = match.start()
                break

        if main_start:
            text = text[main_start:]

    # Для законов пропускаем оглавление
    if doc_type == "law":
        # Ищем начало Part 1 или Article 1
        match = re.search(r"(?:PART 1|Article 1)", text, re.IGNORECASE)
        if match:
            text = text[match.start() :]

    return text.strip()


def split_into_articles(text: str, doc_type: str) -> Optional[List[Tuple[str, str]]]:
    """Разбивает документ на статьи/разделы если они есть.

    Returns:
        List of (article_number, article_text) или None если нет явных статей
    """
    articles = []

    if doc_type == "law":
        # Паттерн для статей типа "Article 1. Title"
        pattern = r"^(Article\s+\d+(?:\.\d+)*\.?\s*[^\n]*)\n(.+?)(?=^Article\s+\d+|$)"
        matches = list(
            re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        )

        if matches:
            for match in matches:
                article_num = match.group(1).strip()
                article_text = match.group(2).strip()

                # Объединяем заголовок и текст
                full_text = f"{article_num}\n\n{article_text}"

                # Минимальная длина статьи - 100 символов
                if len(full_text) > 100:
                    articles.append((article_num, full_text))

    elif doc_type == "regulation":
        # Для регламентов ищем PART или Section
        pattern = r"^((?:PART|Section)\s+\d+[^\n]*)\n(.+?)(?=^(?:PART|Section)\s+\d+|$)"
        matches = list(
            re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        )

        if matches:
            for match in matches:
                section_num = match.group(1).strip()
                section_text = match.group(2).strip()
                full_text = f"{section_num}\n\n{section_text}"

                if len(full_text) > 100:
                    articles.append((section_num, full_text))

    return articles if articles else None


def parse_all_pdfs(
    input_dir: str = "data/raw/docs_corpus",
    output_path: str = "data/processed/difc_documents.csv",
) -> None:
    """Парсит все PDF документы и сохраняет в CSV.

    Каждый документ сохраняется либо целиком, либо разбивается на статьи.
    Текст крупный, без мелкого дробления.

    Args:
        input_dir: Директория с PDF файлами
        output_path: Путь для сохранения результата
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ PDF файлы не найдены в {input_dir}")
        return

    print(f"📄 Найдено {len(pdf_files)} PDF файлов")
    print(f"🔄 Начинаем парсинг...\n")

    all_documents = []
    failed_files = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{idx}/{len(pdf_files)}] Обрабатываем: {pdf_path.name}")

            # Извлекаем текст
            full_text, metadata = extract_text_from_pdf(pdf_path)

            # Document ID
            doc_id = pdf_path.stem

            # Определяем тип документа
            doc_type = detect_document_type(full_text)

            # Извлекаем метаданные дела
            case_metadata = extract_case_metadata(full_text)

            # Извлекаем основной контент
            main_content = extract_main_content(full_text, doc_type)

            # Пробуем разбить на статьи
            articles = split_into_articles(main_content, doc_type)

            if articles:
                # Сохраняем каждую статью отдельно
                print(f"  ✓ Найдено {len(articles)} статей/разделов")

                for article_num, article_text in articles:
                    all_documents.append(
                        {
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "unit_type": "article",
                            "unit_id": f"{doc_id}_{re.sub(r'[^a-zA-Z0-9]', '_', article_num)}",
                            "unit_title": article_num,
                            "text": article_text,
                            "case_number": case_metadata.get("case_number", ""),
                            "case_date": case_metadata.get("case_date", ""),
                            "court": case_metadata.get("court", ""),
                            "doc_title": metadata.get("title", ""),
                            "doc_creation_date": metadata.get("creation_date", ""),
                        }
                    )
            else:
                # Сохраняем весь документ целиком
                print(
                    f"  ✓ Сохранен как единый документ ({len(main_content)} символов)"
                )

                all_documents.append(
                    {
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "unit_type": "full_document",
                        "unit_id": doc_id,
                        "unit_title": metadata.get("title", "")[:200],
                        "text": main_content,
                        "case_number": case_metadata.get("case_number", ""),
                        "case_date": case_metadata.get("case_date", ""),
                        "court": case_metadata.get("court", ""),
                        "doc_title": metadata.get("title", ""),
                        "doc_creation_date": metadata.get("creation_date", ""),
                    }
                )

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            failed_files.append(pdf_path.name)

    # Создаем DataFrame
    if all_documents:
        df = pd.DataFrame(all_documents)

        # Сохраняем в CSV
        df.to_csv(output_path, index=False)

        print(f"\n{'='*80}")
        print(f"✅ УСПЕШНО ЗАВЕРШЕНО")
        print(f"{'='*80}")
        print(f"📊 Всего обработано файлов: {len(pdf_files)}")
        print(f"📝 Всего единиц (документы/статьи): {len(df)}")
        print(f"💾 Результат сохранен: {output_path}")

        # Статистика по типам
        print(f"\n📈 Статистика по типам документов:")
        print(df["doc_type"].value_counts().to_string())

        print(f"\n📈 Статистика по типам единиц:")
        print(df["unit_type"].value_counts().to_string())

        # Длина текстов
        df["text_length"] = df["text"].str.len()
        print(f"\n📏 Длина текста:")
        print(f"  Средняя: {df['text_length'].mean():.0f} символов")
        print(f"  Медиана: {df['text_length'].median():.0f} символов")
        print(f"  Мин: {df['text_length'].min():.0f} символов")
        print(f"  Макс: {df['text_length'].max():.0f} символов")

        if failed_files:
            print(f"\n⚠️  Не удалось обработать {len(failed_files)} файлов:")
            for fname in failed_files:
                print(f"  - {fname}")
    else:
        print("\n❌ Не удалось извлечь ни одного документа")


if __name__ == "__main__":
    fire.Fire(parse_all_pdfs)

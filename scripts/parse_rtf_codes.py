"""Скрипт для парсинга RTF файлов с кодексами и создания pandas DataFrame."""

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from striprtf.striprtf import rtf_to_text


def extract_rtf_text(rtf_filepath: Path) -> str:
    """Извлекает текст из RTF файла."""
    try:
        with open(rtf_filepath, "rb") as f:
            rtf_content = f.read().decode("utf-8", errors="ignore")

        # Используем striprtf для конвертации
        text = rtf_to_text(rtf_content)
        return text
    except Exception as e:
        print(f"Ошибка при чтении RTF через striprtf: {e}")
        # Fallback: простая обработка вручную
        with open(rtf_filepath, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")

        # Декодируем Unicode escape последовательности \uXXXX?
        def decode_unicode(match):
            code = int(match.group(1))
            return chr(code) if code < 0x110000 else ""

        content = re.sub(r"\\u(\d+)\\?", decode_unicode, content)

        # Убираем RTF команды
        content = re.sub(r"\\[a-z]+\d*\s*", " ", content)
        content = re.sub(r"\{|\}", "", content)
        content = re.sub(r"\\par\s*", "\n", content)
        content = re.sub(r"\\line\s*", "\n", content)
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"\n\s*\n+", "\n\n", content)

        return content.strip()


def detect_codex_name(filename: str, filepath: Path = None) -> str:
    """Определяет название кодекса по имени файла или содержимому."""
    # Маппинг известных номеров на названия
    codex_map = {
        "63": "Уголовный кодекс РФ",
        "51": "Гражданский кодекс РФ",
        "145": "Бюджетный кодекс РФ",
        "188": "Жилищный кодекс РФ",
        "197": "Трудовой кодекс РФ",
    }

    # Ищем номер в имени файла
    match = re.search(r"N\s+(\d+)", filename)
    if match:
        num = match.group(1)
        codex_from_map = codex_map.get(num)
        if codex_from_map:
            return codex_from_map

    # Если не нашли по номеру, пытаемся определить по содержимому файла
    if filepath and filepath.exists():
        try:
            text = extract_rtf_text(filepath)
            text_lower = text[:1000].lower()  # Первые 1000 символов достаточно

            if "жилищный кодекс" in text_lower or "жилищное" in text_lower:
                return "Жилищный кодекс РФ"
            elif "трудовой кодекс" in text_lower or "трудов" in text_lower:
                return "Трудовой кодекс РФ"
            elif "уголовный кодекс" in text_lower or "уголовн" in text_lower:
                return "Уголовный кодекс РФ"
            elif "гражданский кодекс" in text_lower or "гражданск" in text_lower:
                return "Гражданский кодекс РФ"
            elif "бюджетный кодекс" in text_lower or "бюджетн" in text_lower:
                return "Бюджетный кодекс РФ"
        except Exception:
            pass

    # Пробуем определить по имени файла
    filename_lower = filename.lower()
    if "уголовн" in filename_lower or "63" in filename:
        return "Уголовный кодекс РФ"
    elif "гражданск" in filename_lower or "51" in filename:
        return "Гражданский кодекс РФ"
    elif "бюджет" in filename_lower or "145" in filename:
        return "Бюджетный кодекс РФ"
    elif "трудов" in filename_lower or "197" in filename:
        return "Трудовой кодекс РФ"
    elif "188" in filename:
        return "Жилищный кодекс РФ"

    return "Неизвестный кодекс"


def extract_articles(text: str) -> List[Tuple[str, str, str]]:
    """Извлекает статьи из текста. Возвращает список (номер статьи, название статьи, текст статьи)."""
    articles = []

    # Нормализуем переносы строк
    text = re.sub(r"\r\n|\r", "\n", text)
    # Убираем множественные пробелы, но сохраняем переносы строк
    text = re.sub(r"[ \t]+", " ", text)

    # Паттерн для поиска статей: "Статья N." или "Статья N" (но не "Глава" или "Раздел")
    # Ищем начало статьи, которая начинается с номера
    pattern = r"(?:^|\n)\s*(?:Статья|СТАТЬЯ|статья)\s+(\d+(?:\.\d+)?)[\.:]?\s*"

    matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))

    for i, match in enumerate(matches):
        article_num = match.group(1)

        # Берем текст от текущей статьи до следующей статьи или до конца
        start_pos = match.end()

        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
            article_text = text[start_pos:end_pos]
        else:
            article_text = text[start_pos:]

        # Очищаем текст статьи
        # Убираем заголовки глав и разделов в начале текста статьи
        article_text = re.sub(
            r"^(?:Глава|ГЛАВА|Раздел|РАЗДЕЛ|Часть|ЧАСТЬ)\s+\d+[\.:]?\s*.*?\n",
            "",
            article_text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Извлекаем название статьи
        # Название обычно идет сразу после номера статьи и до первого пункта (цифра + точка)
        # или до конца строки/абзаца
        article_text_stripped = article_text.strip()

        # Паттерн для поиска начала первого пункта
        # Ищем цифру с точкой в начале строки или после переноса строки
        point_match = re.search(r"^\s*(\d+)\.\s+", article_text_stripped, re.MULTILINE)

        if point_match:
            # Название - это текст до первого пункта
            title_end = point_match.start()
            article_title = article_text_stripped[:title_end].strip()
            # Текст статьи начинается с первого пункта
            article_body = article_text_stripped[point_match.start() :].strip()
        else:
            # Если нет пунктов, название - первая строка или абзац
            # Берем первую строку до переноса строки или до конца, если нет переносов
            lines = article_text_stripped.split("\n", 1)
            if len(lines) > 1 and len(lines[0]) > 0 and len(lines[0]) < 200:
                # Если первая строка не слишком длинная, считаем её названием
                article_title = lines[0].strip()
                article_body = lines[1].strip() if len(lines) > 1 else ""
            else:
                # Если первая строка очень длинная, возможно это весь текст
                # Пробуем найти название по другим признакам (до точки с заглавной буквы следующего предложения)
                title_match = re.match(
                    r"^([^.]+\.[\s\n]+[А-Я])", article_text_stripped, re.DOTALL
                )
                if title_match:
                    article_title = title_match.group(1).split("\n")[0].strip()
                    article_body = article_text_stripped[len(article_title) :].strip()
                else:
                    # Если не удалось выделить название, оставляем пустым
                    article_title = ""
                    article_body = article_text_stripped

        # Нормализуем пробелы, но сохраняем структуру
        article_title = re.sub(r"[ \t]+", " ", article_title)
        article_title = re.sub(r"\n+", " ", article_title).strip()

        article_body = re.sub(r"[ \t]+", " ", article_body)
        article_body = re.sub(r"\n\s*\n+", "\n", article_body)
        article_body = article_body.strip()

        # Убираем очень короткие фрагменты или только заголовки
        if (
            article_body
            and len(article_body) > 30
            and not re.match(
                r"^(?:Глава|Раздел|Часть|ЧАСТЬ|ГЛАВА|РАЗДЕЛ)\s+\d+", article_body
            )
        ):
            articles.append((article_num, article_title, article_body))

    return articles


def parse_rtf_file(filepath: Path) -> List[Tuple[str, str, str, str]]:
    """Парсит RTF файл и возвращает список (номер статьи, название статьи, текст статьи, кодекс)."""
    # Определяем кодекс с учетом содержимого файла
    codex_name = detect_codex_name(filepath.name, filepath)

    # Извлекаем текст из RTF файла
    text = extract_rtf_text(filepath)

    # Извлекаем статьи
    articles = extract_articles(text)

    # Добавляем название кодекса к каждой статье
    result = [(num, title, text, codex_name) for num, title, text in articles]

    return result


def main():
    """Основная функция."""
    data_dir = Path("data/raw")

    if not data_dir.exists():
        print(f"Директория {data_dir} не найдена!")
        return

    all_articles = []

    # Обрабатываем все RTF файлы
    for rtf_file in data_dir.glob("*.rtf"):
        print(f"Обработка файла: {rtf_file.name}")
        try:
            articles = parse_rtf_file(rtf_file)
            all_articles.extend(articles)
            print(f"  Найдено статей: {len(articles)}")
        except Exception as e:
            print(f"  Ошибка при обработке {rtf_file.name}: {e}")
            continue

    # Создаем DataFrame
    if all_articles:
        df = pd.DataFrame(
            all_articles,
            columns=["Номер статьи", "Название статьи", "Текст статьи", "Кодекс"],
        )

        # Убираем дубликаты по номеру статьи и кодексу
        df = df.drop_duplicates(subset=["Номер статьи", "Кодекс"], keep="first")

        # Сортируем по кодексу и номеру статьи
        df["Номер статьи (число)"] = (
            df["Номер статьи"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float, errors="ignore")
        )
        df = df.sort_values(["Кодекс", "Номер статьи (число)"])
        df = df.drop(columns=["Номер статьи (число)"])

        # Сохраняем в CSV
        output_file = Path("data/processed/articles.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\nВсего статей обработано: {len(df)}")
        print(f"Сохранено в: {output_file}")
        print("\nСтатистика по кодексам:")
        print(df["Кодекс"].value_counts())
        print("\nПервые 5 строк:")
        print(df.head(5)[["Номер статьи", "Название статьи", "Кодекс"]].to_string())
        print("\nПример статьи:")
        if len(df) > 0:
            sample = df.iloc[0]
            print(
                f"Статья {sample['Номер статьи']} ({sample['Кодекс']}): {sample['Название статьи']}"
            )
            print(
                f"Текст: {sample['Текст статьи'][:200]}..."
                if len(sample["Текст статьи"]) > 200
                else f"Текст: {sample['Текст статьи']}"
            )
    else:
        print("Статьи не найдены!")


if __name__ == "__main__":
    main()

"""Скрипт для отправки submission в систему оценки."""

import sys
import zipfile
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "starter_kit"))

from arlc import EvaluationClient

load_dotenv()


def create_code_archive(
    archive_path: str = "code_archive.zip",
    root_dir: str = ".",
) -> Path:
    """Создает архив с кодом для submission.

    Args:
        archive_path: Путь к архиву
        root_dir: Корневая директория проекта

    Returns:
        Путь к созданному архиву
    """
    archive_file = Path(archive_path)
    root = Path(root_dir).resolve()

    print(f"📦 Создание архива кода...")

    # Исключаем ненужные директории и файлы
    exclude_dirs = {
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "data",
        "mlruns",
        "logs",
        "notebooks",
    }
    exclude_files = {
        ".env",
        "submission.json",
        "code_archive.zip",
        ".DS_Store",
        "*.pyc",
        "*.pyo",
    }

    # Файлы и папки которые нужно включить
    include_paths = [
        "russian_laws",  # Основной код
        "conf",  # Конфиги
        "scripts/create_submission.py",  # Скрипт генерации
        "scripts/parse_difc_pdfs.py",  # Парсер
        "scripts/index_difc_docs_chunked.py",  # Индексация
        "pyproject.toml",  # Зависимости
        "uv.lock",  # Lock file
        ".env.example",  # Пример env
    ]

    archive_file.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for include_path in include_paths:
            path = root / include_path

            if not path.exists():
                print(f"  ⚠️  Пропущено (не найдено): {include_path}")
                continue

            if path.is_dir():
                # Добавляем директорию рекурсивно
                for file_path in path.rglob("*"):
                    if not file_path.is_file():
                        continue

                    # Проверяем исключения
                    if any(excl in file_path.parts for excl in exclude_dirs):
                        continue
                    if file_path.name in exclude_files:
                        continue

                    arcname = file_path.relative_to(root)
                    zf.write(file_path, arcname)

            elif path.is_file():
                arcname = path.relative_to(root)
                zf.write(path, arcname)

    print(f"✓ Архив создан: {archive_file.absolute()}")
    print(f"  Размер: {archive_file.stat().st_size / 1024:.1f} KB")

    return archive_file


def create_env_example(output_path: str = ".env.example") -> None:
    """Создает .env.example для submission."""
    env_example = """# API Router (для LLM и эмбеддингов)
API_KEY_ROUTER=your-api-key-here

# Qdrant Cloud
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key

# AWS (если используется)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret

# Evaluation API (для соревнования)
EVAL_API_KEY=your-eval-api-key
EVAL_BASE_URL=https://platform.agentic-challenge.ai/api/v1
"""

    with open(output_path, "w") as f:
        f.write(env_example)

    print(f"✓ Создан {output_path}")


def submit_solution(
    submission_path: str = "submission.json",
    code_archive_path: str = "code_archive.zip",
    create_archive: bool = True,
) -> dict:
    """Отправляет submission в систему оценки.

    Args:
        submission_path: Путь к submission.json
        code_archive_path: Путь к архиву с кодом
        create_archive: Создать архив автоматически

    Returns:
        Ответ от API
    """
    print(f"🚀 Отправка submission в систему оценки")
    print(f"{'='*80}\n")

    # Проверяем submission.json
    submission_file = Path(submission_path)
    if not submission_file.exists():
        print(f"❌ Файл не найден: {submission_file}")
        return {}

    print(f"✓ submission.json найден ({submission_file.stat().st_size / 1024:.1f} KB)")

    # Создаем .env.example если его нет
    env_example = Path(".env.example")
    if not env_example.exists():
        create_env_example()

    # Создаем архив с кодом
    archive_file = Path(code_archive_path)
    if create_archive or not archive_file.exists():
        archive_file = create_code_archive(
            archive_path=code_archive_path,
            root_dir=".",
        )
    else:
        print(f"✓ Используем существующий архив: {archive_file}")

    # Отправляем через API
    print(f"\n📤 Отправка submission...")
    try:
        client = EvaluationClient.from_env()
        response = client.submit_submission(submission_file, archive_file)

        print(f"\n{'='*80}")
        print(f"✅ SUBMISSION ОТПРАВЛЕН")
        print(f"{'='*80}")
        print(f"UUID: {response.get('uuid')}")
        print(f"Status: {response.get('status')}")
        print(f"Phase: {response.get('phase')}")
        print(f"Created: {response.get('created_at')}")

        # Проверяем статус
        if response.get("uuid"):
            print(f"\n📊 Для проверки статуса используй:")
            print(f"   UUID: {response['uuid']}")

        return response

    except Exception as e:
        print(f"\n❌ Ошибка отправки: {e}")
        return {}


if __name__ == "__main__":
    fire.Fire(submit_solution)

"""Простой скрипт для отправки submission."""

import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "starter_kit"))

from arlc import EvaluationClient

load_dotenv()


def submit(
    submission_path: str = "submission.json",
    code_archive_path: str = "code_archive.zip",
):
    """Отправляет submission в систему оценки.

    Args:
        submission_path: Путь к submission.json
        code_archive_path: Путь к архиву с кодом
    """
    submission_file = Path(submission_path)
    archive_file = Path(code_archive_path)

    if not submission_file.exists():
        print(f"❌ Не найден: {submission_file}")
        return

    if not archive_file.exists():
        print(f"❌ Не найден: {archive_file}")
        return

    print(f"📤 Отправка submission...")
    print(f"  submission: {submission_file}")
    print(f"  archive: {archive_file}")

    try:
        client = EvaluationClient.from_env()
        response = client.submit_submission(submission_file, archive_file)

        print(f"\n✅ УСПЕШНО ОТПРАВЛЕНО")
        print(f"\nUUID: {response.get('uuid')}")
        print(f"Status: {response.get('status')}")
        print(f"Phase: {response.get('phase')}")

        return response

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return {}


if __name__ == "__main__":
    fire.Fire(submit)

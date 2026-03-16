"""Скрипт для создания submission.json для соревнования ARLC."""

import json
import sys
import time
from pathlib import Path

import fire
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from russian_laws.embeddings import EmbeddingModel
from russian_laws.generator import LLMGenerator
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder

load_dotenv()


def parse_answer_to_json_type(answer_str: str, answer_type: str):
    """Конвертирует строковый ответ в правильный JSON тип для submission.

    Args:
        answer_str: Ответ от LLM (строка)
        answer_type: Тип ответа из вопроса

    Returns:
        Значение правильного типа для JSON (str, int, float, bool, list, None)
    """
    answer_str = answer_str.strip()

    # Null обрабатывается отдельно
    if answer_str.lower() == "null":
        return None

    if answer_type == "number":
        try:
            # Пробуем распарсить как число
            if "." in answer_str:
                return float(answer_str)
            else:
                return int(answer_str)
        except ValueError:
            # Если не удалось - возвращаем строку (для отладки)
            return answer_str

    elif answer_type == "boolean":
        # Конвертируем в JSON boolean
        if answer_str.lower() in ("true", "yes", "1"):
            return True
        elif answer_str.lower() in ("false", "no", "0"):
            return False
        # Если не распознали - возвращаем строку
        return answer_str

    elif answer_type == "names":
        # Парсим JSON массив
        try:
            parsed = json.loads(answer_str)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        # Если не удалось распарсить - возвращаем как есть
        return answer_str

    # Для name, date, free_text возвращаем строку
    return answer_str


def create_submission(
    questions_path: str = "data/raw/docs_corpus/questions.json",
    collection_name: str = "difc_docs_hybrid",
    output_path: str = "submission.json",
    limit: int = 5,
) -> None:
    """Создает submission.json для соревнования.

    Args:
        questions_path: Путь к файлу с вопросами
        collection_name: Название коллекции в Qdrant
        output_path: Путь для сохранения submission.json
        limit: Количество документов для поиска
    """
    print(f"🚀 Создание submission для соревнования")
    print(f"{'='*80}\n")

    # Загружаем конфигурацию
    print("📋 Загрузка конфигурации...")
    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="config")

    config.qdrant.collection_name = collection_name
    print(f"✓ Конфигурация загружена")
    print(f"  Модель: {config.generator.model}")
    print(f"  Коллекция: {collection_name}")
    print(f"  Hybrid search: {config.qdrant.get('hybrid', {}).get('enabled', False)}\n")

    # Инициализируем компоненты
    print("🔧 Инициализация компонентов...")
    embedding_model = EmbeddingModel(config)
    qdrant_manager = QdrantManager(config)

    hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
    sparse_encoder = SparseEncoder(config) if hybrid_enabled else None

    llm_generator = LLMGenerator(config)
    print("✓ Все компоненты инициализированы\n")

    # Загружаем вопросы
    print(f"📄 Загрузка вопросов из {questions_path}...")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"✓ Загружено {len(questions)} вопросов\n")

    # Формируем submission
    answers = []

    print("💭 Генерация ответов...")
    for item in tqdm(questions):
        question_id = item["id"]
        question_text = item["question"]
        answer_type = item.get("answer_type", "free_text")

        # Замеряем время начала
        start_time = time.perf_counter()

        # Поиск релевантных документов
        query_vector = embedding_model.encode([question_text])[0].cpu().tolist()

        if hybrid_enabled and sparse_encoder is not None:
            sparse_query = sparse_encoder.encode(question_text)
            results = qdrant_manager.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_query,
                limit=limit,
            )
        else:
            results = qdrant_manager.search(
                query_vector=query_vector,
                limit=limit,
            )

        # Собираем контексты и страницы для телеметрии
        contexts = []
        retrieval_pages = {}  # doc_id -> set of page_numbers

        for point in results:
            payload = point.payload or {}
            context_text = payload.get("parent_text") or payload.get("text", "")
            contexts.append(context_text)

            # Собираем doc_id и page_number для телеметрии
            # doc_id должен быть БЕЗ расширения .pdf (только хеш)
            doc_id = payload.get("doc_id", "")
            page_number = payload.get("page_number")

            if doc_id and page_number:
                if doc_id not in retrieval_pages:
                    retrieval_pages[doc_id] = set()
                retrieval_pages[doc_id].add(int(page_number))

        # Генерируем ответ с телеметрией
        context_combined = "\n\n---\n\n".join(contexts)

        # Время генерации (без streaming ttft = total_time)
        gen_start = time.perf_counter()

        answer_raw, usage_info = llm_generator.generate(
            query=question_text,
            context=context_combined,
            answer_type=answer_type,
            return_usage=True,
        )

        # Время генерации
        end_time = time.perf_counter()
        total_time_ms = int((end_time - start_time) * 1000)
        ttft_ms = total_time_ms  # Без streaming ttft = total_time

        # Конвертируем ответ в правильный JSON тип
        answer_typed = parse_answer_to_json_type(answer_raw, answer_type)

        # Формируем retrieval refs для submission
        retrieved_chunk_pages = [
            {"doc_id": doc_id, "page_numbers": sorted(list(pages))}
            for doc_id, pages in sorted(retrieval_pages.items())
        ]

        # Формируем telemetry
        telemetry = {
            "timing": {
                "ttft_ms": ttft_ms,
                "tpot_ms": 0,  # Нет streaming
                "total_time_ms": total_time_ms,
            },
            "retrieval": {
                "retrieved_chunk_pages": retrieved_chunk_pages,
            },
            "usage": {
                "input_tokens": usage_info.get("input_tokens", 0),
                "output_tokens": usage_info.get("output_tokens", 0),
            },
            "model_name": config.generator.model,
        }

        # Добавляем ответ
        answers.append(
            {
                "question_id": question_id,
                "answer": answer_typed,
                "telemetry": telemetry,
            }
        )

    # Формируем submission
    submission = {
        "architecture_summary": "Hybrid search (dense multilingual-e5-large + BM25 sparse) with parent-child chunking",
        "answers": answers,
    }

    # Сохраняем
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ SUBMISSION СОЗДАН")
    print(f"{'='*80}")
    print(f"📊 Всего вопросов: {len(questions)}")
    print(f"📊 Всего ответов: {len(answers)}")
    print(f"💾 Сохранено: {output_file.absolute()}")
    print(f"\n📝 Следующие шаги:")
    print(f"  1. Проверь submission.json")
    print(f"  2. Создай архив с кодом (code_archive.zip)")
    print(f"  3. Отправь через систему оценки")


if __name__ == "__main__":
    fire.Fire(create_submission)

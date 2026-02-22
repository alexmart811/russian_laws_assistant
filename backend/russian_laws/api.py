"""FastAPI сервис для RAG-системы поиска статей российского законодательства."""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import BaseModel
from russian_laws.embeddings import EmbeddingModel
from russian_laws.generator import LLMGenerator
from russian_laws.qdrant_manager import QdrantManager
from russian_laws.sparse_encoder import SparseEncoder

load_dotenv()


class EmbedRequest(BaseModel):
    """Запрос на генерацию эмбеддинга."""

    text: str


class EmbedResponse(BaseModel):
    """Ответ с эмбеддингом."""

    embedding: list[float]
    dimension: int


class SearchRequest(BaseModel):
    """Запрос на поиск релевантных фрагментов."""

    query: str
    limit: int = 10
    score_threshold: float | None = None


class SearchResult(BaseModel):
    """Результат поиска."""

    article_id: int
    article_title: str
    article_text: str
    codex: str
    score: float


class SearchResponse(BaseModel):
    """Ответ с результатами поиска."""

    results: list[SearchResult]
    query_embedding: list[float]


class AnswerRequest(BaseModel):
    """Запрос на подготовку контекста для LLM."""

    query: str
    limit: int = 5
    score_threshold: float | None = None


class AnswerContext(BaseModel):
    """Контекст для LLM."""

    query: str
    relevant_articles: list[dict[str, Any]]
    context_text: str


class GenerateRequest(BaseModel):
    """Запрос на генерацию ответа."""

    query: str
    limit: int = 5
    score_threshold: float | None = None


class GenerateResponse(BaseModel):
    """Ответ с сгенерированным текстом."""

    query: str
    answer: str
    sources: list[dict[str, Any]]


app = FastAPI(
    title="Russian Laws RAG Service",
    description="RAG-сервис для поиска статей российского законодательства",
    version="0.1.0",
)

# Настройка CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели и менеджеров
embedding_model: EmbeddingModel | None = None
sparse_encoder: SparseEncoder | None = None
qdrant_manager: QdrantManager | None = None
llm_generator: LLMGenerator | None = None
config: DictConfig | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Инициализация при запуске приложения."""
    global embedding_model, sparse_encoder, qdrant_manager, llm_generator, config

    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="config")

    print(f"Загрузка модели эмбеддингов: {config.embedding.model_name}...")
    try:
        embedding_model = EmbeddingModel(config)
        print("✓ Модель эмбеддингов загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        raise

    # Инициализируем sparse encoder если включен гибридный режим
    hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
    if hybrid_enabled:
        print("Инициализация sparse encoder...")
        try:
            sparse_encoder = SparseEncoder(config)
            print("✓ Sparse encoder инициализирован")
        except Exception as e:
            print(f"✗ Ошибка инициализации sparse encoder: {e}")
            raise

    print("Инициализация Qdrant...")
    qdrant_manager = QdrantManager(config)
    print("✓ Qdrant инициализирован")

    print(f"Инициализация LLM генератора: {config.generator.model}...")
    try:
        llm_generator = LLMGenerator(config)
        print("LLM генератор инициализирован")
    except Exception as e:
        print(f"Ошибка инициализации генератора: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Очистка при завершении приложения."""
    global qdrant_manager
    if qdrant_manager:
        qdrant_manager.close()


@app.get("/")
async def root() -> dict[str, str]:
    """Корневой эндпоинт."""
    return {
        "service": "Russian Laws RAG Service",
        "status": "running",
        "endpoints": "/embed, /search, /answer, /generate",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверка здоровья сервиса."""
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy", "model_loaded": "true"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Генерирует эмбеддинг для текста."""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embedding = embedding_model.encode([request.text])[0]
        embedding_list = embedding.cpu().tolist()

        return EmbedResponse(
            embedding=embedding_list,
            dimension=len(embedding_list),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Ищет релевантные фрагменты по запросу (гибридный поиск)."""
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        query_embedding = embedding_model.encode([request.query])[0]
        query_vector = query_embedding.cpu().tolist()

        # Гибридный поиск если включен
        hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
        if hybrid_enabled and sparse_encoder is not None:
            sparse_query = sparse_encoder.encode(request.query)
            results = qdrant_manager.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_query,
                limit=request.limit * 3,
                score_threshold=request.score_threshold,
            )
        else:
            # Fallback на обычный dense поиск
            results = qdrant_manager.search(
                query_vector=query_vector,
                limit=request.limit * 3,
                score_threshold=request.score_threshold,
            )

        seen_article_ids: set[int] = set()
        search_results: list[SearchResult] = []

        for point in results:
            payload = point.payload or {}
            article_id = payload.get("article_id")

            if article_id is None or article_id in seen_article_ids:
                continue

            seen_article_ids.add(article_id)
            search_results.append(
                SearchResult(
                    article_id=article_id,
                    article_title=payload.get("article_title", ""),
                    article_text=payload.get("article_text", ""),
                    codex=payload.get("codex", ""),
                    score=float(point.score) if hasattr(point, "score") else 0.0,
                )
            )

            if len(search_results) >= request.limit:
                break

        return SearchResponse(
            results=search_results,
            query_embedding=query_vector,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.post("/answer", response_model=AnswerContext)
async def answer(request: AnswerRequest) -> AnswerContext:
    """Подготавливает контекст для LLM на основе релевантных статей."""
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Выполняем прямой поиск для получения parent_text
        query_embedding = embedding_model.encode([request.query])[0]
        query_vector = query_embedding.cpu().tolist()

        # Гибридный поиск если включен
        hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
        if hybrid_enabled and sparse_encoder is not None:
            sparse_query = sparse_encoder.encode(request.query)
            results = qdrant_manager.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_query,
                limit=request.limit,
                score_threshold=request.score_threshold,
            )
        else:
            results = qdrant_manager.search(
                query_vector=query_vector,
                limit=request.limit,
                score_threshold=request.score_threshold,
            )

        relevant_articles = []
        context_parts = []
        seen_parent_ids: set[str] = set()

        for point in results:
            payload = point.payload or {}

            # Для parent-child используем parent_text, иначе article_text
            parent_id = payload.get("parent_id")
            if parent_id and parent_id in seen_parent_ids:
                # Пропускаем дубликаты parent'ов
                continue

            if parent_id:
                seen_parent_ids.add(parent_id)

            # Извлекаем текст для контекста (parent_text если есть, иначе article_text)
            context_text_chunk = payload.get("parent_text") or payload.get(
                "article_text", ""
            )

            article_data = {
                "article_id": payload.get("article_id"),
                "article_title": payload.get("article_title", ""),
                "article_text": context_text_chunk,
                "codex": payload.get("codex", ""),
                "score": float(point.score) if hasattr(point, "score") else 0.0,
            }
            relevant_articles.append(article_data)

            context_parts.append(
                f"Статья {payload.get('article_id')} ({payload.get('codex', 'N/A')}):\n"
                f"Название: {payload.get('article_title', '')}\n"
                f"Текст: {context_text_chunk}\n"
                f"Релевантность: {point.score:.4f}\n"
            )

        context_text = "\n---\n".join(context_parts)

        return AnswerContext(
            query=request.query,
            relevant_articles=relevant_articles,
            context_text=context_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error preparing context: {str(e)}"
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Генерирует ответ на вопрос пользователя на основе релевантных статей."""
    if embedding_model is None or qdrant_manager is None or llm_generator is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        answer_context = await answer(
            AnswerRequest(
                query=request.query,
                limit=request.limit,
                score_threshold=request.score_threshold,
            )
        )

        generated_answer = llm_generator.generate(
            query=request.query,
            context=answer_context.context_text,
        )

        return GenerateResponse(
            query=request.query,
            answer=generated_answer,
            sources=answer_context.relevant_articles,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

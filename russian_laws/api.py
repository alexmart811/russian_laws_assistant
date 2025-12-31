"""FastAPI сервис для RAG-системы поиска статей российского законодательства."""

from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import BaseModel

from russian_laws.qdrant_manager import QdrantManager
from russian_laws.train import RetrievalModel


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


app = FastAPI(
    title="Russian Laws RAG Service",
    description="RAG-сервис для поиска статей российского законодательства",
    version="0.1.0",
)

model: RetrievalModel | None = None
qdrant_manager: QdrantManager | None = None
config: DictConfig | None = None


def load_model(checkpoint_path: str) -> RetrievalModel:
    """Загружает обученную модель из checkpoint.

    Args:
        checkpoint_path: Путь к checkpoint файлу

    Returns:
        Загруженная модель
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    hyperparams = checkpoint_data.get("hyper_parameters", {})
    model_name = hyperparams.get("model_name", cfg.embedding.model_name)

    loaded_model = RetrievalModel.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        config=cfg,
        model_name=model_name,
        strict=False,
    )

    loaded_model.eval()
    for param in loaded_model.parameters():
        param.requires_grad = False

    return loaded_model


@app.on_event("startup")
async def startup_event() -> None:
    """Инициализация при запуске приложения."""
    global model, qdrant_manager, config

    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="config")

    checkpoint_path = Path("data/models/rubert_base/last.ckpt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {checkpoint_path}. "
            "Убедитесь, что обучение завершено и модель сохранена."
        )

    print(f"Загрузка модели из {checkpoint_path}...")
    try:
        model = load_model(str(checkpoint_path))
        device = config.embedding.device
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        print("✓ Модель загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        raise

    print("Инициализация Qdrant...")
    qdrant_manager = QdrantManager(config)
    print("✓ Qdrant инициализирован")


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
        "endpoints": ["/embed", "/search", "/answer"],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверка здоровья сервиса."""
    if model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Генерирует эмбеддинг для текста.

    Args:
        request: Запрос с текстом

    Returns:
        Эмбеддинг текста
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with torch.no_grad():
            embedding = model.encode([request.text])[0]
            embedding_list = embedding.cpu().numpy().tolist()

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
    """Ищет релевантные фрагменты по запросу.

    Args:
        request: Запрос с текстом и параметрами поиска

    Returns:
        Список релевантных статей с метаданными
    """
    if model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        with torch.no_grad():
            query_embedding = model.encode([request.query])[0]
            query_vector = query_embedding.cpu().numpy().tolist()

        results = qdrant_manager.search(
            query_vector=query_vector,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        search_results = []
        for point in results:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    article_id=point.id,
                    article_title=payload.get("article_title", ""),
                    article_text=payload.get("article_text", ""),
                    codex=payload.get("codex", ""),
                    score=float(point.score) if hasattr(point, "score") else 0.0,
                )
            )

        return SearchResponse(
            results=search_results,
            query_embedding=query_vector,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.post("/answer", response_model=AnswerContext)
async def answer(request: AnswerRequest) -> AnswerContext:
    """Подготавливает контекст для LLM на основе релевантных статей.

    Args:
        request: Запрос с текстом вопроса

    Returns:
        Контекст с релевантными статьями для LLM
    """
    if model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        search_response = await search(
            SearchRequest(
                query=request.query,
                limit=request.limit,
                score_threshold=request.score_threshold,
            )
        )

        relevant_articles = []
        context_parts = []

        for result in search_response.results:
            article_data = {
                "article_id": result.article_id,
                "article_title": result.article_title,
                "article_text": result.article_text,
                "codex": result.codex,
                "score": result.score,
            }
            relevant_articles.append(article_data)

            context_parts.append(
                f"Статья {result.article_id} ({result.codex}):\n"
                f"Название: {result.article_title}\n"
                f"Текст: {result.article_text}\n"
                f"Релевантность: {result.score:.4f}\n"
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

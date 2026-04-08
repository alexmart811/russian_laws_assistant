"""FastAPI сервис для RAG-системы поиска статей российского законодательства."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from russian_laws.reranker import Reranker
from russian_laws.sparse_encoder import SparseEncoder

load_dotenv()


# ─── Pydantic модели ───────────────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    score_threshold: float | None = None


class SearchResult(BaseModel):
    article_id: int
    article_title: str
    article_text: str
    codex: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_embedding: list[float]


class AnswerRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float | None = None


class AnswerContext(BaseModel):
    query: str
    relevant_articles: list[dict[str, Any]]
    context_text: str


class GenerateRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float | None = None
    answer_type: str = "free_text"


class GenerateResponse(BaseModel):
    query: str
    answer: str
    answer_type: str
    sources: list[dict[str, Any]]


# ─── Состояние приложения ───────────────────────────────────────────────────────


class AppState:
    """Контейнер для компонентов RAG-пайплайна (без глобальных переменных)."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.embedding_model = EmbeddingModel(config)

        hybrid_enabled = config.qdrant.get("hybrid", {}).get("enabled", False)
        self.sparse_encoder = SparseEncoder(config) if hybrid_enabled else None

        reranker_enabled = config.reranker.get("enabled", False)
        self.reranker = Reranker(config) if reranker_enabled else None

        self.qdrant_manager = QdrantManager(config)
        self.llm_generator = LLMGenerator(config)

    def retrieve(
        self,
        query: str,
        limit: int,
        score_threshold: float | None = None,
    ) -> tuple[list[Any], list[float]]:
        """Единый метод retrieval: embed → hybrid/dense search → rerank."""
        query_vector = self.embedding_model.encode([query])[0].cpu().tolist()

        reranker_enabled = self.config.reranker.get("enabled", False)
        search_limit = (
            self.config.reranker.get("candidates_limit", 15)
            if reranker_enabled
            else limit
        )

        hybrid_enabled = self.config.qdrant.get("hybrid", {}).get("enabled", False)
        if hybrid_enabled and self.sparse_encoder is not None:
            sparse_query = self.sparse_encoder.encode(query)
            results = self.qdrant_manager.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_query,
                limit=search_limit,
                score_threshold=score_threshold,
            )
        else:
            results = self.qdrant_manager.search(
                query_vector=query_vector,
                limit=search_limit,
                score_threshold=score_threshold,
            )

        if reranker_enabled and self.reranker is not None and results:
            documents_for_rerank = [
                (p.payload or {}).get("parent_text")
                or (p.payload or {}).get("article_text", "")
                for p in results
            ]
            ranked_indices = self.reranker.rerank(
                query=query,
                documents=documents_for_rerank,
                top_k=limit,
            )
            results = [results[idx] for idx in ranked_indices]

        return results, query_vector

    def close(self) -> None:
        self.qdrant_manager.close()


# ─── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Инициализация и очистка компонентов."""
    config_dir = Path("conf").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="config")

    _app.state.rag = AppState(config)
    print("Все компоненты инициализированы")

    yield

    _app.state.rag.close()


app = FastAPI(
    title="Russian Laws RAG Service",
    description="RAG-сервис для поиска статей российского законодательства",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_state(app_instance: FastAPI) -> AppState:
    state: AppState | None = getattr(app_instance.state, "rag", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return state


# ─── Эндпоинты ──────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Russian Laws RAG Service",
        "status": "running",
        "endpoints": "/embed, /search, /answer, /generate",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    _get_state(app)
    return {"status": "healthy", "model_loaded": "true"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Генерирует эмбеддинг для текста."""
    state = _get_state(app)
    try:
        embedding = state.embedding_model.encode([request.text])[0]
        embedding_list = embedding.cpu().tolist()
        return EmbedResponse(embedding=embedding_list, dimension=len(embedding_list))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка эмбеддинга: {e}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Ищет релевантные статьи по запросу."""
    state = _get_state(app)
    try:
        results, query_vector = state.retrieve(
            query=request.query,
            limit=request.limit,
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

        return SearchResponse(results=search_results, query_embedding=query_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {e}")


@app.post("/answer", response_model=AnswerContext)
async def answer(request: AnswerRequest) -> AnswerContext:
    """Подготавливает контекст для LLM на основе релевантных статей."""
    state = _get_state(app)
    try:
        results, _ = state.retrieve(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        relevant_articles = []
        context_parts = []
        seen_parent_ids: set[str] = set()

        for point in results:
            payload = point.payload or {}

            parent_id = payload.get("parent_id")
            if parent_id and parent_id in seen_parent_ids:
                continue
            if parent_id:
                seen_parent_ids.add(parent_id)

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
        raise HTTPException(status_code=500, detail=f"Ошибка контекста: {e}")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Генерирует ответ на вопрос пользователя на основе релевантных статей."""
    state = _get_state(app)
    try:
        answer_context = await answer(
            AnswerRequest(
                query=request.query,
                limit=request.limit,
                score_threshold=request.score_threshold,
            )
        )

        result = state.llm_generator.generate(
            query=request.query,
            context=answer_context.context_text,
            answer_type=request.answer_type,
        )
        generated_answer = result if isinstance(result, str) else result[0]

        return GenerateResponse(
            query=request.query,
            answer=generated_answer,
            answer_type=request.answer_type,
            sources=answer_context.relevant_articles,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

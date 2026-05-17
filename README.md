# Юридический ассистент на основе LLM

## Постановка задачи

Разработка модуля интеллектуального поиска (retriever) для юридического ассистента на основе LLM. Модуль находит релевантные фрагменты законов и нормативных актов по запросу пользователя и формирует контекст для генерации ответа.

Для корректного поиска требуется векторная модель, способная:

- понимать контекст нормативно-правовых актов,
- находить семантически близкие фрагменты,
- корректно ранжировать результаты.

## Метрики

### Retrieval

- **Recall@k** — найден ли правильный фрагмент среди первых k результатов
- **MRR** — позиция первого релевантного ответа
- **nDCG@k** — качество ранжирования с учётом позиции

Метрики вычисляются для k = [1, 3, 5, 10].

### RAG

- **Faithfulness** — соответствие ответа контексту (LLM-as-judge)
- **Answer Relevance** — семантическая близость ответа к вопросу (embeddings)
- **ROUGE-1/2/L** — лексическое перекрытие с эталонным текстом

## Конфигурация

- **Поиск:** Hybrid search (dense + sparse / BM25)
- **Чанкирование:** Parent-child (маленькие child-чанки для поиска, большие parent-чанки для LLM)
- **Модель:** `Roflmax/bge-m3-legal-ru-cocktail-40-60` (дообученная BGE-M3)

## Датасет

### [RRNCB RAG Benchmark](https://fractalagents.ai/rrncb-rag-benchmark/)

Публичный бенчмарк для оценки качества поиска по российскому законодательству. Содержит 200 пар «вопрос — эталонный ответ» с привязкой к конкретному документу-источнику (PDF). Охватывает 53 документа: российские кодексы (Налоговый, Арбитражный процессуальный, Жилищный и др.) и нормативно-технические документы.

## Архитектура

- **Embeddings:** `Roflmax/bge-m3-legal-ru-cocktail-40-60`
- **Vector store:** Qdrant (hybrid search) / Milvus / Weaviate
- **Reranker:** `BAAI/bge-reranker-v2-m3`
- **LLM:** через OpenAI-совместимый роутер (vsellm.ru)

### Пайплайн

1. Предобработка текстов статей (обогащение префиксом `[кодекс | ст. № | название]`)
2. Parent-child чанкирование
3. Индексация dense + sparse векторов
4. Hybrid search при запросе (dense + BM25, weighted sum fusion)
5. Опциональный reranking через CrossEncoder
6. Генерация ответа на основе parent-текстов

## Setup

### Требования

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker + Docker Compose
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (рекомендуется)

### Установка

```bash
cd backend
uv sync
```

### Переменные окружения

```bash
# backend/.env
API_KEY_ROUTER="your-api-key"   # ключ роутера vsellm.ru
```

### Запуск (Docker Compose)

Перед запуском убедитесь, что CSV со статьями лежит по пути `backend/data/processed/articles_parsed.csv`.

```bash
docker compose up -d --build
```

При первом запуске:
1. Поднимается стек Milvus (etcd + minio + milvus)
2. Backend ждёт готовности Milvus
3. Если коллекция `legal_docs_bge_m3_pc_hybrid` не существует — индексирует документы
4. Запускает API на порту 8000

При последующих запусках индексация пропускается — коллекция уже есть в Milvus.

Фронтенд доступен на [http://localhost:3000](http://localhost:3000).

### Локальная разработка

```bash
# Backend
cd backend && uv run uvicorn russian_laws.api:app --port 8000 --reload

# Frontend
cd frontend && npm run dev
```

## API

- `POST /embed` — эмбеддинг текста
- `POST /search` — поиск релевантных статей
- `POST /answer` — контекст для LLM
- `POST /generate` — генерация ответа

## Структура проекта

```
├── docker-compose.yml
├── docker-compose.vector-stores.yml  # Запуск альтернативных хранилищ для бенчмарка
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── conf/                          # Hydra-конфиги
│   │   ├── config.yaml
│   │   ├── embedding/
│   │   │   ├── bge_m3_legal_ru.yaml   # Дообученная BGE-M3 (продакшн)
│   │   │   ├── default.yaml           # Базовая BGE-M3
│   │   │   └── jina_v3.yaml
│   │   ├── generator/default.yaml
│   │   ├── qdrant/default.yaml        # Hybrid search параметры
│   │   ├── reranker/default.yaml
│   │   ├── sparse/default.yaml        # BM25 параметры
│   │   ├── train/default.yaml
│   │   └── vector_store/
│   │       ├── milvus.yaml
│   │       ├── qdrant.yaml
│   │       └── weaviate.yaml
│   └── russian_laws/
│       ├── api.py                     # FastAPI
│       ├── embeddings.py              # Dense encoder
│       ├── sparse_encoder.py          # BM25 sparse encoder
│       ├── generator.py               # LLM генератор
│       ├── indexer.py                 # Чанкирование и индексация
│       ├── qdrant_manager.py          # Hybrid search через Qdrant
│       ├── reranker.py                # CrossEncoder reranker
│       ├── metrics.py                 # Retrieval + RAG метрики
│       ├── test.py                    # Тестирование (Lightning + MLflow)
│       └── vector_stores/             # Адаптеры хранилищ
│           ├── base.py
│           ├── factory.py
│           ├── milvus_store.py
│           ├── qdrant_store.py
│           └── weaviate_store.py
└── frontend/
    ├── Dockerfile
    └── nginx.conf
```

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./

RUN uv sync --no-install-project --no-dev

COPY russian_laws ./russian_laws
COPY conf ./conf
COPY scripts ./scripts

RUN uv sync --no-dev

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "russian_laws.api:app", "--host", "0.0.0.0", "--port", "8000"]

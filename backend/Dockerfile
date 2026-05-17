FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./

RUN uv sync --no-install-project --no-dev

COPY russian_laws ./russian_laws
COPY conf ./conf
COPY scripts ./scripts

RUN uv sync --no-dev

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

RUN chmod +x scripts/start.sh

CMD ["bash", "scripts/start.sh"]

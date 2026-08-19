# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HH_RAG_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6" \
    && python -m pip install ".[app,web]"

# The frozen retriever is fully local. These exact artifacts are intentionally baked into the
# demo image so startup never changes the selected model, index, or chunk mapping.
COPY results/final_retriever_config.json ./results/final_retriever_config.json
COPY cache/models/BAAI__bge-m3--5617a9f61b02 ./cache/models/BAAI__bge-m3--5617a9f61b02
COPY cache/indexes/final/3b19f7581e6e195f ./cache/indexes/final/3b19f7581e6e195f
COPY data/processed/23828c1c95c62c20/chunks/final-test-3b19f7581e6e195f.jsonl \
    ./data/processed/23828c1c95c62c20/chunks/final-test-3b19f7581e6e195f.jsonl

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "-m", "uvicorn", "hh_goa_rag.web:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Ledger on a free Hugging Face Space (Docker SDK, CPU basic: 2 vCPU, 16 GB RAM).
# One container: Ollama serving qwen2.5:7b-instruct + the FastAPI app + the frontend.
# CPU inference is slow (a new answer takes 1-3 minutes): answers stream as they are generated,
# the UI shows a CPU notice, and questions pre-answered by scripts/warm_cache.py (shipped in the
# seed's response_cache.sqlite3) return instantly. Rate limits are per visitor IP.
# The Space repo needs only this Dockerfile, start.sh and README.md; the code is cloned from GitHub
# and the pre-built demo corpus (graphs, indexes, router policy) is downloaded from SEED_URL.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git zstd unzip \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
ARG REPO=https://github.com/SamarthZaveri/GraphRAG.git
ARG REF=main
RUN git clone --depth 1 --branch "$REF" "$REPO" /app && chown -R user /app
WORKDIR /app
RUN pip install --no-cache-dir -r backend/requirements.txt

USER user
ENV HOME=/home/user OLLAMA_MODELS=/home/user/.ollama/models PYTHONUNBUFFERED=1
# Bake the 7B and the embedding model into the image so a cold start does not download ~5 GB.
RUN ollama serve > /tmp/ollama-build.log 2>&1 & pid=$!; \
    for i in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags > /dev/null && break; sleep 1; done; \
    ollama pull qwen2.5:7b-instruct; status=$?; kill $pid; exit $status
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"

COPY --chown=user start.sh /home/user/start.sh

# Serving needs only the 7B: it extracts, classifies and answers. The 14B wrote and judged the
# benchmark offline; its name is kept so the policy's model metadata matches.
ENV LEDGER_LLM_BACKEND=ollama \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    LEDGER_EXTRACTION_MODEL=qwen2.5:7b-instruct \
    LEDGER_QUERY_MODEL=qwen2.5:7b-instruct \
    LEDGER_ANSWER_MODEL=qwen2.5:7b-instruct \
    LEDGER_QUESTION_MODEL=qwen2.5:14b \
    LEDGER_JUDGE_MODEL=qwen2.5:14b \
    LEDGER_NUM_CTX_BY_MODEL=qwen2.5:7b-instruct=8192 \
    LEDGER_NUM_CTX=8192 \
    LEDGER_KEEP_ALIVE=24h \
    LEDGER_OLLAMA_TIMEOUT=900 \
    LEDGER_EXTRACTION_CONCURRENCY=1 \
    LEDGER_MAX_CONCURRENT_LLM_REQUESTS=1 \
    LEDGER_INFERENCE_DEVICE=cpu \
    LEDGER_RATE_QUERY=12/600 \
    LEDGER_RATE_COMPARE=4/600 \
    LEDGER_LLM_QUEUE_WAIT_SECONDS=1 \
    LEDGER_PUBLIC_MODE=true \
    LEDGER_GPU_HOURLY_USD=0 \
    LEDGER_DATA_DIR=/home/user/data \
    ANONYMIZED_TELEMETRY=False \
    OLLAMA_NUM_PARALLEL=1

EXPOSE 7860
CMD ["sh", "/home/user/start.sh"]

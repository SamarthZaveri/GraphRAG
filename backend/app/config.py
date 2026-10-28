"""Environment-based configuration for hosted APIs and optional Ollama."""
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LEDGER_DATA_DIR", str(BACKEND_ROOT / "data"))).resolve()
SAMPLE_DOCS_DIR = BACKEND_ROOT / "data" / "sample_docs"
UPLOADS_DIR = DATA_DIR / "uploads"
GRAPH_STATE_DIR = DATA_DIR / "graph_state"
CHROMA_DIR = DATA_DIR / "chroma_db"
BENCHMARK_QUESTIONS_PATH = DATA_DIR / "benchmark_questions.json"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"
for d in (UPLOADS_DIR, GRAPH_STATE_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

LLM_BACKEND = os.environ.get("LEDGER_LLM_BACKEND", "ollama")
if LLM_BACKEND not in ("api", "ollama"):
    raise ValueError("LEDGER_LLM_BACKEND must be api or ollama")
MODEL_PROFILES = {"14b": "Qwen/Qwen2.5-14B-Instruct" if LLM_BACKEND == "api" else "qwen2.5:14b"}
API_BASE_URL = os.environ.get("LEDGER_API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
API_KEY = os.environ.get("LEDGER_API_KEY") or os.environ.get("HF_TOKEN", "")
NVIDIA_THINKING = os.environ.get("LEDGER_NVIDIA_THINKING", "false").lower() == "true"
API_MIN_INTERVAL = max(0.0, float(os.environ.get("LEDGER_API_MIN_INTERVAL", "0")))
API_JSON_MODE = os.environ.get("LEDGER_API_JSON_MODE", "false").lower() == "true"
LLM_RETRIES = int(os.environ.get("LEDGER_LLM_RETRIES", "3"))
RATE_LIMIT_ATTEMPTS = max(1, int(os.environ.get("LEDGER_RATE_LIMIT_ATTEMPTS", "6")))
MODEL_PROFILE = os.environ.get("LEDGER_MODEL_PROFILE", "14b")
if MODEL_PROFILE not in MODEL_PROFILES:
    raise ValueError(f"LEDGER_MODEL_PROFILE must be one of {list(MODEL_PROFILES)}")
DEFAULT_MODEL = MODEL_PROFILES[MODEL_PROFILE]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EXTRACTION_MODEL = os.environ.get("LEDGER_EXTRACTION_MODEL", DEFAULT_MODEL)
ANSWER_MODEL = os.environ.get("LEDGER_ANSWER_MODEL", DEFAULT_MODEL)
QUESTION_MODEL = os.environ.get("LEDGER_QUESTION_MODEL", DEFAULT_MODEL)
JUDGE_MODEL = os.environ.get("LEDGER_JUDGE_MODEL", DEFAULT_MODEL)
OLLAMA_REQUEST_TIMEOUT = int(os.environ.get("LEDGER_OLLAMA_TIMEOUT", "900"))
OLLAMA_NUM_CTX = int(os.environ.get("LEDGER_NUM_CTX", "16384"))
OLLAMA_KEEP_ALIVE = os.environ.get("LEDGER_KEEP_ALIVE", "5m")
EXTRACTION_CONCURRENCY = int(os.environ.get("LEDGER_EXTRACTION_CONCURRENCY", "1"))
EXTRACTION_MAX_TOKENS = int(os.environ.get("LEDGER_EXTRACTION_MAX_TOKENS", "4096"))
EXTRACTION_TOKEN_LIMIT = int(os.environ.get("LEDGER_EXTRACTION_TOKEN_LIMIT", "16384"))
if not 0 < EXTRACTION_MAX_TOKENS <= EXTRACTION_TOKEN_LIMIT:
    raise ValueError("Require 0 < LEDGER_EXTRACTION_MAX_TOKENS <= LEDGER_EXTRACTION_TOKEN_LIMIT")
ANSWER_MAX_TOKENS = int(os.environ.get("LEDGER_ANSWER_MAX_TOKENS", "1500"))
QA_PER_CATEGORY = int(os.environ.get("LEDGER_QA_PER_CATEGORY", "3"))
QA_INPUT_CHARS = int(os.environ.get("LEDGER_QA_INPUT_CHARS", "16000"))
CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200
LEIDEN_RESOLUTION = 1.0
VECTOR_TOP_K = 6
HYBRID_CANDIDATES = 20
HYBRID_TOP_K = 6
RRF_K = 60
RETRIEVAL_VERSION = "pure_graph_vector_bm25_rrf_v1"

# Cost estimation for live metrics. Hosted APIs bill per token; self-hosted Ollama bills GPU time
# (default: AWS g5.xlarge on-demand, enough for qwen2.5:14b Q4).
PRICE_INPUT_PER_M = float(os.environ.get("LEDGER_PRICE_INPUT_PER_M", "0"))
PRICE_OUTPUT_PER_M = float(os.environ.get("LEDGER_PRICE_OUTPUT_PER_M", "0"))
GPU_HOURLY_USD = float(os.environ.get("LEDGER_GPU_HOURLY_USD", "1.006"))
METRICS_DB_PATH = DATA_DIR / "metrics.sqlite3"
RL_RESULTS_DIR = Path(os.environ.get("LEDGER_RL_RESULTS_DIR", str(BACKEND_ROOT.parent / "RL" / "data" / "local-14b"))).resolve()

# Upload guardrails (the frontend reads these from /api/status).
MAX_UPLOAD_FILES = int(os.environ.get("LEDGER_MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_MB = float(os.environ.get("LEDGER_MAX_UPLOAD_MB", "10"))
ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")
CORS_ORIGINS = [o.strip() for o in os.environ.get("LEDGER_CORS_ORIGINS", "*").split(",") if o.strip()]

# Public deployment guardrails. In public mode the shared corpus is read-only without the admin token;
# visitors upload into private, expiring workspaces.
PUBLIC_MODE = os.environ.get("LEDGER_PUBLIC_MODE", "false").lower() == "true"
ADMIN_TOKEN = os.environ.get("LEDGER_ADMIN_TOKEN", "")
WORKSPACES_DIR = DATA_DIR / "workspaces"
WORKSPACE_TTL_HOURS = float(os.environ.get("LEDGER_WORKSPACE_TTL_HOURS", "24"))
MAX_WORKSPACES = int(os.environ.get("LEDGER_MAX_WORKSPACES", "100"))
MAX_WORKSPACE_DOCS = int(os.environ.get("LEDGER_MAX_WORKSPACE_DOCS", "10"))
MAX_WORKSPACE_CHARS = int(os.environ.get("LEDGER_MAX_WORKSPACE_CHARS", "400000"))
# Sliding-window rate limits per client IP, as "requests/seconds".
RATE_LIMITS = {
    "query": os.environ.get("LEDGER_RATE_QUERY", "20/60"),
    "compare": os.environ.get("LEDGER_RATE_COMPARE", "6/60"),
    "ingest": os.environ.get("LEDGER_RATE_INGEST", "6/3600"),
    "default": os.environ.get("LEDGER_RATE_DEFAULT", "240/60"),
}
TRUST_PROXY = os.environ.get("LEDGER_TRUST_PROXY", "false").lower() == "true"
# Concurrent LLM-backed requests admitted at once; the rest get a fast 503 instead of a long hang.
MAX_CONCURRENT_LLM_REQUESTS = int(os.environ.get("LEDGER_MAX_CONCURRENT_LLM_REQUESTS", "4"))
LLM_QUEUE_WAIT_SECONDS = float(os.environ.get("LEDGER_LLM_QUEUE_WAIT_SECONDS", "3"))

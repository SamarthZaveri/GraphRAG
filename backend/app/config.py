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

LLM_BACKEND = os.environ.get("LEDGER_LLM_BACKEND", "api")
if LLM_BACKEND not in ("api", "ollama"):
    raise ValueError("LEDGER_LLM_BACKEND must be api or ollama")
MODEL_PROFILES = ({"14b": "Qwen/Qwen2.5-14B-Instruct", "32b": "Qwen/Qwen2.5-32B-Instruct", "72b": "Qwen/Qwen2.5-72B-Instruct"}
                  if LLM_BACKEND == "api" else
                  {"14b": "qwen2.5:14b", "32b": "qwen2.5:32b", "72b": "qwen2.5:72b"})
API_BASE_URL = os.environ.get("LEDGER_API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
API_KEY = os.environ.get("LEDGER_API_KEY") or os.environ.get("HF_TOKEN", "")
API_JSON_MODE = os.environ.get("LEDGER_API_JSON_MODE", "false").lower() == "true"
LLM_RETRIES = int(os.environ.get("LEDGER_LLM_RETRIES", "3"))
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
OLLAMA_NUM_CTX = int(os.environ.get("LEDGER_NUM_CTX", "32768"))
OLLAMA_KEEP_ALIVE = os.environ.get("LEDGER_KEEP_ALIVE", "5m")
EXTRACTION_CONCURRENCY = int(os.environ.get("LEDGER_EXTRACTION_CONCURRENCY", "1"))
ANSWER_MAX_TOKENS = int(os.environ.get("LEDGER_ANSWER_MAX_TOKENS", "1500"))
QA_PER_CATEGORY = int(os.environ.get("LEDGER_QA_PER_CATEGORY", "3"))
QA_INPUT_CHARS = int(os.environ.get("LEDGER_QA_INPUT_CHARS", "48000"))
CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200
LEIDEN_RESOLUTION = 1.0
VECTOR_TOP_K = 6
HYBRID_CANDIDATES = 20
HYBRID_TOP_K = 6
RRF_K = 60
RETRIEVAL_VERSION = "pure_graph_vector_bm25_rrf_v1"

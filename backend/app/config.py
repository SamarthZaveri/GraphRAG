"""
Central configuration for Ledger.

Runs entirely offline against a local Ollama server — no API key needed.
Install Ollama (https://ollama.com), then pull a model, e.g.:

    ollama pull llama3.1
    ollama serve          # usually already running as a background service

By default Ledger talks to Ollama at http://localhost:11434 and uses the
"llama3.1" model for everything. Override with env vars if you want a
different model per stage (e.g. a smaller/faster model for the high-volume
extraction step and a bigger one for final answers):

    export OLLAMA_HOST=http://localhost:11434
    export LEDGER_EXTRACTION_MODEL=llama3.1
    export LEDGER_ANSWER_MODEL=llama3.1
    export LEDGER_JUDGE_MODEL=llama3.1
"""
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOADS_DIR = DATA_DIR / "uploads"
GRAPH_STATE_DIR = DATA_DIR / "graph_state"
CHROMA_DIR = DATA_DIR / "chroma_db"
BENCHMARK_QUESTIONS_PATH = DATA_DIR / "benchmark_questions.json"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"

for d in (UPLOADS_DIR, GRAPH_STATE_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Ollama (offline) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# One model is enough, but you can point each stage at a different local
# model if you've pulled more than one (e.g. a small model for the
# high-volume extraction pass, a bigger one for final answers/judging).
EXTRACTION_MODEL = os.environ.get("LEDGER_EXTRACTION_MODEL", "qwen2.5:3b-instruct")
ANSWER_MODEL = os.environ.get("LEDGER_ANSWER_MODEL", "qwen2.5:3b-instruct")
JUDGE_MODEL = os.environ.get("LEDGER_JUDGE_MODEL", "qwen2.5:7b-instruct")

OLLAMA_REQUEST_TIMEOUT = int(os.environ.get("LEDGER_OLLAMA_TIMEOUT", "180"))

CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200

# Community detection
LEIDEN_RESOLUTION = 1.0

# Retrieval
VECTOR_TOP_K = 6
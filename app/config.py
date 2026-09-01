"""
Central configuration for Ledger.

The Anthropic API key is read from the ANTHROPIC_API_KEY environment
variable. Set it before starting the server, e.g.:

    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn app.main:app --reload

Nothing in this file should be committed with a real key in it.
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

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Extraction / generation model: fast + cheap is fine for entity extraction,
# a stronger model is used for query answering and judging.
EXTRACTION_MODEL = os.environ.get("LEDGER_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
ANSWER_MODEL = os.environ.get("LEDGER_ANSWER_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("LEDGER_JUDGE_MODEL", "claude-sonnet-4-6")

CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200

# Community detection
LEIDEN_RESOLUTION = 1.0

# Retrieval
VECTOR_TOP_K = 6
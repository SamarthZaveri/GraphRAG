"""Make shared modules importable when running any test file independently."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "RL"))

import pytest


@pytest.fixture(autouse=True)
def _isolated_metrics_db(tmp_path, monkeypatch):
    """Keep request metrics written by API tests out of the real data directory."""
    from app import config
    monkeypatch.setattr(config, "METRICS_DB_PATH", tmp_path / "metrics.sqlite3")


@pytest.fixture(autouse=True)
def _isolated_guardrails(tmp_path, monkeypatch):
    """Fresh rate-limit counters and a temporary workspace root for every test."""
    from app import config, guard
    guard.limiter.hits.clear()
    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "workspaces")


@pytest.fixture(autouse=True)
def _no_experiment_tracking(monkeypatch):
    """Benchmark tests must not write runs into the real MLflow store."""
    monkeypatch.setenv("LEDGER_MLFLOW", "0")

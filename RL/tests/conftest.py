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

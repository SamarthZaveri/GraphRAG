"""Load test for a running Ledger backend (roadmap step 28): real throughput and p95, not estimates.

    pip install locust==2.31.8
    locust -f scripts/loadtest/locustfile.py --host http://localhost:8000 \
           --users 4 --spawn-rate 1 --run-time 10m --headless --csv RL/data/loadtest/local

Read the result with the app's own telemetry (GET /api/metrics) alongside Locust's CSVs: Locust
measures what a client sees, /api/metrics breaks it into routing, retrieval and generation.
With a local LLM the GPU is the bottleneck, so keep --users near LEDGER_MAX_CONCURRENT_LLM_REQUESTS;
extra users only queue (and, by design, receive 429/503 responses from the rate limiter).
"""
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "What was total operating revenue in the most recent quarter, and how did it change year over year?",
    "How did diluted earnings per share compare between GAAP and adjusted results?",
    "Compare operating margin trends across the companies in the corpus.",
    "What drove the change in operating expenses as a share of revenue?",
    "Are non-GAAP adjustments described consistently across these releases?",
]


class Visitor(HttpUser):
    wait_time = between(2, 6)

    @task(8)
    def routed_query(self):
        with self.client.post("/api/query", json={"question": random.choice(QUESTIONS), "engine": "auto"},
                              name="/api/query (auto)", catch_response=True, timeout=600) as r:
            if r.status_code in (429, 503):
                r.success()  # the limiter shedding load is the intended behaviour, not an error
            elif r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(3)
    def read_only(self):
        self.client.get("/api/rl/results", name="/api/rl/results")
        self.client.get("/api/metrics?window_minutes=60", name="/api/metrics")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="/api/health")

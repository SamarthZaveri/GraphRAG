"""Per-request tracing: LLM token usage, stage latency and cost estimates.

A trace is bound to the current context with `trace()`; code underneath records
LLM calls (llm_client.chat) and timed stages (`span`). With no active trace every
hook is a no-op, so offline training/evaluation runs are unaffected."""
from __future__ import annotations
import contextvars
import re
import threading
import time
from contextlib import contextmanager

from . import config

_current = contextvars.ContextVar("ledger_trace", default=None)

# Answers that decline instead of guessing; the answer prompt asks for this wording.
ABSTAIN_PATTERN = re.compile(r"cannot be (established|determined)|insufficient|not (provided|available|mentioned) in the (evidence|context)"
                      r"|no documents have been ingested", re.I)


class Trace:
    def __init__(self):
        self.started = time.perf_counter()
        self.llm_calls = []
        self.stages = {}
        self._stack = []
        self._lock = threading.Lock()

    def add_llm_call(self, call):
        with self._lock:
            call["stage"] = self._stack[-1] if self._stack else "other"
            self.llm_calls.append(call)

    def add_stage(self, name, seconds):
        with self._lock:
            self.stages[name] = self.stages.get(name, 0.0) + seconds

    def elapsed_ms(self):
        return (time.perf_counter() - self.started) * 1000

    def summary(self):
        prompt = sum(c["prompt_tokens"] for c in self.llm_calls)
        completion = sum(c["completion_tokens"] for c in self.llm_calls)
        llm_s = sum(c["seconds"] for c in self.llm_calls)
        gen_s = sum(c.get("generation_seconds") or c["seconds"] for c in self.llm_calls)
        by_stage = {}
        for c in self.llm_calls:
            s = by_stage.setdefault(c["stage"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "ms": 0.0})
            s["calls"] += 1
            s["prompt_tokens"] += c["prompt_tokens"]
            s["completion_tokens"] += c["completion_tokens"]
            s["ms"] += c["seconds"] * 1000
        cost, basis = estimate_cost(prompt, completion, llm_s)
        return {
            "latency_ms": round(self.elapsed_ms(), 1),
            "stage_ms": {k: round(v * 1000, 1) for k, v in self.stages.items()},
            "llm_ms": round(llm_s * 1000, 1),
            "llm_calls": len(self.llm_calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "tokens_estimated": any(c["estimated"] for c in self.llm_calls),
            "output_tokens_per_s": round(completion / gen_s, 2) if gen_s > 0 else None,
            "llm_by_stage": by_stage,
            "cost_usd": round(cost, 6),
            "cost_basis": basis,
        }


def estimate_cost(prompt_tokens, completion_tokens, llm_seconds):
    """Hosted APIs bill per token; self-hosted Ollama bills GPU time."""
    if config.LLM_BACKEND == "api":
        if config.PRICE_INPUT_PER_M or config.PRICE_OUTPUT_PER_M:
            return ((prompt_tokens * config.PRICE_INPUT_PER_M + completion_tokens * config.PRICE_OUTPUT_PER_M) / 1e6,
                    f"api tokens @ ${config.PRICE_INPUT_PER_M}/M in, ${config.PRICE_OUTPUT_PER_M}/M out")
        return 0.0, "api pricing not configured (LEDGER_PRICE_INPUT_PER_M / LEDGER_PRICE_OUTPUT_PER_M)"
    return (config.GPU_HOURLY_USD * llm_seconds / 3600,
            f"self-hosted GPU time @ ${config.GPU_HOURLY_USD}/hr (LEDGER_GPU_HOURLY_USD)")


def estimate_tokens(text):
    # ~4 characters per token for English; used only when the provider omits usage.
    return max(1, len(text or "") // 4)


def current():
    return _current.get()


@contextmanager
def trace():
    t = Trace()
    token = _current.set(t)
    try:
        yield t
    finally:
        _current.reset(token)


@contextmanager
def span(name):
    t = _current.get()
    if t is None:
        yield
        return
    with t._lock:
        t._stack.append(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        t.add_stage(name, time.perf_counter() - start)
        with t._lock:
            if name in t._stack:
                t._stack.remove(name)


def record_llm_call(model, seconds, prompt_tokens=None, completion_tokens=None,
                    prompt_text="", completion_text="", generation_seconds=None):
    t = _current.get()
    if t is None:
        return
    estimated = prompt_tokens is None or completion_tokens is None
    t.add_llm_call({
        "model": model, "seconds": seconds, "generation_seconds": generation_seconds,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else estimate_tokens(prompt_text),
        "completion_tokens": completion_tokens if completion_tokens is not None else estimate_tokens(completion_text),
        "estimated": estimated,
    })


def bind(fn):
    """Run fn in a copy of the caller's context so worker threads report into the same trace."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.copy().run(fn, *args, **kwargs)


def answer_quality(answer, citations):
    """Reference-free quality signals computable on every live request."""
    retrieved = sorted({c.doc_id for c in citations if c.doc_id})
    cited = [d for d in retrieved if d in (answer or "")]
    return {
        "num_citations": len(citations),
        "retrieved_docs": len(retrieved),
        "cited_docs": len(cited),
        # Share of retrieved documents the answer actually cites inline (groundedness proxy).
        "citation_coverage": round(len(cited) / len(retrieved), 3) if retrieved else None,
        "context_chars": sum(len(c.snippet) for c in citations),
        "abstained": bool(ABSTAIN_PATTERN.search(answer or "")),
        "has_numeric_answer": "FINAL_NUMERIC_ANSWER:" in (answer or ""),
        "answer_chars": len(answer or ""),
    }

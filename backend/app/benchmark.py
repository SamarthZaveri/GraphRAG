"""
Runs the same benchmark question set against (a) GraphRAG and (b) vanilla
vector RAG, then scores each answer against a reference answer using an
LLM-judge rubric (0-5). This is the "provable advantage" comparison from
the PRD.

DIAGNOSTIC NOTE (added after the 12-corpus real run): `judge()` silently
returns {"score_a": 0, "score_b": 0, ...} whenever the judge model's
output can't be parsed as JSON -- indistinguishable from a genuine 0/5
grade in the recorded results. `five9_longitudinal` scored 0.50/0.33 avg
across both engines simultaneously in one run, which looks more like a
parsing-failure pattern than genuine answer quality. Added warning prints
below (no scoring behavior changed) so the next run makes this visible
instead of silent -- if the warnings fire repeatedly on a corpus, that
corpus's recorded reward is not trustworthy as-is and needs a real fix
(e.g. raising judge max_tokens, or a retry-on-parse-failure loop) rather
than being fed into bandit training.
"""
from __future__ import annotations
import json
from typing import List

from . import config, query_engine, vector_baseline
from .ollama_client import chat_json
from .models import BenchmarkQuestion, BenchmarkResult, BenchmarkSummary


def load_questions() -> List[BenchmarkQuestion]:
    with open(config.BENCHMARK_QUESTIONS_PATH) as f:
        data = json.load(f)
    return [BenchmarkQuestion(**d) for d in data]


JUDGE_SYSTEM_PROMPT = """You are grading answers from two different QA systems against a \
reference answer, for questions about a set of contracts/financial filings. Score each answer \
0-5 on this rubric:

5 = fully correct, matches all key facts in the reference answer, no hallucination
4 = correct on the main point, missing a minor supporting detail
3 = partially correct — gets some facts right but misses an important one, or is vague
2 = mostly wrong or only tangentially relevant
1 = wrong but at least on-topic
0 = no answer / completely wrong / fabricated facts not in the reference

Penalize hallucinated specifics (numbers, names, dates) that aren't in the reference answer, \
even if the overall gist is right. Reward correct citation of which document a fact came from \
when the reference distinguishes documents.

Return ONLY JSON, no preamble: {"score_a": float, "score_b": float, "rationale": str} where the \
rationale is 1-2 sentences comparing the two answers."""


def judge(question: str, reference: str, answer_a: str, answer_b: str) -> dict:
    prompt = (
        f"Question: {question}\n\nReference answer: {reference}\n\n"
        f"Answer A (GraphRAG):\n{answer_a}\n\nAnswer B (Vector RAG):\n{answer_b}"
    )
    data = chat_json(config.JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, prompt, max_tokens=400)
    if not data:
        print(f"  [judge WARNING] unparseable/empty judge output for question "
              f"{question[:70]!r} -- both scores defaulting to 0. This is a PARSING "
              f"FAILURE, not necessarily a genuine 0/5 grade. If this fires often on "
              f"one corpus, that corpus's recorded reward is not trustworthy as-is.")
        return {"score_a": 0, "score_b": 0, "rationale": "Judge model returned unparseable output."}
    missing = [k for k in ("score_a", "score_b") if k not in data]
    if missing:
        print(f"  [judge WARNING] judge JSON parsed but missing keys {missing} for "
              f"question {question[:70]!r} -- raw parsed data: {data!r}")
    return data


def run_benchmark() -> BenchmarkSummary:
    questions = load_questions()
    results: List[BenchmarkResult] = []
    for q in questions:
        graphrag_resp = query_engine.answer_question(q.question, mode="auto")
        vector_resp = vector_baseline.answer_question(q.question)
        verdict = judge(q.question, q.reference_answer, graphrag_resp.answer, vector_resp.answer)
        results.append(BenchmarkResult(
            question_id=q.id, question=q.question, category=q.category,
            graphrag_answer=graphrag_resp.answer, vector_rag_answer=vector_resp.answer,
            graphrag_score=float(verdict.get("score_a", 0)),
            vector_rag_score=float(verdict.get("score_b", 0)),
            judge_rationale=verdict.get("rationale", ""),
        ))

    graphrag_avg = sum(r.graphrag_score for r in results) / len(results) if results else 0
    vector_avg = sum(r.vector_rag_score for r in results) / len(results) if results else 0
    summary = BenchmarkSummary(results=results, graphrag_avg=graphrag_avg, vector_rag_avg=vector_avg)

    with open(config.BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)
    return summary


def load_last_results() -> BenchmarkSummary | None:
    if not config.BENCHMARK_RESULTS_PATH.exists():
        return None
    with open(config.BENCHMARK_RESULTS_PATH) as f:
        return BenchmarkSummary(**json.load(f))
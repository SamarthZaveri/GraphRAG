"""
Auto-generates benchmark question/reference-answer pairs for a corpus,
grounded in the actual document text -- this is what makes generating
benchmarks for 12-15 corpora tractable instead of hand-authoring each one
(which is what backend/data/benchmark_questions.json is, and doesn't scale).

Every reference answer is required (by prompt instruction) to be verifiable
from the provided text -- not invented. This is a real, standard technique
(document-grounded synthetic QA generation), with the usual honest caveat:
an LLM-generated benchmark is noisier than a hand-checked one, which is
exactly why the bandit's reward signal should be read as directional at
this scale, not gospel.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.ollama_client import chat_json  # noqa: E402
from app import config  # noqa: E402

MAX_CHARS_PER_DOC = 3500
QUESTIONS_PER_CORPUS = 6
MAX_TOTAL_INPUT_CHARS = 10000  # scale per-doc budget down as doc count grows, so total input (and the model's job) stays bounded regardless of corpus size

GENERATE_SYSTEM_PROMPT = f"""You write benchmark question/reference-answer pairs to evaluate a \
financial-document question-answering system, given the full text of several real financial \
filings. Generate exactly {QUESTIONS_PER_CORPUS} questions covering a MIX of these categories:

- "local": answerable from one specific place in one document (a single metric/fact/date).
- "global": requires synthesizing across multiple documents or companies in the set.
- "multi_hop": requires connecting two or more specific facts (may be within one document or \
  across documents) to answer, e.g. comparing two metrics or two periods.
- "conflict": asks whether something is consistent or inconsistent across the documents (a \
  definition, a period boundary, a reported figure) -- the honest answer may be "no conflict \
  found," don't force one.

Every reference_answer MUST be fully grounded in and verifiable from the provided text -- do not \
invent, guess, or use outside knowledge. If the documents don't actually support an interesting \
question in some category, skip that category rather than fabricate one.

Return ONLY JSON, no preamble: {{"questions": [{{"question": str, "category": str, \
"reference_answer": str}}, ...]}}
"""


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n[...truncated...]"


def _per_doc_budget(num_docs: int) -> int:
    """Scales the per-document character budget down as document count
    grows, so total input (and the model's job size) stays roughly bounded
    regardless of corpus size -- this is what broke on the 4-doc
    longitudinal corpora: 4 x 3500 chars of input left too little output
    budget for the model to complete valid JSON."""
    if num_docs <= 0:
        return MAX_CHARS_PER_DOC
    return max(1200, min(MAX_CHARS_PER_DOC, MAX_TOTAL_INPUT_CHARS // num_docs))


def generate_benchmark_for_corpus(doc_texts: List[str], doc_ids: List[str]) -> List[dict]:
    """doc_texts and doc_ids are parallel lists. Returns a list of question dicts
    matching the same shape as backend/data/benchmark_questions.json (minus 'id',
    which the caller should assign)."""
    per_doc_budget = _per_doc_budget(len(doc_texts))
    labeled = [f"=== DOCUMENT: {doc_id} ===\n{_truncate(text, per_doc_budget)}"
               for doc_id, text in zip(doc_ids, doc_texts)]
    combined = "\n\n".join(labeled)

    data = chat_json(config.JUDGE_MODEL, GENERATE_SYSTEM_PROMPT, combined, max_tokens=2800, temperature=0.4)
    questions = data.get("questions", []) if isinstance(data, dict) else []

    if not questions and len(doc_texts) > 2:
        # Fallback: the combined prompt was likely still too large for the
        # model to produce complete JSON. Retry once with a harder per-doc
        # cap and fewer requested questions rather than silently returning
        # nothing.
        tight_budget = max(800, MAX_TOTAL_INPUT_CHARS // (2 * len(doc_texts)))
        labeled = [f"=== DOCUMENT: {doc_id} ===\n{_truncate(text, tight_budget)}"
                   for doc_id, text in zip(doc_ids, doc_texts)]
        combined = "\n\n".join(labeled)
        retry_prompt = GENERATE_SYSTEM_PROMPT.replace(
            f"Generate exactly {QUESTIONS_PER_CORPUS} questions",
            "Generate exactly 4 questions",
        )
        data = chat_json(config.JUDGE_MODEL, retry_prompt, combined, max_tokens=2000, temperature=0.4)
        questions = data.get("questions", []) if isinstance(data, dict) else []

    cleaned = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if not q.get("question") or not q.get("reference_answer"):
            continue
        category = q.get("category", "local")
        if category not in ("local", "global", "multi_hop", "conflict"):
            category = "local"
        cleaned.append({
            "question": q["question"], "category": category,
            "reference_answer": q["reference_answer"],
        })
    return cleaned
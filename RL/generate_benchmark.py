"""
Auto-generates benchmark question/reference-answer pairs for a corpus,
grounded in the actual document text -- this is what makes generating
benchmarks for many corpora tractable instead of hand-authoring each one
(which is what backend/data/benchmark_questions.json is, and doesn't scale).

Every reference answer is required (by prompt instruction) to be verifiable
from the provided text -- not invented. This is a real, standard technique
(document-grounded synthetic QA generation), with the usual honest caveat:
an LLM-generated benchmark is noisier than a hand-checked one, which is
exactly why the bandit's reward signal should be read as directional at
this scale, not gospel.

REVISION (QUESTIONS_PER_CORPUS 6 -> 12): this project already hit a real
bug from under-scaling max_tokens relative to question count once --
4-document longitudinal corpora silently returned zero questions because
the model ran out of output budget mid-JSON-response (see
_per_doc_budget's docstring). Doubling the question count without also
scaling max_tokens would reopen exactly that failure mode. max_tokens
below is scaled up proportionally, and the retry fallback's question count
and budget now scale with QUESTIONS_PER_CORPUS instead of being hardcoded
to a fixed "4 questions" tuned for the old target of 6.

ONE THING THIS FILE CANNOT VERIFY: whether Ollama's context window
(num_ctx) is large enough to hold the combined input + this larger output
budget. That's set in ollama_client.py's chat() call (or Ollama's model
defaults if unset there), not here -- if generation still truncates after
this change, that's the next place to check, not a sign this fix was
wrong.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.ollama_client import chat_json  # noqa: E402
from app import config  # noqa: E402

MAX_CHARS_PER_DOC = 3500
QUESTIONS_PER_CORPUS = 12
MAX_TOTAL_INPUT_CHARS = 10000  # scale per-doc budget down as doc count grows, so total input (and the model's job) stays bounded regardless of corpus size

# Scaled proportionally with QUESTIONS_PER_CORPUS. The old value (2800) was
# tuned for 6 questions; simple linear scaling would suggest ~5600, but
# per-question marginal cost is a bit lower than the first few (shared JSON
# structure overhead doesn't grow linearly), so this is set generously
# rather than exactly linearly, erring toward too much budget rather than
# risking truncation again.
PRIMARY_MAX_TOKENS = 5200

# Retry fallback: previously hardcoded to a flat "4 questions" regardless
# of the primary target, which made sense when the target was 6 (4 is a
# meaningful fallback) but would be a much bigger cut relative to 12. Scale
# it to roughly half the primary target instead, with its own scaled token
# budget.
RETRY_QUESTIONS = max(4, QUESTIONS_PER_CORPUS // 2)
RETRY_MAX_TOKENS = 3200

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

    data = chat_json(config.JUDGE_MODEL, GENERATE_SYSTEM_PROMPT, combined,
                      max_tokens=PRIMARY_MAX_TOKENS, temperature=0.4)
    questions = data.get("questions", []) if isinstance(data, dict) else []

    if not questions and len(doc_texts) > 2:
        # Fallback: the combined prompt was likely still too large for the
        # model to produce complete JSON. Retry once with a harder per-doc
        # cap and fewer requested questions (scaled relative to the primary
        # target, not a fixed number) rather than silently returning
        # nothing.
        tight_budget = max(800, MAX_TOTAL_INPUT_CHARS // (2 * len(doc_texts)))
        labeled = [f"=== DOCUMENT: {doc_id} ===\n{_truncate(text, tight_budget)}"
                   for doc_id, text in zip(doc_ids, doc_texts)]
        combined = "\n\n".join(labeled)
        retry_prompt = GENERATE_SYSTEM_PROMPT.replace(
            f"Generate exactly {QUESTIONS_PER_CORPUS} questions",
            f"Generate exactly {RETRY_QUESTIONS} questions",
        )
        data = chat_json(config.JUDGE_MODEL, retry_prompt, combined,
                          max_tokens=RETRY_MAX_TOKENS, temperature=0.4)
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

    if 0 < len(cleaned) < QUESTIONS_PER_CORPUS:
        # Previously silent: a corpus could quietly return fewer questions
        # than requested (partial JSON, some entries filtered out for
        # missing fields) with no visibility into the shortfall. Not fatal
        # -- callers already handle "fewer than expected" -- but worth
        # knowing about rather than only noticing when counting old logs.
        print(f"  [generate_benchmark] got {len(cleaned)}/{QUESTIONS_PER_CORPUS} valid "
              f"questions after cleaning -- some were dropped or the model returned fewer "
              f"than requested.")

    return cleaned
# RL — Contextual Bandit Router Training

Trains the GraphRAG-vs-vector-RAG router (currently rule-based in
`backend/app/corpus_router.py`) into a learned contextual bandit, using
real benchmark outcomes across multiple corpora as reward signal.

## The honest scope of this

- **The algorithm** (`bandit.py`, LinUCB) is verified correct against
  synthetic environments with known ground truth — see
  `tests/test_bandit_synthetic.py`. This doesn't depend on real data volume
  and is fully trustworthy.
- **Training on real data** (`run_experiments.py` → `train_bandit.py`) is
  only as good as how many corpora you actually run it on. 12 corpora
  (the default manifest in `fetch_corpora.py`) is enough to see the policy
  move in a sensible direction and to demo convincingly, but it is **not**
  enough to call this a validated production policy — `train_bandit.py`
  prints an explicit warning to this effect below 10 recorded corpora.
- **`fetch_corpora.py` was written but not tested against the live network**
  in the environment that wrote it (no path to sec.gov from there). It's
  built against SEC's documented, stable public APIs and should work, but
  expect to debug a ticker or two (a company's most recent 8-K not
  containing an earnings exhibit, an exhibit using a slightly different
  filename convention, etc.) — that's a normal part of scraping a real API,
  not a sign something is fundamentally wrong.

## Pipeline

```
fetch_corpora.py  →  data/corpora/<name>/*.txt   (12 real corpora, mix of
                                                    GraphRAG-favorable and
                                                    deliberately unfavorable)
        │
        ▼
run_experiments.py  →  for each corpus: ingest via the REAL backend
                        pipeline (extraction, graph, communities, R-GCN),
                        auto-generate a benchmark (generate_benchmark.py),
                        run both engines, judge, record
                        data/experiment_results.jsonl
        │
        ▼
train_bandit.py  →  simulates honest online bandit training (shuffled
                     arrival order, bandit-feedback only, not both arms)
                     over the recorded results, reports accuracy/regret,
                     saves data/trained_bandit.json
```

## Running it

```bash
cd RL
pip install -r requirements.txt

# 1. Edit SEC_USER_AGENT at the top of fetch_corpora.py to your real name/email
#    (SEC blocks generic User-Agent strings)
python fetch_corpora.py

# 2. Make sure Ollama is running (same requirement as the main backend)
python run_experiments.py          # slow -- real ingestion + real LLM calls x 12 corpora

# 3.
python train_bandit.py

# Run the test suite (fast, no network, no Ollama needed for the synthetic
# correctness tests; the pipeline-integration tests mock the LLM but use
# real feature-extraction and training-loop code)
pytest tests/ -v
```

## Why LinUCB and not deep RL

There's no labeled data and, more importantly, nowhere near enough corpora
to train a neural policy safely (12-20 examples, not the thousands-to-
millions a deep RL method needs). LinUCB (Li et al. 2010) is the correct
tool at this scale: it maintains an explicit per-arm uncertainty estimate,
so its exploration shrinks automatically as evidence accumulates, and it's
a real, standard, citable contextual-bandit algorithm rather than a
simplified stand-in invented for this project.

## What feeds the bandit

Context (5-dim, see `features.py`): cross-document entity overlap fraction,
community modularity, R-GCN validation AUC, normalized document count, and
a bias term — the same signals `backend/app/corpus_router.py`'s rule-based
router already uses, so the bandit is learning to weight/combine signals a
human already picked as relevant, not starting from nothing.

Reward: the benchmark judge's average score for the chosen engine on that
corpus (normalized 0-1). Since experiments run both engines on every
corpus, we know both arms' true reward for evaluation/regret purposes —
but `train_bandit.py` deliberately simulates genuine bandit feedback
(the policy only gets to see the reward for the arm it actually picked)
so the reported accuracy/regret numbers reflect real online-learning
performance, not an inflated full-information result.

## Once trained

`data/trained_bandit.json` isn't wired into the main backend's
`/api/query` endpoint yet — that's a deliberate next step, not an
oversight, since a policy trained on 12 corpora should earn its way into
being the default via more validation first. The rule-based router in
`backend/app/corpus_router.py` remains the production default until then.

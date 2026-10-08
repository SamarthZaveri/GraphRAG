# RL: Balanced Three-Engine Evaluation and Routing

This folder evaluates pure GraphRAG, pure vector RAG, and BM25 + vector hybrid retrieval through the backend. Hosted APIs are the default. See the [project README](../README.md) for credentials, installation, model roles and deployment.

## Workflow

From the project root after configuring models and `HF_TOKEN`:

```powershell
python RL/fetch_corpora.py --only semiconductors_2025
python RL/run_experiments.py --corpus semiconductors_2025 --output-dir RL/data/run-32b
python RL/train_query_bandit.py --results RL/data/run-32b/query_results.jsonl --seeds 20
```

Check `SEC_USER_AGENT` before fetching. Omit the corpus argument to fetch/evaluate all available corpora. If generation cannot support an even category split, the run fails rather than inventing evidence. Use `--allow-unbalanced` only when deliberately accepting a shortfall.

| File | Purpose |
| --- | --- |
| `generate_benchmark.py` | Generate category-balanced questions, references, reasoning and verified quotes |
| `run_experiments.py` | Reuse a saved benchmark, ingest isolated indexes, answer with all three engines, judge and save rewards |
| `train_query_bandit.py` | Train the live three-arm query router |
| `train_bandit.py` | Historical two-arm corpus-router experiment; not the live policy |
| `run_ablation.py` | Historical separate two-engine diagnostic; shares app state and is no longer needed to disable fallback |
| `features.py`, `bandit.py` | Re-export shared backend features and LinUCB |
| `tests/` | Offline synthetic and mocked-model regression tests |

## Questions and reference answers

The default is 12 questions: three each of `local`, `global`, `multi_hop`, and `conflict`. Generation proceeds per category, deduplicates questions, verifies exact evidence quotations against original filings, and audits references and category validity with the configured judge. Unsupported categories cause explicit shortfalls. Consistency questions may have a supported no-conflict answer; differences in reporting periods or metric bases are not forced into contradictions.

```powershell
python RL/run_experiments.py --corpus semiconductors_2025 --questions-only --per-category 3
```

Questions are cached in `RL/data/balanced_benchmarks/<corpus>.json`, with document fingerprints and generator metadata. Subsequent runs reuse them even when the answer model changes. `--regenerate-questions` explicitly replaces them. A changed source corpus, target count or generator version is rejected as stale. References come from sampled source excerpts, not necessarily every line of long filings; exact-quote checks and an LLM audit reduce errors but do not eliminate them.

The standalone generator writes a question-list file compatible with the app benchmark:

```powershell
python RL/generate_benchmark.py --corpus RL/data/corpora/semiconductors_2025 --output backend/data/benchmark_questions.json --per-category 3
```

Ingest the matching corpus before running that benchmark.

## Compare answer-model sizes fairly

Keep questions, extraction and judge fixed while changing the answer model. These commands use 72B to generate/audit references and judge answers, 14B for extraction, and compare 14B/32B/72B answers on the same saved questions and graph. Using `--state-dir` here deliberately shares experiment indexes between the three single-corpus runs; it does not touch the app's default state.

```powershell
. .\scripts\use-model.ps1 -Profile 32b -Backend api
$env:LEDGER_EXTRACTION_MODEL = "Qwen/Qwen2.5-14B-Instruct"
$env:LEDGER_QUESTION_MODEL = "Qwen/Qwen2.5-72B-Instruct"
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
$env:LEDGER_ANSWER_MODEL = "Qwen/Qwen2.5-14B-Instruct"
python RL/run_experiments.py --corpus semiconductors_2025 --output-dir RL/data/run-14b --state-dir RL/data/shared-runtime
$env:LEDGER_ANSWER_MODEL = "Qwen/Qwen2.5-32B-Instruct"
python RL/run_experiments.py --corpus semiconductors_2025 --output-dir RL/data/run-32b --state-dir RL/data/shared-runtime --reuse-index
$env:LEDGER_ANSWER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
python RL/run_experiments.py --corpus semiconductors_2025 --output-dir RL/data/run-72b --state-dir RL/data/shared-runtime --reuse-index
```

To compare judge sizes, keep answers fixed and rescore saved answers using `rescore_results.py`; do not regenerate questions or answers:

```powershell
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
python RL/rescore_results.py --input RL/data/run-32b/corpora/semiconductors_2025.json --output RL/data/judge-14b.json
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
python RL/rescore_results.py --input RL/data/run-32b/corpora/semiconductors_2025.json --output RL/data/judge-32b.json
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
python RL/rescore_results.py --input RL/data/run-32b/corpora/semiconductors_2025.json --output RL/data/judge-72b.json
```

Question-model sizes can be compared using separate `--benchmark-dir` directories and `--questions-only`; keep that experiment separate from retrieval comparisons, which need one frozen question set.

## Results and isolation

`--output-dir` defaults to `RL/data/three_engine`. It contains:

- `corpora/<name>.json`: questions, references, quotes, reasoning, engine answers/citations, judge scores/rationales and failures.
- `query_results.jsonl`: one reward row per successfully evaluated question, with all three rewards, eight-dimensional context and model/retrieval metadata.
- `experiment_results.jsonl`: corpus-level aggregates.
- `runtime/`: isolated graph, vector and community state unless `--state-dir` overrides it.

Each corpus output replaces its earlier result within that directory. JSONL files are rebuilt from the saved corpus results, avoiding duplicate append-on-rerun rows. Use separate output directories for model configurations. `--reuse-index` requires one corpus and matching corpus/extraction/retrieval metadata. Experiments import the real backend pipeline; R-GCN is optional and graph-derived retrieval remains the same as serving.

An invalid or failed judge call never becomes a zero reward. It is retried, then logged as an excluded failure. The command exits with failure if any corpus or question was incomplete; valid results already collected remain available. Report category coverage and failures alongside averages, since exclusions can bias the evaluated subset.

## Train and serve

```powershell
$env:LEDGER_ANSWER_MODEL = "Qwen/Qwen2.5-32B-Instruct"
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
python RL/train_query_bandit.py --results RL/data/run-32b/query_results.jsonl --seeds 20
```

The default output is `backend/data/graph_state/query_bandit.json` (or the equivalent under `LEDGER_DATA_DIR`). Live serving validates arm names and model/retrieval metadata before using it. Historical two-engine rewards, missing hybrid rewards, invalid rewards and mixed model configurations are rejected.

LinUCB training explores using uncertainty bonuses and updates only the chosen engine's reward. Regret uses the known rewards of all three engines. Serving exploits the highest predicted reward without live updates. The saved policy comes from the final random shuffle, not an average of models. Selection accuracy treats tied best rewards as correct.

The learned context is three corpus features (shared-entity fraction, recurrence/10, document count/10), four category indicators and a bias. Modularity and R-GCN AUC are excluded from that context. Live category classification comes from the extraction LLM, whereas training categories come from the generator, so classification mismatch remains a possible error source.

## Validation

```powershell
python -m pytest RL/tests -q
```

Tests are offline and use mocked model responses where relevant. Recorded training metrics simulate online learning; they are not a held-out generalization study. Manually review benchmark references, and evaluate the trained router on new corpora before treating it as a production policy.

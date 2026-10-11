# Shared-source numerical and textual RAG experiments

The active dataset is `shared_finqa`: FinQA source text/tables and original numeric QA, plus 3 to 5 generated questions per local/global/multi-hop/conflict category for each same-source bundle. Source reports retain the prepared FinQA train/dev/test assignment; generated questions never migrate across splits. Related company/year reports are excluded across selected splits by FinQA preparation. These are document excerpts, not downloaded complete filings.

## Model roles

Use `scripts/use-ollama.ps1`. All LLM roles use `qwen2.5:14b` in separate calls: extraction/classification, question generation, reference auditing, answering and text judging. Calls are sequential with a 16384-token context and zero keep-alive. The 16000-character generation source budget leaves space for instructions and output, though unusual tokenization can still exhaust context; inspect source coverage and model responses.

## Stages

```powershell
. .\scripts\use-ollama.ps1
python RL/prepare_finqa.py --train-limit 40 --dev-limit 20 --test-limit 20
python RL/prepare_shared_benchmark.py --per-category 3
python RL/run_public_benchmark.py --dataset-dir RL/data/shared_finqa --split train --output-dir RL/data/local-14b
python RL/train_shared_router.py --results RL/data/local-14b/train/query_results.jsonl --output RL/data/local-14b/policy.json --numeric-weight 0.5
python RL/run_public_benchmark.py --dataset-dir RL/data/shared_finqa --split dev --output-dir RL/data/local-14b --policy RL/data/local-14b/policy.json
python RL/run_public_benchmark.py --dataset-dir RL/data/shared_finqa --split test --output-dir RL/data/local-14b --policy RL/data/local-14b/policy.json
```

The stop-on-error wrapper is `scripts/run-local-flow.ps1`. It stops after dev by default; after selecting/fixing settings, run `scripts/run-local-flow.ps1 -StartAt test -Through test` for the final frozen test. `-PerCategory 4` or `5` uses distinct dataset/result directories. Choose settings using validation before final testing. Generation is done once, using a fixed 14B model and source plan. For another source subset/generation configuration, choose another `--output-dir`; do not mutate a held-out experiment's manifest. Targeting 5 per category produces 20 text questions per bundle only if supported by source evidence. Partial audited QA is saved across failures. Before updating nonempty progress, its previous state is copied to a `.previous.json` backup. Incompatible checkpoints are rejected before generation rather than overwritten. Local checkpoint matching uses the Ollama endpoint and migrates older fingerprints that included an unused hosted endpoint. Generation cites numbered source passages; the validator attaches exact original quotations before auditing. Rejected tasks are avoided; retries use smaller rotating source sets, prioritizing documents not used by approved questions. Passage IDs are validated against the sources shown in that attempt. Unfinished v4 generation checkpoints can resume under v5; completed datasets require a new output directory for a changed generation plan.

## Scores and learning

Numeric answers end in `FINAL_NUMERIC_ANSWER: <number> <unit>` or an explicit yes/no. The scorer rejects arbitrary numbers extracted from explanatory prose and compares directly with FinQA `exe_ans`, never asking a judge to assign numeric rewards. It handles percent versus ratio, negative values, explicit thousand/million/billion scale conversions and published answer rounding. Tolerance is the maximum of 1e-5, 0.1% of the gold magnitude, and applicable human-answer rounding precision. Ambiguous/unparseable model output scores zero; operational request failures are recorded separately. This is a documented custom scalar metric; it does not execute predicted programs or reproduce official FinQA leaderboard accuracy. Dataset annotation inconsistencies can still affect scores.

Text responses are graded blind against generated references and source evidence by local Qwen2.5 14B. Question/reference audits reject unsupported periods, unlabeled table columns and arithmetic mistakes. Local quotes may include surrounding original source context. A supported consistency finding is allowed for conflict questions; different quarterly values alone are not contradictions.

All three engines receive only the question and their own retrieved context. Source-document gold hints and reference programs are never ingested. Router context is the existing eight-feature vector: three corpus structure values, four predicted category indicators and bias. Task labels and authored text categories are used to balance/report rewards, not as oracle serving features.

Weighted LinUCB observes only the selected engine's reward for each shuffled training example. Numeric rewards receive 50% total weight; each authored text category receives 12.5%. Dev/test choose the maximum predicted reward with no UCB exploration or updates. Training produces a deployable 14B-compatible router policy.

## Artifacts

- `RL/data/public_finqa`: pinned original data and selected report-disjoint source bundles.
- `RL/data/shared_finqa`: shared source copies, source plan, custom QA checkpoints and final numeric/text manifest.
- `RL/data/local-14b`: runtime indexes, extraction checkpoints and separate train/dev/test reports.
- `<run>/<split>/summary.json`: numeric/text results, text-category results, engine/router averages and failure coverage.
- `<run>/train/query_results.jsonl`: train-only rewards; incomplete runs cannot train a policy.
- `<run>/policy.json`: model-compatible router for the app.

Frozen evaluation compares all engines plus the router and a hindsight oracle. Results from older hosted/LLM-judged numeric experiments are incompatible with this workflow. Do not combine them with new numeric rewards. Offline tests verify plumbing; run and review real local-model output before drawing conclusions.

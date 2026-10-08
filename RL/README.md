# RL: Offline Experiments and Contextual Bandit Training

This folder evaluates the backend's GraphRAG and vector-RAG engines and trains a per-question engine selector. It reuses the actual backend ingestion, retrieval, and judge functions.

The query-level policy is connected to the app: training saves `backend/data/graph_state/query_bandit.json`, and `/api/query` loads it when `engine="auto"`. The older corpus-level policy from `train_bandit.py` is a separate experiment and is not loaded by that endpoint.

## Files

| File | Purpose |
| --- | --- |
| `fetch_corpora.py` | Fetch financial-report corpora from SEC EDGAR |
| `generate_benchmark.py` | Generate document-grounded questions, categories, and reference answers |
| `run_experiments.py` | Ingest each corpus, run both engines, judge answers, and record rewards |
| `train_query_bandit.py` | Train the eight-feature per-question policy used by the app |
| `train_bandit.py` | Train the older four-feature corpus-level policy |
| `run_ablation.py` | Compare GraphRAG without its local vector fallback against vector RAG |
| `features.py` | Re-export feature functions from `backend/app/router_features.py` |
| `bandit.py` | Re-export LinUCB from `backend/app/query_bandit.py` |
| `tests/` | Synthetic bandit and mocked-LLM pipeline tests |

## Workflow

From the project root:

```powershell
python -m pip install -r backend/requirements.txt
python -m pip install -r RL/requirements.txt
ollama pull qwen2.5:7b-instruct
```

Start Ollama if it is not running. The extraction, answer, and judge models can be overridden through the backend configuration environment variables described in the [project README](../README.md).

Check `SEC_USER_AGENT` in `fetch_corpora.py` and use an appropriate identifying name and contact email before fetching.

```powershell
cd RL
python fetch_corpora.py
python run_experiments.py
python train_query_bandit.py
```

To process one corpus:

```powershell
python fetch_corpora.py --only semiconductors_2025
python run_experiments.py --corpus semiconductors_2025
```

Fetching requires network access. Experiments require the backend dependencies, local model assets, and a running Ollama server.

## Data flow

```text
fetch_corpora.py
  -> data/corpora/<name>/*.txt

run_experiments.py
  -> backend ingestion + benchmark generation + both engines + LLM judge
  -> data/experiment_results.jsonl  (one aggregate row per corpus)
  -> data/query_results.jsonl       (one row per corpus/question pair)

train_query_bandit.py
  -> backend/data/graph_state/query_bandit.json
  -> backend's automatic engine selection
```

**Experiments share the backend's active storage.** Ingestion resets its persistent vector collection and writes graph state, community summaries, and R-GCN artifacts into the backend's configured data directory. Ablations do the same. Re-ingest the desired app documents after running experiments; corpus folders and JSONL results are separate, but retrieval state is not isolated.

Results append to both experiment JSONL files. Re-running a corpus adds more rows, which can give that corpus extra weight during subsequent training.

## What the policy learns

The two actions are `graphrag` and `vector_rag`. Each action's reward is its LLM-judge answer score divided by five.

The query-level context has eight features:

- Cross-document entity fraction.
- Maximum entity document recurrence divided by ten, capped at one.
- Document count divided by ten, capped at one.
- Four one-hot category indicators: `local`, `global`, `multi_hop`, and `conflict`.
- A bias term.

Training categories come from the benchmark generator's labels. Live categories come from `query_engine.classify_query()`, which prompts the extraction LLM. A shared taxonomy helps consistency but does not guarantee matching classifications.

Community modularity and R-GCN validation AUC are computed during ingestion but excluded from the learned feature vector. R-GCN still supports graph retrieval, and those metrics remain part of the backend's rule-based routing fallback.

## Training and serving

LinUCB estimates a linear reward model for each engine and tracks uncertainty. During training, rows arrive in shuffled order. The policy selects an engine using predicted reward plus an exploration bonus, observes only the selected engine's recorded reward, and updates that engine's model. Both known rewards are used to report regret.

```powershell
python train_query_bandit.py --seeds 20 --alpha 1.0
```

Metrics are averaged across the requested shuffles. The saved policy comes from the final shuffle; the script does not average the models. The backend loads the file on requests, chooses the highest predicted reward without exploration, and does not update the policy from live queries. If routing fails or the policy is missing, it uses corpus-level rules.

The older `train_bandit.py` uses corpus-average rewards and a four-feature corpus context, then saves `data/trained_bandit.json`. That file is not the live query router's policy.

## Ablation

```powershell
python run_ablation.py --corpus semiconductors_2025
```

The ablation passes `use_hybrid=False` to GraphRAG. This disables vector-chunk fallback in local retrieval; global retrieval continues to use community summaries. Both engines answer the same generated question set within the ablation run. The questions need not match a previous experiment run.

Results are stored in `data/ablation_results.jsonl` and do not feed the router-training files.

## Validation and limits

```powershell
python -m pytest tests/ -v
```

Synthetic tests check bandit behavior against known environments. Pipeline tests mock LLM calls; they do not establish answer quality on real filings.

The training metrics describe simulated online learning over recorded data, without a separate held-out corpus or question evaluation. Sparse categories, repeated experiment rows, classifier differences, generated reference errors, and judge errors can affect the learned policy. Unparseable judge output currently defaults to zero scores rather than being excluded or retried.

Treat the recorded results as experimental evidence within the tested data, rather than production accuracy guarantees.

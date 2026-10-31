# Build the pre-ingested demo corpus for public deployments (Hugging Face Space, EC2) as dist/demo_seed.zip:
# graph, vector and BM25 indexes and the trained router policy, so visitors never wait on extraction.
# It reuses the benchmark run's extraction cache, so corpora the benchmark already ingested take minutes.
# Do not run it while a benchmark stage is using Ollama.
#
#   .\scripts\build_demo_corpus.ps1                                   # EDGAR airlines + payments (train sectors)
#   .\scripts\build_demo_corpus.ps1 -Corpora train_airlines,train_retail
#   .\scripts\build_demo_corpus.ps1 -WarmPerCorpus 0                    # skip pre-answering questions
#   gh release upload v1-router dist/demo_seed.zip                    # then set the Space's SEED_URL to the asset URL
param(
    [string[]]$Corpora = @("train_airlines", "train_payments"),
    [string]$Run = "RL/data/runs/mixed",
    [string]$Out = "dist",
    # Questions per corpus pre-answered into the seed's response cache (Auto + Compare), so the
    # CPU deployment answers them instantly. Takes ~1 min per question on the GPU.
    [int]$WarmPerCorpus = 6
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot/use-ollama.ps1" -Profile mixed
$root = Resolve-Path "$PSScriptRoot/.."
$seed = Join-Path $root "$Out/demo_seed"
$docs = Join-Path $root "$Out/demo_docs"
if (Test-Path $seed) { Remove-Item -Recurse -Force $seed }
if (Test-Path $docs) { Remove-Item -Recurse -Force $docs }
New-Item -ItemType Directory -Force $seed, $docs | Out-Null

foreach ($corpus in $Corpora) {
    $source = Join-Path $root "RL/data/edgar_xdoc/corpora/$corpus"
    if (-not (Test-Path $source)) { throw "Unknown corpus $corpus (expected $source)" }
    Copy-Item "$source/*.txt" $docs
}
$cache = Join-Path $root "$Run/runtime/extraction_cache"
if (Test-Path $cache) { Copy-Item -Recurse $cache (Join-Path $seed "extraction_cache") }

$env:LEDGER_DATA_DIR = $seed
python (Join-Path $root "backend/ingest.py") --documents $docs --mode replace
if ($LASTEXITCODE -ne 0) { throw "Ingestion failed" }

$policy = Join-Path $root "$Run/policy.json"
if (Test-Path $policy) { Copy-Item $policy (Join-Path $seed "graph_state/query_bandit.json") }
else { Write-Warning "No trained policy at $policy; Auto mode will use the fallback rule." }

if ($WarmPerCorpus -gt 0) {
    python (Join-Path $root "scripts/warm_cache.py") --corpora $Corpora --per-corpus $WarmPerCorpus --compare
    if ($LASTEXITCODE -ne 0) { throw "Cache warm-up failed" }
}

# The extraction cache and any uploads are build-time only (the response cache ships).
foreach ($transient in @("extraction_cache", "uploads", "workspaces", "metrics.sqlite3")) {
    $path = Join-Path $seed $transient
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}
$zip = Join-Path $root "$Out/demo_seed.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$seed/*" -DestinationPath $zip
Remove-Item -Recurse -Force $docs
Write-Host ("Wrote {0} ({1:N1} MB) from: {2}" -f $zip, ((Get-Item $zip).Length / 1MB), ($Corpora -join ", "))

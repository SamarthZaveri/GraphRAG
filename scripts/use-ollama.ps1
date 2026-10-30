param(
    # mixed: 7B (fits an 8 GB GPU) extracts, classifies and answers; 14B writes questions and judges.
    #        A stronger, different judge also avoids a model grading its own answers.
    # 7b:    every role on the 7B, for fast iteration.
    # 14b:   every role on the 14B (about half on the CPU with 8 GB of VRAM, ~7 tokens/s).
    [ValidateSet("mixed", "7b", "14b")][string]$Profile = "mixed",
    [string]$HostUrl = "http://localhost:11434"
)
$fast = "qwen2.5:7b-instruct"
$strong = "qwen2.5:14b"
switch ($Profile) {
    "mixed" { $worker = $fast; $grader = $strong }
    "7b" { $worker = $fast; $grader = $fast }
    "14b" { $worker = $strong; $grader = $strong }
}
$env:LEDGER_LLM_BACKEND = "ollama"
$env:OLLAMA_HOST = $HostUrl
$env:LEDGER_EXTRACTION_MODEL = $worker
$env:LEDGER_QUERY_MODEL = $worker
$env:LEDGER_ANSWER_MODEL = $worker
$env:LEDGER_QUESTION_MODEL = $grader
$env:LEDGER_JUDGE_MODEL = $grader
# One context size per model: Ollama reloads a model whenever num_ctx changes.
$env:LEDGER_NUM_CTX_BY_MODEL = "$fast=8192,$strong=12288"
$env:LEDGER_NUM_CTX = "8192"
$env:LEDGER_EXTRACTION_TOKEN_LIMIT = "6144"
# Keep models loaded between calls; reloading the 14B costs 10-18 s per call.
$env:LEDGER_KEEP_ALIVE = "30m"
$env:LEDGER_QA_INPUT_CHARS = "16000"
$env:LEDGER_OLLAMA_TIMEOUT = "900"
$env:ANONYMIZED_TELEMETRY = "False"  # chromadb 0.5's telemetry client only logs errors
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
# The 7B runs fully on the GPU, where 4 parallel requests give ~2.5x the throughput of one.
$env:LEDGER_EXTRACTION_CONCURRENCY = $(if ($worker -eq $fast) { "4" } else { "1" })
Write-Host "Ollama profile '$Profile': extraction/query/answer=$worker; questions/judge=$grader"
try {
    $installed = (Invoke-RestMethod -Uri "$HostUrl/api/tags" -TimeoutSec 5).models.name
    foreach ($model in @($worker, $grader) | Select-Object -Unique) {
        if ($installed -notcontains $model) { Write-Warning "Model $model is not installed: run 'ollama pull $model'" }
    }
} catch {
    Write-Warning "Ollama is not reachable at $HostUrl; start it before running a stage."
}
if (-not $env:OLLAMA_NUM_PARALLEL) {
    Write-Host "Tip: parallel requests need OLLAMA_NUM_PARALLEL=4 (and OLLAMA_FLASH_ATTENTION=1) set for the Ollama server; restart Ollama after setting them."
}

# End-to-end local experiment: datasets -> train rewards -> router -> frozen dev/test evaluation.
# Every stage checkpoints its work; rerun with -StartAt <stage> to resume after an interruption.
#
#   finqa   FinQA filings + generated text questions (RL/data/shared_finqa; reused when present)
#   edgar   real cross-document earnings releases from SEC EDGAR + generated questions
#           (needs LEDGER_SEC_USER_AGENT = "Name contact@email")
#   train   answer and score every train question with all three engines, per dataset
#   router  train the LinUCB router on all datasets' train rewards, then analyze train
#   dev     frozen-policy evaluation on dev, then analysis
#   test    frozen-policy evaluation on test, then analysis (run once, on the final system)
#
# Results: RL/data/runs/<profile>/  (per dataset: <dataset>/<split>/; combined: <split>/)
# Experiment tracking: mlflow ui --backend-store-uri sqlite:///RL/mlflow.db
param(
    [ValidateSet("mixed", "7b", "14b")][string]$Profile = "mixed",
    [ValidateSet("finqa", "edgar", "train", "router", "dev", "test")][string]$StartAt = "finqa",
    [ValidateSet("finqa", "edgar", "train", "router", "dev", "test")][string]$Through = "dev",
    [string[]]$Datasets = @("shared_finqa", "edgar_xdoc"),
    [ValidateRange(3, 5)][int]$PerCategory = 3,
    [double]$LatencyWeight = 0
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot/use-ollama.ps1" -Profile $Profile
$run = "RL/data/runs/$Profile"
$policy = "$run/policy.json"
$order = @("finqa", "edgar", "train", "router", "dev", "test")

function Invoke-Stage([string]$Name, [string[]]$Arguments) {
    Write-Host "`n=== $Name :: python $($Arguments -join ' ')" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage $Name failed. Fix the cause, then resume with -Profile $Profile -StartAt $($Name.Split(' ')[0])." }
}

function Invoke-Split([string]$Split) {
    foreach ($dataset in $Datasets) {
        $arguments = @("RL/run_public_benchmark.py", "--dataset-dir", "RL/data/$dataset", "--output-dir", $run, "--split", $Split)
        if ($Split -ne "train") { $arguments += @("--policy", $policy) }
        Invoke-Stage "$Split ($dataset)" $arguments
    }
}

$started = $false
foreach ($stage in $order) {
    if ($stage -eq $StartAt) { $started = $true }
    if (-not $started) { continue }
    switch ($stage) {
        "finqa" {
            if ($Datasets -contains "shared_finqa") {
                if (Test-Path "RL/data/shared_finqa/manifest.json") { Write-Host "shared_finqa questions already prepared." }
                else { Invoke-Stage "finqa" @("RL/prepare_shared_benchmark.py", "--per-category", "$PerCategory", "--output-dir", "RL/data/shared_finqa") }
            }
        }
        "edgar" {
            if ($Datasets -contains "edgar_xdoc") {
                if (-not $env:LEDGER_SEC_USER_AGENT) { throw "Set `$env:LEDGER_SEC_USER_AGENT = 'Name contact@email' (SEC fair-access policy)." }
                Invoke-Stage "edgar" @("RL/prepare_edgar.py", "--output-dir", "RL/data/edgar_xdoc", "--per-category", "4")
            }
        }
        "train" { Invoke-Split "train" }
        "router" {
            $results = $Datasets | ForEach-Object { "$run/$_/train/query_results.jsonl" }
            Invoke-Stage "router" (@("RL/train_shared_router.py", "--results") + $results + @("--output", $policy, "--latency-weight", "$LatencyWeight"))
            Invoke-Stage "analyze train" @("RL/analyze_results.py", "--run-dir", $run, "--split", "train")
        }
        "dev" { Invoke-Split "dev"; Invoke-Stage "analyze dev" @("RL/analyze_results.py", "--run-dir", $run, "--split", "dev") }
        "test" { Invoke-Split "test"; Invoke-Stage "analyze test" @("RL/analyze_results.py", "--run-dir", $run, "--split", "test") }
    }
    if ($stage -eq $Through) { break }
}
Write-Host "`nDone through '$Through'. Deploy the router with:"
Write-Host "  Copy-Item $policy backend/data/graph_state/query_bandit.json"

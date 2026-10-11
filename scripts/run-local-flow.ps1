param(
    [ValidateSet("14b")][string]$Profile = "14b",
    [ValidateSet("prepare", "train", "router", "dev", "test")][string]$StartAt = "prepare",
    [ValidateRange(3, 5)][int]$PerCategory = 3,
    [ValidateSet("router", "dev", "test")][string]$Through = "dev"
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot/use-ollama.ps1" -Profile $Profile
$ledgerRun = "RL/data/local-$Profile"
$ledgerDataset = "RL/data/shared_finqa"
if ($PerCategory -ne 3) {
    $ledgerDataset = "RL/data/shared_finqa-c$PerCategory"
    $ledgerRun = "RL/data/local-$Profile-c$PerCategory"
}
$ledgerStages = @(
    @{ Name = "prepare"; Arguments = @("RL/prepare_shared_benchmark.py", "--per-category", "$PerCategory", "--output-dir", $ledgerDataset) },
    @{ Name = "train"; Arguments = @("RL/run_public_benchmark.py", "--dataset-dir", $ledgerDataset, "--split", "train", "--output-dir", $ledgerRun) },
    @{ Name = "router"; Arguments = @("RL/train_shared_router.py", "--results", "$ledgerRun/train/query_results.jsonl", "--output", "$ledgerRun/policy.json") },
    @{ Name = "dev"; Arguments = @("RL/run_public_benchmark.py", "--dataset-dir", $ledgerDataset, "--split", "dev", "--output-dir", $ledgerRun, "--policy", "$ledgerRun/policy.json") },
    @{ Name = "test"; Arguments = @("RL/run_public_benchmark.py", "--dataset-dir", $ledgerDataset, "--split", "test", "--output-dir", $ledgerRun, "--policy", "$ledgerRun/policy.json") }
)
$ledgerStart = $false
foreach ($ledgerStage in $ledgerStages) {
    if ($ledgerStage.Name -eq $StartAt) { $ledgerStart = $true }
    if (-not $ledgerStart) { continue }
    Write-Host "Stage: $($ledgerStage.Name)"
    & python @($ledgerStage.Arguments)
    if ($LASTEXITCODE -ne 0) { throw "Stage $($ledgerStage.Name) failed. Resume with -Profile $Profile -PerCategory $PerCategory -StartAt $($ledgerStage.Name)." }
    if ($ledgerStage.Name -eq $Through) { break }
}

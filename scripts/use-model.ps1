param(
    [ValidateSet("14b", "32b", "72b")][string]$Profile = "32b",
    [ValidateSet("api", "ollama")][string]$Backend = "api"
)
$env:LEDGER_LLM_BACKEND = $Backend
$env:LEDGER_MODEL_PROFILE = $Profile
$env:LEDGER_OLLAMA_TIMEOUT = "900"
$env:LEDGER_NUM_CTX = "32768"
$env:LEDGER_EXTRACTION_CONCURRENCY = "1"
# Explicit role overrides (LEDGER_ANSWER_MODEL etc.) remain in effect.
if ($Backend -eq "api" -and -not $env:LEDGER_API_BASE_URL) {
    $env:LEDGER_API_BASE_URL = "https://router.huggingface.co/v1"
}
Write-Host "Selected $Backend profile $Profile. Role overrides remain in effect."

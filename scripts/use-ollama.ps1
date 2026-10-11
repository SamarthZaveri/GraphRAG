param(
    [ValidateSet("14b")][string]$Profile = "14b",
    [string]$HostUrl = "http://localhost:11434"
)
$env:LEDGER_LLM_BACKEND = "ollama"
$env:LEDGER_MODEL_PROFILE = $Profile
$env:OLLAMA_HOST = $HostUrl
$ledgerLocalModel = "qwen2.5:14b"
$env:LEDGER_EXTRACTION_MODEL = $ledgerLocalModel
$env:LEDGER_ANSWER_MODEL = $ledgerLocalModel
$env:LEDGER_QUESTION_MODEL = "qwen2.5:14b"
$env:LEDGER_JUDGE_MODEL = "qwen2.5:14b"
$env:LEDGER_EXTRACTION_CONCURRENCY = "1"
$env:LEDGER_NUM_CTX = "16384"
$env:LEDGER_QA_INPUT_CHARS = "16000"
$env:LEDGER_KEEP_ALIVE = "0"
$env:LEDGER_OLLAMA_TIMEOUT = "900"
Write-Host "Local Ollama: all LLM roles=qwen2.5:14b; context=16384."

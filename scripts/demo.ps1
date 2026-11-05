<#
Start the API and UI on this laptop for a demo. No Docker and no Ollama.

  .\scripts\demo.ps1             normal demo: follow-up rewriting off
  .\scripts\demo.ps1 -FollowUps  also understand follow-ups (one extra Gemini call for each one)
  .\scripts\demo.ps1 -Check      show what would run, then stop. Loads nothing, makes no calls
  .\scripts\demo.ps1 -Reindex    rebuild the search index first

Stop the demo by closing the two windows it opens.
#>
param([switch]$FollowUps, [switch]$Check, [switch]$Reindex)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$bin = Join-Path $root ".venv\Scripts"
$answerModel = "gemini-3.5-flash-lite"

function Stop-Demo($message) {
    Write-Host "`nNot started: $message" -ForegroundColor Red
    exit 1
}

foreach ($tool in "shipment-rag.exe", "uvicorn.exe", "streamlit.exe") {
    if (-not (Test-Path (Join-Path $bin $tool))) {
        Stop-Demo "$tool is missing. Install first: see 'Run locally' in README.md."
    }
}
if (-not (Test-Path ".env")) { Stop-Demo "there is no .env. Copy .env.example to .env and set GEMINI_API_KEY." }

# Load .env into this window, because the UI reads only real environment variables. Values are
# never printed. The settings below come after this, so they win over anything in .env.
foreach ($line in Get-Content ".env") {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"', "'"), "Process")
    }
}
if (-not $env:GEMINI_API_KEY) { Stop-Demo "GEMINI_API_KEY is empty in .env." }

$env:GEMINI_MODEL = $answerModel
if ($FollowUps) {
    $env:REWRITE_PROVIDER = "gemini"
    $env:REWRITE_MODEL = $answerModel
} else {
    $env:REWRITE_PROVIDER = "off"
}
$env:API_URL = "http://localhost:8080"
# With API_KEYS set the API rejects a UI that has no key, so give the UI one of them.
if ($env:API_KEYS -and -not $env:API_KEY) { $env:API_KEY = ($env:API_KEYS -split ",")[0].Trim() }

$os = Get-CimInstance Win32_OperatingSystem
$freeGb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$busy = 8080, 8501 | Where-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue }
$hasIndex = Test-Path "data\chroma_db\chroma.sqlite3"

Write-Host "Answer model: $answerModel"
Write-Host "Follow-up rewriting: $env:REWRITE_PROVIDER"
Write-Host "Free RAM: $freeGb GB"
Write-Host "Search index: $(if ($hasIndex) { 'found' } else { 'missing, will be built' })"
if ($freeGb -lt 3) {
    Write-Host "Low memory: the API loads two small local models. Close browser tabs and other apps first." -ForegroundColor Yellow
}
if (Get-Process -Name "ollama" -ErrorAction SilentlyContinue) {
    Write-Host "Ollama is running and the demo doesn't need it. You can quit it from the system tray." -ForegroundColor Yellow
}
if ($busy) { Stop-Demo "port $($busy -join ' and ') is in use. Close the earlier demo windows first." }
if ($Check) {
    Write-Host "`nCheck passed. Nothing was started." -ForegroundColor Green
    exit 0
}

if ($Reindex -or -not $hasIndex) {
    Write-Host "`nBuilding the search index..."
    & (Join-Path $bin "shipment-rag.exe") ingest --rebuild
    if ($LASTEXITCODE -ne 0) { Stop-Demo "indexing failed." }
}

Write-Host "`nStarting the API (loading its models can take a minute)..."
$api = Start-Process -FilePath (Join-Path $bin "uvicorn.exe") -PassThru -WorkingDirectory $root `
    -ArgumentList "shipment_rag.api.main:create_app", "--factory", "--port", "8080"
$ready = $false
for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
    if ($api.HasExited) { Stop-Demo "the API window closed. Run it by hand to see why (README.md)." }
    Start-Sleep -Seconds 2
    try {
        Invoke-WebRequest "http://localhost:8080/ready" -UseBasicParsing -TimeoutSec 3 | Out-Null
        $ready = $true
    } catch { }
}
if (-not $ready) { Stop-Demo "the API isn't ready after 4 minutes. Check its window, or try -Reindex." }

Write-Host "API ready. Starting the UI..."
Start-Process -FilePath (Join-Path $bin "streamlit.exe") -WorkingDirectory $root -ArgumentList `
    "run", "ui/streamlit_app.py", "--server.port", "8501", "--server.headless", "true", `
    "--browser.gatherUsageStats", "false" | Out-Null
Start-Sleep -Seconds 5
Start-Process "http://localhost:8501"

Write-Host "`nOpen http://localhost:8501 and log in with UI_USERNAME and UI_PASSWORD from your .env." -ForegroundColor Green
Write-Host "API docs: http://localhost:8080/docs"
Write-Host @"

Demo questions:
  1. Normal:     What is the shelf life of StableCap Dry Powder?
  2. Table:      How many vials of TL-COLD-9001 were shipped in total?
  3. ID:         Why was shipment SH-2024-007 rejected?
  4. Follow-up:  What corrective action was taken?   (right after 3; needs -FollowUps)
  5. Off-topic:  What is the capital of France?
"@

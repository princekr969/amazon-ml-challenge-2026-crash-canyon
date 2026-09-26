# =====================================================================
# ML Challenge 2026 - crash-canyon - Submission ZIP builder (PROD)
# Run AFTER src/main_fast.py finishes.
# Usage:  .\make_submission_zip.ps1
# =====================================================================

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "ML Challenge 2026 - crash-canyon - Submission ZIP Builder" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# ---------- Step 1: Verify output files ----------
Write-Host "`n[1/5] Verifying output files..." -ForegroundColor Yellow
$matchFile = "output\matching_results.tsv"
$candFile  = "output\candidate_pairs.tsv"
if (-not (Test-Path $matchFile)) { Write-Host "  X  $matchFile missing. Run src\main_fast.py first." -ForegroundColor Red; exit 1 }
if (-not (Test-Path $candFile))  { Write-Host "  X  $candFile missing." -ForegroundColor Red; exit 1 }

$matchRows = (Get-Content $matchFile | Measure-Object -Line).Lines - 1
$matchSize = [math]::Round((Get-Item $matchFile).Length / 1MB, 2)
$candRows  = (Get-Content $candFile  | Measure-Object -Line).Lines - 1
$candSize  = [math]::Round((Get-Item $candFile).Length / 1MB, 2)
Write-Host "  OK matching_results.tsv : $matchRows rows, $matchSize MB" -ForegroundColor Green
Write-Host "  OK candidate_pairs.tsv  : $candRows rows, $candSize MB" -ForegroundColor Green

if ($matchRows -ne 1732544) {
    Write-Host "  !  WARNING: expected 1,732,544 rows for test set, got $matchRows" -ForegroundColor Yellow
}

# ---------- Step 2: Run validator ----------
Write-Host "`n[2/5] Running validator..." -ForegroundColor Yellow
$validator = "data\utils\validate_submission.py"
if (Test-Path $validator) {
    $valOut = & ".\venv\Scripts\python.exe" $validator `
        --matching "output\matching_results.tsv" `
        --candidate "output\candidate_pairs.tsv" `
        --test-dir "data\test" 2>&1
    $valExit = $LASTEXITCODE
    Write-Host $valOut
    if ($valExit -ne 0) {
        Write-Host "`nValidator FAILED. Fix issues before zipping." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ~  Validator not found, skipping." -ForegroundColor DarkYellow
}

# ---------- Step 3: Build ZIP staging folder ----------
Write-Host "`n[3/5] Building ZIP staging..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$zipName   = "crash_canyon_submission.zip"
$stageDir  = "submission\stage_$timestamp"
$zipOut    = "submission\$zipName"
if (Test-Path $stageDir)  { Remove-Item $stageDir  -Recurse -Force }
if (Test-Path $zipOut)    { Remove-Item $zipOut    -Force }
New-Item -ItemType Directory -Force -Path "$stageDir\output" | Out-Null
New-Item -ItemType Directory -Force -Path "$stageDir\code\business_entity_resolution\src" | Out-Null

# output/  (the two TSVs that go into the zip root structure under output/)
Copy-Item $matchFile -Destination "$stageDir\output\matching_results.tsv"
Copy-Item $candFile  -Destination "$stageDir\output\candidate_pairs.tsv"

# code/business_entity_resolution/src/  (your source code, all of it)
$srcFiles = @("data_loader.py","preprocess.py","country_parser.py","blocking_fast3.py","main_fast.py","country_utils.py")
foreach ($f in $srcFiles) {
    $src = "src\$f"
    if (Test-Path $src) { Copy-Item $src -Destination "$stageDir\code\business_entity_resolution\src\$f" }
}
# Also drop the ZIP builder itself for reproducibility
if (Test-Path "make_submission_zip.ps1") {
    Copy-Item "make_submission_zip.ps1" -Destination "$stageDir\code\business_entity_resolution\src\"
}

# code/business_entity_resolution/README.md
$codeReadme = @"
# How to Reproduce crash-canyon's V1 Submission

## Quick start
```bash
# 1. Place train/test data under data/ following README
# 2. Install deps
pip install -r requirements.txt

# 3. Run end-to-end
python -u src/main_fast.py
# (writes output/matching_results.tsv and output/candidate_pairs.tsv)

# 4. Validate
python data/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

## Pipeline stages
1. **Load** (`data_loader.py`) — read TSVs with progress bar
2. **Preprocess** (`preprocess.py`) — Unicode normalization, lowercasing, tokenization
3. **Block** (`blocking_fast3.py`) — token + 2-gram + 3-gram inverted index, top-K=10 per S1
4. **Score & Select** (`main_fast.py:fast_score`) — token overlap, top-K=3 from candidates
5. **Write Outputs** — `candidate_pairs.tsv` (model input) and `matching_results.tsv` (final)

## File layout
```
src/
  data_loader.py           # load TSVs with progress
  preprocess.py            # name/address normalization
  country_parser.py        # per-country parsers (US/IN/FR)
  blocking_fast3.py        # memory-safe blocked inverted index
  main_fast.py             # orchestrator
  make_submission_zip.ps1  # bundle the submission zip
```

## Key design decisions
- **Candidate set = union of 3 blocking strategies**: token, 2-gram, 3-gram.
- **Match top-K=3 from candidates**: precision-heavy per F_0.5 metric.
- **Country treated as soft feature**: never hard filter, must accept test-set France.
- **All blocking done in pure Python dict + heap**: memory-safe on 16GB laptops.
"@
Set-Content -Path "$stageDir\code\business_entity_resolution\README.md" -Value $codeReadme -Encoding UTF8

# code/business_entity_resolution/requirements.txt
if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" -Destination "$stageDir\code\business_entity_resolution\requirements.txt"
}

# Documentation_template.md (filled in) - root of zip
$docTemplateSrc = "..\6ab10eb3b23ba_student_resource\student_resource\Documentation_template.md"
$docDst = "$stageDir\Documentation_template.md"
if (Test-Path $docTemplateSrc) {
    Copy-Item $docTemplateSrc -Destination $docDst
    Write-Host "  Filled Documentation_template.md copied from student_resource" -ForegroundColor Gray
} elseif (Test-Path "Documentation_template.md") {
    Copy-Item "Documentation_template.md" -Destination $docDst
}

# ---------- Step 4: Compress ----------
Write-Host "`n[4/5] Compressing ZIP..." -ForegroundColor Yellow
Compress-Archive -Path "$stageDir\*" -DestinationPath $zipOut -CompressionLevel Optimal
$zipMB = [math]::Round((Get-Item $zipOut).Length / 1MB, 2)

Write-Host "  OK ZIP created: $zipOut ($zipMB MB)" -ForegroundColor Green
Write-Host "`n     ZIP structure:" -ForegroundColor Gray
Get-ChildItem -Recurse -Path (Split-Path $stageDir -Parent) | Where-Object { $_.PSIsContainer -eq $false } | ForEach-Object {
    $rel = $_.FullName.Substring((Resolve-Path $stageDir).Path.Length + 1)
    Write-Host "       $rel" -ForegroundColor Gray
}

# ---------- Step 5: Done ----------
Write-Host "`n[5/5] DONE!" -ForegroundColor Yellow
Write-Host "    ZIP ready: $zipOut" -ForegroundColor White
Write-Host "`nUpload to Unstop:"
Write-Host "  - Live leaderboard portal: upload output/matching_results.tsv (single TSV)"
Write-Host "  - Final package: also submit this full ZIP." -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

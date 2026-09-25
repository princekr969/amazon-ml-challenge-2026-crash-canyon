# fix_data_structure.ps1
# Reorganizes the data folder to match what the script expects

$ErrorActionPreference = 'Stop'
$projectRoot = "D:\Amazon_ML_Challenge\Amazon_ML_Challenge\amazon-ml-challenge-2026-crash-canyon"
$sourceDataset = "D:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset"
$dataDir = Join-Path $projectRoot "data"

Write-Host "=== Fixing data structure ===" -ForegroundColor Cyan

# Create proper subdirectories
$trainDir = Join-Path $dataDir "train"
$testDir  = Join-Path $dataDir "test"
$utilsDir = Join-Path $dataDir "utils"
New-Item -ItemType Directory -Path $trainDir, $testDir, $utilsDir -Force | Out-Null
Write-Host "Created subdirs: train/, test/, utils/"

# Copy test files (existing, may overwrite)
foreach ($f in @("test_source1.tsv","test_source2.tsv","test_source3.tsv")) {
    $src = Join-Path $dataDir $f
    $dst = Join-Path $testDir $f
    if (Test-Path $src) {
        Move-Item -Path $src -Destination $dst -Force
        Write-Host "  Moved: $f -> test/"
    }
}

# Copy train files
foreach ($f in @("train_source1.tsv","train_source2.tsv","train_source3.tsv","train_ground_truth.tsv")) {
    $src = Join-Path $sourceDataset "train\$f"
    $dst = Join-Path $trainDir $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Force
        Write-Host "  Copied: $f -> train/ ($((Get-Item $dst).Length / 1MB.ToString('0.0')) MB)"
    } else {
        Write-Host "  MISSING: $src" -ForegroundColor Red
    }
}

# Copy validator
$valSrc = Join-Path $sourceDataset "..\utils\validate_submission.py"
$valDst = Join-Path $utilsDir "validate_submission.py"
if (Test-Path $valSrc) {
    Copy-Item -Path $valSrc -Destination $valDst -Force
    Write-Host "  Copied: validate_submission.py -> utils/"
} else {
    Write-Host "  Searching for validate_submission.py..." -ForegroundColor Yellow
    $found = Get-ChildItem -Path "D:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource" -Filter "validate_submission.py" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Copy-Item -Path $found.FullName -Destination $valDst -Force
        Write-Host "  Copied: $($found.Name) -> utils/"
    }
}

# Verify final structure
Write-Host "`n=== Final data folder structure ===" -ForegroundColor Cyan
Get-ChildItem -Path $dataDir -Recurse -File | ForEach-Object {
    $relPath = $_.FullName.Substring($dataDir.Length)
    "{0,-50} {1,12:N0} bytes" -f $relPath, $_.Length
}

Write-Host "`n✅ Data structure fixed!" -ForegroundColor Green
Write-Host "Now run: python src\main.py" -ForegroundColor Yellow

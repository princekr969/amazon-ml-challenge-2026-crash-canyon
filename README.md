# Amazon ML Challenge 2026 — Team crash-canyon

Solution for the Business Entity Resolution problem. Match records across three independent business data sources and identify same-real-world-business entities.

## Team

- **Prince Kumar** — pipeline engineering, blocking, scoring
- **Harshit Vishnoi** — feature engineering, country-aware heuristics
- **Shubham Sharma** — data analysis, validation, evaluation harness
- **Yashasvi Jain** — submission packaging, runbook, GitHub release

## Special Thanks

- **Ajay Singh** — debugging support and review feedback during the hackathon

## Quick Start

```powershell
# 1. Run end-to-end pipeline
.\venv\Scripts\python.exe -u src\main_fast.py
# (writes output/matching_results.tsv and output/candidate_pairs.tsv)

# 2. Validate
.\venv\Scripts\python.exe data\utils\validate_submission.py `
    --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv `
    --test-dir data\test

# 3. Bundle the final submission ZIP
.\make_submission_zip.ps1
```

## Pipeline

1. **Load** test sources (S1, S2, S3) with progress bar
2. **Preprocess** names + addresses (Unicode → ASCII, lowercase, tokenize)
3. **Block** S1 → S2 and S1 → S3 with token + 2-gram + 3-gram inverted index
4. **Score** candidates by token overlap (names + addresses)
5. **Output** `candidate_pairs.tsv` (blocking model input) + `matching_results.tsv` (top-K=3 scored)

See `Documentation_template.md` for full methodology.

## File Layout

```
data/                  # train + test TSVs
src/                   # pipeline code
output/                # generated TSVs
make_submission_zip.ps1   # build submission zip
requirements.txt       # pinned deps
```

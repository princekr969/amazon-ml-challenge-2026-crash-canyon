# Amazon ML Challenge 2026 — Team Crash-Canyon

Solution for the **Business Entity Resolution** problem: match records across three independent business data sources and identify same-real-world-business entities. Optimised for the **macro F_0.5** metric (precision weighted 2× recall).

## Team

- **Prince Kumar** — pipeline engineering, blocking, scoring
- **Harshit Vishnoi** — feature engineering, country-aware heuristics
- **Shubham Sharma** — data analysis, validation, evaluation harness
- **Yashasvi Jain** — submission packaging, runbook, GitHub release

## Special Thanks

- **Ajay Singh** — debugging support and review feedback during the hackathon
- **Anthropic Claude Opus 4.5** — V12 architecture design (multi-channel blocking, 2-stage LightGBM, expected F_0.5 subset selection)

## Current Best: V12 (production-grade)

See `src/README_V12.md` for full architecture details and `Documentation_template.md` for the official methodology write-up.

## Quick Start (V12 — recommended)

```powershell
# 1. Install dependencies (run once)
pip install -r requirements.txt

# 2. Run end-to-end V12 pipeline on AWS g5.xlarge or local
python src/er_v12.py --data /path/to/dataset --out output/

# 3. (Optional) Validate locally
python data\utils\validate_submission.py `
    --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv `
    --test-dir data\test

# 4. Submission is at output/outputs_V12.zip
```

## V12 Pipeline (5-stage)

```
RAW ──► [1] Normalize ──► [2] Block ──► [3] 2-Stage LGBM ──► [4] Calibrate ──► [5] Decide ──► ZIP
       name+address      TF-IDF + key    stage1 → stage2       Isotonic         Expected F_0.5
       abbrev expand     joins per       34 features +         regression       subset selection
       +postal+house     country         context (reverse                      (per-entity k)
       extraction                       rank, score gap)
```

**Critical design choices** (each one fixes a V11 bug):

| Choice | Why it wins |
|---|---|
| **Top-1 cap → expected F_0.5 subset selection** | Recovers full recall — picks optimal k per S1 based on calibrated probs |
| **GroupKFold by S1 (5-fold)** | No validation leakage — same S1 never in train+val |
| **Per-country TF-IDF blocking** | France open-set handled automatically (cross-country = 0 candidates) |
| **Isotonic calibration** | Honest probabilities for F_0.5 math |
| **One-to-one enforcement** | Each S2/S3 record → at most one S1 |
| **34 features incl. reverse-rank** | Captures competition among S1s for same target |

## Earlier versions (kept for history)

- **V1** (`src/main_fast.py`) — simple token-inverted-index blocking + jaccard scoring, top-K=3 cap. Submitted, scored 0.133 (precision too low).
- **V1.1 STRICT** (`src/rescore_strict.py`) — top-2 cap with thr=0.70. Same 0.133 score format.
- **V11** (`scripts/aws_v11_NOTEBOOK.py`) — 19 features with per-(country, source) thresholds. Had blocking truncation issues, abandoned.
- **V12** (`src/er_v12.py`) — **current best**, see above.

## File Layout

```
data/                       # train + test TSVs (shared across versions)
src/                        # source code
  main_fast.py              # V1 baseline (legacy)
  rescore_strict.py         # V1.1 STRICT scorer (legacy)
  er_v12.py                 # V12 production pipeline ← RUN THIS
  README_V12.md             # V12 architecture + run instructions
output/                     # generated TSVs (per-version subdirs)
  output/matching_results.tsv    # V12 final predictions
  output/candidate_pairs.tsv    # V12 candidates the model scored
  output/outputs_V12.zip        # V12 submission zip
  output/artifacts/             # LightGBM models + OOF summary
scripts/                    # PowerShell + per-version SageMaker cells
make_submission_zip.ps1     # legacy submission packager
requirements.txt            # pinned deps (lightgbm, rapidfuzz, sparse-dot-topn, ...)
Documentation_template.md   # official methodology write-up (V12)
```

## Performance notes

- **AWS g5.xlarge** (4 vCPU + A10G): V12 runs in ~35-50 min
- **Local 16 GB**: OOM risk during TF-IDF blocking on full data — recommend AWS or `--train-sample 200000` for local debug
- **Memory peak**: ~10-12 GB during stage-2 inference

## License

This codebase is original work by team crash-canyon. Models used are MIT/Apache-licensed (LightGBM, scikit-learn, rapidfuzz, sparse-dot-topn). No external data or paid APIs were used.
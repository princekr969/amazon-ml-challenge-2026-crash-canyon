# V12 Entity Resolution Pipeline

Production-grade 5-stage pipeline for Amazon ML Challenge 2026 — Business Entity Resolution.

## Quick Start

```bash
pip install -r requirements.txt
python src/er_v12.py --data /path/to/dataset --out output/
```

## Pipeline Overview

1. **Normalization** — name (initials collapse, abbrev expansion, legal suffix strip) + address (postal + house number extraction)
2. **Multi-channel blocking** — TF-IDF (word + char 4-gram) + exact key joins, per-country
3. **Stage-1 LightGBM** — 34 features (string + address + postal + house + token + context + reverse-rank)
4. **Stage-2 LightGBM** — stacks stage-1 probabilities + per-S1-group statistics
5. **Isotonic calibration** + **Expected F_0.5 subset selection** (the killer — directly optimizes the leaderboard metric)

## Critical design decisions

- **Top-1 cap removed** — uses per-entity expected F_0.5 subset selection (`best_k` picks optimal k per S1)
- **GroupKFold by S1** — no validation leakage between train/val
- **Per-country blocking** — handles France open-set automatically (no cross-country matches)
- **One-to-one enforcement** — each S2/S3 record assigned to at most one S1
- **Calibrated probabilities** — Isotonic regression on OOF for honest F_0.5 math

## Hyperparameters (CLI flags)

| Flag | Default | Description |
|---|---|---|
| `--train-sample` | 400_000 | # S1 train entities used for model training |
| `--k` | 10 | top-k per TF-IDF channel per source |
| `--max-cands` | 25 | max candidates per S1 fed to the model |
| `--min-cos` | 0.10 | minimum cosine threshold for TF-IDF blocking |
| `--folds` | 5 | # CV folds (GroupKFold by S1) |
| `--seed` | 42 | RNG seed |
| `--smoke` | False | quick debug run (NOT submittable) |

## Output

- `output/matching_results.tsv` — final predictions (upload to portal)
- `output/candidate_pairs.tsv` — candidates the model actually scored
- `output/outputs_V12.zip` — zip for submission
- `output/artifacts/stage1.txt`, `stage2.txt` — LightGBM models
- `output/artifacts/oof_summary.json` — OOF macro-F_0.5 summary

## Performance on g5.xlarge

- Train blocking: ~5-8 min
- Stage-1 5-fold CV: ~5-10 min
- Stage-2 5-fold CV: ~5-7 min
- Test blocking + scoring: ~15-20 min
- **Total: ~35-50 min**

## Requirements

```
pip install lightgbm scikit-learn rapidfuzz unidecode scipy pandas numpy sparse-dot-topn
```
# Amazon ML Challenge 2026 — Team Crash-Canyon

Solution for the **Business Entity Resolution** problem: match records across three independent business data sources and identify same-real-world-business entities. Optimised for the **macro F_0.5** metric (precision weighted 4× recall).

## Final Results (Hackathon 2026)

| Metric | Value | Top 1 Reference |
|---|---|---|
| **Rank** | **2612** | Top 1: 0.990788 |
| **Overall F_0.5** | **0.951853** | Gap to #1: ~0.039 |
| Team | crash-canyon (4 members) | |
| Best pipeline | V31 (5-stage + competition features) | |
| Total iterations | 14 versions (V1, V1.1, V11-V12, V20-V31) | |

## 📈 Score Progression

```
 V10/V11 era baseline (jaccard, top-K=3)  0.133  ███░░░░░░░░░░░░░░░░░░░░░░░░░  ← 26 Sep, 5:31 PM
 V22 reverse blocking + on-disk cache     0.551  █████████████▌░░░░░░░░░░░░░  ← 27 Sep, 9:48 AM (incomplete AWS copy)
 V24 learned pruning + 2-stage LGBM       0.918  ███████████████████████▌░░  ← 27 Sep, 2:14 PM (first official data run)
 V27 + address + transliteration + stage3 0.947  █████████████████████████▌░  ← 27 Sep, 7:25 PM
 V31 FINAL — competition features 🏆      0.952  █████████████████████████▌░  ← 27 Sep, evening
                                         ─────  ────────────────────────────
                                         0.000                            1.000
```

**Score lifted 7.2×** in 14 iterations: **0.133 → 0.952** 📈

| Submitted | Version | Δ F_0.5 | What changed | Wall-time |
|---|---|---|---|---|
| 26 Sep, 5:31 PM | V10/V11 era | — | Baseline jaccard + top-K=3 | — |
| 27 Sep, 9:48 AM | V22 | **+0.418** | Reverse blocking + on-disk cache (incomplete AWS copy → low) | ~16 h |
| 27 Sep, 2:14 PM | V24 | **+0.367** | First official run: learned pruning + 2-stage LGBM | ~4 h |
| 27 Sep, 7:25 PM | V27 | **+0.029** | + address channel + transliteration + name splitting + stage 3 | ~5 h |
| 27 Sep, evening  | **V31** | **+0.005** | + V29 noise fixes + second hop + competition features (`--s1-keep 0.81`) | ~2 h |

> The last 0.005 took ~2 hours but mattered most — that's the precision-4×-weighted slice of F_0.5.

## Team

- **Prince Kumar** — pipeline engineering, blocking, scoring, V1-V31 integration
- **Harshit Vishnoi** — feature engineering, country-aware heuristics, V12+ review
- **Shubham Sharma** — data analysis, validation harness, error analysis
- **Yashasvi Jain** — submission packaging, runbook, GitHub release

## Special Thanks

- **Ajay Singh** — debugging support and review feedback during the hackathon
- **Anthropic Claude Opus 4.5** — V12-V31 architecture design (5-stage pipeline, expected-F_0.5 subset selection, multi-channel blocking, learned equivalences, competition features, second-hop retrieval, transliteration)
- **MiniMax / MiniMax-M3 (Mavis)** — final-iteration code review, documentation polish, score-progression visualization, GitHub release management, V22-V31 push & cleanup

## Current Best: V31 (final production pipeline)

See `src/README_V12.md` for base architecture details and `Documentation_template.md` for the official methodology write-up.

## Quick Start (V31 — final)

```powershell
# 1. Install dependencies (run once)
pip install -r requirements.txt

# 2. Run end-to-end V31 pipeline
python src/er_v31.py --data-dir /path/to/dataset --out-dir output --workers 4

# 3. (Optional) Validate locally
python data\utils\validate_submission.py `
    --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv `
    --test-dir data\test

# 4. Submission is at output/matching_results.tsv + candidate_pairs.tsv
```

## Pipeline Evolution (14 versions)

| Version | Key addition | F_0.5 | Submitted |
|---|---|---|---|
| V1 | Token-inverted-index blocking + jaccard scoring, top-K=3 | **0.133** | 26 Sep, 5:31 PM |
| V1.1 | STRICT threshold (0.70), top-2 cap | 0.133 | (same format, low precision) |
| V11 | 19 features, per-(country, source) thresholds | abandoned | (tuple-unpack bug) |
| V12 | Production 5-stage: TF-IDF blocking + 2-stage LGBM + expected-F_0.5 | — | (AWS OOM) |
| V20 | char 3-gram + optional sparse_dot_topn + top-2 exclusivity | — | (still iterating) |
| V21 | HashingVectorizer (low memory ~6-8 GB) | — | (still iterating) |
| **V22** | Reverse blocking + on-disk cache + stage-2 K-fold OOF | **0.551** | 27 Sep, 9:48 AM |
| V23 | Learned token equivalences + error report | — | (still iterating) |
| **V24** | Learned candidate pruning (smaller candidate_pairs.tsv) | **0.918** | 27 Sep, 2:14 PM |
| V25 | Address-only blocking + state normalization (US + IN) | — | (still iterating) |
| V26 | Stage-2 K-fold OOF + Stage-3 set coherence + filtered equivalences | — | (still iterating) |
| **V27** | Learned transliteration (native-script) + glued-name segmentation + French norm | **0.947** | 27 Sep, 7:25 PM |
| V28 | Fixed exclusivity + French noise + dotted acronyms | — | (still iterating) |
| V29 | Repeated words removed + look-alike digits + ordinals + STREET-WORDS features | — | (still iterating) |
| **V30** | **Second-hop candidates (sibling retrieval) + S1-keep validation** | — | (folded into V31) |
| **V31** | **Competition features + full S1 scoring (FINAL)** | **0.951853 🏆** | 27 Sep, evening |

## V31 Pipeline (final architecture)

```
RAW ──► [1] Normalize ──► [2] Block ──► [3] Second-Hop ──► [4] Stage-1 LGBM
       name+address      TF-IDF + key    anchors retrieve   + context + reverse-rank
       abbrev expand     joins + rev     neighbors via      + competition features
       +postal+house     blocking        TF-IDF
       + transliteration
       + segmentation

       ──► [5] Stage-2 LGBM ──► [6] Stage-3 ──► [7] Calibrate ──► [8] Decide
            K-fold OOF        set coherence   Isotonic         Expected F_0.5
            + group stats                       regression       subset selection
                                                                       (per-entity k)
```

### Critical design choices (each fixes a real V11/V12 bug)

| Choice | Why it wins |
|---|---|
| **Top-1 cap → Expected F_0.5 subset selection** | Recovers full recall — picks optimal k per S1 from calibrated probs |
| **GroupKFold by S1 (5-fold)** | No validation leakage — same S1 never in train+val |
| **Per-country TF-IDF blocking** | France open-set handled automatically (cross-country = 0 candidates) |
| **Second-hop retrieval (V30)** | Catches hidden sibling records of the same business |
| **Learned transliteration (V27)** | Hindi/Tamil/Bengali/Cyrillic → English name matching |
| **Name segmentation (V27)** | "fortunefinance.com" → ["fortune", "finance", "com"] |
| **Competition features (V31)** | Cross-S1 tie-breaking on contested S2/S3 records |
| **Isotonic calibration** | Honest probabilities for F_0.5 math |
| **One-to-one enforcement** | Each S2/S3 record → at most one S1 |
| **34+ features incl. reverse-rank + competition** | Captures S1-vs-S1 competition + reverse-rank context |
| **Learned token equivalences (V23)** | Data-driven abbreviation map (better than hand-written) |
| **On-disk blocking cache (V22)** | Re-runs skip blocking (~20-30 min saved) |

## File Layout

```
data/                       # train + test TSVs (NOT in git, downloaded separately)
src/                        # source code (all V* pipelines here)
  main_fast.py              # V1 baseline (legacy)
  rescore_strict.py         # V1.1 STRICT scorer (legacy)
  rescore_ultra.py          # V1 ULTRA STRICT (legacy)
  er_v12.py                 # V12 production pipeline (5-stage baseline)
  er_v20.py                 # V20 char 3-gram + sparse_dot_topn
  er_v21.py                 # V21 hashed TF-IDF (low memory)
  er_v22.py                 # V22 reverse blocking + cache
  er_v23.py                 # V23 learned equivalences + error report
  er_v24.py                 # V24 learned candidate pruning
  er_v25.py                 # V25 address-only blocking + state normalization
  er_v26.py                 # V26 stage-2 K-fold OOF + set coherence
  er_v27.py                 # V27 transliteration + segmentation + French
  er_v28.py                 # V28 fixed exclusivity + dotted acronyms
  er_v29.py                 # V29 look-alike digits + ordinals + street-words
  er_v30.py                 # V30 second-hop candidates + S1-keep
  er_v31.py                 # V31 FINAL: competition features + full S1 scoring ← FINAL BEST
  README_V12.md             # V12 architecture + run instructions
scripts/                    # PowerShell + per-version SageMaker cells
output/                     # generated TSVs (per-version subdirs)
make_submission_zip.ps1     # legacy submission packager
requirements.txt            # pinned deps (lightgbm, rapidfuzz, sparse-dot-topn, ...)
Documentation_template.md   # official methodology write-up
README.md                   # this file
```

## Performance notes

- **AWS ml.m5.4xlarge** (16 vCPU + 64 GB RAM, CPU-only): primary instance for V22-V31 — all post-hackathon iterations ran here, ~45-60 min for V31
- **AWS g5.xlarge** (4 vCPU + A10G + 16 GB GPU): V12/V20 prototype runs, ~35-50 min for V31
- **Local 16 GB**: OOM risk during blocking — recommend `--train-s1 100000` for local debug
- **Memory peak**: ~10-12 GB during stage-2 inference
- **Cache reuse**: V22-V31 share `--work-dir` for fast re-runs

## License

This codebase is original work by team crash-canyon. Models used are MIT/Apache-licensed (LightGBM, scikit-learn, rapidfuzz, sparse-dot-topn). No external data or paid APIs were used. V12-V31 architecture designed with assistance from Claude Opus 4.5 during the hackathon; implementation and tuning are our own work.

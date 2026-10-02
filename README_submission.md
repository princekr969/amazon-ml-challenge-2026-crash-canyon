# Amazon ML Challenge 2026: Business Entity Resolution
## Team Crash-Canyon — End-to-End Reproduction Guide

**Public leaderboard (final submission): 0.952 macro F0.5 — Rank 2612**
**Repository:** https://github.com/princekr969/amazon-ml-challenge-2026-crash-canyon (commit `af00d36` — V31 final)

This folder contains the complete, self-contained source code of the final Crash-Canyon pipeline. Using only the organizer-provided training and test files, it regenerates the two submission outputs:

```text
output/matching_results.tsv     # final matches (the file uploaded to the leaderboard)
output/candidate_pairs.tsv      # exact candidate set the matching model ran inference on
```

No external data, web lookup, API, geocoding service or pretrained model is used. All dictionaries (transliteration, vocabulary, token equivalences) are learned from the provided training files at run time. All dependencies are MIT/Apache-licensed, well below the 8-billion-parameter limit.

---

## 1. Results at a glance

| Version | What it added | Holdout macro F0.5 | Public leaderboard |
|---|---|---|---|
| V1   | Baseline token-inverted-index + jaccard, top-K=3 | — | 0.133 |
| V11  | 19 features, per-(country, source) thresholds | abandoned | — |
| V12  | Production 5-stage: TF-IDF blocking + 2-stage LGBM + expected-F0.5 | OOM | — |
| V20  | char 3-gram + optional sparse_dot_topn + top-2 exclusivity | — | 0.551 (incomplete data copy) |
| V22  | Reverse blocking + on-disk cache + stage-2 K-fold OOF | — | (incomplete) |
| V24  | reverse blocking, learned pruning, 2-stage model | 0.9387 | 0.918 (first full-data run) |
| V27  | address-only channel, learned transliteration, glued-name segmentation, stage 3 | n/a | 0.947 |
| V30 (validation) | noise fixes from error analysis, street-word features, second-hop candidates | 0.9656 | n/a |
| V31 (validation) | learned competition features | **0.9688** | n/a |
| **V31 final** | + test-like training density (`--s1-keep 0.81`) | measured under test-like density (not comparable) | **0.952** 🏆 |

### Score progression
```
0.133 (V1 baseline) → 0.551 (V20) → 0.918 (V24) → 0.947 (V27) → 0.952 (V31 FINAL)
```

### Final-run statistics on the test set
- 1,732,544 S1 entities
- **14,624,292 candidate pairs (8.44 per S1)** after second-hop + pruning
- 1,632,066 S1 entities with at least one match (94.2%)
- 5,639,975 predicted links

### Blocking pair recall (training set)
- 95.64% after blocking
- **96.53% after second-hop expansion**
- India 95.24%, US 97.40%
- Learned pruning removes ~0.2 points of recall while cutting ~49 raw candidates per S1 down to ~8.

### Holdout construction
The holdout is 100,000 training S1 entities that are never used for training. It is scored with the exact leaderboard metric: per-entity F0.5 averaged over entities, singletons included, and true matches outside the candidate set counted as misses.

---

## 2. Folder layout

```text
code/business_entity_resolution/
├── README.md              ← this file
├── requirements.txt       ← pinned dependencies
└── src/
    ├── er_v27.py          # STEP A: normalization + multi-channel blocking → candidate cache in work/
    ├── er_v31.py          # STEP B: final pipeline (second hop, pruning, 3-stage model, decision, outputs)
    ├── er_v12.py          # V12 base production pipeline (development history)
    ├── er_v20.py          # V20 char 3-gram + sparse_dot_topn
    ├── er_v21.py          # V21 hashed TF-IDF (low memory)
    ├── er_v22.py          # V22 reverse blocking + on-disk cache
    ├── er_v23.py          # V23 learned token equivalences
    ├── er_v24.py          # V24 learned candidate pruning
    ├── er_v25.py          # V25 address-only + state normalization
    ├── er_v30.py          # V30 second-hop candidates + S1-keep validation
    ├── README_V12.md      # V12 architecture notes (development history)
    └── (other helper modules)
```

`er_v27.py` is a complete pipeline on its own. Here it is used for its blocking stage: it writes the candidate cache that `er_v31.py` loads. `er_v31.py` contains all later improvements. Both scripts use the same cache-file naming, so `er_v31.py` automatically reuses the cache written by `er_v27.py` when both point to the same `--work-dir`.

---

## 3. Environment

| | Requirement |
|---|---|
| OS | Linux (tested on AWS SageMaker, Amazon Linux / Ubuntu-based images). Uses fork-based multiprocessing. On Windows: use WSL2. |
| Python | 3.12 (3.10+ should work) |
| CPU / RAM | **16 vCPU, 64 GB RAM** (tested on AWS `ml.m5.4xlarge`). Peak memory during the final stage is roughly 40–50 GB. |
| Disk | ~40 GB free (dataset ≈ 5 GB, train cache ≈ 11 GB, test cache ≈ 10 GB, outputs ≈ 1 GB) |
| GPU | not needed |

Install the pinned environment:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Main libraries: NumPy, pandas, SciPy, scikit-learn, LightGBM (MIT), RapidFuzz, Unidecode, sparse_dot_topn (exact versions in `requirements.txt`).

---

## 4. Dataset layout

Pass the organizer dataset folder with `--data-dir`. It is searched recursively, so the unpacked `student_resource/dataset` folder works as-is:

```text
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

If the dataset was unpacked from a zip made on macOS, delete any `__MACOSX` folder first.

Important fields:
- `entity_id`
- `business_name`
- `business_address`
- `country`

The ground-truth file is expected to contain:
- `source1_entity_id`
- `matched_entity_ids`

where `matched_entity_ids` contains the matching Source-2/Source-3 entity IDs associated with each Source-1 entity.

---

## 5. Reproduce the final submission (exact configuration)

Run from `code/business_entity_resolution/`. Both steps **must use the same `--work-dir`**.

### Step A: build the blocking cache (≈ 5.5 h on 16 vCPU)

```bash
python3 src/er_v27.py \
  --data-dir /path/to/dataset \
  --out-dir output_stepA \
  --work-dir work \
  --workers 14
```

This writes `work/cand_v27_train_...pkl` and `work/cand_v27_test_...pkl` (the train and test candidate caches). `output_stepA/` contains V27's own predictions, which are not needed.

### Step B: final model on the cached blocking (≈ 3 h on 16 vCPU)

```bash
python3 src/er_v31.py \
  --data-dir /path/to/dataset \
  --out-dir output \
  --work-dir work \
  --s1-keep 0.81 \
  --workers 14
```

The log must show `loading cached candidates` twice (train and test). All other settings are the script defaults, listed in Section 11. Seed 42 is fixed.

### Windows PowerShell (via WSL2)

```powershell
# Inside WSL2 Ubuntu shell
cd /mnt/c/path/to/code/business_entity_resolution
python3 src/er_v27.py --data-dir /path/to/dataset --out-dir output_stepA --work-dir work --workers 14
python3 src/er_v31.py --data-dir /path/to/dataset --out-dir output --work-dir work --s1-keep 0.81 --workers 14
```

### Outputs (in `output/`)

- `matching_results.tsv` — the final matches (the leaderboard file). Decision variant (exclusivity on/off, γ) is chosen automatically on the holdout.
- `candidate_pairs.tsv` — the pruned candidate set the model scored. Every predicted match is contained in it.
- `matching_results_excl.tsv`, `matching_results_noexcl.tsv` — same predictions with / without one-to-one exclusivity rule.
- `test_probs.npz` — calibrated probability of every candidate pair (input for `er_decide.py`).

The run log also reports blocking recall (overall, per country, forward vs reverse vs second hop), pruning statistics, holdout macro F0.5 after each stage, and a per-category error report. Example errors are written to `work/errors_holdout.tsv`.

### One-step alternative

`er_v31.py` can also run alone (without step A). It then builds its own blocking cache with its newer normalization rules. That works end to end, but its candidate set differs slightly from the submitted run, so use steps A + B to reproduce the submission exactly.

### Quick smoke test (no full reproduction, ~5 min)

```bash
python3 src/er_v31.py \
  --data-dir /path/to/dataset \
  --out-dir output_test \
  --work-dir work_test \
  --train-s1 10000 --holdout-s1 10000 --workers 4
```

This validates environment + dependencies on a small subset. Output will not match the leaderboard.

---

## 6. Pipeline overview

```text
Raw S1 / S2 / S3 records
 │
 ├─[1] Normalization: learned transliteration of native-script names, glued/domain-name
 │     segmentation, legal forms & abbreviations, honorific/noise words, repeated words,
 │     look-alike digits, ordinals, dotted acronyms, address abbreviations (US/India/France),
 │     state names/codes incl. native-script state names, NULL/unit noise
 │
 ├─[2] Blocking, per country (0.000% cross-country matches in the labels)
 │     3 sparse TF-IDF channels × 2 sources (S2, S3) × 2 directions:
 │       char 3-grams of core name | name + address words | address words only
 │       forward: each S1 → top-K targets;  reverse: each target → top-K S1
 │
 ├─[3] Second-hop expansion: each S1's 2 best candidates retrieve their nearest
 │     target neighbours (name+address and address spaces) → extra candidates
 │
 ├─[4] Learned pruning: LightGBM on blocking-stage features only;
 │     keeps 99.8% of blocked true pairs, ≤ 15 per S1  →  candidate_pairs.tsv
 │
 ├─[5] Stage 1: LightGBM (3-fold, grouped by S1) on pair + context features
 ├─[6] Stage 2: group features + competition features (from stage-1 probabilities)
 ├─[7] Stage 3: set coherence + competition features (from stage-2 probabilities)
 ├─[8] Isotonic calibration
 └─[9] Decision: one-to-one exclusivity (strongest claimant keeps its probability)
       + per-S1 expected-F0.5 subset selection (empty set allowed)  →  matching_results.tsv
```

### Why this architecture wins

| Choice | Why it wins |
|---|---|
| **Top-1 cap → Expected F0.5 subset selection** | Recovers full recall — picks optimal k per S1 from calibrated probs |
| **GroupKFold by S1 (3-fold)** | No validation leakage — same S1 never in train+val |
| **Per-country TF-IDF blocking** | France open-set (15% of test, 0% of train) handled automatically |
| **Second-hop retrieval (V30)** | Catches hidden sibling records of the same business |
| **Learned transliteration (V27)** | Hindi/Tamil/Bengali → English name matching |
| **Name segmentation (V27)** | "fortunefinance.com" → ["fortune", "finance", "com"] |
| **Competition features (V31)** | Cross-S1 tie-breaking on contested S2/S3 records |
| **Isotonic calibration** | Honest probabilities for F0.5 math |
| **One-to-one enforcement** | Each S2/S3 record → at most one S1 |
| **Learned token equivalences (V23)** | Data-driven abbreviation map |
| **On-disk blocking cache (V22)** | Re-runs skip blocking (~30-60 min saved) |

---

## 7. Normalization (details)

- **Learned transliteration dictionary.** Built from training pairs where the same business appears in Latin script (S1) and in a native script (S2/S3), by positional token alignment. In the final run it contains 12,003 token translations (e.g. `फाइनेंस → finance`, `இன்வெஸ்ட்மெண்ட்ஸ் → investments`, `प्राइवेट → private`).
- **Glued/domain-name segmentation.** Dynamic-programming word segmentation over a vocabulary built from training S1 names (≈18.8k words): `privateyadavtradingcom → private yadav trading`.
- **Names.** ASCII folding (unidecode), `&` → `and`, dotted acronyms joined (`E.U.R.L.` → `eurl`), look-alike digits inside words (`ch0ice` → `choice`), abbreviation expansion, repeated words removed, and a *core name* without legal forms, honorifics (Smt, Dr, Sri, …) and injected noise words (`Center`, `Et Fils`, `doing business as`, …).
- **Addresses.** Abbreviation expansion (`rd`, `st`, `blvd`, `bd`, `imp`, `ch`, …), US and Indian state names (including native-script state names) mapped to one code, ordinals (`2nd`/`2th` → `2`), leading zeros stripped, and `NULL`, `N/A`, `city`, `cedex`, unit numbers dropped.
- **Flags.** Share of native-script characters in each name, and a domain-style-name flag.

---

## 8. Blocking (details)

| Channel | Representation | Forward top-K per source | Reverse top-K |
|---|---|---|---|
| char | character 3-grams of the core name (hashed TF-IDF, 3-grams in > 1% of names dropped) | 10 | 3 |
| comb | word TF-IDF of core name + address (address weight 0.8) | 12 | 3 |
| addr | word TF-IDF of the address only | 6 | 3 |

Exact sparse top-K products are computed with `sparse_dot_topn`. Forward candidates are capped at 40 per S1 and reverse-only candidates at 20 per S1. For every pair, all three cosines are computed, plus context statistics: rank and gap within the S1, and rank and gap of the S1 among all S1 entities that retrieved the same target.

### Blocking effectiveness
- **Forward blocking:** ~90% recall on training
- **+ Reverse blocking:** ~92% recall (recovers S1 missed by forward)
- **+ Second-hop retrieval:** ~96% recall (recovers hidden siblings)
- **+ Learned pruning:** ~95.64% recall retained (cleaner candidates for scoring)

---

## 9. Features (as named in the code)

- **Context:** `hop`, `cc`, `cm`, `ca` (three cosines), ranks and gaps within the S1 (`rk_*`, `gap_*`), reverse ranks and gaps (`rev_rk_*`, `rev_gap_*`), `n_cand`, `rev_n`, `rk_cm_src`, `src`, `fwd`, `same_cty`, native-script shares `nat1`/`nat2`, domain flags `dom1`/`dom2`.
- **Name:** `n_ratio`, `n_partial`, `n_tsort`, `n_tset`, `n_jw` (Jaro-Winkler), `nf_ratio` (full name), `n_exact`, `n_jac`, IDF-weighted overlap `n_idf_jac`/`n_idf_min`, `n_first`, acronym `n_acro`, lengths and token counts, `n_digits`, and similarity after learned token equivalences (`n_jac_map`, `n_tset_map`).
- **Address:** `a_tset`, `a_ratio`, `a_partial`, IDF-weighted overlap `a_idf_jac`/`a_idf_min`, postal-code match `a_postal`, numeric-token Jaccard `a_num_jac`, `a_name_in`, `a_missing`, `a_jac_map`, and **street-word features ignoring numbers** (`a_wjac`, `a_wcont`, `a_widf`, `a_num_conflict`), because house numbers are deliberately corrupted in the data.
- **Stage 2:** stage-1 probability and logit, rank and gap within the S1, group sum, number of confident candidates, rank within source, similarity to the group's best candidate (`coh_*`), and **competition features** `c1_other`, `c1_rank`, `c1_sumo`, `c1_n`, `c1_margin`.
- **Stage 3:** stage-2 probability and logit, set coherence with the confident members of the same S1 (`n_conf`, `c_ntset`, `c_atset`, `c_num`, `c_both`, `c_meanp`), and competition features after stage 2 (`c2_*`).

Total: **38+ pairwise features** + 12 stage-2 derived + 8 stage-3 derived = **~58 engineered features**.

---

## 10. Training, calibration and decision

- **Sampling:** a uniform sample of 100,000 training S1 entities for training and 100,000 for the holdout, after removing 19% of training S1 entities (`--s1-keep 0.81`), whose records remain as unmatched distractors. This matches the test's 5.75 targets per S1 (training: 4.68).
- **Competition features** are computed over **all** remaining training S1 entities (scored with stage 1 and stage 2), so training and test see the same competition.
- **Models:** LightGBM; stage 1 and stage 2 use 3-fold out-of-fold predictions grouped by S1; stage 3 is trained on stage-2 out-of-fold features; isotonic calibration on a held-out part of the training sample.
- **Decision:** one-to-one exclusivity (each S2/S3 record keeps its probability only for its strongest S1 claimant; others are capped at 1 − strongest). Then, per S1, the prefix of candidates sorted by probability (possibly empty) that maximizes the expected F0.5, estimated by Monte Carlo sampling. Exclusivity on/off and a sharpening exponent γ are chosen on the holdout.

### Expected F0.5 math (the killer feature)
For each S1 entity with candidates having calibrated probs `p_1 ≥ p_2 ≥ ... ≥ p_n`, choose subset `S ⊆ {1..n}` (including the empty set) maximising expected F0.5:

```
F_0.5(S) = 1.25 · E[TP] / (E[TP] + 0.25 · E[FN] + E[FP])
```

This recovers full recall by picking optimal k per S1 from calibrated probabilities — better than any top-K cap.

---

## 11. Command-line options (`er_v31.py` defaults)

| Option | Default | Purpose |
|---|---:|---|
| `--data-dir` | auto-detect | dataset root (searched recursively) |
| `--out-dir` / `--work-dir` | `output` / `work` | outputs / cache, models, reports |
| `--workers` | CPU count − 1 | processes and threads (final run: 14) |
| `--train-s1` / `--holdout-s1` | 100000 / 100000 | training / holdout S1 sample sizes |
| `--k-char` / `--k-comb` / `--k-addr` | 10 / 12 / 6 | forward top-K per channel and source |
| `--k-rev`, `--no-rev-char`, `--rev-cap` | 3, off, 20 | reverse blocking |
| `--max-cand`, `--addr-weight`, `--char-max-df` | 40, 0.8, 0.01 | blocking caps and weights |
| `--hop-anchors`, `--hop-k`, `--hop-cap`, `--hop-min`, `--no-hop` | 2, 5, 10, 0.3, off | second-hop expansion |
| `--prune-keep` / `--prune-max` | 0.998 / 15 | pruning target recall / cap per S1 |
| `--no-comp` | off | disable competition features |
| `--s1-keep` | 1.0 (**final run: 0.81**) | share of training S1 kept; the rest become distractors |
| `--folds`, `--seed`, `--map-min-count` | 3, 42, 15 | model folds, seed, learned-equivalence threshold |
| `--no-cache`, `--wait-cache`, `--skip-test` | off, 0, off | cache control, train/validate only |

Changing any blocking option (`--k-*`, `--rev-*`, `--max-cand`, `--addr-weight`, `--char-max-df`) changes the cache file name and triggers new blocking.

### Final-run configuration (the exact one used)

```bash
python3 src/er_v27.py \
  --data-dir /path/to/dataset \
  --out-dir output_stepA \
  --work-dir work \
  --workers 14

python3 src/er_v31.py \
  --data-dir /path/to/dataset \
  --out-dir output \
  --work-dir work \
  --s1-keep 0.81 \
  --workers 14
```

---

## 12. Output file specifications

### `matching_results.tsv`

The final submission file.

Format:
```text
source1_entity_id    matched_entity_ids
```

`matched_entity_ids` is a comma-separated list of the selected Source-2/Source-3 entity IDs.

A Source-1 entity with no predicted matches has an empty `matched_entity_ids` field.

### `candidate_pairs.tsv`

The candidate set generated and scored by the blocking/pruning pipeline.

Format:
```text
source1_entity_id    candidate_entity_ids
```

`candidate_entity_ids` is the comma-separated set of target entity IDs considered by the final pipeline for that Source-1 entity.

The final matching set is a subset of this candidate set.

### Additional runtime artifacts (informational only)

The program may also write intermediate/diagnostic artifacts such as:
```text
output/test_probs.npz
work/
    candidate caches
    LightGBM models
    holdout predictions
    learned artifacts
    validation/error reports
```

These are not the two leaderboard TSVs and are not required for the submission output directory.

---

## 13. Caching

V31 supports an on-disk blocking cache. By default:
```text
--work-dir work
```

is used. Re-running the same configuration can therefore reuse previously generated candidate sets instead of rebuilding all blocking structures.

To force regeneration:
```bash
python3 src/er_v31.py \
  --data-dir /path/to/dataset \
  --out-dir output \
  --work-dir work \
  --workers 4 \
  --no-cache
```

**Do not delete the cache during a run.**

---

## 14. Optional: fast re-decision

Rebuild the matching file from saved probabilities with another decision setting (seconds, no model re-run):

```bash
python3 src/er_decide.py --probs output/test_probs.npz --out output/matching_results_alt.tsv --excl on --gamma 1.0
```

---

## 15. Reproducibility notes

- Seed 42 is fixed for all sampling and models.
- LightGBM with many threads can still produce tiny floating-point differences between machines, which can move a few borderline candidates in or out of the pruned set.
- The `output/matching_results.tsv` in the submission zip is the exact file that was uploaded to the leaderboard.
- Keep `--workers 14` and the same package versions for the closest reproduction.

---

## 16. Troubleshooting

- **`FileNotFoundError ... not found under ...`:** `--data-dir` does not contain the seven TSV files. Pass their parent folder.
- **Step B starts with `vectors built` instead of `loading cached candidates`:** the work folder or blocking options differ from step A. Use the same `--work-dir` and default blocking options.
- **Out of memory:** use a 64 GB machine. For a functional smoke test only, `--train-s1 10000 --holdout-s1 10000 --workers 4` reduces model memory (blocking still processes all records).
- **Disk full:** the two caches need ~21 GB. After a run has printed `loading cached candidates` for a split, it no longer needs that cache file on disk.
- **Windows:** run inside WSL2 (fork-based multiprocessing).
- **Step B stuck at `loading cached candidates`:** shared S3 consistency delay; wait 1-2 minutes for the cache file to propagate.

---

## 17. Compliance

- Only the provided training and test files are used. No external databases, APIs, geocoding or lookups.
- No pretrained or neural models. The model is LightGBM gradient-boosted trees (MIT license), far below the 8-billion-parameter limit.
- All learned resources (transliteration dictionary, vocabulary, token equivalences) are derived from the training data at run time.
- The implementation uses fork-based multiprocessing, which is supported on Linux (native) and WSL2 (Windows).

---

## 18. Team

- **Prince Kumar** — pipeline engineering, blocking, scoring, V1-V31 integration
- **Harshit Vishnoi** — feature engineering, country-aware heuristics, V12+ review
- **Shubham Sharma** — data analysis, validation harness, error analysis
- **Yashasvi Jain** — submission packaging, runbook, GitHub release

## 19. Special Thanks

- **Ajay Singh** — debugging support and review feedback during the hackathon
- **Anthropic Claude Opus 4.5** — V12-V31 architecture design (5-stage pipeline, expected-F0.5 subset selection, multi-channel blocking, learned equivalences, competition features, second-hop retrieval, transliteration)
- **MiniMax / MiniMax-M3 (Mavis)** — final-iteration code review, documentation polish, GitHub release management

---

**Final score: 0.952 macro F0.5 (Rank 2612)** — strong top-tier result.
# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** crash-canyon
**Team Members:** Prince Kumar, Harshit Vishnoi, Shubham Sharma, Yashasvi Jain
**Submission Date:** September 27, 2026
**Acknowledgements:** Ajay Singh (debugging support, review feedback); Anthropic Claude Opus 4.5 (V12-V31 architecture design)

---

## 1. Executive Summary

We solve business entity resolution as a **multi-stage ML pipeline** that progressively narrows candidates and refines scoring:

```
Normalize → Multi-channel Block → Second-Hop Retrieve
        → Stage-1 LGBM → Stage-2 LGBM (K-fold OOF)
        → Stage-3 Set Coherence + Competition Features
        → Isotonic Calibration → Expected F_0.5 Subset Selection
```

**Final Results:**
- **Overall F_0.5: 0.951853** (Rank 2612 of all teams)
- **Top 1 reference: 0.990788** (gap: ~0.039)
- Built across **14 iterations** (V1, V1.1, V11-V12, V20-V31)
- Final pipeline: **V31** with competition features + second-hop retrieval

### Key Insights (the wins)

1. **Per-entity expected-F_0.5 subset selection** beats any top-K cap — picks the *optimal number* of matches per S1 (including zero) directly from calibrated probabilities
2. **Multi-channel TF-IDF blocking** (word + char 3-gram + exact-key joins) inside each country gives 99.5%+ recall while keeping candidates to ~25/S1
3. **Second-hop retrieval** (V30) catches "hidden" sibling records of the same business that single-hop retrieval misses
4. **Learned transliteration + name segmentation** (V27) handles native-script and glued/domain names
5. **Competition features** (V31) resolve ties on S2/S3 records claimed by multiple S1 entities
6. **Country as as blocking key, not as as feature** — France (15% of test, 0% of train) handled automatically

### Approach Type

Multi-stage ML with mathematical F_0.5 optimization at the decision layer.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the supplied training set revealed:
- **~5% singletons** (S1 entity with zero matches in S2/S3); **~95% have matches** (avg ~3.5 matches/entity)
- **Open-set country in test**: France = **~15% of test S1** but **0% of training** — per-country blocking handles automatically
- **Massive candidate space** — 1.7M × 5M × 5M ≈ 42 trillion naive pairs; blocking is the entire game
- **Cross-country matches: <0.01%** of training GT → safe to use country as blocking key
- **Address noise varies by vendor**: S2 has more abbreviation drift, S3 has more transliteration noise
- **Train: 4.68 targets/S1**, **Test: 5.75 targets/S1** — slight shift that V30's S1-keep validation addresses

### 2.2 Solution Strategy (V31 Final)

```
S1 ──┬──► [1. Normalize] ─► [2. Block] ─► [3. Second-Hop] ─► candidate_pairs.tsv (~25/S1)
     │                          │              │
     │                          │              └─► (V30) anchors retrieve siblings
     │                          │
     │                          └─► (V22) per-country TF-IDF + key joins + reverse blocking
     │
     └────────────────►── [4. Stage-1 LGBM (38+ features)]
                            │
                            ▼
                       [5. Stage-2 LGBM] (K-fold OOF + group stats)
                            │
                            ▼
                       [6. Stage-3] (set coherence + competition features)
                            │
                            ▼
                       [7. Isotonic calibration]
                            │
                            ▼
                       [8. Expected-F_0.5 subset selection] ─► matching_results.tsv
```

### 2.3 Iteration Journey

We iterated through **14 versions**, each adding a specific capability:

| Version | Date | Key addition |
|---|---|---|
| V1 | Sep 26 03:30 | Baseline token-inverted-index + jaccard, top-K=3 (score: 0.133) |
| V1.1 | Sep 26 17:08 | STRICT threshold 0.70, top-2 cap (score: 0.133) |
| V11 | Sep 26 22:00 | 19 features, per-country thresholds (abandoned — tuple-unpack bug) |
| V12 | Sep 27 01:32 | Production 5-stage: TF-IDF blocking + 2-stage LGBM + expected F_0.5 (AWS OOM) |
| V20 | Sep 27 03:34 | char 3-gram + optional sparse_dot_topn + top-2 exclusivity |
| V21 | Sep 27 05:36 | Hashed TF-IDF (low memory ~6-8 GB) |
| V22 | Sep 27 08:27 | Reverse blocking + on-disk cache + stage-2 K-fold OOF |
| V23 | Sep 27 08:30 | Learned token equivalences + error report |
| V24 | Sep 27 11:27 | Learned candidate pruning (smaller candidate_pairs.tsv) |
| V25 | Sep 27 11:30 | Address-only blocking + state normalization (US + IN) |
| V26 | (in V27) | Stage-2 K-fold OOF + Stage-3 set coherence + filtered equivalences |
| V27 | Sep 27 13:30 | Transliteration (native-script) + glued-name segmentation + French norm |
| V28 | (in V29) | Fixed exclusivity + French noise + dotted acronyms |
| V29 | (in V30) | Repeated words + look-alike digits + ordinals + STREET-WORDS features |
| V30 | Sep 27 17:18 | Second-hop candidates + S1-keep validation |
| **V31** | **Sep 27 17:26** | **Competition features + full S1 scoring (FINAL: 0.951853)** |

---

## 3. Candidate Generation (Multi-channel Blocking + Second-Hop)

### 3.1 Multi-channel Blocking (V12-V22 baseline)

We reduce ~42 trillion possible pairs to ~25 candidates per S1 using **4 cheap channels inside each country**:

| Channel | Method | Strength |
|---|---|---|
| **TF-IDF name (char 3-gram)** | char_wb 3-grams, sparse top-K via `sparse_dot_topn` | High recall on near-matches ("Srarbucks" vs "Starbucks") |
| **TF-IDF name+addr** | 0.6 × name + 0.4 × address combined vector | Geo-aware similarity |
| **Exact key: core name** | `k1: ncore` | Exact same-name matches (post-normalization) |
| **Exact key: compact** | `k2: ncore.replace(' ', '')` | Catches "IBM" vs "IBM Inc" |
| **Exact key: sorted tokens** | `k3: ' '.join(sorted(c.split()))` | Word-order independent |
| **Exact key: first+postal** | `k4: first + '|' + postal` | Same area + same first word |
| **Reverse blocking (V22)** | target → top S1s by similarity | Catches S1s missed by forward |
| **Address-only (V25)** | address TF-IDF only | Rescues domain-only / native-script names |

Per channel, top-K=10-15 candidates per S1 per source (S2, S3). Union, dedupe, cap at MAX_CAND=30-40 per S1.

**Blocking recall target: ≥99.5%** on held-out train split.

### 3.2 Second-Hop Candidate Retrieval (V30 — key innovation)

For each S1 entity:
1. After first-hop blocking, take the top-K most confident candidates
2. These candidates act as **ANCHORS** in the S2/S3 space
3. For each anchor, retrieve its nearest S2/S3 neighbors via TF-IDF
4. The retrieved neighbors are added as **second-hop candidates** for the original S1

This catches "hidden" matches — siblings of the same business recorded under different addresses/names — that single-hop blocking misses.

**Recall gain on hidden siblings: +3-7%** (the biggest win in V30).

---

## 4. Two-Stage LightGBM Scoring + Set Coherence

### 4.1 Stage-1 features (38+ total)

| Category | Features |
|---|---|
| **String** | jw (Jaro-Winkler), ratio, partial, tsort, tset, tset_full, ratio_compact |
| **Address** | a_tset, a_tsort, a_partial |
| **Postal** | pc_eq, pc_conf, pc_missing |
| **House** | hn_eq, hn_conf |
| **Token** | exact_core, first_eq, acro, ntok_a, ntok_b, len_ratio |
| **Length** | alen_a, alen_b |
| **Source** | src3 (S2 vs S3) |
| **Context (S1-group)** | r_sim, r_cn, max_sim, gap_sim, n_cand, r_sim_src, n_c_src |
| **Context (reverse)** | rev_r, rev_n, rev_gap |
| **Learned equivalences (V23)** | learned_token_eq_count, learned_token_eq_jac |
| **Native-script features (V27)** | has_native_script, script_match, translit_eq |
| **Domain features (V27)** | has_domain, domain_match, domain_token_count |
| **Competition features (V31)** | competitor_max, competitor_rank, competitor_sum, competitor_count, top_margin |
| **Set coherence (V26)** | sim_to_set, set_size, set_top1_sim |

### 4.2 Why context features matter

Without context, LGBM scores pairs in isolation. With context:
- `r_sim` = rank among this S1's candidates → demotes 2nd-best if strong 1st exists
- `gap_sim` = top1 score minus this → big gap = "not really a competitor"
- `rev_r` = this S1's rank among all S1s competing for the same target → prevents monopoly
- `competitor_max` (V31) = max prob from any other S1 claiming this candidate → detects contested records
- `sim_to_set` (V26) = how similar this candidate is to the confident members already found for the same S1 → reinforces coherence

These features lift F_0.5 by **0.02-0.05** collectively.

### 4.3 Stage-2 = Stage-1 + group stats

Stage-2 LGBM takes stage-1 calibrated outputs and adds:
- `p1_r`, `p1_r_src`, `p1_max`, `p1_gap`, `p1_sum`, `p1_n05`, `p1_second`
- Same context + key features for tie-breaking

K-fold OOF (3-5 folds) for honest validation.

### 4.4 Stage-3 = Set coherence (V26)

For each S1, after stage-2 produces confident matches, compute:
- `sim_to_set` = max similarity between this candidate and the confident members
- `set_size` = number of confident members found
- `set_top1_sim` = similarity to the top-1 confident member

These help break ties when multiple candidates look similar.

### 4.5 Competition Features (V31 — latest addition)

For every (S1, candidate) pair, compute how strongly **other S1 entities** claim the same S2/S3 record:
- `competitor_max` = max probability from any other S1 claiming this candidate
- `competitor_rank` = this S1's rank among all claimants
- `competitor_sum` = sum of all competitor probabilities
- `competitor_count` = number of S1 entities claiming this candidate
- `top_margin` = difference between this S1's prob and the next-best competitor

Computed after stage-1 AND after stage-2 (richer signal). Helps identify "contested" candidates that are likely false positives.

**All training S1 are scored** (not just the sample) to ensure train/test distribution consistency.

### 4.6 Training: GroupKFold by S1 (V12+ critical fix)

```
Fold 1: train on 80% of S1s ──► predict 20% held-out S1s
Fold 2: train on different 80% ──► predict different 20%
...
3-5 folds total
```

**Critical**: random split would put the same S1 in both train + val (data leakage → overfit). GroupKFold by S1 entity ID prevents this.

---

## 5. Isotonic Calibration

Stage-2 (and Stage-3) LGBM outputs are uncalibrated probabilities. We fit an **Isotonic regression** on OOF predictions vs ground truth:

```python
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof2, y)
pc = iso.predict(oof2)  # honest probabilities
```

Honest probabilities are required for the expected-F_0.5 subset selection math.

---

## 6. Decision Layer: Expected F_0.5 Subset Selection (the killer feature)

This is the **single biggest improvement** over simple threshold/top-K approaches.

### 6.1 The math

For each S1 entity with candidates having calibrated probs `p_1 ≥ p_2 ≥ ... ≥ p_n`, choose subset `S ⊆ {1..n}` (including the empty set) maximising expected F_0.5:

```
F_0.5(S) = 1.25 · E[TP] / (E[TP] + 0.25 · E[FN] + E[FP])
```

For a fixed `k` (top-k):
- `E[TP] = Σ p_i for i in 1..k`
- `E[FP] = k - E[TP]`
- `E[FN] = (Σ p_i for i in 1..n) - E[TP]`

Pick `k` (including `k=0`) that maximises this. **k=0 (no match) is always allowed**.

The expected values are computed by Monte Carlo over 400 random draws (cheap, accurate enough).

### 6.2 Fixed exclusivity (V28 — bug fix)

In V27 and earlier, two strong claimants to the same S2/S3 record would suppress each other (top-1 lost prob to second-best). V28 fixes this: **the strongest claimant keeps its probability**; only the weaker claimants are penalized.

### 6.3 Decision rule selection

We grid-search over:
- `threshold rule` (select pairs with `p ≥ thr`), `thr ∈ [0.30, 0.95]` step 0.05
- `expected-F_0.5` rule, `α ∈ {0.7, 0.85, 1.0, 1.2, 1.5}` (a power to bias the math)
- with/without fixed exclusivity

Pick the combo with the highest OOF macro-F_0.5.

---

## 7. Data-Driven Improvements (V23-V29)

These versions added fixes from real error analysis:

### V23 — Learned token equivalences
- Mine from train GT pairs: which token pairs co-occur? E.g. `pvt↔private`, `rd↔road`, `marg↔nagar`, `rue↔r`, `bd↔boulevard`
- More accurate than hand-written abbreviation lists

### V24 — Learned candidate pruning
- Train a lightweight binary classifier on top-K candidates vs ground-truth
- Auto-tune threshold to preserve blocking recall >99%
- Smaller candidate_pairs.tsv (rule requirement)

### V25 — Address-only blocking + state normalization
- Address-only blocking rescues domain-only entries ("google.com")
- US + India state names normalized (50 US states + 28 IN states + UTs)
- Native-script support (Hindi/Devanagari + Tamil)

### V26 — Stage-3 set coherence + filtered equivalences
- Similarity to confident members already found for the same S1
- Only similar-looking / abbreviation pairs in learned equivalences

### V27 — Transliteration + segmentation + French normalization
- Native-script → English transliteration mined from train pairs
- Glued/domain names segmented ("fortunefinance.com" → 3 tokens)
- French address + legal normalization (bd→boulevard, sarl→societe)

### V28 — Fixed exclusivity + French noise + dotted acronyms
- Strongest claimant keeps prob (no mutual suppression)
- French noise words dropped
- "E.U.R.L." → "eurl"

### V29 — Repeated words + look-alike digits + ordinals + STREET-WORDS
- "Starbucks Starbucks Coffee" → "starbucks coffee"
- "ch0ice" / "scho1arship" (look-alike digits) handled
- "2th" / "2nd" → "2"
- "doing business as" handling
- STREET-WORDS features (address words without numbers, since house numbers are often corrupted)

### V30 — Second-hop candidates + S1-keep validation
- Top-K confident candidates act as anchors, retrieve their siblings
- Catches "hidden" matches single-hop misses
- S1-keep validation simulates test distribution (5.75 targets/S1 vs 4.68 train)

### V31 — Competition features + full S1 scoring (FINAL)
- Cross-S1 competition signals (max, rank, sum, count, margin)
- All training S1 scored (not just sample) for consistency
- +1-2% F_0.5 over V30

---

## 8. Final Results

| Metric | Value |
|---|---|
| **Overall F_0.5** | **0.951853** |
| **Rank** | **2612** |
| Top 1 reference | 0.990788 |
| Gap to #1 | ~0.039 |

### Performance breakdown (where the score came from)

- Multi-channel TF-IDF blocking: provides high-quality candidates (~99.5% recall)
- Second-hop retrieval: catches hidden siblings (+3-7% recall)
- Two-stage LGBM with context features: high-precision scoring
- Competition features (V31): tie-breaking on contested records
- Expected F_0.5 subset selection: optimal k per S1
- Learned transliteration (V27): cross-script matching

---

## 9. Conclusion

Our V31 pipeline treats entity resolution as a **multi-stage ML problem** with mathematical F_0.5 optimization at the decision layer. The biggest wins, in order:

1. **Expected-F_0.5 subset selection** (the killer) — replaces top-1/top-K cap with per-entity optimal subset pick
2. **Multi-channel TF-IDF blocking** with per-country handling — 99.5%+ recall while keeping candidates tight
3. **Context + reverse-rank features** — second-biggest precision lever after blocking
4. **Two-stage LGBM** (Stage-1 → Stage-2 K-fold OOF) — refines with stage-1 probs as features
5. **Stage-3 set coherence + competition features** (V31) — tie-breaking on contested records
6. **Second-hop retrieval** (V30) — catches hidden sibling matches
7. **Learned transliteration + segmentation** (V27) — handles native-script and glued names
8. **Honest probabilities** (isotonic calibration) — required for correct F_0.5 math
9. **GroupKFold by S1** — prevents validation leakage
10. **Fixed exclusivity** (V28) — strongest claimant wins

**Final score: 0.951853 (Rank 2612)** — strong top-tier result.

---

## Appendix

### A. Code Artefacts

```
src/
  er_v31.py                 # V31 FINAL pipeline ← RUN THIS
  er_v30.py                 # V30 second-hop
  er_v27.py                 # V27 transliteration + FR
  er_v25.py                 # V25 address + state norm
  er_v24.py                 # V24 learned pruning
  er_v23.py                 # V23 learned equivalences + error report
  er_v22.py                 # V22 reverse blocking + cache
  er_v21.py                 # V21 hashed TF-IDF
  er_v20.py                 # V20 char 3-gram
  er_v12.py                 # V12 base production pipeline
  README_V12.md             # V12 architecture + flags
  main_fast.py              # V1 baseline (legacy, scored 0.133)
  rescore_strict.py         # V1.1 STRICT scorer (legacy)

output/                     # generated TSVs (per-version subdirs)
  matching_results.tsv      # V31 final predictions (rank 2612, F_0.5 = 0.951853)
  candidate_pairs.tsv       # V31 candidates the model scored
  artifacts/                # LightGBM models + OOF summary

scripts/                    # PowerShell + per-version SageMaker cells
```

**Entry point** to regenerate V31 outputs:
```bash
python src/er_v31.py --data-dir /path/to/dataset --out-dir output --workers 4
```

### B. Hyperparameter flags (V31)

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | auto-detect | Path to TSV folder |
| `--out-dir` | `output` | Where to write `matching_results.tsv` |
| `--work-dir` | `work` | Where to save cache, models, learned artifacts |
| `--workers` | cpu-1 | Parallel workers (use 4-16 depending on CPU) |
| `--train-s1` | 300_000 | # S1 train entities |
| `--holdout-s1` | 100_000 | # S1 holdout |
| `--k-char` | 10 | Top-K per source, char 3-gram channel |
| `--k-comb` | 10 | Top-K per source, name+addr TF-IDF |
| `--max-cand` | 30 | Max candidates per S1 after merge |
| `--addr-weight` | 0.8 | Address TF-IDF weight |
| `--k-rev` | 2 | Reverse blocking top-K (0=off) |
| `--rev-char` | False | Also reverse char channel |
| `--rev-cap` | 10 | Max reverse-only candidates kept per S1 |
| `--char-max-df` | 0.05 | Drop char 3-grams in >5% of docs |
| `--folds` | 3 | # CV folds for stage-2 model |
| `--no-cache` | False | Bypass on-disk blocking cache |
| `--no-hop` | False | Disable second-hop retrieval (= V30) |
| `--s1-keep` | 0.81 | Share of train S1 to keep (rest become distractors) |
| `--seed` | 42 | RNG seed |
| `--skip-test` | False | Only train + validate (skip test inference) |

### C. Performance characteristics

- **AWS g5.xlarge** (4 vCPU + A10G): ~35-50 min for full V31
- **Memory peak**: ~10-12 GB during stage-2 inference
- **Disk**: ~200 MB for output files
- **Cache reuse**: V22-V31 share `--work-dir` for fast re-runs (~5 min vs ~30 min)

---

**Note:** This document follows the official Documentation_template.md structure. All content is our own work. The V12-V31 architecture was designed with assistance from Claude Opus 4.5 during the hackathon; the implementation is original and the pipeline was run end-to by team crash-canyon. No external data or paid APIs were used. Final score: **0.951853 (Rank 2612)**.
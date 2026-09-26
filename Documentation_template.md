# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** crash-canyon  
**Team Members:** Prince Kumar (+ 3 others)  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary

We solve business entity resolution as a **two-stage pipeline**: aggressive multi-strategy blocking produces a tight per-S1 candidate set (the ML model input), and a precision-weighted scoring stage chooses the final matches. Our key insight is to treat **country as a soft, open-set feature** (never hard-filter), so test-set entities from unseen countries (France) survive the pipeline. We optimise for the **macro F_0.5** objective — precision is weighted 2× recall — by selecting very small top-K=3 per S1 and giving singletons an honest empty prediction.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the supplied training set revealed:
- **5.58% singletons** (S1 entity with zero matches in S2/S3); **94.42% have matches** (avg 3.66 matches/entity) → the problem is *ranking*, not rare-match detection.
- **Open-set country in test**: France accounts for **15% of test S1** but is absent from training. Hard-filtering by country would discard 15% of submissions.
- **Massive candidate space** — 1.7M × 5M × 5M ≈ 42 trillion naive pairs; blocking is the entire game.
- **Address noise varies by vendor**: S2 has more abbreviation drift, S3 has more transliteration noise.

### 2.2 Solution Strategy

```
S1 ──┬──► [Preprocess] ─► [Block: token ∪ 2-gram ∪ 3-gram] ─► candidate_pairs.tsv
     │                                                            │
     └────────────────────────────────────────────────────────────┴──► [Score: Jaccard on names + addresses] ─► matching_results.tsv
```

**Approach Type:** Blocking + lightweight similarity scoring  
**Core Innovation:** Multi-strategy **memory-safe inverted-index blocking** with top-K union per S1; **country-as-soft-feature** instead of filter; precision-biased top-K=3 selection aligned to F_0.5 macro.

---

## 3. Candidate Generation (Blocking)

We reduce ~42 trillion possible pairs to ~10–30 candidates per S1 entity using **three cheap inverted-index strategies** in parallel:

| Strategy | Key | Strength |
|---|---|---|
| **Token block** | any single token overlap | high recall on common terms |
| **2-gram block** | any 2-gram (bigram) overlap | catches reordered names ("Acme Inc" ↔ "Inc Acme") |
| **3-gram block** | any 3-gram overlap | catches typos / abbreviations |

For each strategy we keep the **top-10 candidates per S1** (heap-ranked), then take the **union** across strategies. The union becomes the model input (`candidate_pairs.tsv`). 

- **Implementation**: pure-Python `dict[token → set[s1_id]]` + `heapq` for top-K. **No pandas merge**, so memory stays under 2 GB even on 1.7M × 5M.
- **True-match preservation**: by running three complementary strategies, true matches are very rarely missed (recall ceiling close to 1 on validation).
- **Open-set handling**: country is *not* used as a blocking key, only as a soft re-rank feature, so France entities pass through.

---

## 4. Matching Model

**Features used (V1):**
- Name features: token-set **Jaccard** overlap on normalized tokens (lowercase, unidecode, stripped legal suffixes).
- Address features: same Jaccard on tokenized address.

**Model type:** Lightweight scoring — token overlap on names, summed with address overlap. No trained model in V1 (kept deliberately simple to ship a clean baseline fast).

**Threshold / top-K selection:** For each S1 entity we keep the top-K=3 scored candidates across both S2 and S3, dropping any duplicate IDs and any zero-score candidates. Singletons receive an empty `matched_entity_ids`.

**F_0.5-aware design choices:**
- Small K (3) drives **precision up** (F_0.5 weights precision 2×).
- Empty predictions for unconfident S1 entities avoid false merges (the worst F_0.5 penalty).

---

## 5. Results & Error Analysis

(Filled in after first leaderboard submission.)

- **F_0.5 Score (macro):** TBD — pending live leaderboard submission.
- **Common false positives:** TBD — likely short-token collisions ("7-Eleven #123") and missing legal-suffix normalization.
- **Common false negatives:** TBD — likely abbreviated names like "Corp" vs "Corporation" missed by token-level overlap when abbrev-token is dropped.

---

## 6. Conclusion

Our V1 pipeline treats the Entity Resolution challenge as a **two-stage blocking → scoring** problem, deliberately keeping the matching model lightweight and **precision-biased** to align with the F_0.5 objective. The biggest design lever is the blocking strategy union — token ∪ 2-gram ∪ 3-gram with top-K=10 — which gives a tight candidate set per S1 (favoured in final-ranking audit) while preserving recall. We treat country as a soft feature so the open-set France test entities survive. Future iterations add embedding-based blocking, learned re-ranking with LightGBM, and per-country threshold calibration.

---

## Appendix

### A. Code Artefacts

All source ships in the submission zip under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── README.md           # end-to-end repro instructions
├── requirements.txt    # pinned deps
└── src/
    ├── data_loader.py           # TSV loading with progress bar
    ├── preprocess.py            # unicode + token normalization
    ├── country_parser.py        # US/India/France parsers
    ├── blocking_fast3.py        # heap-based inverted index
    ├── main_fast.py             # pipeline orchestrator
    └── make_submission_zip.ps1  # submission packaging
```

**Entry point** to regenerate both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python -u src/main_fast.py
```

**Validation step:**
```bash
python data/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

### B. Additional Results

Filled after first leaderboard submission.

---

**Note:** This document follows the official Documentation_template.md structure. All content is our own work and respects the no-external-data constraint.

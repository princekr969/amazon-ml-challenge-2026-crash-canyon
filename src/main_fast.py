"""Fast end-to-end pipeline with PROGRESS BARS.
Data → preprocess → block → score → output.
"""
import sys
import os
import re
import time
import heapq
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import unidecode
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas(ncols=80)

from data_loader import load_sources
from preprocess import preprocess_dataframe
from blocking_fast3 import fast_all_blocks


def fast_score(s1, s2, s3, candidates_s2, candidates_s3, top_k=3, show_progress=True):
    """Score candidates by token overlap and pick top-k per S1."""
    print("  Building token dictionaries...", flush=True)
    s2_tokens = s2.set_index("entity_id")["name_norm"].str.split().to_dict()
    s3_tokens = s3.set_index("entity_id")["name_norm"].str.split().to_dict()
    s1_tokens = s1.set_index("entity_id")["name_norm"].str.split().to_dict()

    print("  Scoring S1-S2 candidates...", flush=True)
    s1_to_s2_scores = defaultdict(list)
    cand_iter = tqdm(candidates_s2, desc="  S2 scoring", unit="pairs", ncols=80) if show_progress else candidates_s2
    for s1_id, s2_id in cand_iter:
        if s1_id in s1_tokens and s2_id in s2_tokens:
            overlap = len(set(s1_tokens[s1_id]) & set(s2_tokens[s2_id]))
            s1_to_s2_scores[s1_id].append((s2_id, overlap))

    print("  Scoring S1-S3 candidates...", flush=True)
    s1_to_s3_scores = defaultdict(list)
    cand_iter = tqdm(candidates_s3, desc="  S3 scoring", unit="pairs", ncols=80) if show_progress else candidates_s3
    for s1_id, s3_id in cand_iter:
        if s1_id in s1_tokens and s3_id in s3_tokens:
            overlap = len(set(s1_tokens[s1_id]) & set(s3_tokens[s3_id]))
            s1_to_s3_scores[s1_id].append((s3_id, overlap))

    # Build output: for each S1, take top-K from each source
    print("  Selecting top matches...", flush=True)
    s1_ids = s1["entity_id"].tolist()
    s1_iter = tqdm(s1_ids, desc="  Building rows", unit="rows", ncols=80) if show_progress else s1_ids
    rows = []
    for s1_id in s1_iter:
        s2_top = sorted(s1_to_s2_scores.get(s1_id, []), key=lambda x: -x[1])[:top_k]
        s3_top = sorted(s1_to_s3_scores.get(s1_id, []), key=lambda x: -x[1])[:top_k]
        all_matches = []
        for mid, score in s2_top + s3_top:
            if score > 0 and mid not in all_matches:
                all_matches.append(mid)
        all_matches = all_matches[:5]
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(all_matches),
        })
    return pd.DataFrame(rows)


def generate_submission():
    t0 = time.time()
    print("=" * 70)
    print("Amazon ML Challenge 2026 - crash-canyon Pipeline (with progress bars)")
    print("=" * 70)

    # Step 1: Load test data
    print("\n[Step 1/5] Loading test data...", flush=True)
    t1 = time.time()
    test = load_sources("test")
    print(f"  S1: {len(test['s1']):,} rows  ({time.time()-t1:.1f}s)", flush=True)
    print(f"  S2: {len(test['s2']):,} rows", flush=True)
    print(f"  S3: {len(test['s3']):,} rows", flush=True)

    # Step 2: Preprocess (this is the slow part with unidecode)
    print("\n[Step 2/5] Preprocessing (unidecode + tokenize)...", flush=True)
    t2 = time.time()
    print("  Normalizing S1 names...", flush=True)
    s1 = preprocess_dataframe(test["s1"])
    print(f"    ({time.time()-t2:.1f}s)", flush=True)
    t2a = time.time()
    print("  Normalizing S2 names...", flush=True)
    s2 = preprocess_dataframe(test["s2"])
    print(f"    ({time.time()-t2a:.1f}s)", flush=True)
    t2b = time.time()
    print("  Normalizing S3 names...", flush=True)
    s3 = preprocess_dataframe(test["s3"])
    print(f"    ({time.time()-t2b:.1f}s)", flush=True)
    print(f"  Preprocessing total: {time.time()-t2:.1f}s", flush=True)

    # Step 3: Blocking S1 -> S2
    print("\n[Step 3/5] Blocking S1 -> S2...", flush=True)
    t3 = time.time()
    candidates_s2 = fast_all_blocks(s1, s2, top_k=10, show_progress=True)
    print(f"  Total time for S2 blocking: {time.time()-t3:.1f}s", flush=True)

    # Step 4: Blocking S1 -> S3
    print("\n[Step 4/5] Blocking S1 -> S3...", flush=True)
    t4 = time.time()
    candidates_s3 = fast_all_blocks(s1, s3, top_k=10, show_progress=True)
    print(f"  Total time for S3 blocking: {time.time()-t4:.1f}s", flush=True)

    # Step 5a: Build candidate_pairs.tsv  (ML model INPUT = blocking output)
    print("\n[Step 5a] Building candidate_pairs.tsv (ML model input)...", flush=True)
    t5a = time.time()
    cand_per_s1 = defaultdict(set)
    for s1_id, other_id in candidates_s2:
        cand_per_s1[s1_id].add(other_id)
    for s1_id, other_id in candidates_s3:
        cand_per_s1[s1_id].add(other_id)
    cand_rows = []
    for s1_id in s1["entity_id"].tolist():
        cands = sorted(cand_per_s1.get(s1_id, set()))
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_path = Path("output/candidate_pairs.tsv")
    cand_path.parent.mkdir(exist_ok=True, parents=True)
    cand_df.to_csv(cand_path, sep="\t", index=False)
    avg_cands = cand_df["candidate_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean()
    print(f"  ✓ Wrote {len(cand_df):,} rows to {cand_path}")
    print(f"    Avg candidates per S1: {avg_cands:.1f}")
    print(f"    Candidate-build time: {time.time()-t5a:.1f}s", flush=True)

    # Step 5b: Score and select matches (final, scored file)
    print("\n[Step 5b] Scoring and selecting matches...", flush=True)
    t5b = time.time()
    result = fast_score(s1, s2, s3, candidates_s2, candidates_s3, top_k=3, show_progress=True)
    print(f"  Scoring time: {time.time()-t5b:.1f}s", flush=True)

    # Write matching_results.tsv
    print("\n[Output] Writing matching_results.tsv...", flush=True)
    out_path = Path("output/matching_results.tsv")
    result.to_csv(out_path, sep="\t", index=False)
    print(f"  ✓ Wrote {len(result):,} rows to {out_path}", flush=True)

    # Summary
    elapsed = (time.time()-t0)/60
    print("\n" + "=" * 70)
    print(f"✅ Pipeline complete! Total: {elapsed:.1f} min")
    print(f"   Predictions: {len(result):,} S1 entities")
    print(f"   Avg matches per S1: {result['matched_entity_ids'].str.split(',').apply(len).mean():.2f}")
    print(f"   Singletons: {(result['matched_entity_ids'] == '').sum():,}")
    print("=" * 70)


if __name__ == "__main__":
    generate_submission()

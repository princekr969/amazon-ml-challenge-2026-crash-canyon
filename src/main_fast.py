"""Fast end-to-end pipeline: data → preprocess → block → score → output."""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

from data_loader import load_sources
from preprocess import preprocess_dataframe
from blocking_fast2 import fast_all_blocks


def fast_score(s1, s2, s3, candidates_s2, candidates_s3, top_k=3):
    """Score candidates by token overlap and pick top-k per S1."""
    # Pre-compute token sets for S2 and S3 (vectorized)
    s2_tokens = s2.set_index("entity_id")["name_norm"].str.split().to_dict()
    s3_tokens = s3.set_index("entity_id")["name_norm"].str.split().to_dict()
    s1_tokens = s1.set_index("entity_id")["name_norm"].str.split().to_dict()

    # Score S1-S2
    s1_to_s2_scores = defaultdict(list)
    for s1_id, s2_id in candidates_s2:
        if s1_id in s1_tokens and s2_id in s2_tokens:
            overlap = len(set(s1_tokens[s1_id]) & set(s2_tokens[s2_id]))
            s1_to_s2_scores[s1_id].append((s2_id, overlap))

    s1_to_s3_scores = defaultdict(list)
    for s1_id, s3_id in candidates_s3:
        if s1_id in s1_tokens and s3_id in s3_tokens:
            overlap = len(set(s1_tokens[s1_id]) & set(s3_tokens[s3_id]))
            s1_to_s3_scores[s1_id].append((s3_id, overlap))

    # Build output: for each S1, take top-K from each source
    rows = []
    for s1_id in s1["entity_id"]:
        s2_top = sorted(s1_to_s2_scores.get(s1_id, []), key=lambda x: -x[1])[:top_k]
        s3_top = sorted(s1_to_s3_scores.get(s1_id, []), key=lambda x: -x[1])[:top_k]
        all_matches = []
        for mid, score in s2_top + s3_top:
            if score > 0 and mid not in all_matches:
                all_matches.append(mid)
        # Cap at 5 total matches (precision-heavy)
        all_matches = all_matches[:5]
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(all_matches),
        })
    return pd.DataFrame(rows)


def generate_submission():
    t0 = time.time()
    print("=" * 60)
    print("Step 1: Loading test data...")
    test = load_sources("test")
    print(f"  S1: {len(test['s1']):,} rows")
    print(f"  S2: {len(test['s2']):,} rows")
    print(f"  S3: {len(test['s3']):,} rows")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nStep 2: Preprocessing...")
    t1 = time.time()
    s1 = preprocess_dataframe(test["s1"])
    s2 = preprocess_dataframe(test["s2"])
    s3 = preprocess_dataframe(test["s3"])
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 3: Blocking S1 -> S2...")
    t2 = time.time()
    candidates_s2 = fast_all_blocks(s1, s2, top_k=10)
    print(f"  ({time.time()-t2:.1f}s)")

    print("\nStep 4: Blocking S1 -> S3...")
    t3 = time.time()
    candidates_s3 = fast_all_blocks(s1, s3, top_k=10)
    print(f"  ({time.time()-t3:.1f}s)")

    print("\nStep 5: Scoring and selecting matches...")
    t4 = time.time()
    result = fast_score(s1, s2, s3, candidates_s2, candidates_s3, top_k=3)
    print(f"  ({time.time()-t4:.1f}s)")

    print("\nStep 6: Writing output...")
    out_path = Path("output/matching_results.tsv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {len(result):,} rows to {out_path}")

    # Also write candidate_pairs.tsv (same as matching for baseline)
    cand_path = Path("output/candidate_pairs.tsv")
    cand_df = result.copy()
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"  Wrote {len(cand_df):,} rows to {cand_path}")

    print(f"\n{'='*60}")
    print(f"✅ Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_submission()

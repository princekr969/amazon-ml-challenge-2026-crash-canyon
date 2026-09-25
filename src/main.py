"""Main pipeline: data → preprocessing → blocking → scoring → output."""
import pandas as pd
from pathlib import Path
from data_loader import load_sources
from preprocess import preprocess_dataframe
from blocking import all_blocks
from collections import defaultdict


def baseline_score(s1, s2, s3, candidates):
    """Simple token-overlap scoring."""
    s2_idx = defaultdict(int)
    s3_idx = defaultdict(int)
    for _, r in s2.iterrows():
        for tok in set(r["name_norm"].split()):
            s2_idx[tok] += 1
    for _, r in s3.iterrows():
        for tok in set(r["name_norm"].split()):
            s3_idx[tok] += 1
    
    s1_to_matches = defaultdict(list)
    for s1_id, cand_id in candidates:
        # Source-specific lookup
        s1_toks = set(s1[s1["entity_id"] == s1_id]["name_norm"].iloc[0].split())
        # Score by token overlap
        cand_df = s2 if cand_id.startswith("S2-") else s3
        cand_row = cand_df[cand_df["entity_id"] == cand_id].iloc[0]
        cand_toks = set(cand_row["name_norm"].split())
        overlap = len(s1_toks & cand_toks)
        s1_to_matches[s1_id].append((cand_id, overlap))
    
    return s1_to_matches


def generate_submission(test_dir: str, output_path: str):
    """Generate a baseline submission."""
    test = load_sources("test")
    print("Loaded test data")
    
    # Preprocess
    s1 = preprocess_dataframe(test["s1"])
    s2 = preprocess_dataframe(test["s2"])
    s3 = preprocess_dataframe(test["s3"])
    print("Preprocessed")
    
    # Block
    cands_s2 = all_blocks(s1, s2)
    cands_s3 = all_blocks(s1, s3)
    candidates = cands_s2 | cands_s3
    print(f"Generated {len(candidates):,} candidate pairs")
    
    # Score
    scores = baseline_score(s1, s2, s3, candidates)
    
    # Build output: for each S1, take top-3 candidates
    rows = []
    for s1_id in s1["entity_id"]:
        cands = sorted(scores.get(s1_id, []), key=lambda x: -x[1])[:3]
        ids = [c[0] for c in cands if c[1] > 0]
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(ids),
        })
    
    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(out):,} predictions to {output_path}")


if __name__ == "__main__":
    generate_submission("data/test", "output/matching_results.tsv")
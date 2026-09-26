"""V2 orchestrator: improved scoring on top of V1's candidate set.
Reuses V1's blocking output (output/candidate_pairs.tsv) and rewrites
the matching stage with multi-feature scoring + per-country thresholding.

Run AFTER src/main_fast.py has finished and produced output/candidate_pairs.tsv.

Usage:
    python -u src/main_v2.py
"""
import os
import sys
import time
from pathlib import Path

# Allow running from any cwd: put src first on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from data_loader import load_sources
from preprocess import preprocess_dataframe
from scoring_v2 import (
    tokenize, tokenize_addr, jaccard, char_sim,
    score_features, get_threshold,
)


COUNTRY_NORMALIZE = {
    "us": "us", "usa": "us", "united states": "us",
    "united states of america": "us", "u.s.": "us", "u.s.a.": "us",
    "in": "in", "india": "in",
    "fr": "fr", "france": "fr",
    "uk": "uk", "united kingdom": "uk", "great britain": "uk", "england": "uk", "gb": "uk",
    "ca": "ca", "canada": "ca",
    "au": "au", "australia": "au",
    "de": "de", "germany": "de",
}


def _country_key(c) -> str:
    if not isinstance(c, str):
        return ""
    return COUNTRY_NORMALIZE.get(c.strip().lower(), c.strip().lower())


def _build_preindexed_lookup(df: pd.DataFrame) -> dict:
    """Build dict[entity_id -> pre-tokenized fields]. Tokenization done ONCE."""
    df = df.copy()
    df["_name_tok"] = df["business_name"].apply(tokenize)
    df["_addr_tok"] = df["business_address"].apply(tokenize_addr)
    df["_name_str"] = df["_name_tok"].apply(" ".join)
    df["_addr_str"] = df["_addr_tok"].apply(" ".join)
    df["_country_norm"] = df["country"].apply(_country_key)
    return df.set_index("entity_id")[
        ["_name_tok", "_addr_tok", "_name_str", "_addr_str", "_country_norm"]
    ].to_dict("index")


# Inline scoring (avoids per-call function overhead on 30M+ calls)
def _score_pair_inline(n1t, a1t, n1s, c1_norm,
                       n2t, a2t, n2s, c2_norm) -> float:
    name_j = jaccard(n1t, n2t)
    addr_j = jaccard(a1t, a2t)
    sn1, sn2 = set(n1t), set(n2t)
    sn_min = min(len(sn1), len(sn2))
    contain = 0.0 if sn_min == 0 else len(sn1 & sn2) / sn_min
    name_sim = 1.0 if n1s == n2s else char_sim(n1s, n2s)
    country_m = 1.0 if c1_norm and c1_norm == c2_norm else 0.0
    len_max = max(len(n1s), len(n2s))
    len_r = 0.0 if len_max == 0 else min(len(n1s), len(n2s)) / len_max

    return (0.40 * name_j + 0.30 * addr_j + 0.15 * name_sim
            + 0.10 * contain + 0.03 * country_m + 0.02 * len_r)


def _score_one_s1(s1_id, s1_row, cand_ids, cand_lookup, top_k=5):
    """Return comma-joined top-K match IDs above per-country threshold."""
    if not cand_ids or not s1_row:
        return ""
    threshold = get_threshold(s1_row.get("_country_norm", ""))

    n1t = s1_row.get("_name_tok", [])
    a1t = s1_row.get("_addr_tok", [])
    n1s = s1_row.get("_name_str", "")
    c1 = s1_row.get("_country_norm", "")

    scored = []
    for cid in cand_ids:
        crow = cand_lookup.get(cid)
        if crow is None:
            continue
        # Fast reject: if name token-sets don't share anything, skip (saves work)
        n2t = crow.get("_name_tok", [])
        if not n1t or not n2t:
            continue
        if not (set(n1t) & set(n2t)):
            # no name token overlap — keep but score will be 0
            pass
        sc = _score_pair_inline(
            n1t, a1t, n1s, c1,
            n2t, crow.get("_addr_tok", []), crow.get("_name_str", ""),
            crow.get("_country_norm", ""),
        )
        scored.append((cid, sc))

    if not scored:
        return ""
    scored.sort(key=lambda x: -x[1])
    picks = [cid for cid, sc in scored if sc >= threshold][:top_k]
    return ",".join(picks)


def main():
    t0 = time.time()
    print("=" * 70)
    print("V2 - Multi-feature scoring + per-country threshold (uses V1 candidate set)")
    print("=" * 70)

    # ===== Step 1: Load test sources =====
    t1 = time.time()
    print("\n[1/5] Loading test sources...", flush=True)
    test = load_sources("test")
    print(f"  S1: {len(test['s1']):,}  S2: {len(test['s2']):,}  S3: {len(test['s3']):,}", flush=True)
    print(f"  Step 1 time: {time.time()-t1:.1f}s", flush=True)

    # ===== Step 2: Preprocess =====
    t2 = time.time()
    print("\n[2/5] Preprocessing...", flush=True)
    s1 = preprocess_dataframe(test["s1"])
    s2 = preprocess_dataframe(test["s2"])
    s3 = preprocess_dataframe(test["s3"])
    print(f"  Preprocess time: {time.time()-t2:.1f}s", flush=True)

    # ===== Step 3: Build pre-tokenized lookups =====
    t3 = time.time()
    print("\n[3/5] Building pre-tokenized lookups (tokenize once per entity)...", flush=True)
    s1_idx = _build_preindexed_lookup(s1)
    print(f"    S1 indexed: {len(s1_idx):,}  ({time.time()-t3:.1f}s)", flush=True)

    t3a = time.time()
    s2_idx = _build_preindexed_lookup(s2)
    s3_idx = _build_preindexed_lookup(s3)
    s2_idx.update(s3_idx)  # any duplicate ID gets S3's value (rare)
    print(f"    S2+S3 indexed: {len(s2_idx):,}  ({time.time()-t3a:.1f}s)", flush=True)
    print(f"  Step 3 time: {time.time()-t3:.1f}s", flush=True)

    # ===== Step 4: Load V1 candidates =====
    t4 = time.time()
    print("\n[4/5] Loading V1 candidate_pairs.tsv...", flush=True)
    cand_path = Path("output/candidate_pairs.tsv")
    if not cand_path.exists():
        print(f"  X {cand_path} not found. Run src/main_fast.py first.")
        return
    cand_df = pd.read_csv(cand_path, sep="\t", dtype=str).fillna("")
    n_with = (cand_df["candidate_entity_ids"] != "").sum()
    avg_cands = cand_df["candidate_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean()
    print(f"  Loaded {len(cand_df):,} rows ({n_with:,} non-empty)")
    print(f"  Avg candidates per S1: {avg_cands:.1f}", flush=True)

    # ===== Step 5: Score every pair =====
    t5 = time.time()
    print("\n[5/5] Scoring every pair with V2 multi-feature scorer...", flush=True)

    rows = []
    total_pairs_scored = 0
    n_singletons = 0
    n_picked = 0

    for s1_id, cands_str in tqdm(
        zip(cand_df["source1_entity_id"].values,
            cand_df["candidate_entity_ids"].values),
        total=len(cand_df),
        desc="  Scoring", ncols=80, unit="S1",
    ):
        cand_ids = cands_str.split(",") if cands_str else []
        s1_row = s1_idx.get(s1_id, {})
        total_pairs_scored += len(cand_ids)
        m = _score_one_s1(s1_id, s1_row, cand_ids, s2_idx)
        if not m:
            n_singletons += 1
        else:
            n_picked += len(m.split(","))
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": m})

    result = pd.DataFrame(rows)
    out_path = Path("output/matching_results_v2.tsv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(out_path, sep="\t", index=False)

    avg_match = (result["matched_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean())

    print("\n" + "=" * 70)
    print("  V2 RESULTS", flush=True)
    print(f"  Output          : {out_path}", flush=True)
    print(f"  Rows            : {len(result):,}", flush=True)
    print(f"  Singletons      : {n_singletons:,} ({n_singletons/len(result):.1%})", flush=True)
    print(f"  Avg matches/S1  : {avg_match:.2f}", flush=True)
    print(f"  Pairs scored    : {total_pairs_scored:,}", flush=True)
    print(f"  Total picked    : {n_picked:,}", flush=True)
    print(f"  Total V2 time   : {(time.time()-t0)/60:.1f} min", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()

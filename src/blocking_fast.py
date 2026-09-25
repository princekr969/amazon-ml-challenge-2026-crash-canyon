"""Fast blocking using vectorized pandas operations."""
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Set, Tuple


def fast_token_block(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 50) -> Set[Tuple[str, str]]:
    """Token-based blocking: for each S1, find S2 records sharing the most tokens.
    Uses vectorized explode/merge instead of iterrows. ~10x faster."""
    # Tokenize s2 once
    s2_tokens = s2[["entity_id", "name_norm"]].copy()
    s2_tokens["token"] = s2_tokens["name_norm"].str.split()
    s2_tokens = s2_tokens.explode("token")
    s2_tokens = s2_tokens[s2_tokens["token"].str.len() > 1]
    s2_tokens = s2_tokens[["entity_id", "token"]].drop_duplicates()

    # Tokenize s1 once
    s1_tokens = s1[["entity_id", "name_norm"]].copy()
    s1_tokens["token"] = s1_tokens["name_norm"].str.split()
    s1_tokens = s1_tokens.explode("token")
    s1_tokens = s1_tokens[s1_tokens["token"].str.len() > 1]
    s1_tokens = s1_tokens[["entity_id", "token"]].drop_duplicates()

    # Self-merge on shared tokens (vectorized)
    merged = s1_tokens.merge(
        s2_tokens, on="token", suffixes=("_s1", "_s2")
    )
    # Count shared tokens per pair
    pair_counts = merged.groupby(["entity_id_s1", "entity_id_s2"]).size().reset_index(name="overlap")

    # For each S1, take top-k S2 by overlap
    pair_counts = pair_counts.sort_values(["entity_id_s1", "overlap"], ascending=[True, False])
    top = pair_counts.groupby("entity_id_s1").head(top_k)
    return set(zip(top["entity_id_s1"], top["entity_id_s2"]))


def fast_ngram_block(s1: pd.DataFrame, s2: pd.DataFrame, n: int = 2, top_k: int = 50) -> Set[Tuple[str, str]]:
    """Character n-gram blocking, vectorized."""
    # Build n-grams from s2 names
    def make_ngrams(s, n):
        s = s.replace(" ", "_")
        return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

    # Tokenize to n-grams
    s2_ng = s2[["entity_id", "name_norm"]].copy()
    s2_ng["ng"] = s2_ng["name_norm"].apply(lambda s: make_ngrams(s, n))
    s2_ng = s2_ng.explode("ng")
    s2_ng = s2_ng[s2_ng["ng"].notna() & (s2_ng["ng"].str.len() == n)]
    s2_ng = s2_ng[["entity_id", "ng"]].drop_duplicates()

    s1_ng = s1[["entity_id", "name_norm"]].copy()
    s1_ng["ng"] = s1_ng["name_norm"].apply(lambda s: make_ngrams(s, n))
    s1_ng = s1_ng.explode("ng")
    s1_ng = s1_ng[s1_ng["ng"].notna() & (s1_ng["ng"].str.len() == n)]
    s1_ng = s1_ng[["entity_id", "ng"]].drop_duplicates()

    merged = s1_ng.merge(s2_ng, on="ng", suffixes=("_s1", "_s2"))
    pair_counts = merged.groupby(["entity_id_s1", "entity_id_s2"]).size().reset_index(name="overlap")
    pair_counts = pair_counts.sort_values(["entity_id_s1", "overlap"], ascending=[True, False])
    top = pair_counts.groupby("entity_id_s1").head(top_k)
    return set(zip(top["entity_id_s1"], top["entity_id_s2"]))


def fast_all_blocks(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 50) -> Set[Tuple[str, str]]:
    """Union of all blocking strategies (fast)."""
    print("  Token blocking...")
    pairs = fast_token_block(s1, s2, top_k=top_k)
    print(f"    {len(pairs):,} pairs")
    print("  2-gram blocking...")
    pairs |= fast_ngram_block(s1, s2, n=2, top_k=top_k)
    print(f"    {len(pairs):,} pairs (cumulative)")
    print("  3-gram blocking...")
    pairs |= fast_ngram_block(s1, s2, n=3, top_k=top_k)
    print(f"    {len(pairs):,} pairs (cumulative)")
    return pairs

"""Memory-efficient blocking using dict-based inverted index."""
from collections import defaultdict
from typing import Set, Tuple, Dict, List
import pandas as pd


def fast_token_block(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 10) -> Set[Tuple[str, str]]:
    """Token blocking using dict-based inverted index. Memory efficient.

    For each S1 entity, find S2 entities sharing the most tokens.
    Uses dict-of-sets (inverted index) instead of pandas merge to avoid OOM.
    """
    # Build inverted index: token -> set of S2 entity_ids
    token_to_s2: Dict[str, set] = defaultdict(set)
    for _, row in s2[["entity_id", "name_norm"]].iterrows():
        if not isinstance(row["name_norm"], str):
            continue
        for tok in row["name_norm"].split():
            if len(tok) > 1:
                token_to_s2[tok].add(row["entity_id"])

    # Query: for each S1, count overlaps with S2
    s1_to_s2_scores: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in s1[["entity_id", "name_norm"]].iterrows():
        s1_id = row["entity_id"]
        if not isinstance(row["name_norm"], str):
            continue
        tokens = set(row["name_norm"].split())
        for tok in tokens:
            if tok in token_to_s2:
                for s2_id in token_to_s2[tok]:
                    s1_to_s2_scores[s1_id][s2_id] += 1

    # Take top-k S2 per S1
    pairs = set()
    for s1_id, s2_scores in s1_to_s2_scores.items():
        top = sorted(s2_scores.items(), key=lambda x: -x[1])[:top_k]
        for s2_id, score in top:
            if score > 0:
                pairs.add((s1_id, s2_id))
    return pairs


def fast_ngram_block(s1: pd.DataFrame, s2: pd.DataFrame, n: int = 2, top_k: int = 10) -> Set[Tuple[str, str]]:
    """Character n-gram blocking using dict-based inverted index."""
    def make_ngrams(s, n):
        s = s.replace(" ", "_")
        return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

    # Build inverted index
    ng_to_s2: Dict[str, set] = defaultdict(set)
    for _, row in s2[["entity_id", "name_norm"]].iterrows():
        if not isinstance(row["name_norm"], str):
            continue
        for ng in make_ngrams(row["name_norm"], n):
            if len(ng) == n:
                ng_to_s2[ng].add(row["entity_id"])

    # Query
    s1_to_s2_scores: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in s1[["entity_id", "name_norm"]].iterrows():
        s1_id = row["entity_id"]
        if not isinstance(row["name_norm"], str):
            continue
        for ng in make_ngrams(row["name_norm"], n):
            if ng in ng_to_s2:
                for s2_id in ng_to_s2[ng]:
                    s1_to_s2_scores[s1_id][s2_id] += 1

    # Take top-k
    pairs = set()
    for s1_id, s2_scores in s1_to_s2_scores.items():
        top = sorted(s2_scores.items(), key=lambda x: -x[1])[:top_k]
        for s2_id, score in top:
            if score > 0:
                pairs.add((s1_id, s2_id))
    return pairs


def fast_all_blocks(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 10) -> Set[Tuple[str, str]]:
    """Union of all blocking strategies (memory-efficient)."""
    print("  Token blocking...", flush=True)
    pairs = fast_token_block(s1, s2, top_k=top_k)
    print(f"    {len(pairs):,} pairs (cumulative)", flush=True)
    print("  2-gram blocking...", flush=True)
    pairs |= fast_ngram_block(s1, s2, n=2, top_k=top_k)
    print(f"    {len(pairs):,} pairs (cumulative)", flush=True)
    print("  3-gram blocking...", flush=True)
    pairs |= fast_ngram_block(s1, s2, n=3, top_k=top_k)
    print(f"    {len(pairs):,} pairs (cumulative)", flush=True)
    return pairs

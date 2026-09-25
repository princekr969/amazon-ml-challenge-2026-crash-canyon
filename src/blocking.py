"""Multi-strategy blocking for candidate generation."""
import re
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import pandas as pd


def token_ngram_block(s1: pd.DataFrame, s2: pd.DataFrame, n: int = 2) -> Set[Tuple[str, str]]:
    """Block on character n-grams of normalized names."""
    def ngrams(s: str, n: int) -> Set[str]:
        s = s.replace(" ", "_")
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()
    
    s2_idx = defaultdict(set)
    for _, row in s2.iterrows():
        for ng in ngrams(row["name_norm"], n):
            s2_idx[ng].add(row["entity_id"])
    
    pairs = set()
    for _, row in s1.iterrows():
        cands = set()
        for ng in ngrams(row["name_norm"], n):
            if ng in s2_idx:
                cands |= s2_idx[ng]
        for c in cands:
            pairs.add((row["entity_id"], c))
    return pairs


def token_block(s1: pd.DataFrame, s2: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Block on shared tokens in normalized names."""
    s2_idx = defaultdict(set)
    for _, row in s2.iterrows():
        for tok in row["name_norm"].split():
            if len(tok) > 1:
                s2_idx[tok].add(row["entity_id"])
    
    pairs = set()
    for _, row in s1.iterrows():
        cands = set()
        for tok in row["name_norm"].split():
            if tok in s2_idx:
                cands |= s2_idx[tok]
        for c in cands:
            pairs.add((row["entity_id"], c))
    return pairs


def all_blocks(s1: pd.DataFrame, s2: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Union of all blocking strategies."""
    pairs = set()
    pairs |= token_block(s1, s2)
    pairs |= token_ngram_block(s1, s2, n=2)
    pairs |= token_ngram_block(s1, s2, n=3)
    return pairs
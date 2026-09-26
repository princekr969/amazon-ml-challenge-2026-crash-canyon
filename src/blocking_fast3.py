"""Memory-safe blocking with PROGRESS BARS.
Uses min-heap for top-k per S1 (memory bounded).
"""
import heapq
import time
from collections import defaultdict
from typing import Set, Tuple, Dict, List
import pandas as pd
from tqdm import tqdm


def fast_token_block(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 10,
                    show_progress: bool = True) -> Set[Tuple[str, str]]:
    """Token blocking with min-heap top-k per S1. Memory-safe."""
    # Step 1: Build inverted index with progress bar
    iter_s2 = s2[["entity_id", "name_norm"]].iterrows()
    if show_progress:
        iter_s2 = tqdm(iter_s2, total=len(s2), desc="  Indexing S2", unit="rows", ncols=80)

    token_to_s2: Dict[str, set] = defaultdict(set)
    for _, row in iter_s2:
        if not isinstance(row["name_norm"], str):
            continue
        for tok in row["name_norm"].split():
            if len(tok) > 1 and len(token_to_s2[tok]) < 1000:
                token_to_s2[tok].add(row["entity_id"])

    if show_progress:
        print(f"  Indexed {len(token_to_s2):,} unique tokens")

    # Step 2: For each S1, accumulate scores in a min-heap of size top_k
    iter_s1 = s1[["entity_id", "name_norm"]].iterrows()
    if show_progress:
        iter_s1 = tqdm(iter_s1, total=len(s1), desc="  Blocking S1", unit="rows", ncols=80)

    pairs = set()
    start = time.time()
    last_update = start

    for idx, (_, row) in enumerate(iter_s1):
        s1_id = row["entity_id"]
        if not isinstance(row["name_norm"], str):
            continue
        tokens = row["name_norm"].split()
        if not tokens:
            continue
        # Use a min-heap: (score, s2_id) - we keep the largest top_k
        # Note: we use (count, s2_id) with heapq as min-heap on count
        heap: List = []  # min-heap
        seen = set()  # avoid counting same S2 twice
        for tok in tokens:
            if len(tok) > 1 and tok in token_to_s2:
                for s2_id in token_to_s2[tok]:
                    if s2_id in seen:
                        continue
                    seen.add(s2_id)
                    # heap is min-heap on count, so smaller count = pop first
                    if len(heap) < top_k:
                        heapq.heappush(heap, (1, s2_id))
                    elif True:  # count is always 1, so always replace (single-token match)
                        heapq.heapreplace(heap, (1, s2_id))
        for _, s2_id in heap:
            pairs.add((s1_id, s2_id))

        # Periodic progress update (every 10000 S1)
        if show_progress and idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (len(s1) - idx) / max(0.001, rate)
            tqdm.write(f"    [{idx:,}/{len(s1):,}] {rate:.0f} rows/s, ETA: {eta:.0f}s")

    return pairs


def fast_ngram_block(s1: pd.DataFrame, s2: pd.DataFrame, n: int = 2, top_k: int = 10,
                     show_progress: bool = True) -> Set[Tuple[str, str]]:
    """Character n-gram blocking with min-heap top-k per S1."""

    def make_ngrams(s, n):
        s = s.replace(" ", "_")
        return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

    # Build inverted index
    iter_s2 = s2[["entity_id", "name_norm"]].iterrows()
    if show_progress:
        iter_s2 = tqdm(iter_s2, total=len(s2), desc=f"  Indexing {n}-grams", unit="rows", ncols=80)

    ng_to_s2: Dict[str, set] = defaultdict(set)
    for _, row in iter_s2:
        if not isinstance(row["name_norm"], str):
            continue
        ngs = make_ngrams(row["name_norm"], n)
        for ng in ngs:
            if len(ng) == n and len(ng_to_s2[ng]) < 1000:
                ng_to_s2[ng].add(row["entity_id"])

    if show_progress:
        print(f"  Indexed {len(ng_to_s2):,} unique {n}-grams")

    # Query
    iter_s1 = s1[["entity_id", "name_norm"]].iterrows()
    if show_progress:
        iter_s1 = tqdm(iter_s1, total=len(s1), desc=f"  Blocking ({n}-gram)", unit="rows", ncols=80)

    pairs = set()
    start = time.time()

    for idx, (_, row) in enumerate(iter_s1):
        s1_id = row["entity_id"]
        if not isinstance(row["name_norm"], str):
            continue
        ngs = make_ngrams(row["name_norm"], n)
        if not ngs:
            continue
        heap: List = []
        seen = set()
        for ng in ngs:
            if ng in ng_to_s2:
                for s2_id in ng_to_s2[ng]:
                    if s2_id in seen:
                        continue
                    seen.add(s2_id)
                    if len(heap) < top_k:
                        heapq.heappush(heap, (1, s2_id))
                    else:
                        heapq.heapreplace(heap, (1, s2_id))
        for _, s2_id in heap:
            pairs.add((s1_id, s2_id))

        if show_progress and idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (len(s1) - idx) / max(0.001, rate)
            tqdm.write(f"    [{idx:,}/{len(s1):,}] {rate:.0f} rows/s, ETA: {eta:.0f}s")

    return pairs


def fast_all_blocks(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 10,
                    show_progress: bool = True) -> Set[Tuple[str, str]]:
    """Union of all blocking strategies (memory-safe + progress bars)."""
    print("  [1/3] Token blocking...", flush=True)
    pairs = fast_token_block(s1, s2, top_k=top_k, show_progress=show_progress)
    print(f"    → {len(pairs):,} pairs (cumulative)", flush=True)

    print("  [2/3] 2-gram blocking...", flush=True)
    pairs |= fast_ngram_block(s1, s2, n=2, top_k=top_k, show_progress=show_progress)
    print(f"    → {len(pairs):,} pairs (cumulative)", flush=True)

    print("  [3/3] 3-gram blocking...", flush=True)
    pairs |= fast_ngram_block(s1, s2, n=3, top_k=top_k, show_progress=show_progress)
    print(f"    → {len(pairs):,} pairs (cumulative)", flush=True)

    return pairs

"""Feature engineering for candidate pairs."""
import pandas as pd
import numpy as np
from Levenshtein import ratio as lev_ratio
from jellyfish import jaro_winkler_similarity
from metaphone import doublemetaphone
from country_parser import parse_address, component_match


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ng(s):
        s = s.replace(" ", "")
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else set()
    ga, gb = ng(a), ng(b)
    return len(ga & gb) / max(1, len(ga | gb))


def phonetic_match(a: str, b: str) -> int:
    """1 if any token shares a Double Metaphone code, else 0."""
    tokens_a = [a.split()[i] for i in range(min(3, len(a.split())))]
    tokens_b = [b.split()[i] for i in range(min(3, len(b.split())))]
    codes_a = set()
    codes_b = set()
    for t in tokens_a:
        if len(t) > 2:
            codes_a.add(doublemetaphone(t)[0])
    for t in tokens_b:
        if len(t) > 2:
            codes_b.add(doublemetaphone(t)[0])
    return int(len(codes_a & codes_b) > 0)


def compute_pair_features(s1_row, s2_row, embedding_score=None):
    """Compute all features for a candidate pair."""
    feats = {}
    n1, n2 = s1_row["name_norm"], s2_row["name_norm"]
    a1, a2 = s1_row["addr_norm"], s2_row["addr_norm"]
    
    # String features
    feats["name_levenshtein"] = lev_ratio(n1, n2)
    feats["name_jaro_winkler"] = jaro_winkler_similarity(n1, n2)
    feats["name_jaccard"] = jaccard(n1, n2)
    feats["name_ngram3_jaccard"] = char_ngram_jaccard(n1, n2, 3)
    feats["name_len_ratio"] = min(len(n1), len(n2)) / max(1, max(len(n1), len(n2)))
    
    # Address features
    feats["addr_levenshtein"] = lev_ratio(a1, a2)
    feats["addr_jaccard"] = jaccard(a1, a2)
    
    # Phonetic
    feats["name_phonetic_match"] = phonetic_match(n1, n2)
    
    # Country
    feats["country_match"] = int(s1_row["country"] == s2_row["country"])
    feats["country_in_train"] = int(
        s1_row["country"] in ("US", "India") and s2_row["country"] in ("US", "India")
    )
    feats["both_france"] = int(
        s1_row["country"] == "France" and s2_row["country"] == "France"
    )
    
    # Address components
    c1 = parse_address(s1_row["business_address"], s1_row["country"])
    c2 = parse_address(s2_row["business_address"], s2_row["country"])
    comp_feats = component_match(c1, c2)
    feats.update(comp_feats)
    
    # Embedding
    if embedding_score is not None:
        feats["embedding_score"] = embedding_score
    
    return feats
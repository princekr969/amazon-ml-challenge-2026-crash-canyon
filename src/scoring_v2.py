"""V2 multi-feature scoring for candidate pairs.
Stdlib + already-installed deps (unidecode, pandas) only. No rapidfuzz needed.
"""
import re
from difflib import SequenceMatcher
import unidecode
import pandas as pd

# === Normalization helpers (same idea as V1, kept local so V2 stays independent) ===
LEGAL_SUFFIX_MAP = {
    r"\bpvt\.?\b": "private", r"\bltd\.?\b": "limited",
    r"\bcorp\.?\b": "corporation", r"\binc\.?\b": "incorporated",
    r"\bllc\b": "limited liability company", r"\bplc\b": "public limited company",
    r"\b&\b": "and", r"\bco\.?\b": "company",
}
ADDRESS_ABBREV_MAP = {
    r"\brd\.?\b": "road", r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue", r"\bblvd\.?\b": "boulevard",
    r"\bfl\.?\b": "floor", r"\bste\.?\b": "suite",
}


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode.unidecode(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pat, repl, s)
    return s


def normalize_addr(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode.unidecode(s).lower()
    s = re.sub(r"[^\w\s,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in ADDRESS_ABBREV_MAP.items():
        s = re.sub(pat, repl, s)
    return s


def tokenize(s: str):
    """Whitespace tokens of length >= 2."""
    return [t for t in normalize_name(s).split() if len(t) > 1]


def tokenize_addr(s: str):
    return [t for t in normalize_addr(s).split() if len(t) > 1]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    u = sa | sb
    return 0.0 if not u else len(sa & sb) / len(u)


def char_sim(a: str, b: str) -> float:
    """SequenceMatcher similarity in [0, 1]. Stdlib; cheap for short strings."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def compute_features(name1: str, addr1: str, country1: str,
                     name2: str, addr2: str, country2: str) -> dict:
    """Compute 6 features for one (S1, candidate) pair."""
    n1 = tokenize(name1)
    n2 = tokenize(name2)
    a1 = tokenize_addr(addr1)
    a2 = tokenize_addr(addr2)
    n1s = " ".join(n1)
    n2s = " ".join(n2)
    sn1, sn2 = set(n1), set(n2)
    sn_min = min(len(sn1), len(sn2))
    containment = 0.0 if sn_min == 0 else len(sn1 & sn2) / sn_min

    c1 = "" if not isinstance(country1, str) else country1.strip().lower()
    c2 = "" if not isinstance(country2, str) else country2.strip().lower()
    # Normalize common aliases so "US" matches "USA" matches "United States", etc.
    COUNTRY_ALIASES = {
        "us": "us", "usa": "us", "united states": "us",
        "united states of america": "us", "u.s.": "us", "u.s.a.": "us",
        "in": "in", "india": "in",
        "fr": "fr", "france": "fr",
        "uk": "uk", "united kingdom": "uk", "great britain": "uk", "england": "uk",
        "gb": "uk",
        "ca": "ca", "canada": "ca",
        "au": "au", "australia": "au",
        "de": "de", "germany": "de",
    }
    nc1 = COUNTRY_ALIASES.get(c1, c1)
    nc2 = COUNTRY_ALIASES.get(c2, c2)
    country_match = 1.0 if nc1 and nc1 == nc2 else 0.0

    len_max = max(len(n1s), len(n2s))
    length_ratio = 0.0 if len_max == 0 else min(len(n1s), len(n2s)) / len_max

    return {
        "name_jaccard": jaccard(n1, n2),
        "addr_jaccard": jaccard(a1, a2),
        "name_sim": char_sim(n1s, n2s),
        "containment": containment,
        "country_match": country_match,
        "length_ratio": length_ratio,
    }


# === Hand-tuned weights (chosen by intuition; can be re-tuned on train) ===
WEIGHTS = {
    "name_jaccard": 0.40,
    "addr_jaccard": 0.30,
    "name_sim": 0.15,
    "containment": 0.10,
    "country_match": 0.03,
    "length_ratio": 0.02,
}


def score_features(feats: dict) -> float:
    return sum(WEIGHTS[k] * feats[k] for k in WEIGHTS)


# === Per-country acceptance threshold ===
# France is open-set (15% of test, 0% of train) -> lower threshold for recall.
# US/India are well-trained -> higher threshold for precision.
THRESHOLDS = {
    "fr": 0.42, "france": 0.42,
    "us": 0.55, "united states": 0.55, "usa": 0.55,
    "in": 0.55, "india": 0.55,
    "__default__": 0.50,
}


def get_threshold(country) -> float:
    if not isinstance(country, str):
        return THRESHOLDS["__default__"]
    c = country.strip().lower()
    return THRESHOLDS.get(c, THRESHOLDS["__default__"])

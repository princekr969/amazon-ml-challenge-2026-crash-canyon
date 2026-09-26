#!/usr/bin/env python3
"""
rescore_from_pairs.py — Fast V1-baseline re-scorer
==================================================

Inputs:
  - data/test/test_source{1,2,3}.tsv   (entity metadata)
  - output/candidate_pairs.tsv          (S1 -> comma-sep S2/S3 candidates)
Outputs:
  - output/matching_results.tsv         (TAB-separated, scored matches per S1)
  - output/submission.zip               (final zip-ready directory later)

Strategy: skip re-blocking (already done). Just load candidates, compute
features, apply per-country thresholds, write submission.

Features per (s1, cand) pair:
  - jaccard:        |A ∩ B| / |A ∪ B|      (token-set overlap)
  - containment:    |A ∩ B| / |A|         (asymmetric, name-substring proxy)
  - ratio:          rapidfuzz ratio       (Levenshtein-like)
  - partial:        rapidfuzz partial     (best sub-window)
  - token_sort:     rapidfuzz token_sort  (sorted-name Lev)
  - country_match:  bool (s1.country == cand.country)
  - phone_match:    bool (shared digits in phone, length>=7)
  - city_match:     bool (city string ==, normalized)

Final score = weighted combination, then per-country threshold:
  FR: thresh=0.40 (lower; boost recall)
  US/IN: thresh=0.50
  default: 0.45

Tuned on intuition — this is a baseline; LightGBM (V3) will beat this.
"""
import os, sys, time, gc, unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode
import regex

ROOT = r"C:\Users\LENOVO\crash_run"
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data", "test")

os.makedirs(OUT, exist_ok=True)

# ---- logging ----
_t0 = time.time()
def log(msg, force=False):
    elapsed = time.time() - _t0
    print(f"[{elapsed:>7.1f}s] {msg}", flush=True)

# ---- country thresholds (tuned for F_0.5: precision heavy) ----
THRESH_DEFAULT = 0.55   # baseline strict
THRESH_LOW = 0.45       # FR — recall boost (open-set, 15% of test, 0% in train)
THRESH_HIGH = 0.55      # US/IN/GB — default-ish, country boosts help

LOW_THR_COUNTRIES = {"FR", "JP"}
HIGH_THR_COUNTRIES = {"US", "IN", "GB"}

# ---- tokenization helpers ----
TOKEN_RE = regex.compile(r"[\p{L}\p{N}]+", flags=regex.UNICODE | regex.V1)

def tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = unidecode(s).lower()
    return TOKEN_RE.findall(s)

def norm_str(s: str) -> str:
    if not s:
        return ""
    return " ".join(unidecode(s).lower().split())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def containment(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)

def has_phone_digits(s: str, n: int = 7) -> bool:
    digits = "".join(c for c in (s or "") if c.isdigit())
    return len(digits) >= n

def extract_phone_digits(s: str, n: int = 7) -> str:
    digits = "".join(c for c in (s or "") if c.isdigit())
    return digits[-n:] if len(digits) >= n else ""

# ---- data loader: only the columns we need ----
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country", "city", "state", "phone", "postal_code"]

def load_source(path: str) -> Dict[str, dict]:
    log(f"Loading {os.path.basename(path)}")
    df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip", quoting=3).fillna("")
    keep = [c for c in SOURCE_COLS if c in df.columns]
    df = df[keep]
    rows = df.to_dict("records")
    out = {}
    for r in rows:
        eid = r["entity_id"]
        if not eid or eid in out:
            continue
        name_str = norm_str(r.get("business_name", ""))
        addr_str = norm_str(r.get("business_address", ""))
        out[eid] = {
            "name_tok": tokenize(r.get("business_name", "")),
            "name_str": name_str,
            "addr_str": addr_str,
            "addr_tok": tokenize(r.get("business_address", "")),
            "country": (r.get("country", "") or "").strip().upper(),
            "city": norm_str(r.get("city", "")),
            "state": norm_str(r.get("state", "")),
            "phone_digits": extract_phone_digits(r.get("phone", "")),
            "postal": norm_str(r.get("postal_code", "")),
        }
    del df
    gc.collect()
    log(f"  -> {len(out):,} entities indexed")
    return out

# ---- score a single (s1, cand) pair ----
def score_pair(s1: dict, cand: dict) -> Tuple[float, dict]:
    f_jac = jaccard(s1["name_tok"], cand["name_tok"])
    f_con = containment(s1["name_tok"], cand["name_tok"])
    a, b = s1["name_str"], cand["name_str"]
    if not a or not b:
        f_rat = f_par = f_tsr = 0.0
    else:
        f_rat = fuzz.ratio(a, b) / 100.0
        f_par = fuzz.partial_ratio(a, b) / 100.0
        f_tsr = fuzz.token_sort_ratio(a, b) / 100.0

    # Address features (use only when both have address)
    f_ajc = jaccard(s1["addr_tok"], cand["addr_tok"]) if s1["addr_tok"] and cand["addr_tok"] else 0.0
    f_acn = containment(s1["addr_tok"], cand["addr_tok"]) if s1["addr_tok"] and cand["addr_tok"] else 0.0
    addr_a, addr_b = s1["addr_str"], cand["addr_str"]
    if addr_a and addr_b:
        f_ara = fuzz.ratio(addr_a, addr_b) / 100.0
        f_apt = fuzz.partial_ratio(addr_a, addr_b) / 100.0
    else:
        f_ara = f_apt = 0.0

    f_cty = 1.0 if (s1["country"] and s1["country"] == cand["country"]) else 0.0

    # Phone: only meaningful if both have one
    f_phn = 0.0
    if s1["phone_digits"] and cand["phone_digits"]:
        f_phn = 1.0 if s1["phone_digits"] == cand["phone_digits"] else 0.0

    # City: only when both non-empty
    f_cit = 0.0
    if s1["city"] and cand["city"]:
        f_cit = 1.0 if s1["city"] == cand["city"] else 0.0

    # Postal: only when both non-empty
    f_pst = 0.0
    if s1["postal"] and cand["postal"]:
        f_pst = 1.0 if s1["postal"] == cand["postal"] else 0.0

    # Weighted blend (text-heavy, location boost)
    text_score = (
        0.10 * f_jac + 0.05 * f_con +
        0.15 * f_rat + 0.10 * f_par + 0.05 * f_tsr +
        0.05 * f_ajc + 0.05 * f_ara + 0.05 * f_apt
    )
    loc_score = 0.10 * f_cty + 0.10 * f_cit + 0.10 * f_phn + 0.05 * f_pst
    loc_score += 0.10 * f_acn  # address containment bonus

    # Bonus when country + strong text
    bonus = 0.0
    if f_cty > 0 and text_score >= 0.4:
        bonus = 0.10
    elif f_cty > 0 and text_score >= 0.2:
        bonus = 0.05

    return text_score + loc_score + bonus, {
        "jac": f_jac, "con": f_con, "rat": f_rat, "par": f_par, "tsr": f_tsr,
        "cty": f_cty, "phn": f_phn, "cit": f_cit, "pst": f_pst, "bon": bonus,
        "ajc": f_ajc, "acn": f_acn, "ara": f_ara, "apt": f_apt,
    }


def threshold_for_country(c: str) -> float:
    if c in LOW_THR_COUNTRIES:
        return THRESH_LOW
    if c in HIGH_THR_COUNTRIES:
        return THRESH_HIGH
    return THRESH_DEFAULT


def main():
    log("=" * 60)
    log("RESCORE FROM CANDIDATE PAIRS — V1 baseline")
    log("=" * 60)

    # ---- Load all source data into memory-mapped dicts ----
    s1_idx = load_source(os.path.join(DATA, "test_source1.tsv"))
    s2_idx = load_source(os.path.join(DATA, "test_source2.tsv"))
    s3_idx = load_source(os.path.join(DATA, "test_source3.tsv"))
    combined = {**s2_idx, **s3_idx}

    # ---- Load candidate pairs ----
    cand_path = os.path.join(OUT, "candidate_pairs.tsv")
    log(f"Loading {cand_path}")
    cand_df = pd.read_csv(cand_path, sep="\t", dtype=str, on_bad_lines="skip").fillna("")
    log(f"  -> {len(cand_df):,} S1 rows with candidates")

    # ---- Score per S1 ----
    n_total = len(cand_df)
    written = 0
    singletons = 0
    matched = 0
    multi_match = 0

    out_path = os.path.join(OUT, "matching_results.tsv")
    out_f = open(out_path, "w", encoding="utf-8")
    out_f.write("source1_entity_id\tmatched_entity_ids\n")

    t_loop = time.time()
    log("Scoring…")
    for i, (s1_id, cand_str) in enumerate(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"])):
        if i % 50000 == 0 and i:
            elapsed = time.time() - t_loop
            rate = i / max(elapsed, 0.001)
            eta = (n_total - i) / max(rate, 0.001)
            log(f"  scored {i:,}/{n_total:,} S1 rows ({rate:.0f}/s, ETA {eta/60:.1f} min) | written={written:,} matched={matched:,}")

        s1 = s1_idx.get(s1_id)
        if not s1:
            out_f.write(f"{s1_id}\t\n")
            written += 1
            singletons += 1
            continue

        candidates = [c for c in cand_str.split(",") if c] if isinstance(cand_str, str) else []
        country = s1["country"]
        thr = threshold_for_country(country)

        scored = []
        for cid in candidates:
            cand = combined.get(cid)
            if not cand:
                continue
            sc, _ = score_pair(s1, cand)
            scored.append((cid, sc))

        scored.sort(key=lambda x: -x[1])
        picks = [cid for cid, sc in scored if sc >= thr]

        if not picks:
            out_f.write(f"{s1_id}\t\n")
            singletons += 1
        elif len(picks) == 1:
            out_f.write(f"{s1_id}\t{picks[0]}\n")
            matched += 1
        else:
            out_f.write(f"{s1_id}\t{','.join(picks)}\n")
            matched += 1
            multi_match += 1

        written += 1

    out_f.close()
    log(f"Scoring done. Wrote {written:,} rows | matched={matched:,} | multi={multi_match:,} | singletons={singletons:,}")
    log(f"Output: {out_path}")
    log(f"Total time: {(time.time()-_t0)/60:.1f} min")


if __name__ == "__main__":
    main()

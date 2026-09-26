#!/usr/bin/env python3
"""
rescore_strict.py - STRICTER V1 baseline rescore
Same as rescore_fast.py but with HIGHER thresholds + top-3 cap.
Goal: high precision for F_0.5 (precision-weighted metric).
"""
import os, sys, time, gc
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode
import regex

ROOT = r"C:\Users\LENOVO\crash_run"
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data", "test")
os.makedirs(OUT, exist_ok=True)

_t0 = time.time()
def log(m):
    print(f"[{time.time()-_t0:>7.1f}s] {m}", flush=True)

TOKEN_RE = regex.compile(r"[\p{L}\p{N}]+", flags=regex.UNICODE | regex.V1)
def tokset(s):
    return set(TOKEN_RE.findall(unidecode(str(s or "")).lower()))

# STRICTER thresholds (F_0.5 = precision 4× recall)
# Default bumped up significantly
THR_DEFAULT, THR_LOW, THR_HIGH = 0.70, 0.60, 0.70
LOW_C = {"FR", "JP"}
HIGH_C = {"US", "IN", "GB"}
TOP_K = 2  # cap to at most 2 matches per S1
def thr_for(c):
    if c in LOW_C: return THR_LOW
    if c in HIGH_C: return THR_HIGH
    return THR_DEFAULT

def load_minimal(path):
    log(f"  Loading {os.path.basename(path)}")
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "country"],
                     on_bad_lines="skip").fillna("")
    out = {}
    for eid, name, ctry in zip(df["entity_id"].values, df["business_name"].values, df["country"].values):
        if not eid or eid in out: continue
        out[eid] = {
            "toks": tokset(name),
            "name_str": " ".join(unidecode(str(name or "")).lower().split()),
            "country": str(ctry or "").strip().upper(),
        }
    del df; gc.collect()
    log(f"  -> {len(out):,} entities")
    return out

log("=" * 60)
log("STRICT RESCORE (top-{}, thr 0.60/0.70)".format(TOP_K))
log("=" * 60)
s1 = load_minimal(os.path.join(DATA, "test_source1.tsv"))
s2 = load_minimal(os.path.join(DATA, "test_source2.tsv"))
s3 = load_minimal(os.path.join(DATA, "test_source3.tsv"))
combined = {**s2, **s3}

log("Loading candidate_pairs.tsv…")
cp = pd.read_csv(os.path.join(OUT, "candidate_pairs.tsv"), sep="\t", dtype=str).fillna("")
log(f"  {len(cp):,} S1 rows")

out_path = os.path.join(OUT, "matching_results_strict.tsv")
out = open(out_path, "w", encoding="utf-8")
out.write("source1_entity_id\tmatched_entity_ids\n")

n = len(cp)
n_match = n_single = 0
log(f"Scoring with top-K={TOP_K}…")
t_loop = time.time()
for i, (s1_id, cand_str) in enumerate(zip(cp["source1_entity_id"].values, cp["candidate_entity_ids"].values)):
    if i % 200000 == 0:
        elapsed = time.time() - t_loop
        rate = i / max(elapsed, 0.001)
        eta_min = (n - i) / max(rate, 0.001) / 60
        log(f"  {i:,}/{n:,} ({rate:.0f}/s, ETA {eta_min:.1f}min) matches={n_match:,}")

    s = s1.get(s1_id)
    if not s:
        out.write(f"{s1_id}\t\n"); n_single += 1; continue
    cands = [c for c in cand_str.split(",") if c] if isinstance(cand_str, str) else []
    if not cands:
        out.write(f"{s1_id}\t\n"); n_single += 1; continue
    a, ta, ca = s["name_str"], s["toks"], s["country"]
    thr = thr_for(ca)

    scored = []
    for cid in cands:
        c = combined.get(cid)
        if not c: continue
        b = c["name_str"]
        if not a or not b: continue
        tb = c["toks"]
        u = len(ta | tb)
        f_jac = len(ta & tb) / u if u else 0.0
        f_con = len(ta & tb) / len(ta) if ta else 0.0
        f_rat = fuzz.ratio(a, b) / 100.0
        f_par = fuzz.partial_ratio(a, b) / 100.0
        f_tsr = fuzz.token_sort_ratio(a, b) / 100.0
        f_cty = 1.0 if ca and ca == c["country"] else 0.0
        text_score = 0.20 * f_jac + 0.10 * f_con + 0.25 * f_rat + 0.20 * f_par + 0.10 * f_tsr
        loc_score = 0.10 * f_cty
        bonus = 0.10 if (f_cty > 0 and text_score >= 0.3) else 0.0
        scored.append((cid, text_score + loc_score + bonus))
    scored.sort(key=lambda x: -x[1])
    picks = [cid for cid, sc in scored if sc >= thr][:TOP_K]
    if not picks:
        out.write(f"{s1_id}\t\n"); n_single += 1
    else:
        out.write(f"{s1_id}\t{','.join(picks)}\n"); n_match += 1

out.close()
log(f"DONE in {(time.time()-_t0)/60:.1f}min — matched={n_match:,} single={n_single:,}")
log(f"Output: {out_path}")

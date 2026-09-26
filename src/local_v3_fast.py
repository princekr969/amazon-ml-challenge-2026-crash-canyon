# ============================================================
# OPTIMIZED V3-LITE - Streamlined for fast completion
# Pre-tokenized columns, efficient blocking, minimal I/O
# Run: cd C:\Users\LENOVO\crash_run
#      venv\Scripts\python.exe -u src\local_v3_fast.py
# ============================================================
import os, sys, time, re, pickle, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

t0 = time.time()
DATA = Path("data")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

def log(msg):
    """Print + flush to stdout (avoids pipe buffer lock)."""
    print(msg, flush=True)

log(f"[{time.time()-t0:.0f}s] START V3-LITE FAST")

# ============ LOAD ============
log(f"[{time.time()-t0:.0f}s] Loading...")
s1_tr = pd.read_csv(DATA/"train"/"train_source1.tsv", sep="\t", dtype=str).fillna("")
s2_tr = pd.read_csv(DATA/"train"/"train_source2.tsv", sep="\t", dtype=str).fillna("")
s3_tr = pd.read_csv(DATA/"train"/"train_source3.tsv", sep="\t", dtype=str).fillna("")
gt_tr = pd.read_csv(DATA/"train"/"train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
s1_te = pd.read_csv(DATA/"test"/"test_source1.tsv", sep="\t", dtype=str).fillna("")
s2_te = pd.read_csv(DATA/"test"/"test_source2.tsv", sep="\t", dtype=str).fillna("")
s3_te = pd.read_csv(DATA/"test"/"test_source3.tsv", sep="\t", dtype=str).fillna("")
log(f"  TR S1={len(s1_tr):,} S2={len(s2_tr):,} S3={len(s3_tr):,}")
log(f"  TE S1={len(s1_te):,} S2={len(s2_te):,} S3={len(s3_te):,}")

# ============ PREPROCESS - vectorized ============
log(f"[{time.time()-t0:.0f}s] Preprocessing (vectorized)...")

LEGAL_MAP = [
    (r"\bpvt\.?\b", "private"), (r"\bltd\.?\b", "limited"),
    (r"\bcorp\.?\b", "corporation"), (r"\binc\.?\b", "incorporated"),
    (r"\b&\b", "and"), (r"\bco\.?\b", "company"),
    (r"\brd\.?\b", "road"), (r"\bst\.?\b", "street"),
    (r"\bave\.?\b", "avenue"),
]

def preprocess(df):
    """Fast vectorized preprocessing."""
    # Lowercase + ascii
    name = df["business_name"].str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    addr = df["business_address"].str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    # Strip punctuation
    name = name.str.replace(r"[^\w\s]", " ", regex=True)
    addr = addr.str.replace(r"[^\w\s,]", " ", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()
    addr = addr.str.replace(r"\s+", " ", regex=True).str.strip()
    # Legal suffix expansion
    for pat, repl in LEGAL_MAP:
        name = name.str.replace(pat, repl, regex=True)
        addr = addr.str.replace(pat, repl, regex=True)
    return name, addr

log(f"[{time.time()-t0:.0f}s]   preprocess TR S1...")
tr_s1_name, tr_s1_addr = preprocess(s1_tr)
log(f"[{time.time()-t0:.0f}s]   preprocess TR S2...")
tr_s2_name, tr_s2_addr = preprocess(s2_tr)
log(f"[{time.time()-t0:.0f}s]   preprocess TR S3...")
tr_s3_name, tr_s3_addr = preprocess(s3_tr)
log(f"[{time.time()-t0:.0f}s]   preprocess TE S1...")
te_s1_name, te_s1_addr = preprocess(s1_te)
log(f"[{time.time()-t0:.0f}s]   preprocess TE S2...")
te_s2_name, te_s2_addr = preprocess(s2_te)
log(f"[{time.time()-t0:.0f}s]   preprocess TE S3...")
te_s3_name, te_s3_addr = preprocess(s3_te)

# Compute token Series (each row = list of unique tokens)
def toks_series(s):
    """Convert text Series to token-list Series."""
    return s.str.split().apply(lambda lst: list(set(t for t in lst if len(t) > 1)) if isinstance(lst, list) else [])

log(f"[{time.time()-t0:.0f}s] Tokenizing...")
tr_s1_tok = toks_series(tr_s1_name)
tr_s2_tok = toks_series(tr_s2_name)
tr_s3_tok = toks_series(tr_s3_name)
te_s1_tok = toks_series(te_s1_name)
te_s2_tok = toks_series(te_s1_name.iloc[:len(s2_te)])  # placeholder, replaced below
te_s2_tok = toks_series(te_s2_name)
te_s3_tok = toks_series(te_s3_name)
log(f"[{time.time()-t0:.0f}s] Preprocessing DONE")

# ============ Build token-lists for S2/S3 (combined) ============
def build_lk(df, name_tok, addr_norm, country, name_norm):
    """Build a Series indexed by entity_id, value=tuple(name_tok, addr_tok)."""
    return pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "name_tok": name_tok.values,
        "name_n": name_norm.values,
        "country": country.fillna("").values,
    })

log(f"[{time.time()-t0:.0f}s] Building candidate lookups...")
tr23 = pd.concat([
    build_lk(s2_tr, tr_s2_tok, tr_s2_addr, s2_tr["country"], tr_s2_name),
    build_lk(s3_tr, tr_s3_tok, tr_s3_addr, s3_tr["country"], tr_s3_name),
], ignore_index=True).set_index("entity_id")
te23 = pd.concat([
    build_lk(s2_te, te_s2_tok, te_s2_addr, s2_te["country"], te_s2_name),
    build_lk(s3_te, te_s3_tok, te_s3_addr, s3_te["country"], te_s3_name),
], ignore_index=True).set_index("entity_id")
log(f"[{time.time()-t0:.0f}s]   TR candidates: {len(tr23):,}, TE: {len(te23):,}")

# ============ Block ============
def block(s1_df, s1_tok, s23, top_k=5):
    """Token-overlap blocking with memory-safe inverted index."""
    log(f"    building inverted index from {len(s23):,} entities...", flush=True)
    inv = defaultdict(set)
    s23_toks = s23["name_tok"].values
    s23_ids = s23.index.values
    for i, toks in enumerate(s23_toks):
        for t in toks:
            if len(inv[t]) < 500:
                inv[t].add(s23_ids[i])
        if i % 1000000 == 0:
            log(f"      index: {i:,}/{len(s23_toks):,}")

    log(f"    scoring {len(s1_df):,} S1 against index...", flush=True)
    s1_ids = s1_df["entity_id"].values
    s1_tok_vals = s1_tok.values
    pairs = set()
    for i, s1_id in enumerate(s1_ids):
        toks = s1_tok_vals[i]
        if not toks: continue
        scores = defaultdict(int)
        for t in toks:
            if t in inv:
                for cid in inv[t]:
                    scores[cid] += 1
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        for cid, sc in top:
            if sc > 0:
                pairs.add((s1_id, cid))
        if i % 500000 == 0:
            log(f"      score: {i:,}/{len(s1_ids):,}, pairs so far {len(pairs):,}")
    return pairs

log(f"[{time.time()-t0:.0f}s] Blocking TRAIN...")
tr_pairs = block(s1_tr, tr_s1_tok, tr23, top_k=5)
log(f"[{time.time()-t0:.0f}s]   token train pairs: {len(tr_pairs):,}")

# ============ Train features + LightGBM ============
log(f"[{time.time()-t0:.0f}s] Building train features...")

import lightgbm as lgb
gt_dict = {r["source1_entity_id"]: set(m.strip() for m in r["matched_entity_ids"].split(",") if m.strip())
           for _, r in gt_tr.iterrows()}

X_tr, y_tr = [], []
tr23_tok = tr23["name_tok"].to_dict()
tr23_name = tr23["name_n"].to_dict()
tr23_country = tr23["country"].to_dict()
tr_s1_name_dict = tr_s1_name.to_dict()
tr_s1_country_dict = s1_tr["country"].fillna("").to_dict()
tr_s1_id_arr = s1_tr["entity_id"].values
tr_s1_tok_dict = tr_s1_tok.to_dict()

count = 0
for s1_id, c_id in tr_pairs:
    a_tok = tr_s1_tok_dict.get(s1_id)
    a_name = tr_s1_name_dict.get(s1_id)
    a_country = tr_s1_country_dict.get(s1_id, "")
    b_tok = tr23_tok.get(c_id)
    b_name = tr23_name.get(c_id)
    b_country = tr23_country.get(c_id, "")
    if not a_tok or not b_tok:
        continue
    # 6 features
    sn1, sn2 = set(a_tok), set(b_tok)
    n_jacc = len(sn1 & sn2) / len(sn1 | sn2) if sn1 | sn2 else 0
    n_cont = len(sn1 & sn2) / min(len(sn1), len(sn2)) if min(len(sn1), len(sn2)) else 0
    n_eq = 1.0 if a_name == b_name and a_name else 0.0
    first_tok = 1.0 if (a_tok and b_tok and a_tok[0] == b_tok[0]) else 0.0
    country_m = 1.0 if a_country and a_country == b_country else 0.0
    X_tr.append([n_jacc, n_cont, n_eq, first_tok, country_m])
    y_tr.append(1 if c_id in gt_dict.get(s1_id, set()) else 0)
    count += 1
    if count % 1000000 == 0:
        log(f"      feats: {count:,}, pos={sum(y_tr):,}")

X_tr = np.array(X_tr, dtype="float32")
y_tr = np.array(y_tr)
log(f"[{time.time()-t0:.0f}s]   train: {len(X_tr):,} rows, positives: {y_tr.sum():,}")

log(f"[{time.time()-t0:.0f}s] Train LightGBM...")
params = {"objective":"binary","metric":"binary_logloss","learning_rate":0.05,
          "num_leaves":63,"min_child_samples":100,"feature_fraction":0.9,
          "bagging_fraction":0.9,"bagging_freq":5,"verbose":-1,"n_jobs":-1}
model = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=150)
log(f"[{time.time()-t0:.0f}s]   lightgbm trained")

# ============ TEST block + score ============
log(f"[{time.time()-t0:.0f}s] Blocking TEST...")
te_pairs = block(s1_te, te_s1_tok, te23, top_k=5)
log(f"[{time.time()-t0:.0f}s]   test pairs: {len(te_pairs):,}")

# Write candidate_pairs.tsv
log(f"[{time.time()-t0:.0f}s] Writing candidate_pairs.tsv...")
by_s1 = defaultdict(set)
for a, b in te_pairs:
    by_s1[a].add(b)
cand_rows = [{"source1_entity_id": sid,
              "candidate_entity_ids": ",".join(sorted(by_s1.get(sid, set())))}
             for sid in s1_te["entity_id"].tolist()]
pd.DataFrame(cand_rows).to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
log(f"   wrote {len(cand_rows):,} rows", flush=True)

# Score
log(f"[{time.time()-t0:.0f}s] Scoring test pairs...")
te23_tok_d = te23["name_tok"].to_dict()
te23_name_d = te23["name_n"].to_dict()
te23_country_d = te23["country"].to_dict()
te_s1_name_d = te_s1_name.to_dict()
te_s1_country_d = s1_te["country"].fillna("").to_dict()
te_s1_tok_d = te_s1_tok.to_dict()

s1_scores = defaultdict(list)
BATCH = 50000
batchX, batchK = [], []
for s1_id, c_id in te_pairs:
    a_tok = te_s1_tok_d.get(s1_id)
    b_tok = te23_tok_d.get(c_id)
    if not a_tok or not b_tok:
        continue
    sn1, sn2 = set(a_tok), set(b_tok)
    n_jacc = len(sn1 & sn2) / len(sn1 | sn2) if sn1 | sn2 else 0
    n_cont = len(sn1 & sn2) / min(len(sn1), len(sn2)) if min(len(sn1), len(sn2)) else 0
    n_eq = 1.0 if te_s1_name_d.get(s1_id) == te23_name_d.get(c_id) and te_s1_name_d.get(s1_id) else 0.0
    first_tok = 1.0 if (a_tok and b_tok and a_tok[0] == b_tok[0]) else 0.0
    country_m = 1.0 if te_s1_country_d.get(s1_id, "") and te_s1_country_d.get(s1_id, "") == te23_country_d.get(c_id, "") else 0.0
    batchX.append([n_jacc, n_cont, n_eq, first_tok, country_m])
    batchK.append((s1_id, c_id))
    if len(batchX) >= BATCH:
        X = np.array(batchX, dtype="float32")
        probs = model.predict(X)
        for (sid, cid), p in zip(batchK, probs):
            s1_scores[sid].append((cid, float(p)))
        batchX, batchK = [], []
if batchX:
    X = np.array(batchX, dtype="float32")
    probs = model.predict(X)
    for (sid, cid), p in zip(batchK, probs):
        s1_scores[sid].append((cid, float(p)))
log(f"[{time.time()-t0:.0f}s] Scoring complete")

# Threshold + write matching_results.tsv
THRESHOLDS = {"fr":0.40,"france":0.40,"us":0.50,"united states":0.50,
              "usa":0.50,"in":0.50,"india":0.50,"__default__":0.45}
log(f"[{time.time()-t0:.0f}s] Thresholding + writing matching_results.tsv...")
rows = []
for s1_id in s1_te["entity_id"].values:
    country = str(te_s1_country_d.get(s1_id, "")).strip().lower()
    thr = THRESHOLDS.get(country, THRESHOLDS["__default__"])
    sc = s1_scores.get(s1_id, [])
    sc.sort(key=lambda x: -x[1])
    picks = [c for c, p in sc if p >= thr][:3]
    rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(picks)})
pd.DataFrame(rows).to_csv("output/matching_results.tsv", sep="\t", index=False)
n_sing = sum(1 for r in rows if r["matched_entity_ids"]=="")
log(f"[{time.time()-t0:.0f}s]   wrote {len(rows):,} rows, {n_sing:,} singletons")
log(f"\n{'='*60}\nV3-LITE FAST COMPLETE in {(time.time()-t0)/60:.1f} min\n{'='*60}")

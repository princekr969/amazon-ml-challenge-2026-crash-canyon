"""Self-contained lite pipeline. Memory-safe design (heap-based top-K).
Uses V1 proven logic. ~60 min on 16GB local. Outputs matching_results.tsv + candidate_pairs.tsv.
"""
import sys, os, time, re, heapq
from pathlib import Path
from collections import defaultdict

import pandas as pd

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

log("START V1 LITE")
DATA = Path("data")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# === LOAD ===
log("Loading sources...")
s1_te = pd.read_csv(DATA/"test"/"test_source1.tsv", sep="\t", dtype=str).fillna("")
s2_te = pd.read_csv(DATA/"test"/"test_source2.tsv", sep="\t", dtype=str).fillna("")
s3_te = pd.read_csv(DATA/"test"/"test_source3.tsv", sep="\t", dtype=str).fillna("")
log(f"  S1={len(s1_te):,} S2={len(s2_te):,} S3={len(s3_te):,}")

# === PREPROCESS (simple, vectorized) ===
log("Preprocessing...")
LEGAL = {r"\bpvt\.?\b":"private", r"\bltd\.?\b":"limited",
        r"\bcorp\.?\b":"corporation", r"\binc\.?\b":"incorporated",
        r"\b&\b":"and", r"\bco\.?\b":"company"}

def name_norm(s):
    import unicodedata as _ud
    s = _ud.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^\w\s]", " ", s); s = re.sub(r"\s+", " ", s).strip()
    for p, r in LEGAL.items(): s = re.sub(p, r, s)
    return s

def addr_norm(s):
    import unicodedata as _ud
    s = _ud.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^\w\s,]", " ", s); s = re.sub(r"\s+", " ", s).strip()
    return s

log("  S1 names+addresses...")
s1_te["name_norm"] = s1_te["business_name"].astype(str).map(name_norm)
s1_te["name_tok"] = s1_te["name_norm"].str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])
s1_te["addr_tok"] = s1_te["business_address"].astype(str).map(addr_norm).str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])

log("  S2 names+addresses...")
s2_te["name_norm"] = s2_te["business_name"].astype(str).map(name_norm)
s2_te["name_tok"] = s2_te["name_norm"].str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])
s2_te["addr_tok"] = s2_te["business_address"].astype(str).map(addr_norm).str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])

log("  S3 names+addresses...")
s3_te["name_norm"] = s3_te["business_name"].astype(str).map(name_norm)
s3_te["name_tok"] = s3_te["name_norm"].str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])
s3_te["addr_tok"] = s3_te["business_address"].astype(str).map(addr_norm).str.split().apply(
    lambda lst: list(set(t for t in (lst or []) if len(t)>1)) if isinstance(lst, list) else [])
log("  Preprocess done")

# === LOOKUPS ===
log("Building lookups...")
lk_s2 = s2_te.set_index("entity_id")[["name_tok","addr_tok","country"]].to_dict("index")
lk_s3 = s3_te.set_index("entity_id")[["name_tok","addr_tok","country"]].to_dict("index")
lk_s2.update(lk_s3)  # any duplicate ID gets S3
del lk_s3
log(f"  Combined lookup: {len(lk_s2):,}")

# === BLOCK (memory-safe: heap-based top-K per S1) ===
def block(s1_df, lk, top_k=5):
    """Token-blocking with heap-based top-K. O(S1 × tokens × top_k) memory."""
    log("  Building token index from candidate entities...", flush=True)
    inv = defaultdict(set)
    BATCH_COUNT = 500000
    count = 0
    for eid, row in lk.items():
        toks = row.get("name_tok", [])
        for t in toks:
            if len(inv[t]) < 500:
                inv[t].add(eid)
        count += 1
        if count % BATCH_COUNT == 0:
            log(f"    index built: {count:,}/{len(lk):,}", flush=True)
    log(f"  Indexed {len(inv):,} unique tokens", flush=True)
    gc.collect()

    log("  Blocking S1 against candidates...", flush=True)
    pairs = set()
    count = 0
    s1_ids = s1_df["entity_id"].values
    s1_tok = s1_df["name_tok"].values
    for s1_id, toks in zip(s1_ids, s1_tok):
        if not toks: continue
        scores = defaultdict(int)
        for t in toks:
            if t in inv:
                for cid in inv[t]:
                    scores[cid] += 1
        if scores:
            top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
            for cid, sc in top:
                if sc > 0:
                    pairs.add((s1_id, cid))
        count += 1
        if count % 200000 == 0:
            log(f"    blocked: {count:,}/{len(s1_ids):,}, pairs={len(pairs):,}", flush=True)
    return pairs

log("Blocking S1->S2...")
cand_s2 = block(s1_te, lk_s2, top_k=5)
log(f"  pairs S1->S2: {len(cand_s2):,}")
log("Blocking S1->S3...")
cand_s3 = block(s1_te, lk_s2, top_k=5)  # lk_s2 already has S3 merged in
cand_s3 = set()
s3_only_df = s3_te.set_index("entity_id")[["name_tok"]].to_dict("index")
for s1_id, s2_id in cand_s2:
    pass  # don't double-count
# Re-block separately with just S3
log("Building separate S3 lookup...")
del lk_s2
gc.collect()
lk_s3_only = s3_te.set_index("entity_id")[["name_tok","addr_tok","country"]].to_dict("index")
# Rebuild combined with S2+S3 separate
lk_combined = s2_te.set_index("entity_id")[["name_tok","addr_tok","country"]].to_dict("index")
for eid, row in lk_s3_only.items():
    if eid not in lk_combined:
        lk_combined[eid] = row

cand_s3 = block(s1_te, lk_s3_only, top_k=5)
log(f"  pairs S1->S3: {len(cand_s3):,}")

# === COMBINE CANDIDATES (candidate_pairs.tsv) ===
log("Writing candidate_pairs.tsv...")
all_pairs = cand_s2 | cand_s3
by_s1 = defaultdict(set)
for s1_id, c_id in all_pairs:
    by_s1[s1_id].add(c_id)
cand_rows = [{"source1_entity_id": sid,
              "candidate_entity_ids": ",".join(sorted(by_s1.get(sid, set())))}
             for sid in s1_te["entity_id"].tolist()]
pd.DataFrame(cand_rows).to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
log(f"  wrote {len(cand_rows):,} rows")

# === SCORING + WRITE MATCHING ===
log("Scoring and selecting top-K=3 matches...")
THRESH = 0.20  # token overlap / max tokens

rows = []
count = 0
for s1_id in s1_te["entity_id"].values:
    if count % 100000 == 0:
        log(f"  score {count:,}/{len(s1_te):,}", flush=True)
    s1_tok_l = s1_te.set_index("entity_id")["name_tok"].get(s1_id, [])
    if not s1_tok_l:
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
        count += 1
        continue
    s1_set = set(s1_tok_l)
    cands = by_s1.get(s1_id, set())
    scored = []
    for cid in cands:
        crow = lk_combined.get(cid)
        if not crow: continue
        c_tok = crow.get("name_tok", [])
        if not c_tok: continue
        c_set = set(c_tok)
        ovl = len(s1_set & c_set)
        score = ovl / max(len(s1_set), len(c_set))
        scored.append((cid, score))
    scored.sort(key=lambda x: -x[1])
    picks = [c for c, sc in scored if sc >= THRESH][:3]
    rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(picks)})
    count += 1

result = pd.DataFrame(rows)
result.to_csv("output/matching_results.tsv", sep="\t", index=False)
log(f"  wrote {len(result):,} rows to output/matching_results.tsv")
n_sing = sum(1 for r in rows if not r["matched_entity_ids"])
log(f"  Singletons: {n_sing:,}")
log(f"\n{'='*60}")
log(f"V1 LITE COMPLETE in {(time.time()-t0)/60:.1f} min")
log(f"{'='*60}")

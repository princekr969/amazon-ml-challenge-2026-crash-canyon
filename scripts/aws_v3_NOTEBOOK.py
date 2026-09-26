# =============================================================
# AWS SageMaker Studio Notebook — V3 LightGBM Entity Resolution
# Path: C:\Users\LENOVO\crash_run\scripts\aws_v3_NOTEBOOK.py
# Open this file in Notepad, then PASTE each # ===CELL N=== block
# (header + code only) into a separate Jupyter cell in order.
# =============================================================

# ===CELL 1 — SETUP (auto-discover data dir)===
import os, sys, subprocess

print("=" * 60)
print("Cell 1 — Auto-discover data directory")
print("=" * 60)
print(f"Kernel CWD: {os.getcwd()}\n")

candidates = [
    os.getcwd(),
    "/home/ec2-user/SageMaker/crash-canyon",
    "/shared/crash-canyon",
    "/home/ec2-user/SageMaker",
    "/opt/ml",
    "/home/ec2-user",
]
data_dir = None
for d in candidates:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "train_source1.tsv")):
        data_dir = d
        print(f"  ✓ FOUND data at: {d}")
        break

if data_dir is None:
    print("  Common paths failed, searching filesystem…")
    try:
        r = subprocess.run(["find", "/", "-name", "train_source1.tsv", "-type", "f"],
                          capture_output=True, text=True, timeout=60)
        paths = [p for p in r.stdout.split("\n") if p.strip()]
        if paths:
            data_dir = os.path.dirname(paths[0])
            print(f"  ✓ FOUND via find: {data_dir}")
    except Exception as e:
        print(f"  find failed: {e}")

if data_dir is None:
    print("\n❌ Could not find train_source1.tsv")
    print("   Open File Browser, find train_source1.tsv, RIGHT-CLICK → Copy Path")
    user_path = input("   Paste full path here: ").strip()
    if os.path.isdir(os.path.dirname(user_path)):
        data_dir = os.path.dirname(user_path)
        print(f"  Using user-provided directory: {data_dir}")
    else:
        raise SystemExit("Cannot proceed without data")

WORK_DIR = data_dir
DATA_DIR = data_dir

os.chdir(WORK_DIR)
for n in os.listdir(DATA_DIR):
    src = os.path.join(DATA_DIR, n)
    dst = os.path.join(WORK_DIR, n)
    if not os.path.exists(dst) and not os.path.islink(dst):
        try:
            os.symlink(src, dst)
        except: pass

print(f"\nDATA_DIR = {DATA_DIR}")
print(f"WORK_DIR = {WORK_DIR}")
for n in ("train_source1.tsv","train_source2.tsv","train_source3.tsv",
          "test_source1.tsv","test_source2.tsv","test_source3.tsv",
          "train_ground_truth.tsv","validate_submission.py"):
    print(f"  {n:<35} {'OK' if os.path.exists(n) else 'MISSING'}")

globals()["DATA_DIR"] = DATA_DIR
globals()["WORK_DIR"] = WORK_DIR

# ===CELL 2 — INSTALL===
# !pip install --quiet lightgbm==4.7.0 unidecode regex rapidfuzz 2>&1 | tail -2
print("✅ packages installed (if needed)")


# ===CELL 3 — LOAD + INDEX===
import time, gc, pickle, os
import numpy as np
import pandas as pd
import regex
from collections import defaultdict

t0 = time.time()
def log(m): print(f"[{time.time()-t0:>7.1f}s] {m}", flush=True)

DATA = DATA_DIR
WORK = WORK_DIR
os.chdir(WORK)

COLS = ["entity_id", "business_name", "business_address", "country"]
log("Loading sources…")
s1_tr = pd.read_csv(f"{DATA}/train_source1.tsv", sep="\t", dtype=str, usecols=COLS).fillna("")
s2_tr = pd.read_csv(f"{DATA}/train_source2.tsv", sep="\t", dtype=str, usecols=COLS).fillna("")
s3_tr = pd.read_csv(f"{DATA}/train_source3.tsv", sep="\t", dtype=str, usecols=COLS).fillna("")
gt   = pd.read_csv(f"{DATA}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
s1_te = pd.read_csv(f"{DATA}/test_source1.tsv",  sep="\t", dtype=str, usecols=COLS).fillna("")
s2_te = pd.read_csv(f"{DATA}/test_source2.tsv",  sep="\t", dtype=str, usecols=COLS).fillna("")
s3_te = pd.read_csv(f"{DATA}/test_source3.tsv",  sep="\t", dtype=str, usecols=COLS).fillna("")
log(f"  TR S1={len(s1_tr):,} S2={len(s2_tr):,} S3={len(s3_tr):,}, GT={len(gt):,}")
log(f"  TE S1={len(s1_te):,} S2={len(s2_te):,} S3={len(s3_te):,}")

def tokset(s):
    return set(regex.findall(r"[\p{L}\p{N}]+", str(s or ""), flags=regex.UNICODE | regex.V1))

log("Building token sets…")
s1_tr_tok = {eid: tokset(n) for eid, n in zip(s1_tr["entity_id"], s1_tr["business_name"])}
s2_tr_tok = {eid: tokset(n) for eid, n in zip(s2_tr["entity_id"], s2_tr["business_name"])}
s3_tr_tok = {eid: tokset(n) for eid, n in zip(s3_tr["entity_id"], s3_tr["business_name"])}
s1_te_tok = {eid: tokset(n) for eid, n in zip(s1_te["entity_id"], s1_te["business_name"])}
s2_te_tok = {eid: tokset(n) for eid, n in zip(s2_te["entity_id"], s2_te["business_name"])}
s3_te_tok = {eid: tokset(n) for eid, n in zip(s3_te["entity_id"], s3_te["business_name"])}

log("Building inverted indices…")
inv_tr = defaultdict(set)
for eid, t in s2_tr_tok.items():
    for tk in t: inv_tr[tk].add(eid)
for eid, t in s3_tr_tok.items():
    for tk in t: inv_tr[tk].add(eid)
inv_te = defaultdict(set)
for eid, t in s2_te_tok.items():
    for tk in t: inv_te[tk].add(eid)
for eid, t in s3_te_tok.items():
    for tk in t: inv_te[tk].add(eid)
log(f"  inv_tr={len(inv_tr):,}  inv_te={len(inv_te):,}")

gt_set = defaultdict(set)
for _, r in gt.iterrows():
    gt_set[r["source1_entity_id"]].add(r["source2_entity_id"])
    gt_set[r["source1_entity_id"]].add(r["source3_entity_id"])

with open(os.path.join(WORK, "idx.pkl"), "wb") as f:
    pickle.dump({
        "s1_tr_tok":s1_tr_tok, "s2_tr_tok":s2_tr_tok, "s3_tr_tok":s3_tr_tok,
        "s1_te_tok":s1_te_tok, "s2_te_tok":s2_te_tok, "s3_te_tok":s3_te_tok,
        "inv_tr":dict(inv_tr), "inv_te":dict(inv_te),
        "gt_set":dict(gt_set),
        "s1_tr_name":dict(zip(s1_tr["entity_id"], s1_tr["business_name"])),
        "s2_tr_name":dict(zip(s2_tr["entity_id"], s2_tr["business_name"])),
        "s3_tr_name":dict(zip(s3_tr["entity_id"], s3_tr["business_name"])),
        "s1_te_name":dict(zip(s1_te["entity_id"], s1_te["business_name"])),
        "s2_te_name":dict(zip(s2_te["entity_id"], s2_te["business_name"])),
        "s3_te_name":dict(zip(s3_te["entity_id"], s3_te["business_name"])),
        "s1_tr_country":dict(zip(s1_tr["entity_id"], s1_tr["country"].fillna("").astype(str))),
        "s2_tr_country":dict(zip(s2_tr["entity_id"], s2_tr["country"].fillna("").astype(str))),
        "s3_tr_country":dict(zip(s3_tr["entity_id"], s3_tr["country"].fillna("").astype(str))),
        "s1_te_country":dict(zip(s1_te["entity_id"], s1_te["country"].fillna("").astype(str))),
        "s2_te_country":dict(zip(s2_te["entity_id"], s2_te["country"].fillna("").astype(str))),
        "s3_te_country":dict(zip(s3_te["entity_id"], s3_te["country"].fillna("").astype(str))),
    }, f)
log("✅ idx.pkl saved → Cell 3 done.")


# ===CELL 4 — BUILD TRAIN FEATURES (LightGBM)===
import os
WORK = WORK_DIR
os.chdir(WORK)
import numpy as np
from rapidfuzz import fuzz

t1 = time.time()
def log1(m): print(f"[{time.time()-t1:>7.1f}s] {m}", flush=True)

log1("Loading idx.pkl…")
with open("idx.pkl", "rb") as f:
    idx = pickle.load(f)
s1_tr_name, s2_tr_name, s3_tr_name = idx["s1_tr_name"], idx["s2_tr_name"], idx["s3_tr_name"]
s1_tr_country, s2_tr_country, s3_tr_country = idx["s1_tr_country"], idx["s2_tr_country"], idx["s3_tr_country"]
s1_tr_tok, s2_tr_tok, s3_tr_tok = idx["s1_tr_tok"], idx["s2_tr_tok"], idx["s3_tr_tok"]
inv_tr = idx["inv_tr"]
gt_set = idx["gt_set"]

X, y = [], []
train_s1 = list(s1_tr_tok.keys())
log1(f"Building features for {len(train_s1):,} S1…")
N_SAMPLE = 100000
for i, s1 in enumerate(train_s1):
    if i >= N_SAMPLE: break
    if i % 10000 == 0:
        log1(f"  {i:,}/{N_SAMPLE:,}  pairs={len(X):,}  pos={sum(y):,}")
    a = s1_tr_name.get(s1, "")
    ca = str(s1_tr_country.get(s1, "")).upper()
    s1t = s1_tr_tok.get(s1, set())
    if not a or not s1t: continue

    cands = set()
    for tk in s1t: cands |= inv_tr.get(tk, set())
    if not cands: continue

    pre = []
    for cid in cands:
        ct = s2_tr_tok.get(cid) or s3_tr_tok.get(cid)
        if not ct: continue
        ov = len(s1t & ct)
        if ov: pre.append((cid, ct, ov))
    pre.sort(key=lambda x: -x[2])
    pre = pre[:6]

    gt = gt_set.get(s1, set())
    for cid, ct, ov in pre:
        if cid.startswith("S2-"):
            b = s2_tr_name.get(cid, ""); cb = str(s2_tr_country.get(cid, "")).upper()
        else:
            b = s3_tr_name.get(cid, ""); cb = str(s3_tr_country.get(cid, "")).upper()
        if not b: continue
        ta, tb = s1t, ct
        u = len(ta | tb)
        f_jac = len(ta & tb) / u if u else 0.0
        f_con = len(ta & tb) / len(ta) if ta else 0.0
        f_rat = fuzz.ratio(a, b) / 100.0
        f_par = fuzz.partial_ratio(a, b) / 100.0
        f_tsr = fuzz.token_sort_ratio(a, b) / 100.0
        f_cty = 1.0 if ca and ca == cb else 0.0
        inter = ta & tb
        f_long = 1.0 if (inter and max(len(w) for w in inter) >= 5) else 0.0
        X.append([f_jac, f_con, f_rat, f_par, f_tsr, f_cty, f_long])
        y.append(1 if cid in gt else 0)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)
log1(f"X={X.shape}, pos={int(y.sum()):,} ({y.mean()*100:.2f}%)")
np.savez("train_feats.npz", X=X, y=y)
log1("✅ train_feats.npz saved → Cell 4 done.")


# ===CELL 5 — TRAIN LIGHTGBM===
import os, time
import numpy as np
import lightgbm as lgb
WORK = WORK_DIR
os.chdir(WORK)

d = np.load("train_feats.npz")
X, y = d["X"], d["y"]
print(f"X={X.shape}, pos={int(y.sum()):,}")

rng = np.random.default_rng(42)
idx = rng.permutation(len(y)); sp = int(len(y)*0.85)
tr_idx, va_idx = idx[:sp], idx[sp:]

model = lgb.train(
    {"objective":"binary", "metric":"binary_logloss", "learning_rate":0.05,
     "num_leaves":63, "min_data_in_leaf":200, "feature_fraction":0.9,
     "bagging_fraction":0.9, "bagging_freq":5, "lambda_l2":1.0, "verbose":-1,
     "n_jobs":-1, "force_col_wise":True},
    lgb.Dataset(X[tr_idx], y[tr_idx]),
    num_boost_round=600,
    valid_sets=[lgb.Dataset(X[va_idx], y[va_idx])],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)
model.save_model("lgbm_v3.txt")
print(f"✅ model saved (best_iter={model.best_iteration}) → Cell 5 done.")


# ===CELL 6 — V3 INFERENCE (test set)===
import os, time, pickle, numpy as np
from rapidfuzz import fuzz
import lightgbm as lgb
WORK = WORK_DIR
os.chdir(WORK)
t2 = time.time()
def log2(m): print(f"[{time.time()-t2:>7.1f}s] {m}", flush=True)

log2("Loading model + idx…")
model = lgb.Booster(model_file="lgbm_v3.txt")
with open("idx.pkl", "rb") as f: idx = pickle.load(f)
s1_te_tok = idx["s1_te_tok"]; inv_te = idx["inv_te"]
s1_te_name = idx["s1_te_name"]; s2_te_name = idx["s2_te_name"]; s3_te_name = idx["s3_te_name"]
s1_te_country = idx["s1_te_country"]; s2_te_country = idx["s2_te_country"]; s3_te_country = idx["s3_te_country"]
s2_te_tok = idx["s2_te_tok"]; s3_te_tok = idx["s3_te_tok"]

LOW_C, HIGH_C = {"FR","JP"}, {"US","IN","GB"}
def thr(c):
    if c in LOW_C: return 0.55
    if c in HIGH_C: return 0.65
    return 0.62

out_path = "matching_results.tsv"
with open(out_path, "w", encoding="utf-8") as out:
    out.write("source1_entity_id\tmatched_entity_ids\n")
    n = len(s1_te_tok)
    n_match = n_single = 0
    log2(f"Inference over {n:,} S1…")
    for i, (s1, s1t) in enumerate(s1_te_tok.items()):
        if i % 200000 == 0: log2(f"  {i:,}/{n:,} match={n_match:,}")
        a = s1_te_name.get(s1, "")
        ca = str(s1_te_country.get(s1, "")).upper()
        if not a or not s1t:
            out.write(f"{s1}\t\n"); n_single += 1; continue
        cands = set()
        for tk in s1t: cands |= inv_te.get(tk, set())
        if not cands:
            out.write(f"{s1}\t\n"); n_single += 1; continue

        pre = []
        for cid in cands:
            ct = s2_te_tok.get(cid) or s3_te_tok.get(cid)
            if not ct: continue
            ov = len(s1t & ct)
            if ov: pre.append((cid, ct, ov))
        pre.sort(key=lambda x: -x[2]); pre = pre[:6]

        Xb = np.zeros((len(pre), 7), dtype=np.float32)
        for k, (cid, ct, ov) in enumerate(pre):
            if cid.startswith("S2-"):
                b = s2_te_name.get(cid, ""); cb = str(s2_te_country.get(cid, "")).upper()
            else:
                b = s3_te_name.get(cid, ""); cb = str(s3_te_country.get(cid, "")).upper()
            if not b: Xb[k].fill(np.nan); continue
            u = len(s1t | ct)
            Xb[k, 0] = len(s1t & ct) / u if u else 0.0
            Xb[k, 1] = len(s1t & ct) / len(s1t) if s1t else 0.0
            Xb[k, 2] = fuzz.ratio(a, b) / 100.0
            Xb[k, 3] = fuzz.partial_ratio(a, b) / 100.0
            Xb[k, 4] = fuzz.token_sort_ratio(a, b) / 100.0
            Xb[k, 5] = 1.0 if ca and ca == cb else 0.0
            inter = s1t & ct
            Xb[k, 6] = 1.0 if (inter and max(len(w) for w in inter) >= 5) else 0.0

        preds = model.predict(Xb)
        scored = sorted(zip([p[0] for p in pre], preds), key=lambda x: -x[1])
        t = thr(ca)
        picks = [c for c, p in scored if p >= t][:2]
        if not picks:
            out.write(f"{s1}\t\n"); n_single += 1
        else:
            out.write(f"{s1}\t{','.join(picks)}\n"); n_match += 1
log2(f"✅ Done. matched={n_match:,} single={n_single:,}")


# ===CELL 7 — VALIDATE + MAKE ZIP (run LAST)===
import os, subprocess, zipfile
WORK = WORK_DIR
DATA = DATA_DIR
os.chdir(WORK)

print("Validating…")
r = subprocess.run(["python3", "validate_submission.py",
    "--matching", "matching_results.tsv",
    "--candidate", "candidate_pairs.tsv",
    "--test-dir", DATA], capture_output=True, text=True)
print(r.stdout)
print("Return code:", r.returncode)

zp = "submission_v3.zip"
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for n in ("matching_results.tsv", "candidate_pairs.tsv"):
        if os.path.exists(n):
            z.write(n); print(f"  + {n}")
print(f"\n✅ ZIP READY: {zp} ({os.path.getsize(zp)/1e6:.2f} MB)")
print("DOWNLOAD from File Browser → submit to Unstop")

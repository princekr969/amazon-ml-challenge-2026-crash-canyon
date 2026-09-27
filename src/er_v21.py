#!/usr/bin/env python3
"""
Business Entity Resolution — V21 (low-memory V20)
==================================================
Changes vs V20: per-country hashed TF-IDF (no vocab, parallel, ~half peak RAM),
raw text columns dropped after normalization, progress logs during blocking.

Pipeline
  1. Normalize names (legal suffixes, abbreviations) and addresses (abbreviations, numbers).
  2. Multi-channel blocking with sparse TF-IDF top-K (sparse_dot_topn, C++ multithreaded):
       ch A: char 3-gram TF-IDF on core name          (typos, spacing, transliteration)
       ch B: word TF-IDF on name + address combined    (renames / abbreviations rescued by address)
     Per-country blocking only if train GT confirms (almost) no cross-country matches.
     Candidates are scored with both cosines and capped at MAX_CAND per S1.
  3. Pairwise features: RapidFuzz name/address sims, IDF-weighted overlap, postal/number match,
     acronym, + CONTEXT features (rank of candidate within S1, gaps, reverse rank among S1s).
  4. LightGBM on a UNIFORM sample of train S1 (true priors) + isotonic calibration.
  5. Optional exclusivity (each S2/S3 record belongs to at most one S1, if GT confirms).
  6. Per-S1 set selection that maximizes EXPECTED F0.5 (handles singletons + multi-match).
  7. Holdout macro-F0.5 computed exactly like the leaderboard, then test inference.

Outputs: <out-dir>/matching_results.tsv and <out-dir>/candidate_pairs.tsv
  (candidate_pairs = exactly the pairs the model scored; matches are a subset of it)

Run:
  pip install -r requirements.txt
  python er_pipeline.py --data-dir /path/to/data --out-dir output --workers 16
"""
import os, re, sys, gc, csv, time, json, argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression
from unidecode import unidecode
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import lightgbm as lgb

try:
    from sparse_dot_topn import sp_matmul_topn
    HAS_TOPN = True
except Exception:
    HAS_TOPN = False

# ----------------------------------------------------------------------------- config
P = argparse.ArgumentParser()
P.add_argument("--data-dir", default=None, help="folder containing train_*.tsv and test_*.tsv (searched recursively)")
P.add_argument("--out-dir", default="output")
P.add_argument("--work-dir", default="work")
P.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
P.add_argument("--train-s1", type=int, default=300_000, help="uniform S1 sample used to train the model")
P.add_argument("--holdout-s1", type=int, default=100_000, help="uniform S1 sample for honest validation")
P.add_argument("--k-char", type=int, default=10, help="top-K per source, char channel")
P.add_argument("--k-comb", type=int, default=10, help="top-K per source, name+address channel")
P.add_argument("--max-cand", type=int, default=30, help="max candidates per S1 after merge")
P.add_argument("--addr-weight", type=float, default=0.8)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--skip-test", action="store_true", help="only train + validate")
ARGS = P.parse_args()

T0 = time.time()
def log(m): print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)

# ----------------------------------------------------------------------------- normalization
NAME_ABBR = {
    "pvt": "private", "prv": "private", "ltd": "limited", "lmtd": "limited", "corp": "corporation",
    "co": "company", "cos": "companies", "inc": "incorporated", "intl": "international",
    "mfg": "manufacturing", "mfrs": "manufacturers", "svc": "service", "svcs": "services",
    "srvcs": "services", "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "natl": "national", "dept": "department", "grp": "group", "hldgs": "holdings",
    "ent": "enterprises", "entp": "enterprises", "ind": "industries", "inds": "industries",
    "mgmt": "management", "sys": "systems", "sol": "solutions", "solns": "solutions",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc",
    "pllc", "opc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "selarl", "gmbh", "ag",
    "bv", "nv", "pty", "pte", "the", "and", "dba", "of",
}
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "pl": "place", "sq": "square", "cir": "circle", "trl": "trail", "ter": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "opp": "opposite", "nr": "near", "ngr": "nagar",
    "mkt": "market", "rly": "railway", "stn": "station", "sec": "sector", "sect": "sector",
    "no": "number", "chs": "chowk", "fg": "faubourg", "rte": "route", "imm": "immeuble",
}
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_POSTAL = re.compile(r"^\d{5,6}$")

def _clean(s):
    s = unidecode(s or "").lower().replace("&", " and ").replace("'", "").replace(".", "")
    return _NONALNUM.sub(" ", s).split()

def norm_name(s):
    toks = [NAME_ABBR.get(t, t) for t in _clean(s)]
    core = [t for t in toks if t not in LEGAL] or toks
    return " ".join(toks), " ".join(core)

def norm_addr(s):
    return " ".join(ADDR_ABBR.get(t, t) for t in _clean(s))

def _norm_chunk(args):
    names, addrs = args
    out_f, out_c, out_a = [], [], []
    for n, a in zip(names, addrs):
        f, c = norm_name(n); out_f.append(f); out_c.append(c); out_a.append(norm_addr(a))
    return out_f, out_c, out_a

def normalize_frame(df, workers):
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    step = 50_000
    chunks = [(names[i:i+step], addrs[i:i+step]) for i in range(0, len(names), step)]
    F, C, A = [], [], []
    with Pool(workers) as pool:
        for f, c, a in pool.imap(_norm_chunk, chunks):
            F += f; C += c; A += a
    df["n_full"], df["n_core"], df["a_norm"] = F, C, A
    return df

# ----------------------------------------------------------------------------- IO
def find_file(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not found under {root}")

def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    for c in ["entity_id", "business_name", "business_address", "country"]:
        if c not in df.columns: df[c] = ""
    df = df[["entity_id", "business_name", "business_address", "country"]]
    df = df[df["entity_id"] != ""].drop_duplicates("entity_id").reset_index(drop=True)
    df["country"] = df["country"].str.strip().str.upper()
    return df

def load_split(root, prefix, workers):
    s1 = read_tsv(find_file(root, f"{prefix}_source1.tsv"))
    s2 = read_tsv(find_file(root, f"{prefix}_source2.tsv"))
    s3 = read_tsv(find_file(root, f"{prefix}_source3.tsv"))
    s2["src"] = 0; s3["src"] = 1
    tg = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    log(f"  {prefix}: S1={len(s1):,}  targets(S2+S3)={len(tg):,}")
    s1 = normalize_frame(s1, workers); tg = normalize_frame(tg, workers)
    s1 = s1.drop(columns=["business_name", "business_address"])
    tg = tg.drop(columns=["business_name", "business_address"]); gc.collect()
    return s1, tg

# ----------------------------------------------------------------------------- vectors & blocking
HV_CHAR = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**21,
                            alternate_sign=False, norm=None, dtype=np.float32)
HV_WORD = HashingVectorizer(token_pattern=r"[a-z0-9]+", n_features=2**21,
                            alternate_sign=False, norm=None, dtype=np.float32)

def _hash_chunk(args):
    kind, texts = args
    return (HV_CHAR if kind == "char" else HV_WORD).transform(texts)

def hashed_tfidf(texts, kind, workers, max_df, step=200_000):
    jobs = [(kind, texts[s:s+step]) for s in range(0, len(texts), step)]
    with Pool(workers) as pool:
        X = sp.vstack(list(pool.imap(_hash_chunk, jobs)), format="csr")
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    idf[(df < 2) | (df > max_df * n)] = 0.0
    X.data = (1.0 + np.log(X.data)).astype(np.float32) * idf[X.indices]
    X.eliminate_zeros()
    return normalize(X, copy=False)

def country_vectors(core, addr, addr_w, workers):
    Xc = hashed_tfidf(core, "char", workers, 0.05)
    Xw = hashed_tfidf(core, "word", workers, 0.02)
    Xa = hashed_tfidf(addr, "word", workers, 0.02)
    Xm = normalize(sp.hstack([Xw, Xa * addr_w], format="csr")).astype(np.float32)
    del Xw, Xa; gc.collect()
    return Xc, Xm

def _topn(A, B, k, threads):
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    BT = B.T.tocsr()
    if HAS_TOPN:
        C = sp_matmul_topn(A, BT, top_n=k, threshold=1e-6, n_threads=threads).tocoo()
        return C.row.astype(np.int64), C.col.astype(np.int64)
    rows, cols = [], []                       # slow fallback
    for s in range(0, A.shape[0], 5000):
        C = (A[s:s+5000] @ BT).tocsr()
        for r in range(C.shape[0]):
            a, b = C.indptr[r], C.indptr[r+1]
            if a == b: continue
            sel = np.arange(a, b) if b - a <= k else a + np.argpartition(-C.data[a:b], k)[:k]
            rows.append(np.full(len(sel), s + r)); cols.append(C.indices[sel])
    if not rows: return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)

def rowdot(A, B, i, j, chunk=500_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        out[s:s+chunk] = np.asarray(A[i[s:s+chunk]].multiply(B[j[s:s+chunk]]).sum(1)).ravel()
    return out

def _topn_chunked(A, BT, k, threads, step=300_000):
    R, C = [], []
    for s0 in range(0, A.shape[0], step):
        if HAS_TOPN:
            M = sp_matmul_topn(A[s0:s0+step], BT, top_n=k, threshold=1e-6, n_threads=threads).tocoo()
            R.append(M.row.astype(np.int64) + s0); C.append(M.col.astype(np.int64))
        else:
            r, c = _topn(A[s0:s0+step], BT.T.tocsr(), k, threads); R.append(r + s0); C.append(c)
    return np.concatenate(R), np.concatenate(C)

def block(s1, tg, per_country, a):
    """Per-country hashed TF-IDF blocking. Empty-country S1 are searched in every country."""
    c1 = s1["country"].values; ct = tg["country"].values; srcs = tg["src"].values
    s1c, s1a = s1["n_core"].values, s1["a_norm"].values
    tgc, tga = tg["n_core"].values, tg["a_norm"].values
    groups = [g for g in np.unique(c1) if g != ""] if per_country else ["__ALL__"]
    if not groups: groups = ["__ALL__"]
    frames = []
    for g in groups:
        tg0 = time.time()
        if g == "__ALL__":
            ii = np.arange(len(s1)); jj = np.arange(len(tg))
        else:
            ii = np.where((c1 == g) | (c1 == ""))[0]; jj = np.where((ct == g) | (ct == ""))[0]
        if len(ii) == 0 or len(jj) == 0: continue
        n1 = len(ii)
        core = np.concatenate([s1c[ii], tgc[jj]]); addr = np.concatenate([s1a[ii], tga[jj]])
        Xc, Xm = country_vectors(core, addr, a.addr_weight, a.workers); del core, addr
        log(f"    [{g}] vectors built: S1={n1:,} targets={len(jj):,} ({time.time()-tg0:.0f}s)")
        I, J = [], []
        for sv in (0, 1):
            loc = np.where(srcs[jj] == sv)[0]
            if len(loc) == 0: continue
            for name, X, k in (("char", Xc, a.k_char), ("comb", Xm, a.k_comb)):
                BT = X[n1:][loc].T.tocsr()
                r, c = _topn_chunked(X[:n1], BT, k, a.workers); del BT
                I.append(r); J.append(loc[c])
                log(f"    [{g}] src=S{sv+2} {name}: {len(r):,} pairs ({time.time()-tg0:.0f}s)")
        I = np.concatenate(I); J = np.concatenate(J)
        key = np.unique(I * len(jj) + J); I, J = key // len(jj), key % len(jj)
        cc = rowdot(Xc[:n1], Xc[n1:], I, J); cm = rowdot(Xm[:n1], Xm[n1:], I, J)
        del Xc, Xm; gc.collect()
        frames.append(pd.DataFrame({"i": ii[I], "j": jj[J], "cc": cc, "cm": cm}))
        del I, J, cc, cm, key; gc.collect()
    df = pd.concat(frames, ignore_index=True); del frames
    df = df.drop_duplicates(["i", "j"])
    best = np.maximum(df["cc"].values, df["cm"].values)
    order = np.lexsort((-best, df["i"].values)); df = df.iloc[order].reset_index(drop=True)
    I = df["i"].values
    start = np.r_[0, np.flatnonzero(np.diff(I)) + 1]
    rank = np.arange(len(I)) - np.repeat(start, np.diff(np.r_[start, len(I)]))
    df = df[rank < a.max_cand].reset_index(drop=True)
    df["src"] = srcs[df["j"].values].astype(np.int8)
    return df

def add_context(df):
    g = df.groupby("i")
    df["n_cand"] = g["j"].transform("size").astype(np.float32)
    for c in ("cc", "cm"):
        df[f"rk_{c}"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"gap_{c}"] = (g[c].transform("max") - df[c]).astype(np.float32)
    df["rk_cm_src"] = df.groupby(["i", "src"])["cm"].rank(ascending=False, method="min").astype(np.float32)
    h = df.groupby("j")
    df["rev_n"] = h["i"].transform("size").astype(np.float32)
    for c in ("cc", "cm"):
        df[f"rev_rk_{c}"] = h[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"rev_gap_{c}"] = (h[c].transform("max") - df[c]).astype(np.float32)
    return df

# ----------------------------------------------------------------------------- pairwise features
G = {}
STR_FEATS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "nf_ratio", "n_exact", "n_jac",
             "n_idf_jac", "n_idf_min", "n_first", "n_acro", "n_len1", "n_len2", "n_ntok1", "n_ntok2",
             "n_digits", "a_missing", "a_tset", "a_ratio", "a_partial", "a_idf_jac", "a_idf_min",
             "a_postal", "a_num_jac", "a_name_in"]

def build_idf(texts):
    df = Counter()
    for t in texts: df.update(set(t.split()))
    n = len(texts)
    return {w: float(np.log(n / c)) for w, c in df.items()}

def _idf_sims(A, B, idf, dflt):
    if not A or not B: return 0.0, 0.0
    wa = sum(idf.get(t, dflt) for t in A); wb = sum(idf.get(t, dflt) for t in B)
    wi = sum(idf.get(t, dflt) for t in A & B)
    wu = wa + wb - wi
    return (wi / wu if wu else 0.0), (wi / min(wa, wb) if min(wa, wb) else 0.0)

def _acro(ta, tb):
    if len(ta) >= 2 and len(tb) == 1 and len(tb[0]) >= 2:
        return float("".join(t[0] for t in ta) == tb[0])
    if len(tb) >= 2 and len(ta) == 1 and len(ta[0]) >= 2:
        return float("".join(t[0] for t in tb) == ta[0])
    return 0.0

def _feat_chunk(args):
    ii, jj = args
    s1c, s1f, s1a = G["s1c"], G["s1f"], G["s1a"]
    tgc, tgf, tga = G["tgc"], G["tgf"], G["tga"]
    nidf, aidf, nd, ad = G["nidf"], G["aidf"], G["ndflt"], G["adflt"]
    out = np.zeros((len(ii), len(STR_FEATS)), np.float32)
    for r in range(len(ii)):
        a, b = s1c[ii[r]], tgc[jj[r]]; x, y = s1a[ii[r]], tga[jj[r]]
        la, lb = a.split(), b.split(); A, B = set(la), set(lb)
        f = out[r]
        if a and b:
            f[0] = fuzz.ratio(a, b); f[1] = fuzz.partial_ratio(a, b)
            f[2] = fuzz.token_sort_ratio(a, b); f[3] = fuzz.token_set_ratio(a, b)
            f[4] = JaroWinkler.normalized_similarity(a, b)
            f[5] = fuzz.ratio(s1f[ii[r]], tgf[jj[r]])
            f[6] = float(a == b)
            f[7] = len(A & B) / len(A | B)
            f[8], f[9] = _idf_sims(A, B, nidf, nd)
            f[10] = float(la[0] == lb[0]); f[11] = _acro(la, lb)
        f[12], f[13], f[14], f[15] = len(a), len(b), len(la), len(lb)
        da = {t for t in A if t.isdigit()}; db = {t for t in B if t.isdigit()}
        f[16] = -1.0 if not (da or db) else float(da == db)
        if not x or not y:
            f[17] = 2.0 if (not x and not y) else 1.0
            f[18:26] = -1.0
        else:
            X, Y = set(x.split()), set(y.split())
            f[18] = fuzz.token_set_ratio(x, y); f[19] = fuzz.ratio(x, y); f[20] = fuzz.partial_ratio(x, y)
            f[21], f[22] = _idf_sims(X, Y, aidf, ad)
            px = {t for t in X if _POSTAL.match(t)}; py = {t for t in Y if _POSTAL.match(t)}
            f[23] = -1.0 if not (px and py) else float(len(px & py) > 0)
            nx = {t for t in X if t.isdigit()}; ny = {t for t in Y if t.isdigit()}
            f[24] = -1.0 if not (nx and ny) else len(nx & ny) / len(nx | ny)
            f[25] = float(bool(A) and A <= Y) if b else 0.0
    return out

CTX_FEATS = ["cc", "cm", "src", "n_cand", "rk_cc", "gap_cc", "rk_cm", "gap_cm", "rk_cm_src",
             "rev_n", "rev_rk_cc", "rev_gap_cc", "rev_rk_cm", "rev_gap_cm", "same_cty"]
FEATURES = CTX_FEATS + STR_FEATS

def compute_features(df, workers, chunk=20_000):
    ii, jj = df["i"].values, df["j"].values
    jobs = [(ii[s:s+chunk], jj[s:s+chunk]) for s in range(0, len(ii), chunk)]
    with Pool(workers) as pool:
        mats = list(pool.imap(_feat_chunk, jobs, chunksize=1))
    S = np.vstack(mats) if mats else np.zeros((0, len(STR_FEATS)), np.float32)
    C = df[CTX_FEATS].to_numpy(np.float32)
    return np.hstack([C, S])

def set_globals(s1, tg):
    G.update(s1c=s1["n_core"].tolist(), s1f=s1["n_full"].tolist(), s1a=s1["a_norm"].tolist(),
             tgc=tg["n_core"].tolist(), tgf=tg["n_full"].tolist(), tga=tg["a_norm"].tolist())
    G["nidf"] = build_idf(G["s1c"] + G["tgc"]); G["ndflt"] = float(np.log(len(G["s1c"]) + len(G["tgc"])))
    G["aidf"] = build_idf(G["s1a"] + G["tga"]); G["adflt"] = float(np.log(len(G["s1a"]) + len(G["tga"])))

# ----------------------------------------------------------------------------- decision layer
def exclusivity(j, p):
    d = pd.DataFrame({"j": j, "p": p})
    rk = d.groupby("j")["p"].rank(ascending=False, method="first")
    top1 = d.groupby("j")["p"].transform("max")
    top2 = d["p"].where(rk == 2).groupby(d["j"]).transform("max").fillna(0.0)
    other = np.where(rk.values == 1, top2.values, top1.values)
    return np.minimum(p, 1.0 - other)

def decide(i, p, n_mc=256, min_p=0.02, seed=0):
    """Pick, per S1, the candidate subset maximizing expected F0.5 (empty set allowed)."""
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(p), bool)
    idx = np.where(p >= min_p)[0]
    if len(idx) == 0: return sel
    order = idx[np.lexsort((-p[idx], i[idx]))]
    ii, pp = i[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ii)) + 1]; ends = np.r_[starts[1:], len(ii)]
    size = ends - starts
    one = starts[size == 1]
    sel[order[one[pp[one] > 0.5]]] = True           # closed form: E[F]=p vs 1-p
    for s, e in zip(starts[size > 1], ends[size > 1]):
        q = pp[s:e]
        if q[0] < 0.15: continue
        y = rng.random((n_mc, len(q))) < q
        tot = y.sum(1)
        tp = np.cumsum(y, 1); k = np.arange(1, len(q) + 1)
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * (tot[:, None] - tp) + (k - tp)), 0.0).mean(0)
        b = int(np.argmax(f))
        if f[b] > (tot == 0).mean():
            sel[order[s:s + b + 1]] = True
    return sel

def macro_f05(eval_i, sel_i, sel_j, gt_keys, gt_cnt, nt, n1):
    hit = np.isin(sel_i * nt + sel_j, gt_keys).astype(np.float64)
    pc = np.bincount(sel_i, minlength=n1)[eval_i]
    tp = np.bincount(sel_i, weights=hit, minlength=n1)[eval_i]
    gc_ = gt_cnt[eval_i]
    f = ((gc_ == 0) & (pc == 0)).astype(np.float64)
    m = tp > 0
    Pr = tp[m] / pc[m]; Rc = tp[m] / gc_[m]
    f[m] = 1.25 * Pr * Rc / (0.25 * Pr + Rc)
    return f.mean()

# ----------------------------------------------------------------------------- main
def main():
    a = ARGS
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.work_dir, exist_ok=True)
    root = a.data_dir
    if root is None:
        for c in ["/mnt/custom-file-systems/s3/shared/crash-canyon", "/shared/crash-canyon", "dataset", "."]:
            if os.path.isdir(c): root = c; break
    log(f"data root = {root} | workers={a.workers} | sparse_dot_topn={HAS_TOPN}")
    rng = np.random.default_rng(a.seed)

    # ================= TRAIN =================
    log("TRAIN: load + normalize")
    s1, tg = load_split(root, "train", a.workers)
    n1, nt = len(s1), len(tg)
    gt = pd.read_csv(find_file(root, "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, quoting=csv.QUOTE_NONE)
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gt["m"] = gt["m"].str.strip(); gt = gt[gt["m"] != ""]
    gi = pd.Index(s1["entity_id"]).get_indexer(gt["source1_entity_id"])
    gj = pd.Index(tg["entity_id"]).get_indexer(gt["m"])
    ok = (gi >= 0) & (gj >= 0); gi, gj = gi[ok].astype(np.int64), gj[ok].astype(np.int64)
    gt_keys = np.unique(gi * nt + gj)
    gt_cnt = np.bincount(gi, minlength=n1)
    log(f"  GT pairs={len(gt_keys):,} | S1 with matches={int((gt_cnt>0).sum()):,} "
        f"({(gt_cnt>0).mean()*100:.1f}%) | mean matches (non-singleton)={gt_cnt[gt_cnt>0].mean():.2f}")

    cross = (s1["country"].values[gi] != tg["country"].values[gj]).mean()
    per_country = cross < 0.002
    multi = (np.bincount(gj, minlength=nt) > 1).sum() / max(1, len(np.unique(gj)))
    one_to_one = multi < 0.005
    log(f"  cross-country GT rate={cross*100:.3f}% -> per-country blocking={per_country}")
    log(f"  targets matched to >1 S1: {multi*100:.3f}% -> exclusivity={one_to_one}")

    log("TRAIN: vectors + blocking (all S1, needed for reverse-rank features)")
    cand = block(s1, tg, per_country, a); gc.collect()
    cand = add_context(cand)
    cand["same_cty"] = (s1["country"].values[cand["i"].values] == tg["country"].values[cand["j"].values]).astype(np.float32)
    cand["y"] = np.isin(cand["i"].values * nt + cand["j"].values, gt_keys).astype(np.int8)
    rec = cand["y"].sum() / len(gt_keys)
    log(f"  candidates={len(cand):,} ({len(cand)/n1:.1f}/S1) | BLOCKING RECALL={rec*100:.3f}%")

    perm = rng.permutation(n1)
    hold = np.sort(perm[:a.holdout_s1]); train = np.sort(perm[a.holdout_s1:a.holdout_s1 + a.train_s1])
    set_globals(s1, tg)
    tr = cand[np.isin(cand["i"].values, train)].reset_index(drop=True)
    ho = cand[np.isin(cand["i"].values, hold)].reset_index(drop=True)
    del cand; gc.collect()
    log(f"  features: train pairs={len(tr):,}  holdout pairs={len(ho):,}")
    Xtr = compute_features(tr, a.workers); Xho = compute_features(ho, a.workers)
    log("  features done")

    fit_s1 = rng.choice(train, int(len(train) * 0.85), replace=False)
    m_fit = np.isin(tr["i"].values, fit_s1)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=a.workers, seed=a.seed)
    dfit = lgb.Dataset(Xtr[m_fit], tr["y"].values[m_fit], feature_name=FEATURES)
    dcal = lgb.Dataset(Xtr[~m_fit], tr["y"].values[~m_fit], reference=dfit)
    model = lgb.train(params, dfit, 4000, valid_sets=[dcal],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    model.save_model(os.path.join(a.work_dir, "lgbm.txt"))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(model.predict(Xtr[~m_fit], num_iteration=model.best_iteration), tr["y"].values[~m_fit])
    imp = sorted(zip(model.feature_importance("gain"), FEATURES), reverse=True)[:12]
    log("  top features: " + ", ".join(f"{n}" for _, n in imp))

    # ---- holdout evaluation, exactly like leaderboard ----
    p_ho = iso.predict(model.predict(Xho, num_iteration=model.best_iteration))
    hi, hj = ho["i"].values.astype(np.int64), ho["j"].values.astype(np.int64)
    base = p_ho >= 0.5
    log(f"  HOLDOUT macro-F0.5  threshold@0.5         = {macro_f05(hold, hi[base], hj[base], gt_keys, gt_cnt, nt, n1):.6f}")
    best = (-1, None)
    for excl in ([False, True] if one_to_one else [False]):
        pe = exclusivity(hj, p_ho) if excl else p_ho
        for gamma in (0.7, 0.85, 1.0, 1.2, 1.5):
            s = decide(hi, pe ** gamma)
            f = macro_f05(hold, hi[s], hj[s], gt_keys, gt_cnt, nt, n1)
            log(f"  HOLDOUT macro-F0.5  expF0.5 excl={excl!s:5} gamma={gamma:<4} = {f:.6f}")
            if f > best[0]: best = (f, (excl, gamma))
    excl, gamma = best[1]
    ceiling = ((gt_cnt[hold] == 0) | np.isin(hold, np.unique(hi[ho['y'].values == 1]))).mean()
    log(f"  BEST holdout={best[0]:.6f} (excl={excl}, gamma={gamma}) | S1 with >=1 reachable match or singleton={ceiling*100:.2f}%")
    json.dump({"holdout_f05": best[0], "excl": bool(excl), "gamma": gamma, "blocking_recall": float(rec),
               "per_country": bool(per_country)}, open(os.path.join(a.work_dir, "report.json"), "w"), indent=2)
    del s1, tg, tr, ho, Xtr, Xho; G.clear(); gc.collect()
    if a.skip_test: return

    # ================= TEST =================
    log("TEST: load + normalize")
    s1, tg = load_split(root, "test", a.workers)
    n1, nt = len(s1), len(tg)
    cand = block(s1, tg, per_country, a); gc.collect()
    cand = add_context(cand)
    cand["same_cty"] = (s1["country"].values[cand["i"].values] == tg["country"].values[cand["j"].values]).astype(np.float32)
    log(f"  candidates={len(cand):,} ({len(cand)/n1:.1f}/S1) | countries: {s1['country'].value_counts().to_dict()}")
    set_globals(s1, tg)
    probs = np.empty(len(cand), np.float32)
    CH = 2_000_000
    for s in range(0, len(cand), CH):
        X = compute_features(cand.iloc[s:s+CH], a.workers)
        probs[s:s+CH] = iso.predict(model.predict(X, num_iteration=model.best_iteration))
        log(f"  scored {min(s+CH, len(cand)):,}/{len(cand):,}")
    ci, cj = cand["i"].values.astype(np.int64), cand["j"].values.astype(np.int64)
    pe = exclusivity(cj, probs) if excl else probs
    sel = decide(ci, pe.astype(np.float64) ** gamma)

    ids1, idst = s1["entity_id"].values, tg["entity_id"].values
    def write(path, col, ii, jj):
        d = pd.DataFrame({"i": ii, "t": idst[jj]}).groupby("i")["t"].agg(",".join)
        lists = pd.Series("", index=np.arange(n1)); lists.loc[d.index] = d.values
        pd.DataFrame({"source1_entity_id": ids1, col: lists.values}).to_csv(
            path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    write(os.path.join(a.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", ci, cj)
    write(os.path.join(a.out_dir, "matching_results.tsv"), "matched_entity_ids", ci[sel], cj[sel])
    nm = np.bincount(ci[sel], minlength=n1)
    log(f"  DONE: S1 rows={n1:,} | matched S1={int((nm>0).sum()):,} ({(nm>0).mean()*100:.1f}%) "
        f"| total links={int(sel.sum()):,} | files in {a.out_dir}/")

if __name__ == "__main__":
    main()
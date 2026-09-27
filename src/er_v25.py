#!/usr/bin/env python3
"""
Business Entity Resolution — V25
==================================================
V22 = V21 + (1) blocking cache on disk (re-runs skip blocking)
            + (2) reverse blocking: every S2/S3 record also retrieves its top S1s
            + (3) stage-2 model on candidate-group stats + cluster coherence
            + (4) numpy group stats (lower RAM than pandas groupby)
V23 = V22 + learned token equivalences (mined from train GT pairs) as features
            + holdout ERROR REPORT (loss by category/country + example dump)
V24 = V23 + LEARNED CANDIDATE PRUNING (small candidate_pairs.tsv, as the updated rules reward)
            + faster/wider blocking defaults for the full official dataset
            (char max_df 0.01, reverse blocking on char+comb, k_rev 3).
V25 = V24 + data-driven fixes from real examples:
            address-only blocking channel (fwd+rev) for native-script / domain names,
            state-name normalization (US + India incl. native scripts), saint->street,
            leading zeros, dots->spaces, honorific/noise tokens, native-script + domain features.

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
P.add_argument("--train-s1", type=int, default=100_000, help="uniform S1 sample used to train the model")
P.add_argument("--holdout-s1", type=int, default=100_000, help="uniform S1 sample for honest validation")
P.add_argument("--k-char", type=int, default=10, help="top-K per source, char channel")
P.add_argument("--k-comb", type=int, default=12, help="top-K per source, name+address channel")
P.add_argument("--k-addr", type=int, default=6, help="top-K per source, address-only channel (0=off)")
P.add_argument("--max-cand", type=int, default=40, help="max candidates per S1 after merge")
P.add_argument("--addr-weight", type=float, default=0.8)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--skip-test", action="store_true", help="only train + validate")
P.add_argument("--k-rev", type=int, default=3, help="reverse blocking: top-K S1 per target (0=off)")
P.add_argument("--no-rev-char", action="store_true", help="disable reverse blocking on the char channel")
P.add_argument("--rev-cap", type=int, default=20, help="max reverse-only candidates kept per S1")
P.add_argument("--char-max-df", type=float, default=0.01, help="drop char 3-grams in more than this share of docs (lower=faster)")
P.add_argument("--folds", type=int, default=3)
P.add_argument("--no-cache", action="store_true")
P.add_argument("--map-min-count", type=int, default=15, help="min co-occurrences to learn a token equivalence")
P.add_argument("--prune-keep", type=float, default=0.998, help="share of blocked true pairs the pruner must keep")
P.add_argument("--prune-max", type=int, default=15, help="max candidates per S1 after pruning")
ARGS = P.parse_args()
ARGS.rev_char = not ARGS.no_rev_char

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
    # transliterated legal words (Devanagari / Bengali / Malayalam via unidecode)
    "praaivett": "private", "praaibhett": "private", "praivrrrr": "private", "praa": "private",
    "limittedd": "limited", "limirrrrdd": "limited", "li": "limited", "knpnii": "company",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc",
    "pllc", "opc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "selarl", "gmbh", "ag",
    "bv", "nv", "pty", "pte", "the", "and", "dba", "of",
    "smt", "shri", "sri", "dr", "mr", "mrs", "ms", "center", "centre", "www", "com", "net", "org",
}
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "pl": "place", "sq": "square", "cir": "circle", "trl": "trail", "ter": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "opp": "opposite", "nr": "near", "ngr": "nagar",
    "mkt": "market", "rly": "railway", "stn": "station", "sec": "sector", "sect": "sector",
    "chs": "chowk", "fg": "faubourg", "rte": "route", "imm": "immeuble",
    "saint": "street", "suite": "unit", "ste": "unit", "apartment": "unit", "keralam": "kl",
}
ADDR_DROP = {"no", "number", "h", "hno", "city", "na", "the", "of"}
US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"}
IN_STATES = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py"}
NATIVE_STATES = {"महाराष्ट्र": " mh ", "दिल्ली": " dl ", "पश्चिम बंगाल": " wb ", "পশ্চিমবঙ্গ": " wb ",
    "ಕರ್ನಾಟಕ": " ka ", "తెలంగాణ": " tg ", "ఆంధ్ర ప్రదేశ్": " ap ", "தமிழ்நாடு": " tn ", "കേരളം": " kl ",
    "ગુજરાત": " gj ", "उत्तर प्रदेश": " up ", "मध्य प्रदेश": " mp ", "राजस्थान": " rj ", "बिहार": " br ",
    "ਪੰਜਾਬ": " pb ", "ଓଡ଼ିଶା": " od ", "हरियाणा": " hr ", "झारखंड": " jh ", "छत्तीसगढ़": " cg ",
    "उत्तराखंड": " uk ", "অসম": " as ", "गोवा": " ga ", "हिमाचल प्रदेश": " hp "}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, list(US_STATES) + list(IN_STATES)), key=len, reverse=True)) + r")\b")
_STATE_MAP = {**US_STATES, **IN_STATES}
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_POSTAL = re.compile(r"^\d{5,6}$")

def _clean(s, addr=False):
    s = s or ""
    if addr:
        for k, v in NATIVE_STATES.items():
            if k in s: s = s.replace(k, v)
    s = unidecode(s).lower().replace("&", " and ").replace("'", "").replace("n/a", " ").replace(".", " ")
    s = _NONALNUM.sub(" ", s)
    if addr: s = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], s)
    return [t.lstrip("0") or "0" if t.isdigit() else t for t in s.split()]

def name_flags(s):
    s = s or ""
    letters = [ch for ch in s if ch.isalpha()]
    nat = sum(1 for ch in letters if ord(ch) > 0x24F) / len(letters) if letters else 0.0
    low = s.lower()
    dom = float((".com" in low) or ("www" in low) or (len(s) > 12 and " " not in s.strip()))
    return nat, dom

def norm_name(s):
    toks = [NAME_ABBR.get(t, t) for t in _clean(s)]
    core = [t for t in toks if t not in LEGAL] or toks
    return " ".join(toks), " ".join(core)

def norm_addr(s):
    out = []
    for t in _clean(s, addr=True):
        t = ADDR_ABBR.get(t, t)
        if t not in ADDR_DROP: out.append(t)
    return " ".join(out)

def _norm_chunk(args):
    names, addrs = args
    out_f, out_c, out_a, out_n, out_d = [], [], [], [], []
    for n, a in zip(names, addrs):
        f, c = norm_name(n); out_f.append(f); out_c.append(c); out_a.append(norm_addr(a))
        nt, dm = name_flags(n); out_n.append(nt); out_d.append(dm)
    return out_f, out_c, out_a, out_n, out_d

def normalize_frame(df, workers):
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    step = 50_000
    chunks = [(names[i:i+step], addrs[i:i+step]) for i in range(0, len(names), step)]
    F, C, A, N, Dm = [], [], [], [], []
    with Pool(workers) as pool:
        for f, c, a, nt, dm in pool.imap(_norm_chunk, chunks):
            F += f; C += c; A += a; N += nt; Dm += dm
    df["n_full"], df["n_core"], df["a_norm"] = F, C, A
    df["nat"] = np.asarray(N, np.float32); df["dom"] = np.asarray(Dm, np.float32)
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
    Xc = hashed_tfidf(core, "char", workers, ARGS.char_max_df)
    Xw = hashed_tfidf(core, "word", workers, 0.02)
    Xa = hashed_tfidf(addr, "word", workers, 0.02)
    Xm = normalize(sp.hstack([Xw, Xa * addr_w], format="csr")).astype(np.float32)
    del Xw; gc.collect()
    return Xc, Xm, Xa.astype(np.float32)

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
        Xc, Xm, Xa = country_vectors(core, addr, a.addr_weight, a.workers); del core, addr
        log(f"    [{g}] vectors built: S1={n1:,} targets={len(jj):,} ({time.time()-tg0:.0f}s)")
        I, J, F = [], [], []
        S1T = {"comb": Xm[:n1].T.tocsr()} if a.k_rev > 0 else {}
        if a.k_rev > 0 and a.rev_char: S1T["char"] = Xc[:n1].T.tocsr()
        if a.k_rev > 0 and a.k_addr > 0: S1T["addr"] = Xa[:n1].T.tocsr()
        MATS = {"char": Xc, "comb": Xm, "addr": Xa}
        for sv in (0, 1):
            loc = np.where(srcs[jj] == sv)[0]
            if len(loc) == 0: continue
            for name, X, k in (("char", Xc, a.k_char), ("comb", Xm, a.k_comb), ("addr", Xa, a.k_addr)):
                if k <= 0: continue
                BT = X[n1:][loc].T.tocsr()
                r, c = _topn_chunked(X[:n1], BT, k, a.workers); del BT
                I.append(r); J.append(loc[c]); F.append(np.ones(len(r), np.int8))
                log(f"    [{g}] src=S{sv+2} {name}: {len(r):,} pairs ({time.time()-tg0:.0f}s)")
            for name, BT in S1T.items():
                X = MATS[name]
                r, c = _topn_chunked(X[n1:][loc], BT, a.k_rev, a.workers)
                I.append(c); J.append(loc[r]); F.append(np.zeros(len(r), np.int8))
                log(f"    [{g}] src=S{sv+2} REVERSE {name}: {len(r):,} pairs ({time.time()-tg0:.0f}s)")
        del S1T
        I = np.concatenate(I); J = np.concatenate(J); F = np.concatenate(F)
        key = I * len(jj) + J
        fwd_keys = np.unique(key[F == 1]); key = np.unique(key)
        fwd = np.isin(key, fwd_keys).astype(np.int8); del fwd_keys, F
        I, J = key // len(jj), key % len(jj)
        cc = rowdot(Xc[:n1], Xc[n1:], I, J); cm = rowdot(Xm[:n1], Xm[n1:], I, J)
        ca = rowdot(Xa[:n1], Xa[n1:], I, J)
        del Xc, Xm, Xa, MATS; gc.collect()
        frames.append(pd.DataFrame({"i": ii[I], "j": jj[J], "cc": cc, "cm": cm, "ca": ca, "fwd": fwd}))
        del I, J, cc, cm, ca, key, fwd; gc.collect()
    df = pd.concat(frames, ignore_index=True); del frames
    df = df.sort_values("fwd", ascending=False).drop_duplicates(["i", "j"]).reset_index(drop=True)
    best = np.maximum(np.maximum(df["cc"].values, df["cm"].values), df["ca"].values)
    rk_all, _, _ = grp_stats(df["i"].values, best)
    rev_only = df["fwd"].values == 0
    rk_rev = np.full(len(df), 1e9, np.float32)
    if rev_only.any():
        rk_rev[rev_only] = grp_stats(df["i"].values[rev_only], best[rev_only])[0]
    keep = (df["fwd"].values == 1) & (rk_all <= a.max_cand) | (rev_only & (rk_rev <= a.rev_cap))
    df = df[keep].reset_index(drop=True)
    df["src"] = srcs[df["j"].values].astype(np.int8)
    return df

def grp_stats(key, val):
    """Per-group (by key) descending rank (1=best), group max, group size — pure numpy."""
    order = np.lexsort((-val, key))
    ks = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    sizes = np.diff(np.r_[start, len(ks)])
    rank = np.empty(len(key), np.float32); rank[order] = np.arange(len(ks)) - np.repeat(start, sizes) + 1
    gmax = np.empty(len(key), np.float32); gmax[order] = np.repeat(val[order][start], sizes)
    size = np.empty(len(key), np.float32); size[order] = np.repeat(sizes, sizes)
    return rank, gmax, size

def add_context(df):
    i = df["i"].values; j = df["j"].values
    for c in ("cc", "cm", "ca"):
        v = df[c].values
        rk, mx, n = grp_stats(i, v)
        df[f"rk_{c}"] = rk; df[f"gap_{c}"] = mx - v
        if c == "cc": df["n_cand"] = n
        rk, mx, n = grp_stats(j, v)
        df[f"rev_rk_{c}"] = rk; df[f"rev_gap_{c}"] = mx - v
        if c == "cc": df["rev_n"] = n
    df["rk_cm_src"] = grp_stats(i * 2 + df["src"].values, df["cm"].values)[0]
    return df

# ----------------------------------------------------------------------------- pairwise features
G = {}
STR_FEATS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "nf_ratio", "n_exact", "n_jac",
             "n_idf_jac", "n_idf_min", "n_first", "n_acro", "n_len1", "n_len2", "n_ntok1", "n_ntok2",
             "n_digits", "a_missing", "a_tset", "a_ratio", "a_partial", "a_idf_jac", "a_idf_min",
             "a_postal", "a_num_jac", "a_name_in", "n_jac_map", "n_tset_map", "a_jac_map"]

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
        nm, am = G.get("nmap", {}), G.get("amap", {})
        if a and b:
            Am = {nm.get(t, t) for t in A}; Bm = {nm.get(t, t) for t in B}
            f[26] = len(Am & Bm) / len(Am | Bm)
            f[27] = fuzz.token_set_ratio(" ".join(sorted(Am)), " ".join(sorted(Bm)))
        if x and y:
            Xm = {am.get(t, t) for t in X}; Ym = {am.get(t, t) for t in Y}
            f[28] = len(Xm & Ym) / len(Xm | Ym)
        else:
            f[28] = -1.0
    return out

CTX_FEATS = ["cc", "cm", "ca", "rk_ca", "gap_ca", "rev_rk_ca", "rev_gap_ca",
             "nat1", "nat2", "dom1", "dom2", "src", "fwd", "n_cand", "rk_cc", "gap_cc", "rk_cm", "gap_cm", "rk_cm_src",
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

def set_globals(s1, tg, maps=None):
    if maps: G["nmap"], G["amap"] = maps
    G.update(s1c=s1["n_core"].tolist(), s1f=s1["n_full"].tolist(), s1a=s1["a_norm"].tolist(),
             tgc=tg["n_core"].tolist(), tgf=tg["n_full"].tolist(), tga=tg["a_norm"].tolist())
    G["nidf"] = build_idf(G["s1c"] + G["tgc"]); G["ndflt"] = float(np.log(len(G["s1c"]) + len(G["tgc"])))
    G["aidf"] = build_idf(G["s1a"] + G["tga"]); G["adflt"] = float(np.log(len(G["s1a"]) + len(G["tga"])))

# ----------------------------------------------------------------------------- stage 2
S2_BASE = ["cc", "cm", "ca", "nat1", "nat2", "rev_rk_cm", "rev_gap_cm", "fwd", "n_jw", "n_tset", "a_tset", "n_idf_min"]
S2_FEATS = ["p1", "lg1", "rk1", "gap1", "sum1", "cnt1", "other1", "n1", "rk1_src", "cnt1_src",
            "p_ref", "coh_ntset", "coh_nratio", "coh_atset"] + S2_BASE

def base_cols(X):
    return X[:, [FEATURES.index(c) for c in S2_BASE]].astype(np.float32)

def _coh_chunk(args):
    jj, rr = args
    tgc, tga = G["tgc"], G["tga"]
    out = np.full((len(jj), 3), -1.0, np.float32)
    for r in range(len(jj)):
        if rr[r] < 0: continue
        a, b = tgc[jj[r]], tgc[rr[r]]; x, y = tga[jj[r]], tga[rr[r]]
        if a and b: out[r, 0] = fuzz.token_set_ratio(a, b); out[r, 1] = fuzz.ratio(a, b)
        if x and y: out[r, 2] = fuzz.token_set_ratio(x, y)
    return out

def stage2_matrix(i, j, src, p1, base, workers, chunk=50_000):
    n = len(p1)
    order = np.lexsort((-p1, i)); ks = i[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]; sizes = np.diff(np.r_[start, n])
    ps, js = p1[order], j[order]
    def bcast(v):
        out = np.empty(n, np.float32); out[order] = np.repeat(v, sizes); return out
    rk = np.empty(n, np.float32); rk[order] = np.arange(n) - np.repeat(start, sizes) + 1
    top1 = ps[start]; has2 = sizes > 1
    second = np.where(has2, ps[np.minimum(start + 1, n - 1)], 0.0)
    j1 = js[start]; j2 = np.where(has2, js[np.minimum(start + 1, n - 1)], -1)
    gmax, gsec = bcast(top1), bcast(second)
    other = np.where(rk == 1, gsec, gmax)
    ref = np.where(rk == 1, bcast(j2), bcast(j1)).astype(np.int64)
    ref = np.where(rk == 1, ref, ref)  # top1 refs second-best; others ref top1
    sum1 = bcast(np.add.reduceat(ps, start)); cnt1 = bcast(np.add.reduceat((ps > 0.5).astype(np.float32), start))
    rks, _, _ = grp_stats(i * 2 + src, p1)
    o2 = np.lexsort((i * 2 + src,)); k2 = (i * 2 + src)[o2]
    st2 = np.r_[0, np.flatnonzero(np.diff(k2)) + 1]; sz2 = np.diff(np.r_[st2, n])
    cs = np.empty(n, np.float32); cs[o2] = np.repeat(np.add.reduceat((p1[o2] > 0.5).astype(np.float32), st2), sz2)
    jobs = [(j[s:s+chunk], ref[s:s+chunk]) for s in range(0, n, chunk)]
    with Pool(workers) as pool:
        coh = np.vstack(list(pool.imap(_coh_chunk, jobs, chunksize=1))) if jobs else np.zeros((0, 3), np.float32)
    pc = np.clip(p1, 1e-6, 1 - 1e-6)
    M = np.column_stack([p1, np.log(pc / (1 - pc)), rk, gmax - p1, sum1, cnt1, other, bcast(sizes.astype(np.float32)),
                         rks, cs, other, coh, base]).astype(np.float32)
    return M

# ----------------------------------------------------------------------------- learned candidate pruning
PRUNE_FEATS = ["cc", "cm", "ca", "rk_ca", "gap_ca", "rev_rk_ca", "rev_gap_ca",
               "nat1", "nat2", "dom1", "dom2", "src", "fwd", "n_cand", "rk_cc", "gap_cc", "rk_cm", "gap_cm", "rk_cm_src",
               "rev_n", "rev_rk_cc", "rev_gap_cc", "rev_rk_cm", "rev_gap_cm", "same_cty"]

def train_pruner(df, rng, workers, keep):
    """2-fold LightGBM on cheap blocking features. Returns models, OOF scores, threshold."""
    X = df[PRUNE_FEATS].to_numpy(np.float32); y = df["y"].values
    fold = rng.integers(0, 2, int(df["i"].max()) + 1)[df["i"].values]
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
               feature_fraction=0.9, verbose=-1, num_threads=workers)
    oof = np.zeros(len(df), np.float32); models = []
    for f in (0, 1):
        tr, va = fold != f, fold == f
        m = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), 600, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration); models.append(m)
    pos = np.sort(oof[y == 1])
    thr = float(pos[int((1 - keep) * len(pos))]) if len(pos) else 0.0
    return models, oof, thr

def pruner_score(models, df):
    X = df[PRUNE_FEATS].to_numpy(np.float32)
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)

def prune_mask(df, p0, thr, max_k):
    rk = grp_stats(df["i"].values.astype(np.int64), p0)[0]
    return (p0 >= thr) & (rk <= max_k)

# ----------------------------------------------------------------------------- learned equivalences
def mine_equivalences(left, right, gi, gj, min_count, rng, max_pairs=600_000, max_cluster=8):
    """From true pairs differing by exactly one token on each side, learn token substitutions."""
    idx = rng.choice(len(gi), min(len(gi), max_pairs), replace=False)
    cnt = Counter()
    for k in idx:
        A = set(left[gi[k]].split()); B = set(right[gj[k]].split())
        da, db = A - B, B - A
        if len(da) == 1 and len(db) == 1:
            u, v = next(iter(da)), next(iter(db))
            if u != v and not (u.isdigit() or v.isdigit()):
                cnt[(u, v) if u < v else (v, u)] += 1
    parent, size = {}, {}
    def find(x):
        parent.setdefault(x, x); size.setdefault(x, 1)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (u, v), c in cnt.most_common():
        if c < min_count: break
        ru, rv = find(u), find(v)
        if ru == rv or size[ru] + size[rv] > max_cluster: continue
        if ru > rv: ru, rv = rv, ru
        parent[rv] = ru; size[ru] += size[rv]
    mp = {t: find(t) for t in list(parent)}
    mp = {t: r for t, r in mp.items() if t != r}
    top = [f"{u}~{v}({c})" for (u, v), c in cnt.most_common(12)]
    return mp, top

def error_report(hold, hi, hj, p, sel, y, gi, gj, gt_cnt, s1, tg, n1, work_dir, rng):
    ys = y.astype(np.float64)
    pc = np.bincount(hi[sel], minlength=n1); tp = np.bincount(hi[sel], weights=ys[sel], minlength=n1)
    reach = np.bincount(hi, weights=ys, minlength=n1)
    g = gt_cnt
    f = ((g == 0) & (pc == 0)).astype(np.float64)
    m = tp > 0
    Pr = np.zeros(n1); Rc = np.zeros(n1)
    Pr[m] = tp[m] / pc[m]; Rc[m] = tp[m] / g[m]
    f[m] = 1.25 * Pr[m] * Rc[m] / (0.25 * Pr[m] + Rc[m])
    ev = np.zeros(n1, bool); ev[hold] = True
    cats = {
        "singleton_false_merge": ev & (g == 0) & (pc > 0),
        "match_not_in_candidates": ev & (g > 0) & (reach == 0),
        "predicted_empty_but_reachable": ev & (g > 0) & (reach > 0) & (pc == 0),
        "only_wrong_picks": ev & (g > 0) & (reach > 0) & (pc > 0) & (tp == 0),
        "partial_set": ev & (g > 0) & (tp > 0) & (f < 0.999999),
    }
    nh = len(hold); total = (1 - f[hold]).sum() / nh
    log(f"  ERROR REPORT (holdout): total F0.5 lost = {total:.5f}")
    for name, msk in cats.items():
        log(f"    {name:32s} entities={int(msk.sum()):7,}  F0.5 lost={(1 - f[msk]).sum() / nh:.5f}")
    cty = s1["country"].values
    for c in sorted(set(cty[hold])):
        mc = ev & (cty == c)
        log(f"    country {c:10s} n={int(mc.sum()):7,}  F0.5={f[mc].mean():.5f}")
    # example dump
    ids1, idst = s1["entity_id"].values, tg["entity_id"].values
    n1n, n1a, tn, ta = s1["n_full"].values, s1["a_norm"].values, tg["n_full"].values, tg["a_norm"].values
    gt_by = {}
    for name, msk in cats.items():
        pick = np.flatnonzero(msk)
        if len(pick) > 25: pick = rng.choice(pick, 25, replace=False)
        for i in pick: gt_by[i] = name
    rows = []
    order = np.argsort(-p)
    for k in order:
        i = hi[k]
        if i in gt_by and (y[k] or sel[k] or p[k] > 0.2):
            rows.append((gt_by[i], ids1[i], n1n[i], n1a[i], "cand", idst[hj[k]], tn[hj[k]], ta[hj[k]],
                         f"{p[k]:.3f}", int(y[k]), int(sel[k])))
    mg = np.isin(gi, np.fromiter(gt_by.keys(), np.int64))
    cand_keys = set((hi * len(tg) + hj).tolist())
    for i, j in zip(gi[mg], gj[mg]):
        if i * len(tg) + j not in cand_keys:
            rows.append((gt_by[i], ids1[i], n1n[i], n1a[i], "MISSED_TRUE", idst[j], tn[j], ta[j], "", 1, 0))
    rows.sort(key=lambda r: (r[0], r[1]))
    out = os.path.join(work_dir, "errors_holdout.tsv")
    pd.DataFrame(rows, columns=["category", "s1_id", "s1_name", "s1_addr", "kind", "target_id", "target_name",
                                "target_addr", "p", "is_true", "selected"]).to_csv(out, sep="\t", index=False)
    log(f"  error examples -> {out}")

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
def get_candidates(s1, tg, per_country, a, split):
    key = (f"v25_{split}_n{len(s1)}_t{len(tg)}_ka{a.k_addr}_kc{a.k_char}_km{a.k_comb}_kr{a.k_rev}_rc{int(a.rev_char)}"
           f"_rv{a.rev_cap}_mc{a.max_cand}_aw{a.addr_weight}_cdf{a.char_max_df}_pc{int(per_country)}")
    path = os.path.join(a.work_dir, f"cand_{key}.pkl")
    if os.path.exists(path) and not a.no_cache:
        log(f"  loading cached candidates: {path}")
        return pd.read_pickle(path)
    cand = block(s1, tg, per_country, a); gc.collect()
    cand = add_context(cand)
    cand["same_cty"] = (s1["country"].values[cand["i"].values] == tg["country"].values[cand["j"].values]).astype(np.float32)
    cand["nat1"] = s1["nat"].values[cand["i"].values]; cand["nat2"] = tg["nat"].values[cand["j"].values]
    cand["dom1"] = s1["dom"].values[cand["i"].values]; cand["dom2"] = tg["dom"].values[cand["j"].values]
    for c in cand.columns:
        if cand[c].dtype == np.float64: cand[c] = cand[c].astype(np.float32)
    cand.to_pickle(path, protocol=5); log(f"  cached candidates -> {path}")
    return cand

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
    cand = get_candidates(s1, tg, per_country, a, "train")
    cand["y"] = np.isin(cand["i"].values * nt + cand["j"].values, gt_keys).astype(np.int8)
    rec = cand["y"].sum() / len(gt_keys)
    rec_f = cand["y"].values[cand["fwd"].values == 1].sum() / len(gt_keys)
    log(f"  candidates={len(cand):,} ({len(cand)/n1:.1f}/S1) | BLOCKING RECALL={rec*100:.3f}% "
        f"(forward-only {rec_f*100:.3f}%, reverse adds {(rec-rec_f)*100:.3f}%)")
    for g in sorted(set(s1["country"].values)):
        m = s1["country"].values[cand["i"].values] == g
        ng = gt_cnt[s1["country"].values == g].sum()
        if ng: log(f"    recall [{g}] = {cand['y'].values[m].sum()/ng*100:.3f}%")

    perm = rng.permutation(n1)
    hold = np.sort(perm[:a.holdout_s1]); train = np.sort(perm[a.holdout_s1:a.holdout_s1 + a.train_s1])
    nmap, top_n = mine_equivalences(s1["n_core"].values, tg["n_core"].values, gi, gj, a.map_min_count, rng)
    amap, top_a = mine_equivalences(s1["a_norm"].values, tg["a_norm"].values, gi, gj, a.map_min_count, rng)
    log(f"  learned name equivalences={len(nmap):,} e.g. {', '.join(top_n[:8])}")
    log(f"  learned addr equivalences={len(amap):,} e.g. {', '.join(top_a[:8])}")
    MAPS = (nmap, amap)
    set_globals(s1, tg, MAPS)
    tr = cand[np.isin(cand["i"].values, train)].reset_index(drop=True)
    ho = cand[np.isin(cand["i"].values, hold)].reset_index(drop=True)
    del cand; gc.collect()
    log("  training candidate pruner (cheap blocking features only)...")
    pruners, oof0, pthr = train_pruner(tr, rng, a.workers, a.prune_keep)
    ktr = prune_mask(tr, oof0, pthr, a.prune_max)
    kho = prune_mask(ho, pruner_score(pruners, ho), pthr, a.prune_max)
    gt_h = gt_cnt[hold].sum()
    log(f"  PRUNE thr={pthr:.4f}: holdout cand/S1 {len(ho)/len(hold):.1f} -> {kho.sum()/len(hold):.2f} | "
        f"holdout recall {ho['y'].sum()/gt_h*100:.3f}% -> {ho['y'].values[kho].sum()/gt_h*100:.3f}%")
    tr = tr[ktr].reset_index(drop=True); ho = ho[kho].reset_index(drop=True); gc.collect()
    log(f"  features: train pairs={len(tr):,}  holdout pairs={len(ho):,}")
    Xtr = compute_features(tr, a.workers); Xho = compute_features(ho, a.workers)
    log("  features done")

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=a.workers, seed=a.seed)
    # ---- stage 1: K-fold by S1 -> OOF probabilities on train sample
    ytr = tr["y"].values; fold_of = rng.integers(0, a.folds, n1)[tr["i"].values]
    oof = np.zeros(len(tr), np.float32); models = []
    for f in range(a.folds):
        mt, mv = fold_of != f, fold_of == f
        m = lgb.train(params, lgb.Dataset(Xtr[mt], ytr[mt], feature_name=FEATURES), 4000,
                      valid_sets=[lgb.Dataset(Xtr[mv], ytr[mv])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[mv] = m.predict(Xtr[mv], num_iteration=m.best_iteration); models.append(m)
        m.save_model(os.path.join(a.work_dir, f"stage1_f{f}.txt"))
        log(f"  stage1 fold {f}: best_iter={m.best_iteration}")
    imp = sorted(zip(sum(m.feature_importance("gain") for m in models), FEATURES), reverse=True)[:12]
    log("  top stage1 features: " + ", ".join(n for _, n in imp))
    p1_of = lambda X: np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)

    # ---- stage 2: group stats + coherence on top of OOF p1
    ti, tj, ts = tr["i"].values.astype(np.int64), tr["j"].values.astype(np.int64), tr["src"].values.astype(np.int64)
    S2tr = stage2_matrix(ti, tj, ts, oof, base_cols(Xtr), a.workers)
    fit_s1 = rng.choice(train, int(len(train) * 0.85), replace=False)
    m_fit = np.isin(ti, fit_s1)
    model2 = lgb.train({**params, "num_leaves": 63}, lgb.Dataset(S2tr[m_fit], ytr[m_fit], feature_name=S2_FEATS), 3000,
                       valid_sets=[lgb.Dataset(S2tr[~m_fit], ytr[~m_fit])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
    model2.save_model(os.path.join(a.work_dir, "stage2.txt"))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(model2.predict(S2tr[~m_fit], num_iteration=model2.best_iteration), ytr[~m_fit])
    imp2 = sorted(zip(model2.feature_importance("gain"), S2_FEATS), reverse=True)[:10]
    log(f"  stage2 best_iter={model2.best_iteration}; top: " + ", ".join(n for _, n in imp2))
    predict_final = lambda S2: iso.predict(model2.predict(S2, num_iteration=model2.best_iteration))

    # ---- holdout evaluation, exactly like leaderboard ----
    hi, hj = ho["i"].values.astype(np.int64), ho["j"].values.astype(np.int64)
    p1_ho = p1_of(Xho)
    s1only = decide(hi, p1_ho.astype(np.float64))
    log(f"  HOLDOUT macro-F0.5  stage1 only (expF0.5)   = {macro_f05(hold, hi[s1only], hj[s1only], gt_keys, gt_cnt, nt, n1):.6f}")
    p_ho = predict_final(stage2_matrix(hi, hj, ho["src"].values.astype(np.int64), p1_ho, base_cols(Xho), a.workers))
    best = (-1, None)
    for excl in ([False, True] if one_to_one else [False]):
        pe = exclusivity(hj, p_ho) if excl else p_ho
        for gamma in (0.85, 1.0, 1.2, 1.5):
            s = decide(hi, pe ** gamma)
            f = macro_f05(hold, hi[s], hj[s], gt_keys, gt_cnt, nt, n1)
            log(f"  HOLDOUT macro-F0.5  stage2 excl={excl!s:5} gamma={gamma:<4} = {f:.6f}")
            if f > best[0]: best = (f, (excl, gamma))
    excl, gamma = best[1]
    ceiling = ((gt_cnt[hold] == 0) | np.isin(hold, np.unique(hi[ho['y'].values == 1]))).mean()
    log(f"  BEST holdout={best[0]:.6f} (excl={excl}, gamma={gamma}) | S1 with >=1 reachable match or singleton={ceiling*100:.2f}%")
    pe = exclusivity(hj, p_ho) if excl else p_ho
    s_best = decide(hi, pe ** gamma)
    np.savez(os.path.join(a.work_dir, "holdout_preds.npz"), i=hi, j=hj, p=p_ho, sel=s_best, y=ho["y"].values)
    error_report(hold, hi, hj, p_ho, s_best, ho["y"].values, gi, gj, gt_cnt, s1, tg, n1, a.work_dir, rng)
    json.dump({"holdout_f05": best[0], "excl": bool(excl), "gamma": gamma, "blocking_recall": float(rec),
               "per_country": bool(per_country)}, open(os.path.join(a.work_dir, "report.json"), "w"), indent=2)
    del s1, tg, tr, ho, Xtr, Xho; G.clear(); gc.collect()
    if a.skip_test: return

    # ================= TEST =================
    log("TEST: load + normalize")
    s1, tg = load_split(root, "test", a.workers)
    n1, nt = len(s1), len(tg)
    cand = get_candidates(s1, tg, per_country, a, "test")
    n_before = len(cand)
    cand = cand[prune_mask(cand, pruner_score(pruners, cand), pthr, a.prune_max)].reset_index(drop=True)
    log(f"  test pruning: {n_before/n1:.1f} -> {len(cand)/n1:.2f} candidates per S1")
    log(f"  candidates={len(cand):,} ({len(cand)/n1:.1f}/S1) | countries: {s1['country'].value_counts().to_dict()}")
    set_globals(s1, tg, MAPS)
    p1 = np.empty(len(cand), np.float32); base = np.empty((len(cand), len(S2_BASE)), np.float32)
    CH = 2_000_000
    for s in range(0, len(cand), CH):
        X = compute_features(cand.iloc[s:s+CH], a.workers)
        p1[s:s+CH] = p1_of(X); base[s:s+CH] = base_cols(X); del X
        log(f"  stage1 scored {min(s+CH, len(cand)):,}/{len(cand):,}")
    ci, cj = cand["i"].values.astype(np.int64), cand["j"].values.astype(np.int64)
    csrc = cand["src"].values.astype(np.int64); del cand; gc.collect()
    probs = predict_final(stage2_matrix(ci, cj, csrc, p1, base, a.workers)); del base; gc.collect()
    log("  stage2 scored")
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

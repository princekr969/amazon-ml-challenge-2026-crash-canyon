#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution - V12
============================================================
Pipeline:
  1. Normalization (names: initials collapse, abbreviation expansion, legal-suffix strip;
     addresses: abbreviation expansion, postal code + house number extraction)
  2. Blocking inside each country (country treated as an open set; verified on train GT):
       - TF-IDF top-K on name (word tokens + char 4-grams)          per source (S2, S3)
       - TF-IDF top-K on name+address combined vector                per source (S2, S3)
       - exact key joins: core name, compact name, sorted tokens, first-token+postal
     union -> capped at MAX_CANDS per S1  == exactly what the model scores (candidate_pairs.tsv)
  3. Stage-1 LightGBM on string/number/context features (incl. reverse-rank features)
  4. Stage-2 LightGBM on stage-1 probabilities within each S1's candidate group
  5. Isotonic calibration on out-of-fold predictions
  6. Decision layer tuned on OOF macro F0.5 (the exact leaderboard metric):
       threshold rule  vs  expected-F0.5 subset selection, with/without one-to-one resolution
Usage:
  python er_v12.py --data /path/to/dataset --out output            # full run
  python er_v12.py --data /path/to/dataset --out output_smoke --smoke   # quick debug run
"""
import os, sys, gc, re, csv, time, json, zipfile, argparse, subprocess
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode
from sparse_dot_topn import sp_matmul_topn

# ----------------------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="folder containing the tsv files (searched recursively)")
ap.add_argument("--out", default="output")
ap.add_argument("--smoke", action="store_true", help="small subsample run to check the pipeline")
ap.add_argument("--train-sample", type=int, default=400_000, help="#S1 train entities used for model training")
ap.add_argument("--k", type=int, default=10, help="top-k per TF-IDF channel per source")
ap.add_argument("--max-cands", type=int, default=25, help="max candidates per S1 fed to the model")
ap.add_argument("--min-cos", type=float, default=0.10)
ap.add_argument("--key-cap", type=int, default=50, help="skip exact keys shared by more targets than this")
ap.add_argument("--df-cap-min", type=int, default=3000)
ap.add_argument("--df-cap-frac", type=float, default=0.002)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--seed", type=int, default=42)
args, _ = ap.parse_known_args()
NPROC = max(1, args.threads)
F32 = np.float32
T0 = time.time()


def V(df, col):
    """version-proof access to a string column as a numpy object array (pandas 2 and 3)."""
    return np.asarray(df[col].to_numpy(dtype=object), dtype=object)


def log(msg):
    print(f"[{(time.time() - T0) / 60:7.2f} min] {msg}", flush=True)


# ----------------------------------------------------------------------------- IO
def find_file(name):
    roots = [args.data] if args.data else []
    roots += ["/mnt/custom-file-systems/s3/shared/crash-canyon", "/shared/crash-canyon", os.getcwd()]
    for r in roots:
        if r and os.path.isdir(r):
            for dp, _, fn in os.walk(r):
                if name in fn:
                    return os.path.join(dp, name)
    sys.exit(f"ERROR: cannot find {name}. Pass --data /folder/with/tsvs")


def read_tsv(name):
    return pd.read_csv(find_file(name), sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_filter=False)


def load_source(name):
    df = read_tsv(name)
    for c in ["entity_id", "business_name", "business_address", "country"]:
        if c not in df.columns:
            df[c] = ""
    df = df[["entity_id", "business_name", "business_address", "country"]]
    df = df[df.entity_id != ""].drop_duplicates("entity_id").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- normalization
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_ZIP4 = re.compile(r"\b(\d{5})-\d{4}\b")
_PIN_SPLIT = re.compile(r"\b(\d{3})\s(\d{3})\b")

NAME_ABBR = {
    "pvt": "private", "pv": "private", "ltd": "limited", "co": "company",
    "cos": "companies", "corp": "corporation", "inc": "incorporated", "incorp": "incorporated",
    "intl": "international", "int": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "bros": "brothers", "assoc": "associates", "assocs": "associates", "ent": "enterprises",
    "ents": "enterprises", "entp": "enterprises", "svc": "services", "svcs": "services",
    "serv": "services", "mgmt": "management", "mgt": "management", "dept": "department",
    "natl": "national", "govt": "government", "tech": "technologies", "techs": "technologies",
    "sys": "systems", "sol": "solutions", "solns": "solutions", "grp": "group", "hldgs": "holdings",
    "hldg": "holding", "ind": "industries", "inds": "industries", "indus": "industries",
    "engg": "engineering", "eng": "engineering", "dist": "distributors", "distr": "distributors",
    "mkt": "marketing", "univ": "university", "hosp": "hospital", "ctr": "center", "centre": "center",
    "cie": "compagnie", "ste": "societe", "ste": "societe", "st": "saint",
}
LEGAL_TAIL = {
    "private", "limited", "company", "corporation", "incorporated", "llc", "llp", "lp", "plc",
    "pllc", "pc", "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "bv", "nv",
    "pty", "pte", "ltda", "spa", "srl", "opc", "and", "compagnie", "et", "the",
}
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "blk": "block", "sec": "sector", "sect": "sector",
    "ph": "phase", "nr": "near", "opp": "opposite", "mkt": "market", "ngr": "nagar", "clny": "colony",
    "extn": "extension", "ext": "extension", "n": "north", "s": "south", "e": "east", "w": "west",
    "mt": "mount", "ft": "fort", "hts": "heights", "cir": "circle", "ctr": "center", "centre": "center",
    "jn": "junction", "jct": "junction", "ch": "chemin", "imp": "impasse", "fbg": "faubourg",
    "rte": "route", "sq.": "square", "no": "number", "num": "number", "ofc": "office",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk", "po": "postoffice",
}


def _collapse_initials(toks):
    out, buf = [], []
    for t in toks + [None]:
        if t is not None and len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if len(buf) >= 2:
            out.append("".join(buf))
        else:
            out.extend(buf)
        buf = []
        if t is not None:
            out.append(t)
    return out


def norm_name(s):
    s = unidecode(s or "").lower().replace("&", " and ").replace("+", " and ").replace("'", "")
    s = _NONALNUM.sub(" ", s)
    toks = [NAME_ABBR.get(t, t) for t in _collapse_initials(s.split())]
    toks = [t for t in toks if t not in ("dba", "aka")]
    full = " ".join(toks)
    core = list(toks)
    if len(core) > 1 and core[0] == "the":
        core = core[1:]
    while len(core) > 1 and core[-1] in LEGAL_TAIL:
        core.pop()
    return full, " ".join(core)


def norm_addr(s):
    s = unidecode(s or "").lower()
    s = _ZIP4.sub(r"\1", s)
    s = _PIN_SPLIT.sub(r"\1\2", s)
    s = _NONALNUM.sub(" ", s)
    toks = [ADDR_ABBR.get(t, t) for t in s.split()]
    postal = ""
    for t in reversed(toks):
        if t.isdigit() and len(t) in (5, 6):
            postal = t
            break
    house = ""
    for t in toks:
        if t != postal and len(t) <= 5 and t[:1].isdigit() and (t.isdigit() or t[:-1].isdigit()):
            house = t
            break
    return " ".join(toks), postal, house


def _prep_chunk(payload):
    names, addrs = payload
    out = []
    for n, a in zip(names, addrs):
        full, core = norm_name(n)
        an, pc, hn = norm_addr(a)
        out.append((full, core, an, pc, hn))
    return out


def prep(df):
    step = 50_000
    chunks = [(df.business_name.to_numpy(dtype=object)[k:k + step].tolist(), df.business_address.to_numpy(dtype=object)[k:k + step].tolist())
              for k in range(0, len(df), step)]
    with Pool(NPROC) as pool:
        res = pool.map(_prep_chunk, chunks)
    flat = [r for ch in res for r in ch]
    cols = list(zip(*flat)) if flat else [[], [], [], [], []]
    for name, col in zip(["nfull", "ncore", "anorm", "postal", "house"], cols):
        df[name] = np.array(col, dtype=object)
    df["country"] = df.country.str.strip().str.upper()
    core = V(df, "ncore")
    df["first"] = np.array([c.split(" ", 1)[0] if c else "" for c in core], dtype=object)
    df["compact"] = np.array([c.replace(" ", "") for c in core], dtype=object)
    df["acro"] = np.array(["".join(w[0] for w in c.split()) if " " in c else "" for c in core], dtype=object)
    df["ntok"] = np.array([c.count(" ") + 1 if c else 0 for c in core], dtype=F32)
    df.drop(columns=["business_name", "business_address"], inplace=True)
    return df


# ----------------------------------------------------------------------------- vectors
def name_analyzer(core):
    if not core:
        return []
    f = ["w" + t for t in core.split()]
    c = "#" + core.replace(" ", "") + "#"
    f += ["c" + c[k:k + 4] for k in range(max(1, len(c) - 3))]
    return f


def addr_analyzer(doc):
    a, _, pc = doc.partition("|")
    f = ["a" + t for t in a.split() if len(t) > 1 or t.isdigit()]
    if pc:
        f.append("p" + pc)
    return f


def tfidf(docs, analyzer):
    try:
        v = TfidfVectorizer(analyzer=analyzer, min_df=2, sublinear_tf=True, dtype=F32)
        return v.fit_transform(docs).tocsr()
    except ValueError:  # empty vocabulary
        return sp.csr_matrix((len(docs), 1), dtype=F32)


def cap_df(X, cap):
    df = np.bincount(X.indices, minlength=X.shape[1])
    keep = (df <= cap).astype(F32)
    Xb = (X @ sp.diags(keep)).tocsr()
    Xb.eliminate_zeros()
    return normalize(Xb).astype(F32).tocsr()


def rowdot(M, a, b, chunk=1_000_000):
    out = np.empty(len(a), F32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.asarray(M[a[s:s + chunk]].multiply(M[b[s:s + chunk]]).sum(1)).ravel()
    return out


def topn(Mq, Mt, k, thr):
    if Mq.shape[0] == 0 or Mt.shape[0] == 0 or Mq.nnz == 0 or Mt.nnz == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    C = sp_matmul_topn(Mq, Mt.T.tocsr(), top_n=k, threshold=thr, sort=False, n_threads=NPROC).tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64)


def group_cumcount(sorted_keys):
    n = len(sorted_keys)
    if n == 0:
        return np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    return np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))


# ----------------------------------------------------------------------------- blocking
def key_join(S1c, Tc):
    """exact-key candidates; S1c/Tc are the country slices (reset index = local position)."""
    def keys(df):
        core, comp, pc, fs = V(df, "ncore"), V(df, "compact"), V(df, "postal"), V(df, "first")
        kk, pos = [], []
        for p, (c, cp, z, f) in enumerate(zip(core, comp, pc, fs)):
            if len(c) >= 3:
                kk.append("k1" + c); pos.append(p)
            if len(cp) >= 4:
                kk.append("k2" + cp); pos.append(p)
            if " " in c:
                kk.append("k3" + " ".join(sorted(c.split()))); pos.append(p)
            if z and len(f) >= 3:
                kk.append("k4" + f + "|" + z); pos.append(p)
        return pd.DataFrame({"k": kk, "p": np.asarray(pos, np.int64)})
    a, b = keys(S1c), keys(Tc)
    vc = b.k.value_counts()
    b = b[b.k.map(vc).values <= args.key_cap]
    m = a.merge(b, on="k")
    return m.p_x.values.astype(np.int64), m.p_y.values.astype(np.int64)


def block_country(S1, T, s1_idx, t_idx):
    nq, nt = len(s1_idx), len(t_idx)
    S1c = S1.iloc[s1_idx].reset_index(drop=True)
    Tc = T.iloc[t_idx].reset_index(drop=True)
    name_docs = np.concatenate([V(S1c, "ncore"), V(Tc, "ncore")]).tolist()
    addr_docs = [a + "|" + z for a, z in zip(np.concatenate([V(S1c, "anorm"), V(Tc, "anorm")]),
                                            np.concatenate([V(S1c, "postal"), V(Tc, "postal")]))]
    NM = tfidf(name_docs, name_analyzer)
    AM = tfidf(addr_docs, addr_analyzer)
    cap = max(args.df_cap_min, int(args.df_cap_frac * (nq + nt)))
    NMb, AMb = cap_df(NM, cap), cap_df(AM, cap)
    CMb = normalize(sp.hstack([NMb, 0.7 * AMb]).tocsr()).astype(F32).tocsr()
    qs, ts = [], []
    tsrc = Tc.src.values
    for M in (NMb, CMb):
        Mq = M[:nq]
        for s in (2, 3):
            tl = np.flatnonzero(tsrc == s)
            if len(tl) == 0:
                continue
            r, c = topn(Mq, M[nq + tl], args.k, args.min_cos)
            qs.append(r)
            ts.append(tl[c])
    r, c = key_join(S1c, Tc)
    qs.append(r)
    ts.append(c)
    del NMb, AMb, CMb
    q = np.concatenate(qs)
    t = np.concatenate(ts)
    key = np.unique(q * nt + t)
    q, t = key // nt, key % nt
    cn = rowdot(NM, q, nq + t)
    ca = rowdot(AM, q, nq + t)
    score = np.maximum(cn, 0.6 * cn + 0.4 * ca)
    order = np.lexsort((-score, q))
    q, t, cn, ca = q[order], t[order], cn[order], ca[order]
    keep = group_cumcount(q) < args.max_cands
    return pd.DataFrame({"i": s1_idx[q[keep]].astype(np.int64), "j": t_idx[t[keep]].astype(np.int64),
                         "cn": cn[keep], "ca": ca[keep]})


def block_all(S1, T, tag):
    parts = []
    countries = sorted(set(S1.country.unique()))
    t_country = V(T, "country")
    for c in countries:
        s1_idx = np.flatnonzero(V(S1, "country") == c)
        t_idx = np.flatnonzero(t_country == c)
        if len(s1_idx) == 0 or len(t_idx) == 0:
            log(f"  [{tag}] country '{c}': S1={len(s1_idx):,} targets={len(t_idx):,} -> skipped")
            continue
        P = block_country(S1, T, s1_idx, t_idx)
        log(f"  [{tag}] country '{c}': S1={len(s1_idx):,} targets={len(t_idx):,} pairs={len(P):,} "
            f"({len(P) / len(s1_idx):.1f}/S1)")
        parts.append(P)
        gc.collect()
    P = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"i": np.zeros(0, np.int64), "j": np.zeros(0, np.int64), "cn": np.zeros(0, F32), "ca": np.zeros(0, F32)})
    P["src"] = T.src.values[P.j.values].astype(np.int8)
    return P


# ----------------------------------------------------------------------------- features
CTX = ["cn", "ca", "sim", "r_sim", "r_cn", "max_sim", "gap_sim", "n_c", "r_sim_src", "n_c_src",
       "rev_r", "rev_n", "rev_gap"]


def add_context(P):
    P["sim"] = (0.6 * P.cn + 0.4 * P.ca).astype(F32)
    g = P.groupby("i", sort=False)
    P["r_sim"] = g.sim.rank(ascending=False, method="first").astype(F32)
    P["r_cn"] = g.cn.rank(ascending=False, method="first").astype(F32)
    P["max_sim"] = g.sim.transform("max").astype(F32)
    P["gap_sim"] = (P.max_sim - P.sim).astype(F32)
    P["n_c"] = g.sim.transform("size").astype(F32)
    gs = P.groupby(["i", "src"], sort=False)
    P["r_sim_src"] = gs.sim.rank(ascending=False, method="first").astype(F32)
    P["n_c_src"] = gs.sim.transform("size").astype(F32)
    h = P.groupby("j", sort=False)
    P["rev_r"] = h.sim.rank(ascending=False, method="first").astype(F32)
    P["rev_n"] = h.sim.transform("size").astype(F32)
    P["rev_gap"] = (h.sim.transform("max") - P.sim).astype(F32)
    return P


def cpd(a, b, scorer):
    try:
        return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32).astype(F32)
    except Exception:
        return np.fromiter((scorer(x, y) for x, y in zip(a, b)), F32, len(a))


def slen(a):
    return np.fromiter(map(len, a), F32, len(a))


STR_FEATS = ["jw", "ratio", "partial", "tsort", "tset", "tset_full", "ratio_compact", "a_tset", "a_tsort",
             "a_partial", "pc_eq", "pc_conf", "pc_missing", "hn_eq", "hn_conf", "exact_core", "first_eq",
             "acro", "ntok_a", "ntok_b", "len_ratio", "alen_a", "alen_b", "src3"]


def string_feats(S1, T, i, j):
    a, b = V(S1, "ncore")[i], V(T, "ncore")[j]
    F = {}
    F["jw"] = cpd(a, b, JaroWinkler.normalized_similarity)
    F["ratio"] = cpd(a, b, fuzz.ratio) / 100
    F["partial"] = cpd(a, b, fuzz.partial_ratio) / 100
    F["tsort"] = cpd(a, b, fuzz.token_sort_ratio) / 100
    F["tset"] = cpd(a, b, fuzz.token_set_ratio) / 100
    F["tset_full"] = cpd(V(S1, "nfull")[i], V(T, "nfull")[j], fuzz.token_set_ratio) / 100
    ca_, cb_ = V(S1, "compact")[i], V(T, "compact")[j]
    F["ratio_compact"] = cpd(ca_, cb_, fuzz.ratio) / 100
    aa, ab = V(S1, "anorm")[i], V(T, "anorm")[j]
    F["a_tset"] = cpd(aa, ab, fuzz.token_set_ratio) / 100
    F["a_tsort"] = cpd(aa, ab, fuzz.token_sort_ratio) / 100
    F["a_partial"] = cpd(aa, ab, fuzz.partial_ratio) / 100
    pa, pb = V(S1, "postal")[i], V(T, "postal")[j]
    has = (pa != "") & (pb != "")
    F["pc_eq"] = (has & (pa == pb)).astype(F32)
    F["pc_conf"] = (has & (pa != pb)).astype(F32)
    F["pc_missing"] = (~has).astype(F32)
    ha, hb = V(S1, "house")[i], V(T, "house")[j]
    hh = (ha != "") & (hb != "")
    F["hn_eq"] = (hh & (ha == hb)).astype(F32)
    F["hn_conf"] = (hh & (ha != hb)).astype(F32)
    F["exact_core"] = ((a == b) & (a != "")).astype(F32)
    fa, fb = V(S1, "first")[i], V(T, "first")[j]
    F["first_eq"] = ((fa == fb) & (fa != "")).astype(F32)
    xa, xb = V(S1, "acro")[i], V(T, "acro")[j]
    F["acro"] = (((xa == cb_) & (xa != "")) | ((ca_ == xb) & (xb != ""))).astype(F32)
    F["ntok_a"] = S1.ntok.values[i]
    F["ntok_b"] = T.ntok.values[j]
    la, lb = slen(a), slen(b)
    F["len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    F["alen_a"], F["alen_b"] = slen(aa), slen(ab)
    F["src3"] = (T.src.values[j] == 3).astype(F32)
    return np.column_stack([F[k].astype(F32) for k in STR_FEATS])


S1_FEATS = STR_FEATS + CTX
S2_KEEP = ["jw", "tset", "a_tset", "pc_eq", "pc_conf", "exact_core", "src3"]
S2_P = ["p1", "p1_r", "p1_r_src", "p1_max", "p1_gap", "p1_sum", "p1_n05", "p1_second"]
S2_FEATS = S2_P + CTX + S2_KEEP


def stage1_matrix(S1, T, P, chunk=2_000_000):
    """returns full stage-1 feature matrix (used for train sample)."""
    out = np.empty((len(P), len(S1_FEATS)), F32)
    ctx = P[CTX].to_numpy(F32)
    for s in range(0, len(P), chunk):
        e = min(s + chunk, len(P))
        out[s:e, :len(STR_FEATS)] = string_feats(S1, T, P.i.values[s:e], P.j.values[s:e])
        out[s:e, len(STR_FEATS):] = ctx[s:e]
    return out


def stage2_matrix(P, p1, keep):
    D = pd.DataFrame({"i": P.i.values, "src": P.src.values, "p1": p1.astype(F32)})
    g = D.groupby("i", sort=False)
    D["p1_r"] = g.p1.rank(ascending=False, method="first")
    D["p1_r_src"] = D.groupby(["i", "src"], sort=False).p1.rank(ascending=False, method="first")
    D["p1_max"] = g.p1.transform("max")
    D["p1_gap"] = D.p1_max - D.p1
    D["p1_sum"] = g.p1.transform("sum")
    D["gt05"] = (D.p1 > 0.5).astype(F32)
    D["p1_n05"] = D.groupby("i", sort=False).gt05.transform("sum")
    D["p1_second"] = D.p1.where(D.p1_r == 2).groupby(D.i).transform("max").fillna(0)
    return np.column_stack([D[S2_P].to_numpy(F32), P[CTX].to_numpy(F32), keep.astype(F32)])


# ----------------------------------------------------------------------------- model
LGB_PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, num_threads=NPROC, seed=args.seed)


def cv_train(X, y, groups, names, tag):
    oof = np.zeros(len(y), F32)
    iters = []
    n_groups = len(np.unique(groups))
    folds = max(2, min(args.folds, n_groups))
    for f, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X[va], y[va], feature_name=names)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(max(m.best_iteration, 50))
        log(f"  {tag} fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    n_round = int(np.mean(iters) * 1.1)
    final = lgb.train(LGB_PARAMS, lgb.Dataset(X, y, feature_name=names), num_boost_round=n_round)
    imp = sorted(zip(names, final.feature_importance("gain")), key=lambda x: -x[1])[:12]
    log(f"  {tag} final rounds={n_round}; top gain: " + ", ".join(f"{k}" for k, _ in imp))
    return oof, final


# ----------------------------------------------------------------------------- decision + metric
RNG_U = np.random.default_rng(args.seed).random((400, args.max_cands + 1))


def best_k(pp):
    n = len(pp)
    y = RNG_U[:, :n] < pp
    tot = y.sum(1)
    tp = np.cumsum(y, 1)
    k = np.arange(1, n + 1)
    fp = k - tp
    fn = tot[:, None] - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * fn + fp), 0.0).mean(0)
    f0 = (tot == 0).mean()
    kb = int(np.argmax(f))
    return kb + 1 if f[kb] > f0 else 0


def one_to_one(j, p):
    mx = pd.Series(p).groupby(j).transform("max").values
    return np.where(p >= mx, p, 0.0).astype(F32)


def decide(i, j, p, method, param):
    order = np.lexsort((-p, i))
    i, j, p = i[order], j[order], p[order]
    if method == "thr":
        sel = p >= param
        return i[sel], j[sel]
    n = len(i)
    sel = np.zeros(n, bool)
    if n == 0:
        return i, j
    starts = np.flatnonzero(np.r_[True, i[1:] != i[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if p[s] < 0.02:
            continue
        pp = p[s:e]
        pp = pp[pp >= 0.005] ** param
        sel[s:s + best_k(pp)] = True
    return i[sel], j[sel]


def macro_f05(sel_i, sel_j, eval_s1, gt_i, gt_j, return_vec=False):
    idx = pd.Index(eval_s1)
    ev = np.isin(gt_i, eval_s1)
    G = pd.DataFrame({"i": gt_i[ev], "j": gt_j[ev]})
    Pr = pd.DataFrame({"i": sel_i, "j": sel_j})
    npd = Pr.groupby("i").size().reindex(idx, fill_value=0).values.astype(float)
    ngt = G.groupby("i").size().reindex(idx, fill_value=0).values.astype(float)
    tp = Pr.merge(G, on=["i", "j"]).groupby("i").size().reindex(idx, fill_value=0).values.astype(float)
    f = np.zeros(len(idx))
    f[(npd == 0) & (ngt == 0)] = 1.0
    m = tp > 0
    pr, rc = tp[m] / npd[m], tp[m] / ngt[m]
    f[m] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f if return_vec else float(f.mean())


# ============================================================================= MAIN
def main():
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    art = os.path.join(args.out, "artifacts")
    os.makedirs(art, exist_ok=True)
    log(f"V12 start | threads={NPROC} | args={vars(args)}")

    # ------------------------------------------------------------------ TRAIN load
    S1 = load_source("train_source1.tsv")
    T = pd.concat([load_source("train_source2.tsv").assign(src=2),
                   load_source("train_source3.tsv").assign(src=3)], ignore_index=True)
    gt = read_tsv("train_ground_truth.tsv")
    ex = gt[gt.matched_entity_ids.str.strip() != ""].copy()
    ex["m"] = ex.matched_entity_ids.str.split(",")
    ex = ex.explode("m")
    ex["m"] = ex.m.str.strip()
    ex = ex[ex.m != ""]

    if args.smoke:
        keep_s1 = S1.sample(min(30_000, len(S1)), random_state=args.seed)
        need = set(ex[ex.source1_entity_id.isin(keep_s1.entity_id)].m)
        rest = T[~T.entity_id.isin(need)].sample(min(200_000, len(T)), random_state=args.seed)
        S1 = keep_s1.reset_index(drop=True)
        T = pd.concat([T[T.entity_id.isin(need)], rest], ignore_index=True).drop_duplicates("entity_id")
        T = T.reset_index(drop=True)
        log("SMOKE MODE: train subsampled")

    log(f"train S1={len(S1):,} targets={len(T):,}; normalizing...")
    S1, T = prep(S1), prep(T)
    s1_code = pd.Series(np.arange(len(S1)), index=V(S1, "entity_id"))
    t_code = pd.Series(np.arange(len(T)), index=V(T, "entity_id"))
    gi = s1_code.reindex(ex.source1_entity_id.values).values
    gj = t_code.reindex(ex.m.values).values
    ok = ~(np.isnan(gi) | np.isnan(gj))
    gt_i, gt_j = gi[ok].astype(np.int64), gj[ok].astype(np.int64)

    # ---- data facts that drive decisions
    n_match_s1 = len(np.unique(gt_i))
    multi_t = (pd.Series(gt_j).value_counts() > 1).mean() if len(gt_j) else 0
    cross = (V(S1, "country")[gt_i] != V(T, "country")[gt_j]).mean() if len(gt_i) else 0
    log(f"GT pairs={len(gt_i):,}; S1 with matches={n_match_s1:,} ({n_match_s1 / len(S1):.1%}); "
        f"singletons={1 - n_match_s1 / len(S1):.1%}; avg matches/matched S1={len(gt_i) / max(n_match_s1, 1):.2f}")
    log(f"targets matched to >1 S1: {multi_t:.4%} | cross-country GT pairs: {cross:.4%}")
    log(f"train countries: {S1.country.value_counts().to_dict()}")
    if cross > 0.01:
        log("WARNING: >1% cross-country matches; same-country blocking will lose recall!")

    # ------------------------------------------------------------------ TRAIN blocking (all S1 -> correct reverse features)
    log("Blocking train (all S1, needed for reverse-rank features)...")
    P = block_all(S1, T, "train")
    P = add_context(P)
    sample = rng.choice(len(S1), size=min(args.train_sample, len(S1)), replace=False)
    sample.sort()
    P = P[np.isin(P.i.values, sample)].reset_index(drop=True)
    gkey = set((gt_i * len(T) + gt_j).tolist())
    y = np.fromiter(((a * len(T) + b) in gkey for a, b in zip(P.i.values, P.j.values)), np.int8, len(P))
    in_s = np.isin(gt_i, sample)
    rec = y.sum() / max(in_s.sum(), 1)
    ceil_ = macro_f05(P.i.values[y == 1], P.j.values[y == 1], sample, gt_i, gt_j)
    log(f"BLOCKING on sample: pairs={len(P):,} ({len(P) / len(sample):.1f}/S1) pair-recall={rec:.4%} "
        f"| macro-F0.5 ceiling={ceil_:.5f}")

    # ------------------------------------------------------------------ stage 1
    log("Stage-1 features...")
    X1 = stage1_matrix(S1, T, P)
    groups = P.i.values
    oof1, m1 = cv_train(X1, y, groups, S1_FEATS, "stage1")
    keep_idx = [S1_FEATS.index(k) for k in S2_KEEP]
    X2 = stage2_matrix(P, oof1, X1[:, keep_idx])
    del X1
    gc.collect()
    oof2, m2 = cv_train(X2, y, groups, S2_FEATS, "stage2")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof2, y)
    pc = iso.predict(oof2).astype(F32)
    m1.save_model(os.path.join(art, "stage1.txt"))
    m2.save_model(os.path.join(art, "stage2.txt"))

    # ------------------------------------------------------------------ decision tuning on OOF (exact metric)
    log("Tuning decision rule on OOF macro F0.5...")
    results = []
    for oto in (False, True):
        pp = one_to_one(P.j.values, pc) if oto else pc
        for t in np.arange(0.30, 0.951, 0.05):
            si, sj = decide(P.i.values, P.j.values, pp, "thr", float(t))
            results.append((macro_f05(si, sj, sample, gt_i, gt_j), "thr", round(float(t), 2), oto))
        for a in (0.8, 1.0, 1.25, 1.5):
            si, sj = decide(P.i.values, P.j.values, pp, "ef", a)
            results.append((macro_f05(si, sj, sample, gt_i, gt_j), "ef", a, oto))
    results.sort(key=lambda r: -r[0])
    for r in results[:8]:
        log(f"   F0.5={r[0]:.5f} method={r[1]} param={r[2]} one_to_one={r[3]}")
    best = results[0]
    # uncalibrated p1-only reference so you can see what stage2+calibration buys
    si, sj = decide(P.i.values, P.j.values, oof1, "thr", 0.5)
    log(f"   reference: stage1 thr=0.5 F0.5={macro_f05(si, sj, sample, gt_i, gt_j):.5f}")
    pp = one_to_one(P.j.values, pc) if best[3] else pc
    si, sj = decide(P.i.values, P.j.values, pp, best[1], best[2])
    fvec = macro_f05(si, sj, sample, gt_i, gt_j, return_vec=True)
    ctry = V(S1, "country")[sample]
    for c in np.unique(ctry):
        log(f"   OOF F0.5 [{c}] = {fvec[ctry == c].mean():.5f} (n={int((ctry == c).sum()):,})")
    has_gt = np.isin(sample, gt_i)
    log(f"   OOF F0.5 singletons={fvec[~has_gt].mean():.5f} matched={fvec[has_gt].mean():.5f}")
    json.dump({"best": best, "ceiling": ceil_, "top": results[:8]}, open(os.path.join(art, "oof_summary.json"), "w"),
              default=str, indent=1)
    del P, X2, S1, T, oof1, oof2, pc, y
    gc.collect()

    # ------------------------------------------------------------------ TEST
    S1 = load_source("test_source1.tsv")
    T = pd.concat([load_source("test_source2.tsv").assign(src=2),
                   load_source("test_source3.tsv").assign(src=3)], ignore_index=True)
    if args.smoke:
        S1 = S1.sample(min(20_000, len(S1)), random_state=args.seed).reset_index(drop=True)
        T = T.sample(min(200_000, len(T)), random_state=args.seed).reset_index(drop=True)
        log("SMOKE MODE: test subsampled -> output is NOT submittable")
    log(f"test S1={len(S1):,} targets={len(T):,}; countries={S1.country.str.strip().str.upper().value_counts().to_dict()}")
    S1, T = prep(S1), prep(T)
    P = block_all(S1, T, "test")
    P = add_context(P)
    log(f"test pairs={len(P):,} ({len(P) / max(len(S1), 1):.1f}/S1)")

    chunk = 2_000_000
    p1 = np.empty(len(P), F32)
    keep = np.empty((len(P), len(S2_KEEP)), F32)
    ctx = P[CTX].to_numpy(F32)
    for s in range(0, len(P), chunk):
        e = min(s + chunk, len(P))
        Xs = np.hstack([string_feats(S1, T, P.i.values[s:e], P.j.values[s:e]), ctx[s:e]])
        p1[s:e] = m1.predict(Xs)
        keep[s:e] = Xs[:, keep_idx]
        log(f"  test stage1 {e:,}/{len(P):,}")
    del ctx
    X2 = stage2_matrix(P, p1, keep)
    p2 = iso.predict(m2.predict(X2)).astype(F32)
    del X2, keep
    pp = one_to_one(P.j.values, p2) if best[3] else p2
    si, sj = decide(P.i.values, P.j.values, pp, best[1], best[2])
    log(f"test: matched S1={len(np.unique(si)):,}/{len(S1):,} ({len(np.unique(si)) / max(len(S1), 1):.1%}), "
        f"match pairs={len(si):,}")

    # ------------------------------------------------------------------ write outputs
    def write(path, i, j, col):
        d = pd.DataFrame({"i": i, "e": V(T, "entity_id")[j]})
        lists = d.groupby("i").e.agg(",".join).reindex(np.arange(len(S1)), fill_value="")
        pd.DataFrame({"source1_entity_id": V(S1, "entity_id"), col: lists.values}).to_csv(
            path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")

    mpath = os.path.join(args.out, "matching_results.tsv")
    cpath = os.path.join(args.out, "candidate_pairs.tsv")
    write(mpath, si, sj, "matched_entity_ids")
    write(cpath, P.i.values, P.j.values, "candidate_entity_ids")
    log(f"wrote {mpath} and {cpath}")

    val = None
    for dp, _, fn in os.walk(args.data or os.getcwd()):
        if "validate_submission.py" in fn:
            val = os.path.join(dp, "validate_submission.py")
            break
    if val and not args.smoke:
        test_dir = os.path.dirname(find_file("test_source1.tsv"))
        r = subprocess.run([sys.executable, val, "--matching", mpath, "--candidate", cpath, "--test-dir", test_dir],
                           capture_output=True, text=True)
        log("validator:\n" + r.stdout + r.stderr)
    if not args.smoke:
        zp = os.path.join(args.out, "outputs_V12.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(mpath, "output/matching_results.tsv")
            z.write(cpath, "output/candidate_pairs.tsv")
        log(f"zip: {zp}")
    log("DONE. Upload matching_results.tsv to the portal.")


if __name__ == "__main__":
    main()
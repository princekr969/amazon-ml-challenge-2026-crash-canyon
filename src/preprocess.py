"""Text preprocessing and normalization."""
import re
import pandas as pd
import unidecode
from typing import List

LEGAL_SUFFIX_MAP = {
    r'\bpvt\.?\b': 'private', r'\bltd\.?\b': 'limited',
    r'\bcorp\.?\b': 'corporation', r'\binc\.?\b': 'incorporated',
    r'\bllc\b': 'limited liability company', r'\bplc\b': 'public limited company',
    r'\b&\b': 'and', r'\bco\.?\b': 'company',
}

ADDRESS_ABBREV_MAP = {
    r'\brd\.?\b': 'road', r'\bst\.?\b': 'street',
    r'\bave\.?\b': 'avenue', r'\bblvd\.?\b': 'boulevard',
    r'\bfl\.?\b': 'floor', r'\bste\.?\b': 'suite',
    r'\bno\.?\b': 'number', r'\bnr\.?\b': 'near',
    r'\bopp\.?\b': 'opposite', r'\bbldg\.?\b': 'building',
}


def normalize_name(name: str) -> str:
    """Normalize business name: lowercase, transliterate, expand legal suffixes."""
    if not isinstance(name, str):
        return ""
    s = unidecode.unidecode(name).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    for pat, repl in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pat, repl, s)
    return s


def normalize_address(addr: str) -> str:
    """Normalize address: lowercase, transliterate, expand abbreviations."""
    if not isinstance(addr, str):
        return ""
    s = unidecode.unidecode(addr).lower()
    s = re.sub(r'[^\w\s,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    for pat, repl in ADDRESS_ABBREV_MAP.items():
        s = re.sub(pat, repl, s)
    return s


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized name and address columns to a DataFrame."""
    df = df.copy()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    return df
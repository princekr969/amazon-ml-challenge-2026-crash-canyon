"""Data loading utilities for Amazon ML Challenge 2026."""
import pandas as pd
from pathlib import Path
from typing import Dict

DATA_DIR = Path("data")

def load_sources(split: str = "train") -> Dict[str, pd.DataFrame]:
    """Load all 3 source files for a given split.
    
    Args:
        split: "train" or "test"
    
    Returns:
        Dict with keys "s1", "s2", "s3" mapping to DataFrames.
        For train, also returns "ground_truth".
    """
    data = {}
    for src in ["source1", "source2", "source3"]:
        path = DATA_DIR / split / f"{split}_{src}.tsv"
        data[src.replace("source", "s")] = pd.read_csv(
            path, sep="\t", dtype=str
        )
    if split == "train":
        data["ground_truth"] = pd.read_csv(
            DATA_DIR / "train" / "train_ground_truth.tsv",
            sep="\t", dtype=str
        )
    return data


def basic_stats(df: pd.DataFrame, name: str) -> None:
    """Print basic stats about a source DataFrame."""
    print(f"\n=== {name} ===")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing name: {df['business_name'].isna().sum():,}")
    print(f"  Missing address: {df['business_address'].isna().sum():,}")
    print(f"  Country distribution:")
    for country, count in df["country"].value_counts().head().items():
        print(f"    {country}: {count:,}")
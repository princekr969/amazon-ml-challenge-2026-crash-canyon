"""Exploratory data analysis for Amazon ML Challenge 2026."""
import pandas as pd
from pathlib import Path
from data_loader import load_sources


def run_eda():
    train = load_sources("train")
    test = load_sources("test")
    
    print("=" * 60)
    print("TRAINING SET")
    print("=" * 60)
    for name in ["s1", "s2", "s3"]:
        print(f"\n--- train_{name} ---")
        print(f"Rows: {len(train[name]):,}")
        print(f"Countries: {dict(train[name]['country'].value_counts())}")
    
    print("\n--- ground_truth ---")
    gt = train["ground_truth"]
    print(f"Rows: {len(gt):,}")
    gt["n_matches"] = gt["matched_entity_ids"].fillna("").apply(
        lambda s: len([x for x in s.split(",") if x])
    )
    print(f"Match count distribution: {dict(gt['n_matches'].value_counts().sort_index())}")
    print(f"Singletons: {(gt['n_matches'] == 0).sum():,} ({(gt['n_matches'] == 0).mean():.2%})")
    
    print("\n" + "=" * 60)
    print("TEST SET")
    print("=" * 60)
    for name in ["s1", "s2", "s3"]:
        print(f"\n--- test_{name} ---")
        print(f"Rows: {len(test[name]):,}")
        print(f"Countries: {dict(test[name]['country'].value_counts())}")


if __name__ == "__main__":
    run_eda()
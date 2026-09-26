#!/usr/bin/env python3
"""
limit_topk.py - Re-limits matching_results.tsv to top-K per S1
Reads existing matching_results.tsv, keeps only the first K IDs per row
(sorted by original order = score desc). Writes back to same path.
Usage: python src/limit_topk.py [K=1]
"""
import sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"Limiting to top-{K} per S1…", flush=True)

PATH = r"C:\Users\LENOVO\crash_run\output\matching_results.tsv"

with open(PATH, encoding="utf-8") as f:
    header = f.readline()
    rows = f.readlines()

out = []
empty = non_empty = total = 0
for line in rows:
    s1, _, rest = line.rstrip().partition("\t")
    if not rest:
        out.append(line); empty += 1; continue
    ids = rest.split(",")
    keep = ids[:K]
    out.append(f"{s1}\t{','.join(keep)}\n")
    non_empty += 1
    total += len(keep)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(header)
    f.writelines(out)

print(f"Done. rows={len(rows):,} empty={empty:,} non_empty={non_empty:,} total_kept={total:,}")

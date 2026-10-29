import csv
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

key = lambda s: unicodedata.normalize("NFC", s).strip()

d = pd.read_csv("datasets/deepcut.csv").drop_duplicates("Thai")
cnt = Counter()
for g in d.ThSL:
    for t in str(g).split():
        cnt[key(t)] += 1

poses = set()
for f in os.listdir("e-tsl-pose"):
    if f.endswith(".pose"):
        poses.add(key(re.sub(r"_sign\d+$", "", f[:-5])))

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
bad = 0
for r in rows:
    for t in r["ThSL"].split():
        t = key(t)
        if t in ("[start]", "[end]"):
            continue
        alt = sorted(p for p in poses if p.startswith(t + "("))
        issues = []
        if t not in poses:
            issues.append("numbered-only " + str(alt) if alt else "no pose")
        if cnt[t] == 0:
            issues.append("not in train")
        if issues:
            bad += 1
            print(r["Thai"], "|", t, "|", "; ".join(issues), "| train:", cnt[t])
print(len(rows), "sentences,", bad, "flagged tokens")

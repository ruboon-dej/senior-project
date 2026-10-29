import glob
import os
import re
import unicodedata

import pandas as pd

nfc = lambda s: unicodedata.normalize("NFC", s)

stems = {nfc(os.path.splitext(os.path.basename(f))[0]) for f in glob.glob("etsl_dataset/e-tsl-pose/*.pose")}
df = pd.read_csv("datasets/deepcut_dedup.csv")

glosses = set()
for s in df.ThSL:
    for t in s.split():
        if t not in ("[start]", "[end]"):
            glosses.add(nfc(t))

missing = sorted(g for g in glosses if g not in stems)
resolved = {}
unresolved = []
for g in missing:
    base = g.strip("()")
    found = None
    for c in (base, re.sub(r"\(\d+\)", "", base)):
        if c in stems:
            found = c
            break
    if found:
        resolved[g] = found
    else:
        unresolved.append(g)

digits = [g for g in unresolved if re.fullmatch(r"[0-9]+", g)]
others = [g for g in unresolved if g not in digits]

print("missing:", len(missing), "resolved by stripping parentheses:", len(resolved))
print("unresolved digits:", len(digits), "other unresolved:", len(others))
print("other unresolved:", others)
print("pose files starting with a digit:", [s for s in stems if re.match(r"[0-9๐-๙]", s)][:20])

bad = set(unresolved)
hit = df.ThSL.apply(lambda s: any(nfc(t) in bad for t in s.split()))
print("rows using an unresolved gloss:", int(hit.sum()), "of", len(df))

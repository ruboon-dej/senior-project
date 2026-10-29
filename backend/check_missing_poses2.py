import glob
import json
import os
import re
import unicodedata

import pandas as pd

nfc = lambda s: unicodedata.normalize("NFC", s)

stems = {}
for f in glob.glob("etsl_dataset/e-tsl-pose/*.pose"):
    s = nfc(os.path.splitext(os.path.basename(f))[0])
    stems[s.lower()] = s

df = pd.read_csv("datasets/deepcut_dedup.csv")
glosses = set()
for s in df.ThSL:
    for t in s.split():
        if t not in ("[start]", "[end]"):
            glosses.add(nfc(t))

with open("etsl_dataset/word_index.json", encoding="utf8") as fh:
    idx = json.load(fh)

th2en = {}
for key, v in idx.items():
    th = nfc(str(v.get("wrd_word_th", "")))
    en = str(v.get("wrd_word_en", ""))
    th2en.setdefault(th, []).append(en)


def candidates(en):
    e = en.lower().strip()
    return [e, e.replace(" ", "_"), re.sub(r"[^0-9a-z]+", "_", e).strip("_")]


def find(g):
    base = re.sub(r"\(\d+\)", "", g.strip("()"))
    for name in (g, g.strip("()"), base):
        if name.lower() in stems:
            return stems[name.lower()]
    for name in (g.strip("()"), base):
        for en in th2en.get(name, []):
            for c in candidates(en):
                if c in stems:
                    return stems[c]
    return None


missing = [g for g in sorted(glosses) if find(g) is None]
print("glosses:", len(glosses), "still unresolved:", len(missing))

digits = [g for g in missing if re.fullmatch(r"[0-9]+", g)]
others = [g for g in missing if g not in digits]
print("digits:", len(digits), "others:", len(others))

print("index entries for digits:", {d: th2en.get(d) for d in ["1", "2", "10", "100"]})

subs = {}
for g in others:
    key = g.strip("()").lower()
    hits = [s for l, s in stems.items() if key and key in l][:3]
    if hits:
        subs[g] = hits
print("unresolved with partial filename matches:", len(subs))
for g, h in list(subs.items())[:25]:
    print(g, "->", h)

truly = [g for g in others if g not in subs]
print("no match at all:", len(truly))
print(truly)

bad = set(missing)
hit = df.ThSL.apply(lambda s: any(nfc(t) in bad for t in s.split()))
print("rows using an unresolved gloss:", int(hit.sum()), "of", len(df))

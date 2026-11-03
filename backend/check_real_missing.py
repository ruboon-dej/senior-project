import ast
import glob
import itertools
import os
import re
import string
from collections import Counter

import pandas as pd

src = open("thai2thsl.py", encoding="utf8").read()
tree = ast.parse(src)

fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "merge_multiword_glosses")
from typing import List, Dict, Set, Tuple, Optional
ns = {"re": re, "os": os, "itertools": itertools, "glob": glob, "string": string, "List": List, "Dict": Dict, "Set": Set, "Tuple": Tuple, "Optional": Optional}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "merge", "exec"), ns)
merge = ns["merge_multiword_glosses"]

syn = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "gloss_synonyms" for t in n.targets):
        syn = ast.literal_eval(n.value)
print("synonym entries:", len(syn))

pose_dir = "e-tsl-pose"
available = {f[:-5] for f in os.listdir(pose_dir) if f.endswith(".pose")}

strip_chars = (string.punctuation + "¿").replace("[", "").replace("]", "")
pat = re.compile("[%s]" % re.escape(strip_chars))


def resolve(gloss):
    name = gloss
    if not name.isnumeric():
        name = re.sub(r"^(\D+)([0-9])$", r"\1(\2)", name)
    path = os.path.join(pose_dir, name + ".pose")
    if os.path.exists(path):
        return True
    if glob.glob(os.path.join(pose_dir, "{}([0-9]*).pose".format(gloss))):
        return True
    if gloss in syn:
        return os.path.exists(os.path.join(pose_dir, syn[gloss] + ".pose"))
    return False


df = pd.read_csv("datasets/deepcut_dedup.csv")
miss = Counter()
total = Counter()
rows_bad = 0
for s in df.ThSL:
    toks = [pat.sub("", t) for t in s.split() if t not in ("[start]", "[end]")]
    toks = [t for t in toks if t]
    merged = merge(list(toks), available)
    total.update(merged)
    bad = [g for g in merged if not resolve(g)]
    if bad:
        rows_bad += 1
    miss.update(bad)

digits = [g for g in miss if g.isnumeric()]
print("distinct glosses used:", len(total), "unresolved:", len(miss))
print("unresolved digits:", len(digits), "other:", len(miss) - len(digits))
print("rows with at least one unresolved gloss:", rows_bad, "of", len(df))
print("top unresolved by frequency:")
for g, c in miss.most_common(40):
    print(g, c)

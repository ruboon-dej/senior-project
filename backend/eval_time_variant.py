import csv

import thai2thsl as T
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie

trie = Trie(T.thai_known_words)
TIME = ["เมื่อวาน", "เช้า", "บ่าย", "กลางคืน", "เมื่อก่อน", "ตลอดคืน"]
NEED_TON = {"เช้า", "บ่าย", "กลางคืน"}


def strip(g):
    return g.replace("[start]", "").replace("[end]", "").split()


def run_a(raw):
    return [t for t in T.thai_to_tsl_glosses(raw) if t]


def run_b(raw):
    toks = [t for t in word_tokenize(raw, custom_dict=trie, engine="newmm") if t.strip()]
    return T.decode_sequence(" ".join(toks)).split()


rows = list(csv.DictReader(open("eval_real.csv", encoding="utf-8-sig")))
items = []
for r in rows:
    ref = strip(r["ThSL"])
    t = ref[0]
    if t not in TIME:
        continue
    raw = r["Thai_raw"]
    if raw.startswith(t):
        base = raw[len(t):]
    elif raw.endswith(t):
        base = raw[: -len(t)]
    else:
        continue
    items.append((raw, base + ("ตอน" if t in NEED_TON else "") + t, ref))

for name, fn in (("A_app_deepcut", run_a), ("B_dict_segmented", run_b)):
    o = sum(fn(raw) == ref for raw, _, ref in items)
    v = sum(fn(new) == ref for _, new, ref in items)
    print("RESULT", name, "original:", "{}/{}".format(o, len(items)), "rewritten:", "{}/{}".format(v, len(items)))
for raw, new, ref in items[:6]:
    print("RESULT", raw, "->", new, "| B:", " ".join(run_b(new)), "| REF:", " ".join(ref))

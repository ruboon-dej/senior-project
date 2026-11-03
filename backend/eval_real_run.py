import csv
import sys

import thai2thsl as T
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie

trie = Trie(T.thai_known_words)


def strip(g):
    return g.replace("[start]", "").replace("[end]", "").split()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(ref, hyp):
    exact = ref == hyp
    tok = 1 - edit(ref, hyp) / max(len(ref), len(hyp), 1)
    common = sum(min(ref.count(t), hyp.count(t)) for t in set(ref))
    p = common / len(hyp) if hyp else 0
    r = common / len(ref) if ref else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return exact, tok, f1


rows = list(csv.DictReader(open("eval_real.csv", encoding="utf-8-sig")))
systems = {
    "A_app_deepcut": lambda raw: T.thai_to_tsl_glosses(raw),
    "A2_app_skip_unknown": lambda raw: T.thai_to_tsl_glosses(raw, skip_unknown=True),
    "B_dict_segmented": lambda raw: T.decode_sequence(
        " ".join(t for t in word_tokenize(raw, custom_dict=trie, engine="newmm") if t.strip())
    ).split(),
}

results = []
summary = {}
for name, fn in systems.items():
    ex = tk = fs = 0
    for i, r in enumerate(rows):
        ref = strip(r["ThSL"])
        hyp = [t for t in fn(r["Thai_raw"]) if t]
        e, t, f = score(ref, hyp)
        ex += e
        tk += t
        fs += f
        results.append([name, i, r["Thai_raw"], " ".join(ref), " ".join(hyp), int(e), round(t, 3), round(f, 3)])
    n = len(rows)
    summary[name] = (ex / n, tk / n, fs / n)

with open("results_real.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["system", "id", "thai", "reference", "hypothesis", "exact", "token_acc", "f1"])
    w.writerows(results)

print("RESULTS")
for name, (e, t, f) in summary.items():
    print("{:22s} exact={:.3f} token_acc={:.3f} f1={:.3f}".format(name, e, t, f))
print("RESULTS_END")
for r in results:
    if r[0] == "A_app_deepcut" and not r[5]:
        print(r[2], "| REF:", r[3], "| HYP:", r[4])
        if r[1] > 12:
            break

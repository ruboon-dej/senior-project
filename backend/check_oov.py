import csv, json
from pythainlp.tokenize import word_tokenize

d = json.load(open("etsl_dataset/word_index.json", encoding="utf-8"))
vocab = {k.split("__")[0] for k in d}
rows = list(csv.DictReader(open("eval_real.csv", encoding="utf-8-sig")))
bad = 0
for r in rows:
    toks = word_tokenize(r["Thai_raw"], engine="newmm")
    oov = [t for t in toks if t not in vocab]
    if oov:
        bad += 1
        print(r["Thai_raw"], toks, oov)
print(bad, "of", len(rows))

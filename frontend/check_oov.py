import csv, json
from pythainlp.tokenize import word_tokenize

vocab = set(json.load(open("etsl_dataset/word_index.json", encoding="utf-8")))
rows = list(csv.DictReader(open("eval_real.csv", encoding="utf-8-sig")))
for r in rows:
    toks = word_tokenize(r["Thai_raw"], engine="newmm")
    oov = [t for t in toks if t not in vocab]
    if oov:
        print(r["Thai_raw"], toks, oov)

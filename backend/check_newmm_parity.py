import csv
import pandas as pd
import thai2thsl as app
from fair import segment, ref_tokens, score_set

with open("eval_real.csv", encoding="utf-8-sig") as f:
    real = list(csv.DictReader(f))

preds = []
for r in real:
    src = " ".join(segment(r["Thai_raw"], "newmm"))
    preds.append(app.decode_sequence(src, block_repeats=False).split())
refs = [ref_tokens(r["ThSL"]) for r in real]
ex, acc = score_set(preds, refs)

res = pd.read_csv("fair_results.csv")
row = res[(res.tokenizer == "newmm") & (res.seed == int(app.RUN["seed"]))].iloc[0]
print("run", app.RUN["model"])
print("app  real_exact", round(ex, 4), "real_acc", round(acc, 4))
print("fair real_exact", round(float(row.real_exact), 4), "real_acc", round(float(row.real_acc), 4))

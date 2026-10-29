import pandas as pd

src = "/home/suppavich/Documents/Senior_Project(tee)/SP2023-ThSL-main/thai2thsl/datasets/newmm.csv"
out = "datasets/newmm_dedup.csv"
vocab_out = "datasets/th_vocab-newmm-dedup.txt"

d = pd.read_csv(src).drop_duplicates()
g = d.groupby("Thai")["ThSL"].nunique()
print("conflicting Thai:", (g > 1).sum())

d["k"] = d.ThSL.map(lambda x: x.replace("[end]", "").split()[-1] == "ไม่")
d = d.sort_values("k", kind="stable")
d = d.drop_duplicates(subset=["Thai"], keep="first")
d = d.sort_index().drop(columns="k")
d.to_csv(out, index=False)

tokens = set(" ".join(d.Thai).split()) | set(" ".join(d.ThSL).split())
with open(vocab_out, "w", encoding="utf8") as f:
    f.write("\n".join(sorted(tokens)) + "\n")

print(len(d), d.Thai.duplicated().sum(), len(tokens))

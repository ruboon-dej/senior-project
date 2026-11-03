import csv

import thai2thsl as T
from time_normalize import move_time_to_end


def strip(g):
    return g.replace("[start]", "").replace("[end]", "").split()


rows = list(csv.DictReader(open("eval_real.csv", encoding="utf-8-sig")))
hit = 0
total = 0
for r in rows:
    raw = r["Thai_raw"]
    new = move_time_to_end(raw)
    if new == raw:
        continue
    total += 1
    ok = [t for t in T.thai_to_tsl_glosses(new) if t] == strip(r["ThSL"])
    hit += ok
    print("RESULT", raw, "->", new, "|", "ok" if ok else "wrong")
print("RESULT total:", "{}/{}".format(hit, total))

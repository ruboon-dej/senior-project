import glob
import json
import os
import unicodedata

import numpy as np
import pandas as pd
from pose_format import Pose

nfc = lambda s: unicodedata.normalize("NFC", s)

files = glob.glob("etsl_dataset/e-tsl-pose/*.pose")
stems = {nfc(os.path.splitext(os.path.basename(f))[0]): f for f in files}
print("pose files:", len(files))

df = pd.read_csv("datasets/deepcut_dedup.csv")
glosses = set()
for s in df.ThSL:
    for t in s.split():
        if t not in ("[start]", "[end]"):
            glosses.add(nfc(t))
missing = sorted(g for g in glosses if g not in stems)
print("glosses in data:", len(glosses), "without pose file:", len(missing))
print(missing[:30])

with open("etsl_dataset/word_index.json", encoding="utf8") as fh:
    idx = json.load(fh)
print("word_index type:", type(idx).__name__, "size:", len(idx))
sample = list(idx.items())[:3] if isinstance(idx, dict) else idx[:3]
print("word_index sample:", sample)

bad = []
frames_all = []
for stem, path in stems.items():
    try:
        with open(path, "rb") as fh:
            p = Pose.read(fh.read())
        data = np.ma.filled(p.body.data.astype("float64"), np.nan)
        frames = data.shape[0]
        frames_all.append(frames)
        conf = float(np.nanmean(np.ma.filled(p.body.confidence.astype("float64"), np.nan)))
        reasons = []
        if frames < 5:
            reasons.append("frames<5")
        if np.isnan(data).all():
            reasons.append("all_nan")
        elif np.nansum(np.abs(data)) == 0:
            reasons.append("all_zero")
        if conf < 0.3:
            reasons.append("low_conf=%.2f" % conf)
        if reasons:
            bad.append((stem, ",".join(reasons)))
    except Exception as e:
        bad.append((stem, "unreadable:" + type(e).__name__))

print("checked:", len(stems), "flagged:", len(bad))
for b in bad[:30]:
    print(b)
if frames_all:
    a = np.array(frames_all)
    print("frames min/median/max:", a.min(), int(np.median(a)), a.max())

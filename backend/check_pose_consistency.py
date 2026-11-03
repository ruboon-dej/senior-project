import glob
import os
import unicodedata
from collections import Counter

import numpy as np
from pose_format import Pose

nfc = lambda s: unicodedata.normalize("NFC", s)

sigs = {}
hand_low = []
errors = []
files = sorted(glob.glob("e-tsl-pose/*.pose"))
for path in files:
    name = nfc(os.path.splitext(os.path.basename(path))[0])
    try:
        with open(path, "rb") as fh:
            p = Pose.read(fh.read())
        comps = [(c.name, len(c.points)) for c in p.header.components]
        sig = (round(float(p.body.fps), 2), p.body.data.shape[1], p.body.data.shape[2], p.body.data.shape[3], tuple(comps))
        sigs[name] = sig
        conf = np.ma.filled(p.body.confidence.astype("float64"), np.nan)
        offset = 0
        hand_means = []
        for cname, n in comps:
            if "HAND" in cname.upper():
                hand_means.append(float(np.nanmean(conf[:, :, offset:offset + n])))
            offset += n
        if hand_means and max(hand_means) < 0.3:
            hand_low.append((name, round(max(hand_means), 2)))
    except Exception as e:
        errors.append((name, type(e).__name__))

count = Counter(sigs.values())
print("files:", len(files), "read errors:", len(errors))
print("distinct formats:", len(count))
for sig, c in count.most_common(5):
    print(c, "fps=%s people=%s points=%s dims=%s" % sig[:4], "components:", [x[0] for x in sig[4]])
major = count.most_common(1)[0][0]
odd = sorted(n for n, s in sigs.items() if s != major)
print("files not matching the majority format:", len(odd), odd[:30])
print("files where hands are barely detected:", len(hand_low))
print(sorted(hand_low, key=lambda x: x[1])[:40])

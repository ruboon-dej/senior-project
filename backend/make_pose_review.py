import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from spoken_to_signed.gloss_to_pose.concatenate import reduce_holistic

key = lambda s: unicodedata.normalize("NFC", s).strip()

import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
out = "pose_review_{}".format(seed)
os.makedirs(os.path.join(out, "clips"), exist_ok=True)

by_th = {}
for f in glob.glob("etsl_dataset/words/*/metadata.json"):
    d = os.path.dirname(f)
    try:
        m = json.load(open(f, encoding="utf8"))
    except Exception:
        continue
    by_th.setdefault(key(str(m.get("wrd_word_th", ""))), []).append(d)

orig = {unicodedata.normalize("NFC", f[:-5]): f[:-5] for f in os.listdir("e-tsl-pose") if f.endswith(".pose")}
sample = random.Random(seed).sample(sorted(orig), 100)


def find_dirs(base):
    for cand in (base, re.sub(r"\(\d+\)$", "", base).strip()):
        if cand in by_th:
            return by_th[cand]
    folder = os.path.join("etsl_dataset/words", base)
    if os.path.isdir(folder):
        return [folder]
    return []


def source_videos(name):
    m = re.match(r"^(.*?)(?:_sign(\d+))?$", name)
    base, num = key(m.group(1)), m.group(2)
    vids = sorted(v for d in find_dirs(base) for v in glob.glob(os.path.join(d, "*.mp4")))
    if not vids:
        return [], "none"
    prefix = "sign{}_".format(num if num is not None else "0")
    exact = [v for v in vids if os.path.basename(v).startswith(prefix)]
    if exact:
        return exact, "exact"
    return vids, "fallback"


have_ffmpeg = shutil.which("ffmpeg") is not None
rows = []
cards = []
for i, name in enumerate(sample):
    with open("e-tsl-pose/{}.pose".format(orig[name]), "rb") as fh:
        p = reduce_holistic(Pose.read(fh.read()))
    v = PoseVisualizer(p)
    raw = os.path.join(out, "clips", "{:03d}_raw.mp4".format(i))
    final = os.path.join(out, "clips", "{:03d}.mp4".format(i))
    v.save_video(raw, v.draw())
    if have_ffmpeg:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", final], check=True)
        os.remove(raw)
    else:
        os.replace(raw, final)
    srcs, match = source_videos(name)
    rows.append([i, name, len(srcs), "", "", match])
    src_html = "".join(
        '<video src="{}" controls loop width="320"></video>'.format(os.path.relpath(s, out)) for s in srcs[:2]
    ) or "<p>no source video found</p>"
    cards.append(
        '<div style="margin:16px 0;border-bottom:1px solid #999"><h3>{} - {} [{}]</h3>'
        '<video src="clips/{:03d}.mp4" controls loop width="320"></video> {}</div>'.format(i, name, match, i, src_html)
    )
    print(i, name, match)

with open(os.path.join(out, "review.html"), "w", encoding="utf8") as fh:
    fh.write('<meta charset="utf-8"><body style="font-family:sans-serif">' + "".join(cards))
with open(os.path.join(out, "review_sheet.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "name", "source_videos_found", "verdict", "note", "match"])
    w.writerows(rows)
print("seed:", seed, "output:", out)
print("ffmpeg used:", have_ffmpeg, dict(Counter(r[5] for r in rows)))

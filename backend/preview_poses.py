import os
import random

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from spoken_to_signed.gloss_to_pose.concatenate import reduce_holistic

os.makedirs("pose_preview", exist_ok=True)
names = ["เท้า", "เข่า"]
pool = sorted(f[:-5] for f in os.listdir("e-tsl-pose") if f.endswith(".pose") and f[:-5] not in names)
names += random.Random(1).sample(pool, 10)

for n in names:
    with open("e-tsl-pose/{}.pose".format(n), "rb") as fh:
        p = Pose.read(fh.read())
    p = reduce_holistic(p)
    v = PoseVisualizer(p)
    v.save_video("pose_preview/{}.mp4".format(n), v.draw())
    print("saved", n)

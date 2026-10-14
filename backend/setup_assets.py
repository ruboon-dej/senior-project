import os
import sys
import tarfile

import gdown

BASE = os.path.dirname(os.path.abspath(__file__))

ASSETS = [
    ("e-tsl-pose", "ID_POSE", "e-tsl-pose.tar"),
    ("models/thai2thsl-deepcut.hdf5", "ID_MODELS", "models-datasets.tar"),
]

for marker, file_id, archive in ASSETS:
    if os.path.exists(os.path.join(BASE, marker)):
        print("skip", marker)
        continue
    target = os.path.join(BASE, archive)
    gdown.download(id=file_id, output=target, quiet=False)
    with tarfile.open(target) as tar:
        tar.extractall(BASE)
    os.remove(target)
    print("done", marker)

if not os.path.isdir(os.path.join(BASE, "e-tsl-pose")):
    sys.exit("e-tsl-pose missing")

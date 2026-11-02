import random, re
import pandas as pd

def thai_raw(s):    return re.sub(r"\s+", "", s)
def gloss_clean(s): return s.replace("[start]", "").replace("[end]", "").strip()

def load_split(path="datasets/deepcut.csv"):
    """Same shuffle/seed as thai2thsl.py -> identical train/val/test."""
    pairs = pd.read_csv(path).values.tolist()
    random.Random(10).shuffle(pairs)
    n_val = int(0.1 * len(pairs))
    n_train = len(pairs) - 2 * n_val
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]
import argparse, random, re
import pandas as pd
from evaluate_model import clean_reference, edit_distance, bag_f1

def norm(g):  # compare "แขก(2)" with "แขก2", "จ.ภูเก็ต" with "จภูเก็ต"
    return re.sub(r"[()./]", "", g)

def load_split():
    pairs = pd.read_csv("datasets/deepcut.csv").values.tolist()
    random.Random(10).shuffle(pairs)                 # same as thai2thsl.py
    n_val = int(0.1 * len(pairs))
    n_train = len(pairs) - 2 * n_val
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mt5")      # mt5 | llm | auto
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    import gloss_backends
    gloss_backends.BACKEND = a.backend
    _, _, test = load_split()
    test = random.Random(0).sample(test, min(a.n, len(test)))

    exact = wer = f1 = 0
    for thai, thsl in test:
        raw = re.sub(r"\s+", "", thai)               # what a user would really type
        ref = [norm(g) for g in clean_reference(thsl)]
        hyp = [norm(g) for g in gloss_backends.get_glosses(raw)]
        exact += ref == hyp
        wer += edit_distance(ref, hyp) / max(len(ref), 1)
        f1 += bag_f1(ref, hyp)
    n = len(test)
    print(f"{a.backend}: exact {exact/n:.1%}  WER {wer/n:.3f}  bagF1 {f1/n:.3f}")
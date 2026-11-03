import random
import re
import string
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding

CONFIGS = {
    "deepcut": (
        "datasets/deepcut_dedup.csv",
        "datasets/th_vocab-deepcut-dedup.txt",
        "models/thai2thsl-deepcut-dedup.hdf5",
    ),
    "newmm": (
        "datasets/newmm_dedup.csv",
        "datasets/th_vocab-newmm-dedup.txt",
        "models/thai2thsl-newmm-dedup.hdf5",
    ),
}

SEQ_LEN = 20
CHUNK = 256
STRIP = (string.punctuation + "¿").replace("[", "").replace("]", "")


def standardize_py(s):
    return re.sub("[%s]" % re.escape(STRIP), "", s)


def standardize_tf(x):
    return tf.strings.regex_replace(x, "[%s]" % re.escape(STRIP), "")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def clean_ref(s):
    toks = standardize_py(s).split()
    return [t for t in toks if t not in ("[start]", "[end]")]


def evaluate(name):
    csv_path, vocab_path, model_path = CONFIGS[name]
    df = pd.read_csv(csv_path)
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab_size = len(f.read().split("\n"))

    pairs = df.values.tolist()
    random.Random(10).shuffle(pairs)
    n_val = int(0.1 * len(pairs))
    n_train = len(pairs) - 2 * n_val
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train + n_val:]

    thai_vec = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=SEQ_LEN)
    thsl_vec = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=SEQ_LEN + 1,
        standardize=standardize_tf,
    )
    thai_vec.adapt([p[0] for p in train_pairs])
    thsl_vec.adapt([p[1] for p in train_pairs])
    vocab = thsl_vec.get_vocabulary()
    start_id = vocab.index("[start]")

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding,
        },
        compile=False,
    )

    rows = []
    for s in range(0, len(test_pairs), CHUNK):
        batch = test_pairs[s:s + CHUNK]
        enc = thai_vec(tf.constant([p[0] for p in batch])).numpy()
        dec = np.zeros((len(batch), SEQ_LEN), dtype="int64")
        dec[:, 0] = start_id
        out = np.zeros((len(batch), SEQ_LEN), dtype="int64")
        for i in range(SEQ_LEN):
            preds = model([enc, dec], training=False).numpy()
            nxt = preds[:, i, :].argmax(-1)
            out[:, i] = nxt
            if i + 1 < SEQ_LEN:
                dec[:, i + 1] = nxt
        for p, o in zip(batch, out):
            toks = []
            for idx in o:
                t = vocab[idx]
                if t == "[end]":
                    break
                toks.append(t)
            ref = clean_ref(p[1])
            rows.append(
                {
                    "thai": p[0],
                    "ref": " ".join(ref),
                    "pred": " ".join(toks),
                    "exact": toks == ref,
                    "ter": edit_distance(toks, ref) / max(len(ref), 1),
                    "has_neg": "ไม่" in p[0].split(),
                }
            )
    res = pd.DataFrame(rows)
    res.to_csv(f"compare_{name}.csv", index=False)
    return res


summary = {}
results = {}
for name in CONFIGS:
    r = evaluate(name)
    results[name] = r
    summary[name] = {
        "n": len(r),
        "exact_match": round(r.exact.mean(), 4),
        "token_error_rate": round(r.ter.mean(), 4),
        "exact_match_with_NEG": round(r[r.has_neg].exact.mean(), 4),
    }
    print(name, summary[name])

same_ref = (results["deepcut"].ref.values == results["newmm"].ref.values).mean()
print("same test references across both:", round(same_ref, 4))
print(results["deepcut"][~results["deepcut"].exact].head(10)[["thai", "ref", "pred"]].to_string())
print(results["newmm"][~results["newmm"].exact].head(10)[["thai", "ref", "pred"]].to_string())

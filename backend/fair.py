import argparse
import csv
import glob
import json
import os
import random
import re
import string
import subprocess
import sys

import numpy as np
import pandas as pd
from pythainlp.tokenize import subword_tokenize, word_tokenize

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ALL_TOKENIZERS = ["deepcut", "newmm", "attacut", "longest", "tcc", "char"]
WORD_ENGINES = {"deepcut", "newmm", "attacut", "longest", "newmm-safe", "nercut"}
SRC = "datasets/deepcut_dedup.csv"
REAL = "eval_real.csv"
SEQ = 20
EPOCHS = 30
STRIP = (string.punctuation + "¿").replace("[", "").replace("]", "")


def segment(text, name):
    text = text.strip()
    if name == "char":
        toks = list(text)
    elif name == "tcc":
        toks = subword_tokenize(text, engine="tcc_p")
    else:
        toks = word_tokenize(text, engine=name)
    return [t for t in toks if t.strip()]


def detok(s):
    return "".join(str(s).split())


def split_pairs(df):
    pairs = df.values.tolist()
    random.Random(10).shuffle(pairs)
    n = int(0.1 * len(pairs))
    return pairs[:len(pairs) - 2 * n], pairs[len(pairs) - 2 * n: len(pairs) - n], pairs[len(pairs) - n:]


def paths(name, seed=None):
    base = {"csv": "datasets/fair_%s.csv" % name, "vocab": "datasets/th_vocab-fair-%s.txt" % name}
    if seed is not None:
        base["model"] = "models/fair-%s-s%d.hdf5" % (name, seed)
        base["meta"] = "models/fair-%s-s%d.json" % (name, seed)
    return base


def build(name):
    orig = pd.read_csv(SRC)
    d = orig.copy()
    d["Thai"] = d["Thai"].map(lambda s: " ".join(segment(detok(s), name)))
    p = paths(name)
    d.to_csv(p["csv"], index=False)
    tokens = set(" ".join(d.Thai).split()) | set(" ".join(d.ThSL).split())
    with open(p["vocab"], "w", encoding="utf8") as f:
        f.write("\n".join(sorted(tokens)) + "\n")
    lens = d.Thai.map(lambda s: len(s.split()))
    print(name, "rows", len(d), "vocab", len(tokens), "max_len", int(lens.max()),
          "mean_len", round(float(lens.mean()), 1), "dup_thai", int(d.Thai.duplicated().sum()))
    if name == "deepcut":
        bad = orig[orig.Thai != d.Thai]
        print("deepcut round-trip match", len(d) - len(bad), "/", len(d))
        for _, r in bad.head(5).iterrows():
            print("MISMATCH", r.Thai, "|", " ".join(segment(detok(r.Thai), name)))


def vectorizers(df, vocab_path, source_len, train_pairs):
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization

    def standardize_tf(x):
        return tf.strings.regex_replace(x, "[%s]" % re.escape(STRIP), "")

    with open(vocab_path, encoding="utf8") as f:
        vocab_size = len(f.read().split("\n"))
    thai_vec = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=source_len)
    thsl_vec = TextVectorization(max_tokens=vocab_size, output_mode="int",
                                 output_sequence_length=SEQ + 1, standardize=standardize_tf)
    thai_vec.adapt([p[0] for p in train_pairs])
    thsl_vec.adapt([p[1] for p in train_pairs])
    return thai_vec, thsl_vec, vocab_size


def train(name, seed):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.utils.set_random_seed(seed)
    p = paths(name, seed)
    df = pd.read_csv(p["csv"])
    train_pairs, val_pairs, _ = split_pairs(df)
    source_len = max(SEQ, max(len(str(s).split()) for s in df["Thai"]))
    thai_vec, thsl_vec, vocab_size = vectorizers(df, p["vocab"], source_len, train_pairs)
    embed_dim, dense_dim, num_heads, batch_size, epochs = 256, 2048, 8, 64, EPOCHS

    def format_dataset(thai, thsl):
        thai = thai_vec(thai)
        thsl = thsl_vec(thsl)
        return ({"encoder_inputs": thai, "decoder_inputs": thsl[:, :-1]}, thsl[:, 1:])

    def make_dataset(pairs):
        thai_texts, thsl_texts = zip(*pairs)
        ds = tf.data.Dataset.from_tensor_slices((list(thai_texts), list(thsl_texts)))
        ds = ds.batch(batch_size).map(format_dataset)
        return ds.shuffle(2048).prefetch(16).cache()

    train_ds, val_ds = make_dataset(train_pairs), make_dataset(val_pairs)

    enc_in = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(source_len, vocab_size, embed_dim)(enc_in)
    enc_out = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    dec_in = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    state_in = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(SEQ, vocab_size, embed_dim)(dec_in)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, state_in)
    x = layers.Dropout(0.5)(x)
    dec_out = layers.Dense(vocab_size, activation="softmax")(x)
    decoder = keras.Model([dec_in, state_in], dec_out)

    model = keras.Model([enc_in, dec_in], decoder([dec_in, enc_out]), name="transformer")
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    meta = {"tokenizer": name, "seed": seed, "source_len": source_len,
            "csv": p["csv"], "vocab": p["vocab"], "model": p["model"]}
    with open(p["meta"], "w") as f:
        json.dump(meta, f)

    class MetaWriter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            vl = float(logs["val_loss"])
            if "best_val_loss" not in meta or vl < meta["best_val_loss"]:
                meta["best_epoch"] = epoch + 1
                meta["best_val_loss"] = vl
                meta["best_val_accuracy"] = float(logs["val_accuracy"])
            meta["epochs_done"] = epoch + 1
            with open(p["meta"], "w") as f:
                json.dump(meta, f)

    ckpt = keras.callbacks.ModelCheckpoint(p["model"], monitor="val_loss", save_best_only=True, verbose=1)
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[ckpt, MetaWriter()])


def meta_done(m):
    if "best_val_loss" not in m:
        return False
    return m.get("epochs_done", EPOCHS) >= EPOCHS


def is_done(name, seed):
    meta = paths(name, seed)["meta"]
    if not os.path.exists(meta):
        return False
    with open(meta) as f:
        return meta_done(json.load(f))


def decode(model, thai_vec, vocab, sources, chunk=256):
    import tensorflow as tf
    start_id = vocab.index("[start]")
    results = []
    for a in range(0, len(sources), chunk):
        enc = thai_vec(tf.constant(sources[a:a + chunk])).numpy()
        dec = np.zeros((len(enc), SEQ), dtype="int64")
        dec[:, 0] = start_id
        pred = None
        for i in range(SEQ):
            pred = model.predict([enc, dec], batch_size=chunk, verbose=0).argmax(-1)
            if i + 1 < SEQ:
                dec[:, i + 1] = pred[:, i]
        for row in pred:
            toks = []
            for k in row:
                t = vocab[int(k)]
                if t == "[end]":
                    break
                if t:
                    toks.append(t)
            results.append(toks)
    return results


def ref_tokens(s):
    from eval_gold import standardize_py
    return standardize_py(s).replace("[start]", "").replace("[end]", "").split()


def score_set(preds, refs):
    from eval_gold import edit
    ex = acc = 0.0
    for p, r in zip(preds, refs):
        ex += int(p == r)
        acc += 1 - edit(r, p) / max(len(r), len(p), 1)
    n = max(len(refs), 1)
    return ex / n, acc / n


def evaluate():
    from tensorflow import keras
    from eval_gold import load_gold, score_row
    from pronouns import normalize_pronouns
    from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding

    metas = []
    for path in sorted(glob.glob("models/fair-*-s*.json")):
        with open(path) as f:
            m = json.load(f)
        if meta_done(m):
            metas.append(m)
    if not metas:
        print("no finished models found")
        return

    real = None
    if os.path.exists(REAL):
        with open(REAL, encoding="utf-8-sig") as f:
            real = list(csv.DictReader(f))
    gold = load_gold()

    rows, gold_rows = [], []
    for m in metas:
        name = m["tokenizer"]
        df = pd.read_csv(m["csv"])
        train_pairs, _, test = split_pairs(df)
        thai_vec, thsl_vec, _ = vectorizers(df, m["vocab"], m["source_len"], train_pairs)
        vocab = thsl_vec.get_vocabulary()
        model = keras.models.load_model(
            m["model"],
            custom_objects={"TransformerEncoder": TransformerEncoder,
                            "TransformerDecoder": TransformerDecoder,
                            "PositionalEmbedding": PositionalEmbedding},
            compile=False)

        t_ex, t_acc = score_set(decode(model, thai_vec, vocab, [p[0] for p in test]),
                                [ref_tokens(p[1]) for p in test])
        r_ex = r_acc = float("nan")
        if real:
            src = [" ".join(segment(r["Thai_raw"], name)) for r in real]
            r_ex, r_acc = score_set(decode(model, thai_vec, vocab, src), [ref_tokens(r["ThSL"]) for r in real])

        eng = name if name in WORD_ENGINES else "newmm"
        g_src = [" ".join(segment(normalize_pronouns(g[1], eng), name)) for g in gold]
        g_ex = g_acc = 0.0
        for (rid, thai, refs), hyp in zip(gold, decode(model, thai_vec, vocab, g_src)):
            clean, ex, acc, _ = score_row(rid, hyp, refs)
            g_ex += ex
            g_acc += acc
            gold_rows.append([name, m["seed"], rid, thai, " ".join(hyp), " ".join(clean), ex, acc])
        rows.append({"tokenizer": name, "seed": m["seed"], "val_loss": m["best_val_loss"],
                     "test_exact": t_ex, "test_acc": t_acc, "real_exact": r_ex, "real_acc": r_acc,
                     "gold_exact": g_ex, "gold_acc": g_acc / len(gold)})
        print("done", name, m["seed"])
        keras.backend.clear_session()

    res = pd.DataFrame(rows)
    res.to_csv("fair_results.csv", index=False, encoding="utf-8-sig")
    with open("fair_gold_rows.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tokenizer", "seed", "id", "thai", "raw_hypothesis", "scored_hypothesis", "exact", "token_acc"])
        w.writerows(gold_rows)

    cols = ["val_loss", "test_exact", "test_acc", "real_exact", "real_acc", "gold_exact", "gold_acc"]
    print("\nmean +- std over seeds (gold_exact is out of 12; others are fractions)")
    print("{:10s} {:>2s} ".format("tokenizer", "n") + " ".join("{:>15s}".format(c) for c in cols))
    for tk, g in res.groupby("tokenizer"):
        cells = " ".join("{:>7.3f}+-{:<6.3f}".format(g[c].mean(), g[c].std(ddof=0)) for c in cols)
        print("{:10s} {:>2d} ".format(tk, len(g)) + cells)
    print("\nper-model results in fair_results.csv, gold predictions in fair_gold_rows.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "train", "train_one", "eval", "all"])
    ap.add_argument("--tokenizers", nargs="+", default=ALL_TOKENIZERS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    a = ap.parse_args()

    if a.step == "train_one":
        train(a.tokenizers[0], a.seeds[0])
        return
    if a.step in ("build", "all"):
        for n in a.tokenizers:
            build(n)
    if a.step in ("train", "all"):
        for s in a.seeds:
            for n in a.tokenizers:
                if is_done(n, s):
                    print("skip", n, s)
                    continue
                code = subprocess.call([sys.executable, os.path.abspath(__file__), "train_one",
                                        "--tokenizers", n, "--seeds", str(s)])
                if not is_done(n, s):
                    print("FAILED", n, s, "exit code", code)
    if a.step in ("eval", "all"):
        evaluate()


if __name__ == "__main__":
    main()

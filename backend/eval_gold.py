"""Evaluate Thai2ThSL systems against gold_thsl_suppasin2026.csv.

Run from the backend/ folder:
    python eval_gold.py                 # all systems
    python eval_gold.py --skip-models   # tokenizer-only variants (no extra model loading)

Scored rows: tier in {core, partial, phrase} (12 rows). evidence_only and
excluded rows have no word-level reference and are never scored.

Systems
  A_app_deepcut       app pipeline (thai2thsl.py, thai2thsl-deepcut.hdf5)
  A2_app_skip_unknown same, skip_unknown=True
  B_deepcut_dedup     models/thai2thsl-deepcut-dedup.hdf5, deepcut tokenizer
  C_newmm_dedup       models/thai2thsl-newmm-dedup.hdf5, newmm tokenizer
  D_<engine>          model A decoded from input segmented by <engine>
                      (D_newmm == B_dict_segmented in eval_real_run.py)

IGNORE below is my reading of each row's scoring_note. Check it before
trusting the numbers.
"""
import argparse
import csv
import re
import string
import sys

import numpy as np

GOLD = "gold_thsl_suppasin2026.csv"
OUT = "gold_results.csv"
SEQ_LEN = 20
SCORED_TIERS = {"core", "partial", "phrase"}
TOKENIZER_ENGINES = ["newmm", "newmm-safe", "longest", "attacut", "nercut"]
CUSTOM_DICT_ENGINES = {"newmm", "newmm-safe", "longest"}

# Hypothesis tokens that the scoring_note says are not signed / not penalised.
# ฉัน is added automatically wherever ผม is ignored (pronoun normalisation).
IGNORE = {
    "G01": {"นะ", "หา"},
    "G02": {"ผม", "ไว้"},
    "G03": {"เป็น", "ผม"},
    "G04": {"ผม"},
    "G05": {"ผม", "ขึ้น"},
    "G06": set(),  # trailing คุณ handled by TRAILING_IGNORE
    "G07": {"ผม", "คุณ", "ดี", "ที่"},
    "P01": {"คัน"},
    "P02": {"คัน"},
    "P03": {"คัน"},
    "P04": {"ของ"},
    "P05": set(),
}
TRAILING_IGNORE = {"G06": {"คุณ"}}
# Time phrase that must open the sentence (position check)
TIME_PHRASE = {"G02": ["วันนี้", "ตอนเย็น"], "G07": ["เมื่อวาน"]}

DIGITS = {"1": "หนึ่ง", "2": "สอง", "3": "สาม", "4": "สี่", "5": "ห้า",
          "6": "หก", "7": "เจ็ด", "8": "แปด", "9": "เก้า"}


# ---------------------------------------------------------------- scoring
def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def clean_hyp(toks, rid):
    toks = [t for t in toks if t and t.strip() and t not in ("[start]", "[end]")]
    toks = [DIGITS.get(t, t) for t in toks]
    out, i = [], 0
    while i < len(toks):  # merge "ของ"+"คุณ" -> "ของคุณ" (single sign in P05)
        if toks[i] == "ของ" and i + 1 < len(toks) and toks[i + 1] == "คุณ" and rid == "P05":
            out.append("ของคุณ")
            i += 2
        else:
            out.append(toks[i])
            i += 1
    ign = set(IGNORE.get(rid, set()))
    if "ผม" in ign:
        ign.add("ฉัน")
    out = [t for t in out if t not in ign]
    for t in TRAILING_IGNORE.get(rid, ()):
        if out and out[-1] == t:
            out.pop()
    return out


def score_row(rid, hyp, refs):
    hyp = clean_hyp(hyp, rid)
    best = min(refs, key=lambda r: edit(r, hyp))
    exact = any(hyp == r for r in refs)
    tok_acc = 1 - edit(best, hyp) / max(len(best), len(hyp), 1)
    tp = TIME_PHRASE.get(rid)
    time_ok = "" if tp is None else int(hyp[:len(tp)] == tp)
    return hyp, int(exact), round(tok_acc, 3), time_ok


# ---------------------------------------------------------------- systems
def load_gold():
    rows = list(csv.DictReader(open(GOLD, encoding="utf-8-sig")))
    out = []
    for r in rows:
        if r["tier"] not in SCORED_TIERS:
            continue
        refs = [r["gloss_reference"].split()]
        if r["alt_reference"].strip():
            refs.append(r["alt_reference"].split())
        out.append((r["id"], r["thai_sentence"], refs))
    return out


STRIP = (string.punctuation + "¿").replace("[", "").replace("]", "")


def standardize_py(s):
    return re.sub("[%s]" % re.escape(STRIP), "", s)


def make_seq2seq(csv_path, vocab_path, model_path, engine):
    """Rebuild vectorizers exactly as compare_models.py does, then greedy-decode."""
    import random
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import TextVectorization
    from pythainlp import word_tokenize
    from pronouns import normalize_pronouns
    from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding

    def standardize_tf(x):
        return tf.strings.regex_replace(x, "[%s]" % re.escape(STRIP), "")

    df = pd.read_csv(csv_path)
    vocab_size = len(open(vocab_path, encoding="utf8").read().split("\n"))
    pairs = df.values.tolist()
    random.Random(10).shuffle(pairs)
    n_val = int(0.1 * len(pairs))
    train_pairs = pairs[:len(pairs) - 2 * n_val]

    thai_vec = TextVectorization(max_tokens=vocab_size, output_mode="int",
                                 output_sequence_length=SEQ_LEN)
    thsl_vec = TextVectorization(max_tokens=vocab_size, output_mode="int",
                                 output_sequence_length=SEQ_LEN + 1,
                                 standardize=standardize_tf)
    thai_vec.adapt([p[0] for p in train_pairs])
    thsl_vec.adapt([p[1] for p in train_pairs])
    vocab = thsl_vec.get_vocabulary()
    start_id = vocab.index("[start]")
    model = keras.models.load_model(
        model_path,
        custom_objects={"TransformerEncoder": TransformerEncoder,
                        "TransformerDecoder": TransformerDecoder,
                        "PositionalEmbedding": PositionalEmbedding},
        compile=False)

    def run(raw):
        text = normalize_pronouns(raw, engine)
        src = " ".join(t for t in word_tokenize(text, engine=engine) if t.strip())
        enc = thai_vec(tf.constant([src])).numpy()
        dec = np.zeros((1, SEQ_LEN), dtype="int64")
        dec[:, 0] = start_id
        toks = []
        for i in range(SEQ_LEN):
            nxt = int(model([enc, dec], training=False).numpy()[0, i].argmax())
            t = vocab[nxt]
            if t == "[end]":
                break
            toks.append(t)
            if i + 1 < SEQ_LEN:
                dec[0, i + 1] = nxt
        return toks

    return run


def build_systems(skip_models):
    systems = {}
    import thai2thsl as T  # loads the app model (thai2thsl-deepcut.hdf5)
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import Trie

    systems["A_app_deepcut"] = lambda raw: T.thai_to_tsl_glosses(raw)
    systems["A2_app_skip_unknown"] = lambda raw: T.thai_to_tsl_glosses(raw, skip_unknown=True)

    trie = Trie(T.thai_known_words)

    def dict_system(engine):
        def run(raw):
            kw = {"custom_dict": trie} if engine in CUSTOM_DICT_ENGINES else {}
            segs = [t for t in word_tokenize(raw, engine=engine, **kw) if t.strip()]
            return T.decode_sequence(" ".join(segs)).split()
        return run

    for eng in TOKENIZER_ENGINES:
        systems["D_" + eng] = dict_system(eng)

    if not skip_models:
        for name, cfg, eng in (
            ("B_deepcut_dedup",
             ("datasets/deepcut_dedup.csv", "datasets/th_vocab-deepcut-dedup.txt",
              "models/thai2thsl-deepcut-dedup.hdf5"), "deepcut"),
            ("C_newmm_dedup",
             ("datasets/newmm_dedup.csv", "datasets/th_vocab-newmm-dedup.txt",
              "models/thai2thsl-newmm-dedup.hdf5"), "newmm"),
        ):
            try:
                systems[name] = make_seq2seq(*cfg, eng)
            except Exception as e:  # keep going if one model fails to load
                print("SKIP", name, type(e).__name__, e, file=sys.stderr)
    return systems


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    gold = load_gold()
    systems = build_systems(args.skip_models)

    records, summary = [], {}
    for name, fn in systems.items():
        ex = tk = 0
        t_ok = t_n = 0
        for rid, thai, refs in gold:
            try:
                raw_hyp = list(fn(thai))
            except Exception as e:
                raw_hyp = ["<ERR:%s>" % type(e).__name__]
            hyp, exact, tok_acc, time_ok = score_row(rid, raw_hyp, refs)
            ex += exact
            tk += tok_acc
            if time_ok != "":
                t_n += 1
                t_ok += time_ok
            records.append([name, rid, thai, " | ".join(" ".join(r) for r in refs),
                           " ".join(raw_hyp), " ".join(hyp), exact, tok_acc, time_ok])
        n = len(gold)
        summary[name] = (ex, n, tk / n, t_ok, t_n)

    with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "id", "thai", "reference(s)", "raw_hypothesis",
                    "scored_hypothesis", "exact", "token_acc", "time_initial"])
        w.writerows(records)

    print("\nSUMMARY (12 scored gold rows; supporting evidence, not an accuracy figure)")
    print("{:22s} {:>9s} {:>9s} {:>10s}".format("system", "exact", "token_acc", "time_first"))
    for name, (ex, n, ta, tok, tn) in summary.items():
        print("{:22s} {:>6d}/{:<2d} {:>9.3f} {:>7d}/{:<2d}".format(name, ex, n, ta, tok, tn))

    print("\nPER ROW (exact=1/0), columns follow the summary order")
    names = list(systems)
    print("{:5s} ".format("id") + " ".join("{:>8s}".format(n[:8]) for n in names))
    for rid, _, _ in gold:
        cells = []
        for n in names:
            rec = next(r for r in records if r[0] == n and r[1] == rid)
            cells.append("{:>8s}".format("OK" if rec[6] else "x"))
        print("{:5s} ".format(rid) + " ".join(cells))
    print("\nDetails in", OUT)


if __name__ == "__main__":
    main()

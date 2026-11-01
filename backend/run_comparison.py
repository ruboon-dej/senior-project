"""
Compare gloss backends on the SAME test sentences and log everything.

  python run_comparison.py --backends transformer mt5 llm --n 300 --tag thesis_v1

Output: logs/<timestamp>_<tag>/
  run.log                    full console/file log
  config.json                settings (reproducibility)
  per_sentence_<backend>.csv every sentence: ref, prediction, scores, latency
  summary.csv / summary.json / summary.md   table for the thesis
"""
import argparse, json, logging, os, random, re, time
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from data_split import load_split, thai_raw, gloss_clean

def norm(g):                      # แขก(2) == แขก2, จ.ภูเก็ต == จภูเก็ต, หนุ่ม/สาว == หนุ่มสาว
    return re.sub(r"[()./]", "", g)

def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]

def bag_f1(ref, hyp):
    if not ref and not hyp: return 1.0
    common = sum((Counter(ref) & Counter(hyp)).values())
    if common == 0: return 0.0
    p, r = common / len(hyp), common / len(ref)
    return 2 * p * r / (p + r)

def bootstrap_ci(values, iters=2000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["transformer", "mt5"])
    ap.add_argument("--n", type=int, default=300, help="0 = whole test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    out_dir = os.path.join("logs", f"{datetime.now():%Y%m%d_%H%M%S}_{a.tag}")
    os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf8"),
                                  logging.StreamHandler()])
    log = logging.getLogger("compare")

    train, val, test = load_split()
    if a.n and a.n < len(test):
        test = random.Random(a.seed).sample(test, a.n)
    train_tokens = {t for p in train for t in p[0].split()}
    vocab = {g for p in train + val + test for g in gloss_clean(p[1]).split()}
    json.dump({**vars(a), "n_test_used": len(test), "n_train": len(train)},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    log.info("Testing %d sentences, backends=%s", len(test), a.backends)

    import gloss_backends
    summary = []
    for backend in a.backends:
        gloss_backends.BACKEND = backend
        log.info("=== backend: %s ===", backend)
        rows = []
        for k, (thai, thsl) in enumerate(test, 1):
            ref = gloss_clean(thsl).split()
            ref_n = [norm(g) for g in ref]
            err, hyp = "", []
            t0 = time.perf_counter()
            try:
                hyp = list(gloss_backends.get_glosses(thai_raw(thai)))
            except Exception as e:
                err = repr(e); log.error("sentence %d failed: %s", k, err)
            latency = time.perf_counter() - t0
            hyp_n = [norm(g) for g in hyp]
            rows.append({
                "thai": thai_raw(thai), "reference": " ".join(ref), "prediction": " ".join(hyp),
                "exact_strict": int(ref == hyp), "exact_norm": int(ref_n == hyp_n),
                "wer_strict": edit_distance(ref, hyp) / max(len(ref), 1),
                "wer_norm": edit_distance(ref_n, hyp_n) / max(len(ref_n), 1),
                "bag_f1": bag_f1(ref_n, hyp_n),
                "order_only": int(ref_n != hyp_n and sorted(ref_n) == sorted(hyp_n)),
                "has_oov": int(any(t not in train_tokens for t in thai.split())),
                "ref_len": len(ref),
                "invalid_gloss": int(any(g not in vocab for g in hyp)),
                "repeated": int(any(x == y for x, y in zip(hyp, hyp[1:]))),
                "latency_s": latency, "error": err,
            })
            if k % 25 == 0:
                log.info("  %d/%d", k, len(test))
                pd.DataFrame(rows).to_csv(
                    os.path.join(out_dir, f"per_sentence_{backend}.csv"),
                    index=False,
                    encoding="utf-8-sig"
                )
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"per_sentence_{backend}.csv"), index=False, encoding="utf-8-sig")

        bleu = BLEU(tokenize="none").corpus_score([" ".join(norm(g) for g in r.split()) for r in df.prediction],
                 [[" ".join(norm(g) for g in r.split()) for r in df.reference]])
        chrf = CHRF().corpus_score([" ".join(norm(g) for g in r.split()) for r in df.prediction],
                 [[" ".join(norm(g) for g in r.split()) for r in df.reference]])
        lo, hi = bootstrap_ci(df.wer_norm)
        oov, seen = df[df.has_oov == 1], df[df.has_oov == 0]
        s = {
            "backend": backend, "n": len(df),
            "exact_norm": df.exact_norm.mean(), "exact_strict": df.exact_strict.mean(),
            "wer_norm": df.wer_norm.mean(), "wer_norm_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "wer_strict": df.wer_strict.mean(),
            "bleu4": bleu.score, "chrf": chrf.score, "bag_f1": df.bag_f1.mean(),
            "order_only": df.order_only.mean(), "invalid_gloss": df.invalid_gloss.mean(),
            "repeated": df.repeated.mean(), "errors": int((df.error != "").sum()),
            "n_oov": len(oov), "wer_oov": oov.wer_norm.mean() if len(oov) else None,
            "wer_seen": seen.wer_norm.mean() if len(seen) else None,
            "latency_mean_s": df.latency_s.mean(), "latency_p95_s": df.latency_s.quantile(0.95),
        }
        summary.append(s)
        log.info("%s", json.dumps(s, ensure_ascii=False, default=float))

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=float)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf8") as f:
        f.write(sdf.round(3).to_markdown(index=False))
    log.info("Saved to %s", out_dir)

if __name__ == "__main__":
    main()
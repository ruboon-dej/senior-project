"""
Measure how well the Thai -> gloss model does on its held-out test set.

Run from the backend folder, inside the (sign) env:
    python evaluate_model.py            # 300 random test sentences
    python evaluate_model.py --n 1000   # more sentences (slower)
    python evaluate_model.py --n 0      # the whole test set (slow)

For every sentence it decodes the Thai text with the model and compares the
predicted glosses with the reference glosses from the dataset:

  exact match   the predicted gloss sequence is identical to the reference
  WER           word error rate = edit distance / reference length (lower is better)
  bag F1        overlap of glosses ignoring order (is the vocabulary right?)
  order-only    sentences whose glosses are right but in the wrong order
  breakdown     the same scores split by whether the Thai input contains a token
                the model never saw in training (out-of-vocabulary)

Everything is written to gloss_eval/results.csv (open it in Excel / Google Sheets,
sort by wer to read the worst sentences first).
"""

import argparse
import csv
import os
import random
import re
from collections import Counter
from typing import List


def clean_reference(thsl_text: str) -> List[str]:
    """'[start] a b c [end]' -> ['a', 'b', 'c']"""
    return thsl_text.replace("[start]", "").replace("[end]", "").split()


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def bag_f1(ref: List[str], hyp: List[str]) -> float:
    if not ref and not hyp:
        return 1.0
    common = sum((Counter(ref) & Counter(hyp)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(hyp), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def summarize(rows: list) -> dict:
    """Scores for a group of result rows (empty group -> zeros)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "exact": 0.0, "wer": 0.0}
    return {
        "n": n,
        "exact": sum(r["exact"] for r in rows) / n,
        "wer": sum(r["wer"] for r in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="sentences to test (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-repeats",
        action="store_true",
        help="decode like before (no guard against repeating a gloss back to back)",
    )
    args = parser.parse_args()

    # imported here so the metric functions above can be reused without loading the model
    import thai2thsl

    pairs = list(thai2thsl.test_pairs)
    total = len(pairs)
    if args.n and args.n < total:
        pairs = random.Random(args.seed).sample(pairs, args.n)
    print(f"\nTesting {len(pairs)} of {total} test sentences ...\n")

    rows = []
    for k, (thai_text, thsl_text) in enumerate(pairs, start=1):
        ref = clean_reference(thsl_text)
        hyp = thai2thsl.decode_sequence(
            thai_text, block_repeats=not args.allow_repeats
        ).split()
        wer = edit_distance(ref, hyp) / max(len(ref), 1)
        unknown = [t for t in thai_text.split() if thai2thsl._is_unknown_token(t)]
        rows.append(
            {
                "thai": thai_text,
                "unknown_tokens": " ".join(unknown),
                "has_unknown": int(bool(unknown)),
                "reference": " ".join(ref),
                "prediction": " ".join(hyp),
                "repeated": int(any(a == b for a, b in zip(hyp, hyp[1:]))),
                "exact": int(ref == hyp),
                "wer": round(wer, 3),
                "bag_f1": round(bag_f1(ref, hyp), 3),
                "order_only": int(ref != hyp and sorted(ref) == sorted(hyp)),
            }
        )
        if k % 25 == 0:
            print(f"  {k}/{len(pairs)}")

    n = len(rows)
    exact = sum(r["exact"] for r in rows) / n
    wer = sum(r["wer"] for r in rows) / n
    f1 = sum(r["bag_f1"] for r in rows) / n
    order_only = sum(r["order_only"] for r in rows) / n

    os.makedirs("gloss_eval", exist_ok=True)
    out = os.path.join("gloss_eval", "results.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r["wer"]))

    print("\n" + "=" * 60)
    print(f"sentences tested : {n}")
    print(f"exact match      : {exact:.1%}")
    print(f"mean WER         : {wer:.3f}   (0 = perfect)")
    print(f"mean bag F1      : {f1:.3f}   (gloss choice, ignoring order)")
    print(f"order-only errors: {order_only:.1%}   (right glosses, wrong order)")
    print(f"repeated gloss   : {sum(r['repeated'] for r in rows)} sentences "
          f"({'guard OFF' if args.allow_repeats else 'guard ON'})")
    known = summarize([r for r in rows if not r["has_unknown"]])
    oov = summarize([r for r in rows if r["has_unknown"]])
    print("\nbreakdown by input vocabulary:")
    print(f"  all tokens seen in training : {known['n']:4d} sentences, "
          f"exact {known['exact']:.1%}, WER {known['wer']:.3f}")
    print(f"  has unseen (OOV) token      : {oov['n']:4d} sentences, "
          f"exact {oov['exact']:.1%}, WER {oov['wer']:.3f}")
    print("\nworst 5 sentences:")
    for r in sorted(rows, key=lambda r: -r["wer"])[:5]:
        print(f"  thai : {r['thai']}   [unseen: {r['unknown_tokens'] or '-'}]")
        print(f"  ref  : {r['reference']}")
        print(f"  pred : {r['prediction']}")
        print("  " + "-" * 40)
    print(f"\nfull results: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

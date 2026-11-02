import random, re
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback)

MODEL = "google/mt5-small"
OUT = "models/thai2gloss-mt5"

def thai(s):  return re.sub(r"\s+", "", s)                    # drop deepcut spaces
def gloss(s): return s.replace("[start]", "").replace("[end]", "").strip()

pairs = pd.read_csv("datasets/deepcut.csv").values.tolist()
random.Random(10).shuffle(pairs)
n_val = int(0.1 * len(pairs)); n_train = len(pairs) - 2 * n_val
split = {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val]}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def encode(batch):
    enc = tok(batch["src"], max_length=64, truncation=True)
    enc["labels"] = tok(text_target=batch["tgt"], max_length=64, truncation=True)["input_ids"]
    return enc

def make(rows):
    ds = Dataset.from_dict({"src": [thai(r[0]) for r in rows],
                            "tgt": [gloss(r[1]) for r in rows]})
    return ds.map(encode, batched=True, remove_columns=["src", "tgt"])

args = Seq2SeqTrainingArguments(
    output_dir=OUT, learning_rate=5e-4, num_train_epochs=40,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    bf16=True,                      # mT5 is unstable in fp16; 3080 Ti supports bf16
    warmup_ratio=0.05, weight_decay=0.01, report_to="none",
)
Seq2SeqTrainer(
    model=model, args=args, train_dataset=make(split["train"]),
    eval_dataset=make(split["val"]),
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
).train()
model.save_pretrained(OUT); tok.save_pretrained(OUT)
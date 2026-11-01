import os, re
BACKEND = os.getenv("GLOSS_BACKEND", "transformer")   # transformer | mt5 | llm | auto
_mt5 = None

def _load_mt5():
    global _mt5
    if _mt5 is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        p = "models/thai2gloss-mt5"
        _mt5 = (AutoTokenizer.from_pretrained(p), AutoModelForSeq2SeqLM.from_pretrained(p).eval())
    return _mt5

def mt5_glosses(text, beams=5):
    import torch
    from llm_gloss import VOCAB
    tok, model = _load_mt5()
    enc = tok(re.sub(r"\s+", "", text), return_tensors="pt")
    with torch.no_grad():
        outs = model.generate(**enc, num_beams=beams, num_return_sequences=beams,
                              max_new_tokens=48)
    cands = [tok.decode(o, skip_special_tokens=True).split() for o in outs]
    for c in cands:                          # first n-best entry fully inside the vocabulary
        if c and all(g in VOCAB for g in c):
            return c, True
    return cands[0].copy(), False

def get_glosses(text):
    if BACKEND == "transformer":
        from thai2thsl import thai_to_tsl_glosses
        return list(thai_to_tsl_glosses(text))

    if BACKEND == "llm":
        from llm_gloss import llm_glosses
        return list(llm_glosses(text))

    glosses, ok = mt5_glosses(text)

    if BACKEND == "auto" and not ok:
        from llm_gloss import llm_glosses
        return list(llm_glosses(text))

    return list(glosses)
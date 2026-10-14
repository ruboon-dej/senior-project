from pythainlp import word_tokenize

HAIR_WORDS = {"ร่วง", "ยาว", "สั้น", "หงอก", "ดำ", "หยิก", "สวย", "ยุ่ง"}
NO_DATA = {"เรา", "มัน"}


def normalize_pronouns(text, engine):
    tokens = word_tokenize(text, engine=engine)
    out = []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t == "ดิฉัน":
            out.append("ฉัน")
        elif t == "ผม" and i == 0 and nxt not in HAIR_WORDS:
            out.append("ฉัน")
        else:
            out.append(t)
    return "".join(out)


def unsupported_pronouns(text, engine):
    return [t for t in word_tokenize(text, engine=engine) if t in NO_DATA]

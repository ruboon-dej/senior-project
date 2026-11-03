from pythainlp import word_tokenize

HAIR_WORDS = {"ร่วง", "ยาว", "สั้น", "หงอก", "ดำ", "หยิก", "สวย", "ยุ่ง"}
HAIR_VERBS = {"ตัด", "ย้อม", "สระ", "หวี", "ดัด", "ไดร์", "เป่า", "ซอย", "ถัก", "มัด"}
TO_FIRST_PERSON = {"ดิฉัน", "ข้าพเจ้า"}
DROPPED = {"มัน", "ท่าน"}
NO_DATA = DROPPED


def is_pronoun_pom(tokens, i):
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    return nxt not in HAIR_WORDS and prev not in HAIR_VERBS


def normalize_pronouns(text, engine):
    tokens = word_tokenize(text, engine=engine)
    out = []
    for i, t in enumerate(tokens):
        if t in TO_FIRST_PERSON:
            out.append("ฉัน")
        elif t == "ผม" and is_pronoun_pom(tokens, i):
            out.append("ฉัน")
        elif t == "เรา":
            out.append("พวกเรา")
        elif t in DROPPED:
            continue
        else:
            out.append(t)
    return "".join(out)


def unsupported_pronouns(text, engine):
    found = [t for t in word_tokenize(text, engine=engine) if t in NO_DATA]
    return list(dict.fromkeys(found))
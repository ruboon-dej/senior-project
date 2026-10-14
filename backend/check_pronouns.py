import contextlib
import io
import os

from thai2thsl import thai_to_tsl_glosses, lookup, thai_vectorization

PRONOUNS = [
    "ฉัน", "ผม", "ดิฉัน", "เรา", "พวกเรา", "คุณ",
    "เธอ", "เขา", "พวกเขา", "มัน", "ท่าน", "หนู",
]
TEMPLATES = ["{}กินข้าว", "{}ไปตลาด"]

vocab = set(thai_vectorization.get_vocabulary())
pose_names = {f[:-5] for f in os.listdir("e-tsl-pose") if f.endswith(".pose")}

for pronoun in PRONOUNS:
    has_pose = pronoun in pose_names or any(n.startswith(pronoun + "(") for n in pose_names)
    print(f"{pronoun}: in_vocab={pronoun in vocab} pose_file={has_pose}")
    for template in TEMPLATES:
        sentence = template.format(pronoun)
        with contextlib.redirect_stdout(io.StringIO()):
            glosses = thai_to_tsl_glosses(sentence)
            _, not_found = lookup(list(glosses))
        print(f"    {sentence} -> {glosses} not_found={not_found}")

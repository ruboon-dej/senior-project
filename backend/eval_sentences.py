import csv
import json
import sys
import urllib.request

URL = "http://localhost:8000/api/translate/thai-to-thsl"

SENTENCES = [
    "ผมกินข้าว",
    "ดิฉันกินข้าว",
    "ฉันกินข้าว",
    "ผมไปตลาด",
    "ฉันไปตลาด",
    "ผมชอบกินข้าว",
    "ดิฉันชอบไปโรงเรียน",
    "ผมร่วง",
    "ผมยาว",
    "ผมสวย",
    "ผมตัดเสร็จแล้ว",
    "ผมหลุดออกมา",
    "ผมหงอก",
    "เขาเห็นผม",
    "เขาตัดผม",
    "เขาถามผม",
    "แม่หวีผม",
    "เรากินข้าว",
    "เราไปโรงเรียน",
    "มันกินข้าว",
    "มันฝรั่งอร่อย",
    "พวกเรากินข้าว",
    "เขากินข้าว",
    "คุณกินข้าวหรือยัง",
    "ผู้หญิงกินข้าว",
    "ผู้ชายไปตลาด",
    "แม่ไปตลาด",
    "พ่อกินข้าว",
    "ฉันไปโรงพยาบาล",
    "ฉันชอบเธอ",
    "เขาชอบฉัน",
    "ฉันรักแม่",
    "วันนี้ฝนตก",
    "ฉันหิวข้าว",
    "ฉันเรียนหนังสือ",
    "เด็กกินข้าว",
    "ครูสอนหนังสือ",
    "เขาไปโรงเรียน",
    "ฉันซื้อข้าว",
    "ฉันอยากกินข้าว",
]


def translate(text):
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def run(label):
    path = f"results_{label}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["input", "glosses", "not_found", "message"])
        for s in SENTENCES:
            try:
                d = translate(s)
                w.writerow([s, " ".join(d.get("glosses", [])), " ".join(d.get("notFound", [])), d.get("missingMessage") or ""])
            except Exception as e:
                w.writerow([s, "ERROR", "", str(e)])
    print("wrote", path)


def read(label):
    with open(f"results_{label}.csv", newline="", encoding="utf-8-sig") as f:
        return {r["input"]: r for r in csv.DictReader(f)}


def diff(a, b):
    ra, rb = read(a), read(b)
    changed = 0
    print("| input | " + a + " | " + b + " |")
    print("| --- | --- | --- |")
    for s in SENTENCES:
        if s in ra and s in rb and ra[s]["glosses"] != rb[s]["glosses"]:
            changed += 1
            print(f"| {s} | {ra[s]['glosses']} | {rb[s]['glosses']} |")
    print(f"\n{changed} of {len(SENTENCES)} sentences changed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        print("usage: eval_sentences.py run <label> | diff <label_a> <label_b>")

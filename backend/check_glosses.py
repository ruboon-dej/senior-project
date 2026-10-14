"""
Check a Thai sentence word by word (backend only, no frontend needed).

Run from the backend folder, inside the (sign) env:
    python check_glosses.py "แนคชอบขับรถ"

It will:
  1. print the tokens, the unknown words and the predicted glosses
  2. render one short video per gloss (and one for the whole sentence)
  3. write gloss_check/index.html -> open it in a browser to review each sign
"""

import html
import os
import re
import shutil
import sys

from pythainlp import word_tokenize

from thai2thsl import (
    TOKENIZER_ENGINE,
    find_unknown_words,
    lookup,
    merge_multiword_glosses,
    pose_directory,
    pose_sequence,
    thai_to_tsl_glosses,
)

OUT_DIR = "gloss_check"


def safe_name(text: str) -> str:
    return re.sub(r"[^\w()\-]+", "_", text, flags=re.UNICODE).strip("_") or "gloss"


def render(pose_list, out_path: str) -> None:
    """pose_sequence() always writes quick_test.mp4, so move it afterwards."""
    pose_sequence(pose_list)
    shutil.move("quick_test.mp4", out_path)


def main() -> None:
    sentence = " ".join(sys.argv[1:]).strip()
    if not sentence:
        sys.exit('usage: python check_glosses.py "ประโยคภาษาไทย"')

    tokens = word_tokenize(sentence, engine=TOKENIZER_ENGINE)
    unknown = find_unknown_words(sentence)
    glosses = thai_to_tsl_glosses(sentence)

    available = {
        f[:-5]
        for f in os.listdir(os.path.join(os.getcwd(), pose_directory))
        if f.endswith(".pose")
    }
    # same merging that lookup() applies, so each item matches what the web renders
    merged = merge_multiword_glosses(list(glosses), available)

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR)

    rows = []
    for i, gloss in enumerate(merged, start=1):
        status, video = "", None
        pose_list, not_found = lookup([gloss])
        if not pose_list:
            status = "ไม่มีไฟล์ท่า (pose) สำหรับคำนี้"
        else:
            name = f"{i:02d}_{safe_name(gloss)}.mp4"
            try:
                render(pose_list, os.path.join(OUT_DIR, name))
                video = name
                status = "ok"
            except Exception as err:  # keep going so one bad sign doesn't stop the check
                status = f"เรนเดอร์ไม่สำเร็จ: {err}"
        rows.append((i, gloss, status, video))

    # whole sentence, for comparison with the single signs
    full_video, full_status = None, ""
    pose_list, not_found = lookup(list(glosses))
    if pose_list:
        try:
            render(pose_list, os.path.join(OUT_DIR, "full_sentence.mp4"))
            full_video = "full_sentence.mp4"
        except Exception as err:
            full_status = f"เรนเดอร์ไม่สำเร็จ: {err}"
    else:
        full_status = "ไม่มีท่าให้เรนเดอร์"

    esc = html.escape
    token_html = " | ".join(
        f'<b style="color:#d9730d">{esc(t)}</b>' if t in unknown else esc(t)
        for t in tokens
    )
    body = []
    for i, gloss, status, video in rows:
        if video:
            cell = f'<video src="{esc(video)}" controls loop width="260"></video>'
        else:
            cell = f'<span style="color:#d9730d">{esc(status)}</span>'
        body.append(f"<tr><td>{i}</td><td>{esc(gloss)}</td><td>{cell}</td></tr>")

    full_cell = (
        f'<video src="{full_video}" controls loop width="320"></video>'
        if full_video
        else f'<span style="color:#d9730d">{esc(full_status)}</span>'
    )

    page = f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><title>Gloss check</title>
<style>
body{{font-family:sans-serif;margin:24px;max-width:760px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px;vertical-align:middle}}
th{{background:#f4f4f8;text-align:left}}
</style></head><body>
<h2>{esc(sentence)}</h2>
<p><b>ตัดคำ (deepcut):</b> {token_html}</p>
<p><b>คำที่ระบบไม่รู้จัก:</b> {esc(", ".join(unknown)) or "-"}</p>
<p><b>Glosses ที่โมเดลทำนาย:</b> {esc(" · ".join(merged))}</p>
<h3>วิดีโอทั้งประโยค</h3>{full_cell}
<h3>ทีละคำ</h3>
<table><tr><th>#</th><th>gloss</th><th>ท่า</th></tr>{"".join(body)}</table>
</body></html>"""
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf8") as f:
        f.write(page)

    print("\n" + "=" * 60)
    print("tokens  :", tokens)
    print("unknown :", unknown)
    print("glosses :", merged)
    for i, gloss, status, video in rows:
        print(f"  {i:02d} {gloss}: {status}")
    print(f"\nเปิดดูได้ที่ {os.path.abspath(os.path.join(OUT_DIR, 'index.html'))}")


if __name__ == "__main__":
    main()

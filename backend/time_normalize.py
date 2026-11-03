TIME_WORDS = sorted(["เมื่อวาน", "เช้า", "บ่าย", "กลางคืน", "เมื่อก่อน", "ตลอดคืน"], key=len, reverse=True)
NEED_TON = {"เช้า", "บ่าย", "กลางคืน"}


def move_time_to_end(text):
    s = text.strip()
    for t in TIME_WORDS:
        for p in ("ตอน" + t, t):
            if s.startswith(p) and len(s) > len(p):
                rest = s[len(p):].strip()
                return rest + ("ตอน" if t in NEED_TON else "") + t
    return s

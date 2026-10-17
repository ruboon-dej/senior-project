from typing import List, Set, Tuple

from pythainlp.corpus import thai_female_names, thai_male_names

MIN_NAME_LENGTH = 3
MAX_WINDOW = 4

NAME_SET = {
    name
    for name in set(thai_male_names()) | set(thai_female_names())
    if len(name) >= MIN_NAME_LENGTH
}

THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")


def extract_names(tokens: List[str], vocab: Set[str]) -> Tuple[List[str], List[str]]:
    remaining = []
    names = []
    i = 0
    while i < len(tokens):
        matched = False
        for width in range(min(MAX_WINDOW, len(tokens) - i), 0, -1):
            candidate = "".join(tokens[i:i + width])
            if candidate in NAME_SET and candidate not in vocab:
                names.append(candidate)
                i += width
                matched = True
                break
        if not matched:
            remaining.append(tokens[i])
            i += 1
    return remaining, names


def fingerspell(name: str) -> List[str]:
    return [char for char in name if char in THAI_CONSONANTS]

MAX_SPELL_LETTERS = 5


def split_by_names(tokens, is_unknown):
    segments = []
    buffer = []
    i = 0
    while i < len(tokens):
        matched = False
        for width in range(min(MAX_WINDOW, len(tokens) - i), 0, -1):
            candidate = "".join(tokens[i:i + width])
            if candidate in NAME_SET and is_unknown(candidate):
                if buffer:
                    segments.append(("text", "".join(buffer)))
                    buffer = []
                segments.append(("name", candidate))
                i += width
                matched = True
                break
        if not matched:
            buffer.append(tokens[i])
            i += 1
    if buffer:
        segments.append(("text", "".join(buffer)))
    return segments
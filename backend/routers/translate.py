import os
import shutil
import threading
import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pythainlp import word_tokenize
from models.schemas import (
    ThaiToThSLRequest,
    ThaiToThSLResponse,
    ThSLToThaiResponse,
)
from thai2thsl import thai_to_tsl_glosses, lookup, pose_sequence, find_unknown_words
from pronouns import normalize_pronouns, unsupported_pronouns
from thai2thsl import TOKENIZER_ENGINE
from name_handling import split_by_names, fingerspell, MAX_SPELL_LETTERS

router = APIRouter(prefix="/api/translate", tags=["translate"])

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pose_sequence_lock = threading.Lock()


@router.post("/thai-to-thsl", response_model=ThaiToThSLResponse)
def thai_to_thsl(payload: ThaiToThSLRequest, request: Request) -> ThaiToThSLResponse:
    text = normalize_pronouns(payload.text, TOKENIZER_ENGINE)
    if not text.strip():
        raise HTTPException(
            status_code=404,
            detail="No translatable words after removing pronouns without signs.",
        )

    tokens = word_tokenize(text, engine=TOKENIZER_ENGINE)
    segments = split_by_names(tokens, lambda word: bool(find_unknown_words(word)))

    glosses_decoded = []
    unknown_words = []
    spelled_names = []
    skipped_names = []

    for kind, value in segments:
        if kind == "name":
            letters = fingerspell(value)
            _, missing_letters = lookup(letters)
            if 0 < len(letters) <= MAX_SPELL_LETTERS and not missing_letters:
                glosses_decoded.extend(letters)
                spelled_names.append(value)
            else:
                skipped_names.append(value)
        elif value.strip():
            unknown_words.extend(find_unknown_words(value))
            glosses_decoded.extend(thai_to_tsl_glosses(value, skip_unknown=True))

    pose_list, not_found_list = lookup(list(glosses_decoded))

    if len(pose_list) == 0:
        if unknown_words:
            detail = "Unknown word(s): " + ", ".join(unknown_words) + ". Cannot translate."
        elif skipped_names:
            detail = "Name(s) cannot be fingerspelled: " + ", ".join(skipped_names) + ". Cannot translate."
        else:
            detail = "No glosses matched the sign database."
        raise HTTPException(status_code=404, detail=detail)

    unique_name = f"{uuid.uuid4().hex}.mp4"
    unique_path = os.path.join(OUTPUT_DIR, unique_name)
    with pose_sequence_lock:
        pose_sequence(pose_list)
        shutil.move("quick_test.mp4", unique_path)

    video_url = str(request.base_url).rstrip("/") + f"/videos/{unique_name}"

    messages = []
    if spelled_names:
        messages.append("Fingerspelled name(s): " + ", ".join(spelled_names))
    if skipped_names:
        messages.append(
            "Name(s) skipped (too long or missing letter signs): " + ", ".join(skipped_names)
        )
    if unknown_words:
        messages.append(
            "Unknown word(s): " + ", ".join(unknown_words) + " (skipped in translation)"
        )
    if not_found_list:
        messages.append("No sign available for: " + ", ".join(not_found_list))
    no_data = unsupported_pronouns(payload.text, TOKENIZER_ENGINE)
    if no_data:
        messages.append(
            "No sign data for pronoun(s): " + ", ".join(no_data) + " (skipped in translation)"
        )
    missing_message = " | ".join(messages) if messages else None

    return ThaiToThSLResponse(
        inputText=payload.text,
        glosses=list(glosses_decoded),
        videoUrl=video_url,
        notFound=not_found_list,
        missingMessage=missing_message,
    )


@router.post("/thsl-to-thai", response_model=ThSLToThaiResponse)
async def thsl_to_thai(video: UploadFile = File(...)) -> ThSLToThaiResponse:
    _ = await video.read()

    return ThSLToThaiResponse(
        text="Example translated text (mock)",
        confidence=0.87,
    )
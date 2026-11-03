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
from thai2thsl import lookup, pose_sequence, find_unknown_words
from pronouns import normalize_pronouns, unsupported_pronouns
from thai2thsl import TOKENIZER_ENGINE
from gloss_backends import get_glosses
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
            detail="ไม่มีคำที่แปลได้หลังตัดสรรพนามที่ไม่มีข้อมูลท่ามือ",
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
            glosses_decoded.extend(get_glosses(value))

    pose_list, not_found_list = lookup(list(glosses_decoded))

    if len(pose_list) == 0:
        if unknown_words:
            detail = "ระบบไม่รู้จักคำ: " + ", ".join(unknown_words) + " จึงแปลไม่ได้"
        elif skipped_names:
            detail = "สะกดชื่อด้วยนิ้วมือไม่ได้: " + ", ".join(skipped_names) + " จึงแปลไม่ได้"
        else:
            detail = "There is no glosses matched with the database"
        raise HTTPException(status_code=404, detail=detail)

    unique_name = f"{uuid.uuid4().hex}.mp4"
    unique_path = os.path.join(OUTPUT_DIR, unique_name)
    with pose_sequence_lock:
        pose_sequence(pose_list)
        shutil.move("quick_test.mp4", unique_path)

    video_url = str(request.base_url).rstrip("/") + f"/videos/{unique_name}"

    messages = []
    if spelled_names:
        messages.append("สะกดชื่อด้วยนิ้วมือ: " + ", ".join(spelled_names))
    if skipped_names:
        messages.append(
            "ข้ามชื่อ (ยาวเกินหรือไม่มีท่าอักษร): " + ", ".join(skipped_names)
        )
    if unknown_words:
        messages.append(
            "ระบบไม่รู้จักคำ: " + ", ".join(unknown_words) + " (ข้ามในการแปล)"
        )
    if not_found_list:
        messages.append("ไม่มีท่าสำหรับ: " + ", ".join(not_found_list))
    no_data = unsupported_pronouns(payload.text, TOKENIZER_ENGINE)
    if no_data:
        messages.append(
            "ไม่มีข้อมูลสรรพนาม: " + ", ".join(no_data) + " (ข้ามในการแปล)"
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
        text="ตัวอย่างข้อความที่แปลได้ (mock)",
        confidence=0.87,
    )

"""
Translation endpoints.

thai-to-thsl now calls the real Thai2ThSL pipeline (thai2thsl.py, copied
into this backend folder). thsl-to-thai is still mocked — nobody has built
that direction yet.
"""

import os
import shutil
import threading
import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from models.schemas import (
    ThaiToThSLRequest,
    ThaiToThSLResponse,
    ThSLToThaiResponse,
)
from thai2thsl import thai_to_tsl_glosses, lookup, pose_sequence, find_unknown_words
from pronouns import normalize_pronouns, unsupported_pronouns
from thai2thsl import TOKENIZER_ENGINE

router = APIRouter(prefix="/api/translate", tags=["translate"])

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serializes pose_sequence() calls: it always writes to the fixed filename
# "quick_test.mp4" in the current working directory, so two requests
# rendering at the same time would overwrite each other's output.
pose_sequence_lock = threading.Lock()


@router.post("/thai-to-thsl", response_model=ThaiToThSLResponse)
def thai_to_thsl(payload: ThaiToThSLRequest, request: Request) -> ThaiToThSLResponse:
    text = normalize_pronouns(payload.text, TOKENIZER_ENGINE)
    unknown_words = find_unknown_words(text)
    glosses_decoded = thai_to_tsl_glosses(text)

    pose_list, not_found_list = lookup(list(glosses_decoded))

    if len(pose_list) == 0:
        raise HTTPException(
            status_code=404,
            detail="There is no glosses matched with the database",
        )

    unique_name = f"{uuid.uuid4().hex}.mp4"
    unique_path = os.path.join(OUTPUT_DIR, unique_name)
    with pose_sequence_lock:
        pose_sequence(pose_list)
        shutil.move("quick_test.mp4", unique_path)

    video_url = str(request.base_url).rstrip("/") + f"/videos/{unique_name}"

    messages = []
    if unknown_words:
        messages.append(
            "ระบบไม่รู้จักคำ: " + ", ".join(unknown_words) + " (ผลแปลอาจไม่ถูกต้อง)"
        )
    if not_found_list:
        messages.append("ไม่มีท่าสำหรับ: " + ", ".join(not_found_list))
    no_data = unsupported_pronouns(payload.text, TOKENIZER_ENGINE)
    if no_data:
        messages.append(
            "ไม่มีข้อมูลสรรพนาม: " + ", ".join(no_data) + " (ผลแปลอาจไม่ถูกต้อง)"
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
    """
    TODO: replace with a real call, e.g.:

        from thsl2thai.pipeline import recognize
        text, confidence = recognize(await video.read())

    For now: returns a fixed placeholder sentence regardless of the
    uploaded video's actual content.
    """
    _ = await video.read()  # read to simulate handling the upload

    return ThSLToThaiResponse(
        text="ตัวอย่างข้อความที่แปลได้ (mock)",
        confidence=0.87,
    )

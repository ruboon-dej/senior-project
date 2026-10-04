"""
Translation endpoints.

Everything in this file is currently MOCKED so the frontend has something
real to talk to. When the actual pipeline (see the `thai2thsl/` directory in
the SP2023-ThSL model repo) is ready, replace the body of each function with
a call into that pipeline — the request/response shapes (schemas.py) are
already what the frontend expects, so no frontend changes should be needed.
"""

from fastapi import APIRouter, UploadFile, File
from models.schemas import (
    ThaiToThSLRequest,
    ThaiToThSLResponse,
    ThSLToThaiResponse,
)

router = APIRouter(prefix="/api/translate", tags=["translate"])

# A small, freely-embeddable sample video used as a stand-in for a real
# generated ThSL clip until the video-generation pipeline is wired in.
MOCK_VIDEO_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
)


@router.post("/thai-to-thsl", response_model=ThaiToThSLResponse)
def thai_to_thsl(payload: ThaiToThSLRequest) -> ThaiToThSLResponse:
    """
    TODO: replace with a real call, e.g.:

        from thai2thsl.pipeline import translate
        glosses, video_path = translate(payload.text)
        video_url = upload_or_serve(video_path)

    For now: naive whitespace split stands in for the gloss sequence, and a
    fixed sample video stands in for the generated ThSL clip.
    """
    glosses = payload.text.strip().split() or ["(empty)"]

    return ThaiToThSLResponse(
        inputText=payload.text,
        glosses=glosses,
        videoUrl=MOCK_VIDEO_URL,
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

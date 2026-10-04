from pydantic import BaseModel
from typing import List, Optional


class ThaiToThSLRequest(BaseModel):
    text: str


class ThaiToThSLResponse(BaseModel):
    inputText: str
    glosses: List[str]
    videoUrl: str


class ThSLToThaiResponse(BaseModel):
    text: str
    confidence: Optional[float] = None

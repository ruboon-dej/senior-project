from pydantic import BaseModel
from typing import List, Optional


class ThaiToThSLRequest(BaseModel):
    text: str


class ThaiToThSLResponse(BaseModel):
    inputText: str
    glosses: List[str]
    videoUrl: str
    notFound: Optional[List[str]] = None
    missingMessage: Optional[str] = None


class ThSLToThaiResponse(BaseModel):
    text: str
    confidence: Optional[float] = None

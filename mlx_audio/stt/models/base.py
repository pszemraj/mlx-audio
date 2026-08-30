from dataclasses import dataclass
from typing import List, Optional


@dataclass
class STTOutput:
    text: str
    segments: List[dict] = None
    language: str = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    total_time: float = 0.0
    finish_reason: Optional[str] = None
    complete: Optional[bool] = None
    raw_text: Optional[str] = None

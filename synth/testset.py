from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestExample:
    question: str
    reference_answer: str
    reference_contexts: List[str]
    metadata: dict
    valid: bool = True
    errors: Optional[List[str]] = None

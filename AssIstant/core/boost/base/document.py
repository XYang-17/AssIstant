from dataclasses import dataclass
from typing import Any

@dataclass
class Document:
    title: Any
    content: Any

@dataclass
class SearchResult(Document):
    score: float
    chunk: Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class Severity(Enum):
    INFO        = 'info'
    WARNING     = 'warning'
    ERROR       = 'error'
    CRITICAL    = 'critical'


@dataclass
class Alert:
    severity: Severity
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda : datetime.now().isoformat())
    resolved: bool = False


@dataclass
class Suggestion:
    title: str
    detail: str
    action: str
    priority: int


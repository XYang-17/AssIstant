from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from json import dumps


from AssIstant.common.utils import clean_text


class Role(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class Context:
    recent_message: List[Message]
    related_history: List[str]
    portrait: Dict[str, Any]
    summary: str

    def as_prompt(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{clean_text(self.summary)}")
        if self.related_history:
            rh_text = "\n".join(f"- {clean_text(rh)}" for rh in self.related_history[:3])
            parts.append(f"[相关历史]\n{rh_text}")
        if self.portrait:
            parts.append(f"[用户画像]\n{dumps(self.portrait, ensure_ascii=True)}")
        if self.recent_message:
            parts.append("[最近对话]")
            for rm in self.recent_message[-8:]:
                parts.append(f"{rm.role.value}: {clean_text(rm.content)}")
        return "\n\n".join(parts)
    
    
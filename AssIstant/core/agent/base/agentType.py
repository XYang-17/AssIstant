from enum import Enum
from dataclasses import dataclass
from typing import Iterable, List, Set
from json import load

from AssIstant.common.config import AGENT_TYPE_INFO_PATH

with open(AGENT_TYPE_INFO_PATH, 'r', encoding='utf-8') as f:
    _agent_type_info = load(f)


@dataclass
class TypeInfo:
    name: str = None
    static_score: float = None
    intent_score: float = None
    keywords_score_factor: float = None
    max_keywords_score: float = None
    keywords: list[str] = None

    def __init__(
        self,
        name: str
    ):
        self.name = name
        attrs = (
            "static_score",
            "intent_score",
            "keywords_score_factor",
            "max_keywords_score",
            "keywords"
        )
        info = _agent_type_info.get(self.name) or {}
        for attr in attrs:
            setattr(self, attr, info.get(attr) or None)

    # def collaboration(self, msg: str, res: Set):
    #     if self in res: return
    #
    #     if any(kw in msg for kw in self.keywords):
    #         res.add()


class AgentType(Enum):
    GENERAL      = TypeInfo("GENERAL")
    TECHNICAL    = TypeInfo("TECHNICAL")
    ESCALATION   = TypeInfo("ESCALATION")

    @classmethod
    def members(cls) -> Iterable:
        return cls.__members__.values()


if __name__ == "__main__":
    # for aType in AgentType.members():
    #     print(aType)

    a = AgentType.GENERAL
    print(a.value.intent_score)
    print(TypeInfo.intent_score)

    ...
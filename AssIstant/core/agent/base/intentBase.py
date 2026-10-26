from enum import Enum
from typing import List, Optional, Any, Iterable
from abc import ABC, ABCMeta, abstractmethod
from weakref import WeakValueDictionary, finalize
from json import load

from AssIstant.core.agent.base.agentType import AgentType
from AssIstant.common.config import INTENT_PATH
from AssIstant.common.classproperty import classproperty


with open(INTENT_PATH, 'r', encoding='utf-8') as f:
    _intent_cfg = load(f)
_urgency = _intent_cfg["URGENCY"]


class UrgencyLevel(Enum):
    CRITICAL    = _urgency["CRITICAL"]
    HIGH        = _urgency["HIGH"]
    MEDIUM      = _urgency["MEDIUM"]
    LOW         = []


def _unregister_intent(base: Any, name: str):
    base._registry.pop(name, None)

class IntentMeta(ABCMeta):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        if getattr(cls, '__abstractmethods__', None):
            return

        for base in cls.__mro__[1:]:
            if '_registry' in base.__dict__:
                base._registry[cls.name().upper()] = cls
                finalize(cls, _unregister_intent, base, cls.name().upper())


class IntentBase(metaclass=IntentMeta):
    _intent_path: List[str] = ["INTENT"]
    _registry = WeakValueDictionary()

    _urgency: UrgencyLevel = UrgencyLevel.LOW
    _keywords: Optional[List[str]] = None
    _template: Optional[List[str]] = None
    _group: Optional[str] = None
    _agent_type: AgentType = AgentType.GENERAL

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @classproperty
    def urgency(cls) -> UrgencyLevel:
        return cls._urgency

    @classmethod
    @classproperty
    def keywords(cls) -> List[str]:
        if cls._keywords is None:
            cls._load()
        return cls._keywords

    @classmethod
    @classproperty
    def template(cls) -> List[str]:
        if cls._template is None:
            cls._load()
        return cls._template

    @classmethod
    @classproperty
    def group(cls) -> Optional[str]:
        if cls._group is None:
            cls._load()
        return cls._group

    @classmethod
    @classproperty
    def agent_type(cls) -> AgentType:
        return cls._agent_type

    @classmethod
    def names(cls) -> Iterable[Any]:
        return cls._registry.keys()

    @classmethod
    def intents(cls) -> Iterable[Any]:
        return cls._registry.values()

    @classmethod
    def intent(cls, name: str) -> Any:
        try:
            return cls._registry[name.upper()]
        except:
            return None

    @classmethod
    def _find(cls, cfg, name):
        for ip in cls._intent_path:
            cfg = cfg.get(ip, None)
            if cfg is None: return
        return cfg.get(name, None)


    @classmethod
    def _load(cls):
        with open(INTENT_PATH, 'r', encoding='utf-8') as f:
            cfg = load(f)
        cls._group = cls.name().upper()
        intent = cls._find(cfg, cls._group)
        if intent is not None:
            cls._keywords = intent.get("KEYWORDS", [])
            cls._template = intent.get("TEMPLATE", [])
            if (u := intent.get("URGENCY", None)) is not None:
                cls._urgency = UrgencyLevel.__members__[u.upper()]

        group = cfg.get("GROUP", None)
        if group is not None:
            cls._group = group.get(cls._group, "").upper() or cls._group


class GenericIntentBase(IntentBase, ABC):
    _intent_path: List[str] = ["INTENT", "GENERIC"]

    _registry = WeakValueDictionary()


class SpecificIntentBase(IntentBase, ABC):
    _intent_path: List[str] = ["INTENT", "SPECIFIC"]

    _registry = WeakValueDictionary()
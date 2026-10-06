from dataclasses import dataclass, field
from typing import Dict
from json import dumps
from hashlib import md5

from AssIstant.core.agent.base.intentBase import *
from AssIstant.common.cache import LRUCache
from AssIstant.common.utils import clean_text


class OtherIntent(Intent):
    _keywords: Optional[List[str]] = []
    _template: Optional[List[str]] = []
    _group: Optional[str] = "OTHER"

    @classmethod
    def name(cls) -> str:
        return "other"


class QueryIntent(GenericIntent):
    @classmethod
    def name(cls) -> str:
        return "query"


class TechnologyIntent(GenericIntent):
    @classmethod
    def name(cls) -> str:
        return "technology"


class AccountIntent(GenericIntent):
    @classmethod
    def name(cls) -> str:
        return "account"


class EscalationIntent(GenericIntent):
    _urgency: UrgencyLevel = UrgencyLevel.HIGH

    @classmethod
    def name(cls) -> str:
        return "escalation"


class HumanHandoffIntent(SpecificIntent):
    _urgency: UrgencyLevel = UrgencyLevel.HIGH
    @classmethod
    def name(cls) -> str:
        return "human_handoff"


class AccountSecurityIntent(SpecificIntent):
    @classmethod
    def name(cls) -> str:
        return "account_security"


class LogisticsIntent(SpecificIntent):
    @classmethod
    def name(cls) -> str:
        return "logistics"


class ComplainIntent(Intent):
    _urgency: UrgencyLevel = UrgencyLevel.MEDIUM

    @classmethod
    def name(cls) -> str:
        return "complain"


class RequestIntent(Intent):
    @classmethod
    def name(cls) -> str:
        return "request"


class GreetingIntent(Intent):
    @classmethod
    def name(cls) -> str:
        return "greeting"


class FeedbackIntent(Intent):
    @classmethod
    def name(cls) -> str:
        return "feedback"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    urgency: UrgencyLevel
    intent_group: Intent
    entities: Dict[str, List[str]]
    reasoning: str
    latency: float
    source_scores: Dict[str, float] = field(default_factory=dict)


class IntentCache(LRUCache):
    @staticmethod
    @LRUCache.make_key
    def _make_key(message: str, history: Optional[List[Dict[str, str]]]) -> str:
        payload: Dict[Any] = {"content": clean_text(message)[:20]}
        if history:
            payload["history"] = [
                {
                    "role": clean_text(item.get("role", ""))[:20],
                    "content": clean_text(item.get("content", ""))[:100]
                }
                for item in history[:-3]
            ]
        raw = dumps(payload, ensure_ascii=False, sort_keys=True)
        return md5(raw.encode("utf-8")).hexdigest()


def urgency(message: str, intent: Intent) -> UrgencyLevel:
    msg = message.lower()
    for level in UrgencyLevel.__members__.values():
        if any(kw in msg for kw in level.value):
            return level
    return intent.urgency


def pattern_match(message: str, patterns: Dict[Intent, List[str]]) -> tuple[Intent, float]:
    best_cat, score = OtherIntent, 0.0
    for cat, kws in patterns.items():
        hits = sum(1 for kw in kws if kw in message)
        if 0 == hits: continue
        sc = min(1.0, 0.5 + 0.25 * (hits - 1))
        if sc > score:
            best_cat, score = cat, sc
    return best_cat, score

def need_escalation(content: str) -> bool:
    return any(kw in content for kw in (EscalationIntent.keywords + HumanHandoffIntent.keywords))


if __name__ == "__main__":
    for inte in Intent.names():
        print(Intent.intent(inte).keywords)
        print(Intent.intent(inte).template)
        print(Intent.intent(inte).group)
        print(Intent.intent(inte).urgency)

    print(next(Intent._registry.items()))
    print(Intent.intent('other'))


    # for name in GenericIntent.intents():
    #     print(name)
    # print('----------')
    # for name in Intent.intents():
    #     print(name)

    # for u in UrgencyLevel.__members__.keys():
    #     print(type(u))
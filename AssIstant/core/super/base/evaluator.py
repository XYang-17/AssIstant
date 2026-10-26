from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from statistics import mean
from anthropic import AsyncAnthropic
from json import loads

from AssIstant.core.agent.base.intentBase import IntentBase
from AssIstant.core.agent.utils import extract_text_content
from AssIstant.common.prompt import PROMPTS
from AssIstant.common.utils import clean_text
from AssIstant.common.logger import FormatLog


@dataclass
class IntentTestCase:
    message: str
    expected_intent: IntentBase
    context: Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    relevance: float
    accuracy: float
    completeness: float
    helpfulness: float
    failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])

    @classmethod
    def all_scores(cls) -> List[str]:
        return ["relevance", "accuracy", "completeness", "helpfulness"]


@dataclass
class EvalResult:
    test_id: str
    passed: bool
    scores: Dict[str, float]
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    timestamp: str
    total: int
    passed: int
    pass_rate: float
    average_scores: Dict[str, float]
    regressions: List[str]
    recommendations: List[str]
    results: List[EvalResult]


class Judge(FormatLog):
    def __init__(self, client: AsyncAnthropic, model: str, name: str = "judge"):
        FormatLog.__init__(self, name)
        self._client = client
        self._model = model

    async def judge(
        self,
        req: str,
        resp: str,
        context: Optional[str] = None
    ) -> QualityScores:
        prompt = clean_text(PROMPTS["judge"].format(
            question=req,
            response=resp,
            supplements=f"背景信息：\n{context}" if context else ""
        ))
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = loads(raw[s: e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5))
            )
        except Exception as e:
            self.exception(f"failed to judge because of exception: {e}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                failed=True,
                error=str(e)
            )



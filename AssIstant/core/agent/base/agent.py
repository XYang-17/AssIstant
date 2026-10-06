from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from uuid import uuid4

from AssIstant.core.agent.base.intentBase import Intent, UrgencyLevel
from AssIstant.core.agent.base.agentType import AgentType
from AssIstant.core.boost.base.tool import ToolTrace
from AssIstant.common.base import State


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AgentState(State):
    monitor_penalty: float = 0.0

    def score(self) -> float:
        sc = 1.0 / (1.0 + self.average_latency / 1000)
        sc = self.success_rate * 0.7 + sc * 0.3
        return sc * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentRequest:
    content: str
    user_id: str
    conv_id: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    intent: Optional[Intent] = None
    intent_group: Optional[str] = None
    intent_confidence: float = 1.0
    urgency: Optional[UrgencyLevel] = None
    request_id: str = field(default_factory=lambda : str(uuid4())[:8])


@dataclass
class AgentResponse:
    agent_type: AgentType
    success: bool
    content: str
    confidence: float = 1.0
    latency: float = 0.0
    escalate: bool = False
    tools: List[str] = field(default_factory=list)
    traces: List[ToolTrace] = field(default_factory=list)











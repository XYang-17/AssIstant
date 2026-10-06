from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from AssIstant.core.agent.base.agent import AgentType
from AssIstant.core.agent.base.intentBase import Intent


@dataclass
class RoutingDecision:
    primary_agent_type: Optional[AgentType]
    supporting_agents_type: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_type(self) -> List[AgentType]:
        return [self.primary_agent_type] + self.supporting_agents_type

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents_type)


@dataclass
class DirectorResponse:
    request_id: str
    agent_type: AgentType
    response: str
    intent: Optional[Intent]
    latency: float = 0.0
    escalated: bool = False
    agent_types: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tools_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_decision: RoutingDecision = RoutingDecision(None)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from AssIstant.core.agent.base.intent import OtherIntent, IntentBase


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None


class ChatResponse(BaseModel):
    conv_id: str
    request_id: str
    response: str
    latency: float
    intent: str
    agent_type: str
    primary_agent: str
    intent_confidence: float = 0.0
    intent_group: str = OtherIntent.name()
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    agent_types: List[str] = Field(default_factory=list)
    supporting_agent: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated: bool = False
    used_knowledge_library: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)


class ToolTracesResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items:List[Dict[str, Any]] = Field(default_factory=list)


class DocInput(BaseModel):
    title: str
    content: str


class BatchDocInput(BaseModel):
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    message: str
    expected_intent: IntentBase
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
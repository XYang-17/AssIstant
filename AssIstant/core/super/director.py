import asyncio
from typing import Optional, Dict, List, Any
from anthropic import AsyncAnthropic
from collections import deque
from dataclasses import replace
from datetime import datetime
from time import monotonic

from composer import Composer
from base.director import *
from AssIstant.core.agent.discover import all_agents
from AssIstant.core.agent.agent import AgentType, AgentBase, AgentRequest, AgentResponse
from AssIstant.core.agent.base.intent import EscalationIntent, HumanHandoffIntent, UrgencyLevel, OtherIntent
from AssIstant.core.agent.interpreter import IntentInterpreter
from AssIstant.core.boost.base.toolInterface import ToolInterface
from AssIstant.core.boost.skillManager import SkillManager
from AssIstant.core.boost.toolManager import ToolManager, build_tool_interface
from AssIstant.common.config import MAX_TOOL_TRACES, AGENT_MODEL
from AssIstant.common.logger import FormatLog


class Director(FormatLog):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[SkillManager] = None,
        tool_manager: Optional[ToolManager] = None,
        name: str = "Director"
    ):
        FormatLog.__init__(self, name)
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._skill_manager = skill_manager

        self._intent_interpreter = IntentInterpreter(
            api_key=api_key, base_url=base_url, model=self._model
        )
        self._composer = Composer(
            client=self._client, model=model, skill_manager=self._skill_manager
        )

        self._toolInterfaces: Dict[str, ToolInterface] = dict()
        self._tools_trace = deque(maxlen=MAX_TOOL_TRACES)
        self._agents_pool: Dict[AgentType, List[AgentBase]] = dict()

        self._init_agents_pool()
        self.update_tools(build_tool_interface(tool_manager, "knowledgeLibrary"))

    def update_tools(self, tools: Dict[str, ToolInterface]):
        self._toolInterfaces = dict(tools or {})
        for agents in self._agents_pool.values():
            for agent in agents:
                agent.set_tools(self._toolInterfaces)

    def set_skill_manager(self, skill_manager: Optional[SkillManager]):
        self._skill_manager = skill_manager
        self._composer._skill_manager = self._skill_manager
        for agents in self._agents_pool.values():
            for agent in agents:
                agent.set_skill_manager(self._skill_manager)

    def get_tools_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._tools_trace):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_tools_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._tools_trace: return []
        limit = max(1, min(int(limit or 20), len(self._tools_trace)))
        return list(reversed(list(self._tools_trace)[-limit:]))

    def get_states(self) -> Dict[str, Any]:
        res = dict()
        for agent_type, agents in self._agents_pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value.name}_{i}"
                res[key] = {
                    "total": agent.state.total,
                    "success_rate": round(agent.state.success_rate, 3),
                    "average_latency": round(agent.state.average_latency, 1),
                    "monitor_penalty": round(agent.state.monitor_penalty, 3),
                    "routing_score": round(agent.state.score, 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent.model,
                }
        return res

    def update_routing_penalties(self, penalties: Dict[str, float]):
        for agent_type, agents in self._agents_pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value.name}_{i}"
                agent.state.monitor_penalty = min(0.9, max(0.0, penalties.get(key, 0.0)))

    async def interpret(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ):
        return await self._intent_interpreter.interpret(message, history=history)

    async def run(self, req: AgentRequest) -> DirectorResponse:
        time_b = monotonic()

        if req.intent is None:
            intent = await self._intent_interpreter.interpret(req.content, history=req.history)
            req.intent = intent.intent
            req.intent_group = intent.intent_group
            req.intent_confidence = intent.confidence
            req.urgency = intent.urgency

        if self._need_clarification(req):
            res = DirectorResponse(
                request_id=req.request_id,
                agent_type=AgentType.GENERAL,
                response="我无法确认您要处理的问题类型，请补充说明是技术问题或账户问题？",
                intent=req.intent,
                latency=(monotonic()-time_b)*1000,
                escalated=False,
                agent_types=[AgentType.GENERAL],
                routing_decision=RoutingDecision(
                    primary_agent_type=AgentType.GENERAL,
                    reason="解析所得意图的置信度过低，需要用户澄清具体问题"
                )
            )
            self._record_tools_trace(res)
            return res

        decision = self._make_routing_decision(req)
        if decision.multi_agents:
            return await self._run_multi_agents(req, decision)

        resp = await self._execute(req, decision.primary_agent_type)

        escalated = False
        if resp.escalate or UrgencyLevel.CRITICAL == req.urgency or req.intent in (
            EscalationIntent, HumanHandoffIntent
        ):
            escalated = True
            self.warning(f"trigger escalation: request_id = {req.request_id}, urgency = {req.urgency}")

        res = DirectorResponse(
            request_id=req.request_id,
            agent_type=resp.agent_type,
            response=resp.content,
            intent=req.intent,
            latency=(monotonic()-time_b)*1000,
            escalated=escalated,
            agent_types=[resp.agent_type],
            tools_used=list(resp.tools),
            tools_trace=list(resp.trace),
            routing_decision=RoutingDecision(
                primary_agent_type=decision.primary_agent_type,
                supporting_agents_type=decision.supporting_agents_type,
                reason=decision.reason,
                confidence=decision.confidence
            )
        )
        self._record_tools_trace(res)
        return res

    @staticmethod
    def _need_clarification(req: AgentRequest) -> bool:
        if req.intent != OtherIntent:
            return False
        msg = (req.content or "").strip()
        if len(msg) <= 2:
            return False
        return req.intent_confidence < 0.5


    def _record_tools_trace(self, res: DirectorResponse):
        routing_decision = res.routing_decision
        trace = {
            "request_id": res.request_id,
            "intent": res.intent.name() if res.intent else None,
            "latency": round(res.latency, 1),
            "escalated": res.escalated,
            "tools_used": res.tools_used,
            "tools_trace": res.tools_trace,
            "primary_agent": routing_decision.primary_agent_type.value.name if routing_decision.primary_agent_type else None,
            "supporting_agents": [agent_type.value.name for agent_type in routing_decision.supporting_agents_type],
            "timestamp": datetime.now().isoformat(),
        }
        self._tools_trace.append(trace)

    def _init_agents_pool(self):
        def make_agent(
            agent_cls: type(AgentBase),
            client: AsyncAnthropic,
            default_model: str,
            skill_manager: Optional[Any]
        ):
            profile = agent_cls.profile,
            model = AGENT_MODEL.get(agent_cls.agent_types.value.name.strip().upper()) or profile.model
            if model:
                profile = replace(profile, model=model)
            return agent_cls(client, default_model, skill_manager, profile=profile)

        for agent in all_agents:
            if self._agents_pool.get(agent.agent_types) is None:
                self._agents_pool[agent.agent_types] = [
                    make_agent(agent, self._client, self._model, self._skill_manager)
                ]
            else:
                self._agents_pool[agent.agent_types].append(
                    make_agent(agent, self._client, self._model, self._skill_manager)
                )

    def _make_routing_decision(self, req: AgentRequest) -> RoutingDecision:
        if UrgencyLevel.CRITICAL == req.urgency:
            return RoutingDecision(
                primary_agent_type=AgentType.ESCALATION,
                reason="识别到紧急情况",
                confidence=1.0
            )

        if req.intent in (EscalationIntent, HumanHandoffIntent):
            return RoutingDecision(
                primary_agent_type=AgentType.ESCALATION,
                reason=f"识别到意图为{req.intent.name()}",
                confidence=max(req.intent_confidence, 0.8)
            )

        scores = self._domain_scores(req)
        scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if AgentType.GENERAL == agent_type or self._agents_pool.get(agent_type)
        }
        if not scores:
            return RoutingDecision(
                primary_agent_type=AgentType.GENERAL,
                reason="无专用Agent，使用通用Agent",
                confidence=1.0
            )

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        supporting_agents = [
            agent_type
            for agent_type in self._collaboration(req)
            if agent_type != primary_agent and agent_type in scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                   and score >= max(0.45, primary_score*0.55)
            ]

        return RoutingDecision(
            primary_agent_type=primary_agent,
            supporting_agents_type=supporting_agents,
            reason=self._reason(req, scores, primary_agent, supporting_agents),
            confidence=round(min(primary_score, 1.0), 3)
        )

    @staticmethod
    def _domain_scores(req: AgentRequest) -> Dict[AgentType, float]:
        msg = req.content.lower()
        scores = {
            aType: aType.value.static_score
            for aType in AgentType.members()
        }

        aType = req.intent.agent_type
        info = aType.value
        scores[aType] += info.intent_score

        scores[aType] += min(
            info.max_keywords_score,
            sum(1 for kw in info.keywords if kw in msg) * info.keywords_score_factor
        )

        entities = req.entities or {}
        if scores.get(AgentType.TECHNICAL) is not None and entities.get("error_code"):
            scores[AgentType.TECHNICAL] += 0.2

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    def _collaboration(self, req: AgentRequest) -> List[AgentType]:
        msg = req.content.lower()
        res = set()

        aType = req.intent.agent_type
        info = aType.value
        if AgentType.TECHNICAL == aType \
                or any(kw in msg for kw in info.keywords):
            res.add(AgentType.TECHNICAL)

        return [agent_type for agent_type in res if agent_type in self._agents_pool]


    @staticmethod
    def _reason(
        req: AgentRequest,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType]
    ) -> str:
        score_content = ", ".join(
            f"{agent_type.value.name}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )
        support_content = ", ".join(agent_type.value.name for agent_type in supporting_agents) or "None"
        return (f"intent={req.intent.name() if req.intent else 'unkown'}, "
                f"group={req.intent_group.name() if req.intent_group else 'unknown'}, "
                f"primary={primary_agent.value.name}, "
                f"supporting=[{support_content}], "
                f"scores=[{score_content}]")


    async def _run_multi_agents(self, req: AgentRequest, decision: RoutingDecision) -> DirectorResponse:
        time_b = monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, agent_type) for agent_type in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        responses = [resp for resp in responses if isinstance(resp, AgentResponse)]
        combined = await(self._composer.compose(req, responses))
        escalated = any(resp.escalate for resp in responses)
        tools_used = list(dict.fromkeys(
            tool
            for resp in responses
            for tool in resp.tools
        ))
        tools_trace = [
            trace
            for resp in responses
            for trace in resp.trace
        ]

        res = DirectorResponse(
            request_id=req.request_id,
            agent_type=decision.primary_agent_type,
            response=combined,
            intent=req.intent,
            latency=(monotonic()-time_b)*1000,
            escalated=escalated,
            agent_types=[resp.agent_type for resp in responses if resp.success],
            tools_used=tools_used,
            tools_trace=tools_trace,
            routing_decision=RoutingDecision(
                primary_agent_type=decision.primary_agent_type,
                supporting_agents_type=decision.supporting_agents_type,
                reason=decision.reason,
                confidence=decision.confidence
            )
        )
        self._record_tools_trace(res)
        return res

    async def _execute(self, req: AgentRequest, agent_type: AgentType) -> AgentResponse:
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="Agent暂时不可用，请稍后重试",
                success=False
            )

        resp = await agent.handle(req)
        if not resp.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            self.warning(f"{agent_type.value.name}执行失败，尝试通用Agent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                resp = await fallback.handle(req)
        return resp

    def _best_agent(self, agent_type: AgentType) -> Optional[AgentBase]:
        agents = self._agents_pool.get(agent_type, [])
        if not agents: return None
        return max(agents, key=lambda x: x.state.score)


from anthropic import AsyncAnthropic
from time import monotonic
from json import dumps
from inspect import isawaitable
from typing import Any

from base.agent import *
from base.intent import need_escalation
from utils import extract_text_content
from AssIstant.core.boost.base.tool import ToolTrace
from AssIstant.core.boost.base.toolInterface import ToolInterface, ToolInterfaceManager
from AssIstant.common.logger import FormatLog
from AssIstant.common.utils import clean_text, extract_value
from AssIstant.common.config import MAX_CALL_TOOLS_TIMES
from AssIstant.common.prompt import PROMPTS
from AssIstant.core.boost.skillManager import SkillManager


class AgentBase(ToolInterfaceManager, FormatLog):
    agent_type: AgentType = NotImplemented
    system_prompt: str = NotImplemented
    profile: AgentProfile = NotImplemented

    profile_prompt = PROMPTS["profile"]

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        profile: Optional[AgentProfile] = None,
        skill_manager: Any = None,
        name: str = "agent"
    ):
        ToolInterfaceManager.__init__(self)
        self._client = client
        self._profile = profile or self.profile
        self._model = getattr(self._profile, "model", None) or model
        self._skill_manager = skill_manager
        FormatLog.__init__(self, name)

        self._state = AgentState()
        self._tools_used: List[str] = []
        self._tools_traces: List[ToolTrace] = []

    def set_skill_manager(self, skill_manager: Optional[SkillManager]):
        self._skill_manager = skill_manager

    @property
    def model(self) -> Any:
        return self._model

    @property
    def state(self) -> AgentState:
        return self._state

    async def handle(self, req: AgentRequest) -> AgentResponse:
        ms = None
        time_b = monotonic()
        self._state.total += 1
        self._tools_used = []
        self._tools_traces = []
        try:
            content = await self._call_llm(req)
            ms = (monotonic() - time_b) * 1000
            self._state.success += 1
            self._state.total += ms
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=content,
                latency=ms,
                escalate=need_escalation(content),
                tools=list(self._tools_used),
                trace=list(self._tools_traces)
            )
        except Exception as e:
            if ms is None:
                ms = (monotonic() - time_b) * 1000
            self._state.total += ms
            self.exception(f"failed to handle {self.agent_type.value.name} because of exception: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content="暂时无法处理请求，请稍后重试",
                latency=ms,
                trace=list(self._tools_traces)
            )

    async def _call_llm(self, req: AgentRequest) -> str:
        messages = []
        if req.context:
            messages.extend([
                {"role": "user", "content": f"[背景信息]\n{clean_text(req.context)}"},
                {"role": "assistant", "content": f"好的，我已了解背景信息。"}
            ])
        if req.entities:
            entities_text = dumps(req.entities, ensure_ascii=False)
            messages.extend([
                {"role": "user", "content": f"[结构化实体]\n{clean_text(entities_text)}"},
                {"role": "assistant", "content": f"好的，我会结合这些结构化实体处理。"}
            ])
        if role_packet := self._build_role_packet(req):
            messages.extend([
                {"role": "user", "content": f"[角色输入契约]\n{clean_text(role_packet)}"},
                {"role": "assistant", "content": f"好的，我会按照该角色的输入契约处理。"}
            ])
        messages.append({"role": "user", "content": f"{clean_text(req.content)}"})

        tool_interfaces = self.get_tools()
        tools_used = []
        tools_traces = []
        for _ in range(MAX_CALL_TOOLS_TIMES):
            client_kwargs = {
                "model": self._model,
                "max_tokens": self.profile.max_tokens,
                "temperature": self.profile.temperature,
                "system": self._build_system_prompt(req),
                "messages": messages
            }
            if tool_interfaces:
                client_kwargs["tools"] = [
                    {
                        "name": ti.name,
                        "description": ti.description,
                        "input_scheme": ti.input_scheme
                    }
                    for ti in tool_interfaces.values()
                ]
            resp = await self._client.messages.create(**client_kwargs)
            tools_need = [block for block in (resp.content or []) if "tool_use" == extract_value(block, "type")]
            if not tools_need:
                self._tools_used = tools_used
                return extract_text_content(resp.content)

            messages.append({"role": "assistant", "content": resp.content})
            tools_result = []
            for block in tools_need:
                name = extract_value(block, "name")
                tool_id = extract_value(block, "id")
                args = extract_value(block, "input") or {}

                call_success, result_success, result, error_msg, latency = self._call_llm_once(
                    tool_interfaces, name, req, args, tools_used
                )
                tools_traces.append(
                    ToolTrace(
                        name=name,
                        agent_type=self.agent_type,
                        tool_id=tool_id,
                        input=dict(args),
                        success=call_success,
                        result_success=result_success,
                        latency=latency,
                        cached=bool(result.get("cached", False)) if isinstance(result, Dict) else False,
                        reranked=bool(result.get("reranked", False)) if isinstance(result, Dict) else False,
                        error=error_msg
                    )
                )
                tools_result.append({
                    "type": "tool_result",
                    "tool_id": tool_id,
                    "content": dumps(result, ensure_ascii=False)
                })
            messages.append({"role": "user", "content": tools_result})

        self._tools_used = tools_used
        self._tools_traces = tools_traces

        msg = f"reach maximum calling times({MAX_CALL_TOOLS_TIMES})"
        self.exception(msg)
        raise RuntimeError(msg)


    async def _call_llm_once(self,
        tool_interfaces: Dict[str, ToolInterface],
        name: str,
        req: AgentRequest,
        args: Dict[str, Any],
        tools_used: List[str]
    ):
        call_success = False
        result_success: Optional[bool] = None
        error_msg = ""

        interface = tool_interfaces.get(name)
        time_b = monotonic()
        if interface is None:
            result = {"succsee": False, "error": f"{self.agent_type.value.name}未注册名为（{name}）的工具"}
            error_msg = result['error']
        else:
            try:
                result = interface.handle(req, args)
                if isawaitable(result):
                    result = await result
                tools_used.append(name)
                if isinstance(result, Dict) and "success" in result:
                    result_success = bool(result.get("success"))
            except Exception as e:
                self.exception(f"failed to handle request by tool(\'{name}\') because of exception: {e}")
                error_msg = str(e)
                result = {"success": False, "error": error_msg}
        latency = (monotonic() - time_b) * 1000
        if not error_msg and isinstance(result, Dict):
            error_msg = str(result.get("error", "") or "")

        return call_success, result_success, result, error_msg, latency

    def _build_system_prompt(self, req: AgentRequest) -> str:
        base_prompt = self.system_prompt + self.profile_prompt.format(
            self.profile.role,
            self.profile.mission,
            ' -> '.join(self.profile.workflow),
            '；'.join(self.profile.input_contract),
            '；'.join(self.profile.output_contract),
            '；'.join(self.profile.handoff_conditions) or "不升级",
            '；'.join(self.profile.tool_scope) or "仅允许使用当前的请求上下文"
        )
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt4(req.content, self.agent_type)
        if not skill_prompt:
             return base_prompt
        return f"{base_prompt}\n\n[动态Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: AgentRequest) -> str:
        packet = {
            "agent_types": self.agent_type.value.name,
            "intent": req.intent.name() if req.intent else None,
            "intent_group": req.intent_group.name() if req.intent_group else None,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {}
        }
        return dumps(packet, ensure_ascii=False)


if __name__ == "__main__":

    class Test:
        model = 1

    class Agent(AgentBase):
        agent_type: AgentType = None
        system_prompt: str = None
        profile: AgentProfile = None

    a = Agent(None, None, Test())

    ...

from dataclasses import dataclass
from typing import Dict, Any, Callable, Union, Awaitable, Optional, List
from json import load

from AssIstant.core.agent.base.agent import AgentRequest
from AssIstant.common.config import TYPES

ToolHandler = Callable[[AgentRequest, Dict[str, Any]], Union[Any, Awaitable[Any]]]

@dataclass(frozen=True)
class ToolInterface:
    name: str
    description: str
    input_scheme: Dict[str, Any]
    handler: ToolHandler

    def handle(self, req: AgentRequest, args: Dict[str, Any]) -> Union[Any, Awaitable[Any]]:
        self.check_args_type(args)
        return self.handler(req, args)

    def check_args_type(self, args: Dict[str, Any]):
        if not isinstance(args, Dict):
            raise TypeError("args必须使用Dict格式")
        scheme = self.input_scheme
        for field_name in scheme.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺失必要参数: {field_name}")
        properties = scheme.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and scheme.get("additionalProperties", False) is False:
            raise ValueError(f"意料外的参数: {', '.join(unknown)}")
        for k, v in args.items():
            expected = properties.get(k, {}).get("type")
            if expected in TYPES and not isinstance(v, TYPES[expected]):
                raise ValueError(f"参数\'{k}\'期望类型{', '.join(expected)}，但实际接收到{v}")


class ToolInterfaceManager:
    def __init__(self):
        self._tools: Dict[str, ToolInterface] = dict()

    def get_tools(self) -> Dict[str, ToolInterface]:
        return dict(self._tools)

    def set_tools(self, tools: Optional[Dict[str, ToolInterface]]):
        self._tools = dict(tools or {})

    def set_tool(self, name: str, tool: ToolInterface):
        self._tools[name] = tool

    def del_tool(self, name: str):
        self._tools.pop(name)


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: ToolHandler,
    required: Optional[List[str]] = None
) -> ToolInterface:
    return ToolInterface(
        name=name,
        description=description,
        input_scheme={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler
    )


def inspect_request_context(req: AgentRequest, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent": req.intent.name() if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }

with open("./error_code.json", 'r', encoding='utf-8') as f:
    ERROR_CODE_INFO = load(f)
# ERROR_CODE_INFO = {}


def extract_error_code(req: AgentRequest, args: Dict[str, Any]) -> Dict[str, Any]:
    code = str(args.get("error_code", "000")).upper().strip()
    message, steps = ERROR_CODE_INFO.get(code)
    return {
        "error_code": code,
        "content": message,
        "next_steps": steps,
        "server_log_checked": False
    }


def handoff_summary(req: AgentRequest, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "人工核验"))[:100],
        "intent": req.intent.name() if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }







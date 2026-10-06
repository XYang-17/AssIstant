import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict, Tuple
from time import monotonic
from hashlib import md5
from inspect import iscoroutinefunction, isawaitable

from AssIstant.core.boost.base.circuitBreaker import CircuitBreaker
from AssIstant.core.agent.base.agentType import AgentType
from AssIstant.common.logger import logger
from AssIstant.common.base import State
from AssIstant.common.cache import LRUCache


@dataclass
class ToolResult:
    name: str
    data: Any
    success: bool
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0
    reranked: bool = False


@dataclass
class ToolState(State):
    consecutive_failures: int = 0

@dataclass
class ToolTrace:
    name: str
    agent_type: AgentType
    tool_id: str
    input: Dict[str, Any]
    success: bool
    result_success: bool
    latency: float
    cached: bool
    reranked: bool
    error: str


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable
    schema: Dict[str, Any]
    cache_ttl: float = 0.0
    timeout: float = 30.0
    can_rerank: bool = False
    fallback: Optional[Callable] = None

    state: ToolState = field(default_factory=ToolState, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)
    
    async def run_fallback(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        err: str
    ) -> ToolResult:
        """
        尝试执行备用方案
        :param params:
        :param context:
        :param err:
        :return:
        """
        if self.fallback is None:
            return ToolResult(
                name=self.name,
                data=None,
                success=False,
                error=err
            )
    
        try:
            data = self.fallback(params, context, err)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                name=self.name,
                data=data,
                success=True,
                error=err
            )
        except Exception as e:
            err_msg = f"downgrade due to '{err}', but failed due to exception: {e}"
            logger.exception(f"Tool({self.name}) - {err_msg}")
            return ToolResult(
                name=self.name,
                data=None,
                success=False,
                error=err_msg
            )
    
    async def run_handler(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        if iscoroutinefunction(self.handler):
            return await self.handler(params, context)
        res = await asyncio.to_thread(self.handler, params, context)
        if isawaitable(res):
            return await res
        return res

class ToolCache(LRUCache):
    @staticmethod
    @LRUCache.make_key
    def make_key(name: str, params: Dict[str, Any], rerank_top_k: int = 0 ) -> str:
        from json import dumps
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        return f"{name}: {md5(dumps(payload, sort_keys=True).encode()).hexdigest()}"

    @staticmethod
    @LRUCache.extract_value
    def extract_value(cached_value: Tuple[Any, float, bool]) -> Tuple[Any, bool]:
        return cached_value[0], cached_value[2]

    @staticmethod
    @LRUCache.make_value
    def make_value(data: Any, ttl: float, reranked: bool = False) -> Tuple[Any, float, bool]:
        return data, monotonic() + ttl, reranked

    @staticmethod
    @LRUCache.is_expired
    def is_expired(cached_value: Tuple[Any, float, bool]) -> bool:
        return monotonic() > cached_value[1]
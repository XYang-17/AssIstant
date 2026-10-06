from anthropic import AsyncAnthropic
from typing import List

from AssIstant.core.boost.base.tool import *
from AssIstant.core.boost.base.toolInterface import ToolInterface, AgentRequest, make_tool
from AssIstant.core.agent.utils import extract_text_content
from AssIstant.common.logger import FormatLog
from AssIstant.common.utils import clean_text, truncate_text
from AssIstant.common.config import TYPES

class ToolManager(FormatLog):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        name: str = "toolManager"
    ):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._tools: Dict[str, Tool] = dict()
        self._cache = ToolCache(5000, 1250)
        FormatLog.__init__(self, name)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        self.info(f"register {tool.name}")

    def unregister(self, name: str):
        self._tools.pop(name, None)
        self.info(f"unregister {name}")

    def statistics(self) -> Dict[str, Any]:
        return {
            name: {
                "total": tool.state.total,
                "success_rate": round(tool.state.success_rate, 3),
                "average_latency": round(tool.state.average_latency, 1),
                "consecutive_failures": tool.state.consecutive_failures,
                "circuit_state": tool.breaker.state(),
            }
            for name, tool in self._tools.items()
        }

    def description(self, name: str) -> str:
        tool = self._tools.get(name)
        if not tool: return ""
        return tool.description

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0
    ) -> ToolResult:
        """
        调用工具获取结果
        :param name:
        :param params:
        :param context:
        :param use_cache:
        :param rerank_top_k: 重排后取top-k
        :return:
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                name=name,
                data=None,
                success=False,
                error=f"not found"
            )

        if rerank_top_k < 0 or not tool.can_rerank:
            rerank_top_k = 0

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._cache.get(self._cache.cache_key(name, params, rerank_top_k))
            if cached is not None:
                data, reranked = cached
                tool.state.total += 1
                tool.state.success += 1
                return ToolResult(
                    name=name,
                    data=data,
                    success=True,
                    reranked=reranked
                )

        # 熔断处理
        if not tool.breaker.allow():
            return await tool.run_fallback(params, context, f"breaking")

        time_b = monotonic()
        tool.state.total += 1
        try:
            self._validate_params(tool, params)

            data = await asyncio.wait_for(tool.run_handler(params, context), timeout=tool.timeout)
            latency = (monotonic() - time_b) * 1000

            tool.state.success += 1
            tool.state.consecutive_failures = 0
            tool.state.latency += latency
            tool.breaker.record_success()

            reranked = False
            if rerank_top_k > 0 and tool.can_rerank and isinstance(data, List):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

            if tool.cache_ttl > 0:
                self._cache.set(
                    self._cache.cache_key(name, params, rerank_top_k),
                    data, tool.cache_ttl, reranked
                )

            return ToolResult(
                name=name,
                data=data,
                success=True,
                latency_ms=latency,
                reranked=reranked
            )
        except Exception as e:
            tool.state.consecutive_failures += 1
            tool.breaker.record_failure()
            if isinstance(e, asyncio.TimeoutError):
                err = f"timeout({tool.timeout}s)"
            else:
                err = f"exception: {e}"
            self.exception(err)
            return await tool.run_fallback(params, context, err)

    async def broader_call(
        self,
        name: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        rewrite_times: int = 3
    ) -> ToolResult:
        """
        重写query，多线程并行调用工具获取结果
        :param name:
        :param query:
        :param context:
        :param top_k:
        :param rewrite_times:
        :return:
        """
        queries = await self._rewrite_query(query, n=rewrite_times)

        recall_k = max(top_k, 5)
        tasks = [
            self.call(name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in queries
        ]
        raw_res = await  asyncio.gather(*tasks, return_exceptions=True)

        keys, res = set(), []
        for r in raw_res:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    key = md5(str(item).encode()).hexdigest()
                    if key not in keys:
                        keys.add(key)
                        res.append(item)

        if not res:
            return ToolResult(
                name=name,
                data=[],
                success=False,
                error="empty result for all queries"
            )

        reranked = await self._rerank(query, res, top_k)
        return ToolResult(
            name=name,
            data=reranked,
            success=True,
            reranked=True
        )

    async def _rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        从n个角度改写query
        :param query:
        :param n:
        :return:
        """
        from AssIstant.common.prompt import PROMPTS, json
        prompt = clean_text(PROMPTS["rewrite_query"].format(n, query))
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]")+1
            queries = json.loads(raw[s:e])
            return list(dict.fromkeys([query]+queries))
        except Exception as e:
            self.exception(f"failed to rewrite query because of exception: {e}")
            return [query]

    async def _rerank(
        self,
        query: str,
        items: List[Any],
        top_k: int
    ) -> List[Any]:
        if len(items) <= top_k:
            return items

        from AssIstant.common.prompt import PROMPTS, json
        items_str = '\n'.join(f"{i}. {truncate_text(json.dumps(item, ensure_ascii=False), 200)}"
                              for i, item in enumerate(items))
        prompt = clean_text(PROMPTS["rerank"].format(query, items_str))
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]")+1
            order = json.loads(raw[s:e])
            reranked = [items[i] for i in order if 0 <= i < len(items)]
            return reranked[:top_k]
        except Exception as e:
            self.exception(f"failed to rerank because of exception: {e}")
            return items[:top_k]

    def _validate_params(
        self,
        tool: Tool,
        params: Dict[str, Any]
    ):
        schema = tool.schema
        required = schema.get("required", "")
        properties = schema.get("properties", "")

        for r in required:
            if r not in params:
                msg = f"loss required parameter {r} when calling Tool({tool.name})"
                self.exception(msg)
                raise ValueError(msg)

        for k, v in params.items():
            if k in properties:
                expected_types = properties[k].get("type")
                if expected_types and expected_types in TYPES:
                    if not isinstance(v, TYPES[expected_types]):
                        msg = f"type of {k} is {v} when calling Tool({tool.name}), but {TYPES[expected_types]} are expected"
                        self.exception(msg)
                        raise TypeError(msg)


def build_tool_interface(tool_manager: ToolManager, name: str) -> Dict[str, ToolInterface]:
    """
    为tool_manager管理的指定tool构建相应的ToolInterface
    :param tool_manager:
    :param name:
    :return:
    """
    async def do(req: AgentRequest, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_manager is None:
            return {"success": False, "error": "工具管理器未初始化", "results": []}

        query = str(args.get("query") or req.content or "").strip()
        if not query:
            return {"success": False, "error": "无法提取用户查询", "results": []}
        top_k = int(args.get("top_k", 5))
        results = await tool_manager.broader_call(name, query, top_k=top_k)

        if not getattr(results, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(results, "error", "检索失败"),
                "results": [],
                "reranked": False
            }
        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": results.data,
            "reranked": bool(getattr(results, "reranked", False))
        }

    return {
        name: make_tool(
            name=name,
            description=tool_manager.description(name) if tool_manager else "",
            properties={
                "query": {"type": "string", "description": "用户检索"},
                "top_k": {"type": "integer", "description": "匹配条目数"}
            },
            handler=do,
            required=["query"]
        )
    }
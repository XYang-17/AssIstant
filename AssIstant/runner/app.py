import asyncio
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from functools import wraps
from uuid import uuid4
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from base.runner import Runner
from base.io import (ChatResponse, ChatRequest, ToolTracesResponse, RecentToolTracesResponse,
                     BatchDocInput, EvalRunInput)
from AssIstant.core.agent.base.agent import AgentRequest
from AssIstant.core.agent.base.intent import OtherIntent
from AssIstant.core.super.base.memory import Role


runner = Runner()


app = FastAPI(
    title="AssIstant",
    version="1.0.0",
    lifespan=runner.lifespan,
    docs_url='/docs'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def check_service(*, items: List[Any]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if None in items:
                raise HTTPException(503, "Service not ready")
            return func(*args, **kwargs)
        return wrapper
    return decorator


@app.get('/health')
@check_service(items=[runner.director])
async def health() -> Dict[str, Any]:
    return {"status": "ok", "agents": runner.director.get_states()}

@app.get('/skills', tags=['Skills'])
async def skills_summary() -> Dict[str, Any]:
    if runner.skill_manager is None:
        raise HTTPException(503, "Skills not ready")
    return runner.skill_manager.summary()

@app.post('/skills/reload', tags=['Skills'])
async def reload_skills() -> Dict[str, Any]:
    if runner.skill_manager is None:
        raise HTTPException(503, "Skills not ready")
    runner.skill_manager.load()
    if runner.director is not None:
        runner.director.set_skill_manager(runner.skill_manager)
    return runner.skill_manager.summary()

@app.post('/chat', response_model=ChatResponse)
@check_service(items=[runner.director, runner.memory_manager])
async def chat(req: ChatRequest) -> ChatResponse:
    conv_id = req.conv_id or str(uuid4())
    context = await runner.memory_manager.get_context(req.user_id, conv_id, query=req.message)
    history = [
        {"role": m.role.value, "content": m.content}
        for m in context.recent_message[-5:]
    ]

    intent = await runner.director.interpret(req.message, history=history)
    full_context = context.as_prompt()

    request = AgentRequest(
        content=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        entities=intent.entities,
        intent=intent.intent,
        intent_group=intent.intent_group,
        urgency=intent.urgency,
        intent_confidence=intent.confidence
    )

    res = await runner.director.run(request)
    await runner.memory_manager.remember(req.user_id, conv_id, Role.USER, req.message)
    await runner.memory_manager.remember(req.user_id, conv_id, Role.AGENT, res.response)

    asyncio.create_task(runner.memory_manager.update_portrait(req.user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        request_id=res.request_id,
        response=res.response,
        latency=round(res.latency, 1),
        intent=res.intent.intent.name() if res.intent else OtherIntent.name(),
        agent_type=res.agent_type.value.name,
        primary_agent=res.routing_decision.primary_agent_type.value.name,
        intent_confidence=round(intent.confidence, 3),
        intent_group=intent.intent_group.name(),
        intent_source_scores=intent.source_scores,
        agent_types=[at.value.name for at in res.agent_types],
        supporting_agent=[sat.value.name for sat in res.routing_decision.supporting_agents_type],
        tools_used=res.tools_used,
        routing_reason=res.routing_decision.reason,
        routing_confidence=res.routing_decision.confidence,
        escalated=res.escalated,
        used_knowledge_library=any(t.endswith("_library") for t in res.tools_used),
        entities=intent.entities
    )

@app.get('/monitor')
@check_service(items=[runner.monitor])
async def monitor_summary() -> Dict[str, Any]:
    return runner.monitor.summary()

@app.get('/trace/tool/{request_id}', response_model=[ToolTracesResponse])
@check_service(items=[runner.director])
async def get_tool_trace(request_id: str) -> ToolTracesResponse:
    trace = runner.director.get_tools_traces(request_id)
    return ToolTracesResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {}
    )

@app.get('/trace/tools', response_model=RecentToolTracesResponse)
@check_service(items=[runner.director])
async def list_recent_tool_traces(limit: int = 10) -> RecentToolTracesResponse:
    return RecentToolTracesResponse(items=runner.director.get_tools_traces(limit=limit))

@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/search')
@check_service(items=[runner.tool_manager])
async def search(query: str, top_k: int = 5) -> Dict[str, Any]:
    res = await runner.tool_manager.broader_call("knowledge_library", query, top_k)
    return {"query": query, "results": res.data, "reranked": res.reranked}

@app.post('/knowledge/add', tags=['knowledge_library'])
async def add_knowledge(body: BatchDocInput) -> Dict[str, Any]:
    tool = runner.tool_manager.get_tool("knowledge_library") if runner.tool_manager else None
    if tool is None: raise HTTPException(503, "knowledge library not ready")
    kl = tool.handler.__self__
    count = await kl.aimport_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {
        "message": f"successfully loaded {count} document fragments",
        "added_chunks": count,
        "total_chunks": await kl.acount()
    }

@app.post('/knowledge/upload', tags=['knowledge_library'])
async def upload_knowledge(file: UploadFile = File(...)) -> Dict[str, Any]:
    tool = runner.tool_manager.get_tool("knowledge_library") if runner.tool_manager else None
    if tool is None: raise HTTPException(503, "knowledge library not ready")
    kl = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, f"The file size exceeds the limit (10GB)")

    text = content.decode(encoding='utf-8', errors='ignore')
    filename = file.filename or "unknown"

    if filename.endswith('.json'):
        from json import loads, JSONDecodeError
        try:
            docs = loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON file format error")
        except JSONDecodeError as e:
            raise HTTPException(400, f"JSON file parsing failed because of exception: {e}")
    else:
        title = filename.rsplit('.', 1)[0] if '.' in filename else filename
        docs = [{"title": title, "content": text}]

    return {
        "message": "successfully upload",
        "added_chunks": await kl.aimport_documents(docs),
        "total_chunks": await kl.acount()
    }

@app.get('/knowledge/states', tags=["knowledge_library"])
async def knowledge_states() -> Dict[str, Any]:
    tool = runner.tool_manager.get_tool("knowledge_library") if runner.tool_manager else None
    if tool is None: raise HTTPException(503, "knowledge library not ready")
    kl = tool.handler.__self__
    return {"total_chunks": await kl.acount()}

@app.post('/eval/run')
@check_service(items=[runner.evaluator])
async def run_eval(body: Optional[EvalRunInput] = None) -> Dict[str, Any]:
    ...

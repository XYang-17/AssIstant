import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI


from AssIstant.common.logger import logger
from AssIstant.core.agent.interpreter import IntentInterpreter
from AssIstant.core.boost.skillManager import SkillManager
from AssIstant.core.boost.toolManager import ToolManager, Tool
from AssIstant.core.boost.knowledgeLibrary import KnowledgeLibrary
from AssIstant.core.super.monitor import Monitor
from AssIstant.core.super.memoryManager import MemoryManager
from AssIstant.core.super.director import Director, build_tool_interface
from AssIstant.common.config import (SKILLS_DIR, MAX_SKILLS_CHARS, REDIS_URL,
                                     CHROMA_HOST, CHROMA_PORT, CHROMA_PERSIST_DIRECTORY,
                                     PROMETHEUS_PORT, MONITOR_INTERVAL, ALERT_WEBHOOK_URL)



BANNER = ("\n"
          "╔═════════════════════╗"
          "║      AssIstant      ║"
          "╚═════════════════════╝"
          "\n")


def anthropic_config() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key: raise RuntimeError("Missing environment variable 'ANTHROPIC_API_KEY'")
    config = {
        "api_key": key,
        "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url: config["base_url"] = base_url
    return config


@dataclass
class Runner:
    director = None
    memory_manager = None
    tool_manager = None
    monitor = None
    evaluator = None
    skill_manager = None


    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        print(BANNER, flush=True)

        config = anthropic_config()
        logger.info(f"model: {config['model']}\n"
                    f"base_url: {config.get('base_url', 'official')}")

        interpreter = IntentInterpreter(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config["model"]
        )

        self.skill_manager = SkillManager(
            root_dir=SKILLS_DIR,
            max_prompt_chars=MAX_SKILLS_CHARS
        )
        self.skill_manager.load()

        self.director = Director(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config["model"],
            skill_manager=self.skill_manager
        )

        self.memory_manager = MemoryManager(
            redis_url=REDIS_URL,
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT,
            chroma_path=CHROMA_PERSIST_DIRECTORY,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config["model"]
        )

        self.tool_manager = ToolManager(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config["model"]
        )

        kl = KnowledgeLibrary(
            collection="knowledge_library",
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            path=CHROMA_PERSIST_DIRECTORY
        )
        logger.info(f"load {await kl.acount()} document fragments from knowledge library")

        self.tool_manager.register(Tool(
            name="knowledge_library",
            description="Knowledge library (ChromaDB) retrieval service",
            handler=kl.search,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"]
            },
            cache_ttl=300.0,
            can_rerank=True,
            fallback=lambda params, context, error:
            [{
                "title": "knowledge library fallback results",
                "content": f"Knowledge library is temporarily unavailable for service. Please try again later or request manual processing.",
                "score": 0.0,
                "fallback": True,
                "error": error
            }]
        ))

        if self.director is not None:
            self.director.update_tools(build_tool_interface(self.tool_manager, "knowledge_library"))


        self.monitor = Monitor(
            director=self.director,
            tool_manager=self.tool_manager,
            interval=MONITOR_INTERVAL,
            webhook_url=ALERT_WEBHOOK_URL,
            prometheus_port=PROMETHEUS_PORT
        )
        await self.monitor.start()

        ############# evaluator

        logger.info("[AssIstant] Ready!")
        yield

        await self.monitor.stop()
        if self.memory_manager is not None:
            await self.memory_manager.close()

        logger.info("[AssIstant] Bye!")

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from AssIstant.core.agent.base.agentType import AgentType

SKILL_SUFFIXES = [".md", ".txt", ".json"]

@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: str
    keywords: List[str] = field(default_factory=list)
    agent_types: List[AgentType] = field(default_factory=list)
    enabled: bool = True

    def applicable(
        self,
        massage: str,
        agent_type: Optional[AgentType] = None
    ) -> (bool, List[str]):
        """
        该skill是否适用message请求
        :param massage:
        :param agent_type:
        :return:
        """
        if not self.enabled:
            return False, [] # 未启用，不适用

        if self.agent_types and agent_type and agent_type not in self.agent_types:
            return False, [] # agents列表中没有该类型，不适用

        if not self.keywords:
            return True, [] # 未定义关键词，默认适用

        lowered = (massage or "").lower()
        matched_keywords = [kw for kw in self.keywords if kw.lower() in lowered] # message中出现任意关键词即适用
        return bool(matched_keywords), matched_keywords

    def to_prompt_block(self, max_chars: int = 3200) -> str:
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n..."
        description = f"说明：{self.description}" if self.description else ""
        return f"### {self.name}\n{description}\n{body}"

    def to_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "content_chars": len(self.content),
            "path": self.path,
            "keywords": self.keywords,
            "agent_types": self.agent_types,
            "enabled": self.enabled,
        }

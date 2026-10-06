from anthropic import AsyncAnthropic
from typing import Optional, Any, List

from AssIstant.core.agent.base.agent import AgentRequest, AgentResponse
from AssIstant.core.agent.utils import extract_text_content
from AssIstant.common.prompt import PROMPTS
from AssIstant.common.config import MAX_COMPOSE_TOKENS, COMPOSE_TEMPERATURE
from AssIstant.common.logger import FormatLog


class Composer(FormatLog):
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        name: str = "composer"
    ):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager
        FormatLog.__init__(self, name)

    async def compose(
        self,
        request: AgentRequest,
        responses: List[AgentResponse]
    ) -> str:
        success = [resp for resp in responses if resp.success and resp.content.strip()]
        if not success:
            return "所有Agent都失败"
        if 1 == len(success):
            return success[0].content

        evidence = "\n\n".join(
            f"[{resp.agent_type.value} 输出]\n{resp.content}"
            for resp in responses
        )
        prompt = PROMPTS["compose"].format(request.content, evidence, success[0].agent_type.value)
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt4(request.content, "general")
            if skill:
                prompt += f"\n\n[输出边界]\n{skill}"
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=MAX_COMPOSE_TOKENS,
                temperature=COMPOSE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            content = extract_text_content(resp.content).strip()
            if content: return content
        except Exception as e:
            self.exception(f"failed to compose because of exception: {e}")
        result = success[0].content
        result += "\n\n" + "\n\n".join(f"补充说明：\n{resp.content}" for resp in success[1:])
        return result


if __name__ == "__main__":
    ...

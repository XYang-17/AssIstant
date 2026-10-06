from pathlib import Path
from typing import List, Iterable

from AssIstant.core.boost.base.skill import *
from AssIstant.core.boost.utils import discover_skills, load_skill
from AssIstant.common.logger import logger
from AssIstant.common.utils import truncate_text

class SkillManager:
    def __init__(self, root_dir: str, max_prompt_chars: int = 5000, name: str = "skillManager"):
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._max_prompt_chars = max_prompt_chars
        self._name = name

        self._skills: List[Skill] = []
        self._errors: List[str] = []

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def load(self) -> List[Skill]:
        """
        加载可用Skill文档
        :return:
        """
        loaded, errors = [], []
        if not self._root_dir.exists():
            logger.info(f"{type(self).__name__}({self._name}) - root dir(\"{self._root_dir}\") is not exist")
            self._skills, self._errors = [], []
            return []

        for path in discover_skills(self._root_dir):
            try:
                skill = load_skill(path)
                if skill is not None:
                    loaded.append(skill)
            except Exception as e:
                err = f"{path}: {e}"
                errors.append(err)
                logger.exception(f"{type(self).__name__}({self._name}) - {err}")

        self._skills = loaded
        self._errors = errors
        self._log_skills()
        return self._skills

    def prompt4(self, message: str, agent_type: Optional[AgentType] = None):
        """
        根据输入匹配可用Skill，并转换为prompt
        :param message:
        :param agent_type:
        :return:
        """
        prompt_blocks, matched_keywords = [], []
        remaining = self._max_prompt_chars

        for skill in self._skills:
            applicable, matched_kws = skill.applicable(message, agent_type)
            if not applicable:
                continue
            prompt_block = skill.to_prompt_block()
            if len(prompt_block) > remaining:
                prompt_block = prompt_block[:remaining].rstrip() + "\n..."
            prompt_blocks.append(prompt_block)
            matched_keywords.append((skill, matched_kws))
            remaining -= len(prompt_block)
            if remaining <= 0:
                break

        if not prompt_blocks:
            logger.debug(
f"{type(self).__name__}({self._name}) - no skill applicable to content=\"{truncate_text(message or '')}\" and agent_type={agent_type.value or 'all'}")
            return ""

        detail = "; ".join(
            f"{skill.name}(keywords={', '.join(keywords) if keywords else 'all'}"
            for skill, keywords in matched_keywords
        )
        logger.info(
f"{type(self).__name__}({self._name}) - content=\"{truncate_text(message or '')}\" and agent_type={agent_type.value or 'all'}\napplicable skills: {detail}"
        )

        from AssIstant.common.prompt import PROMPTS
        return PROMPTS["applicable_skills"].format("\n\n".join(prompt_blocks))

    def summary(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self._root_dir),
            "skill_count": len(self._skills),
            "skills": [skill.to_summary() for skill in self._skills],
            "errors": self._errors,
        }

    def _log_skills(self):
        lines = [
            "\n⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ Loaded AssIstant Skills ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇",
            f"root_dir: {self._root_dir}",
            f"skills_count: {len(self._skills)}",
        ]
        if self._skills:
            for i, skill in enumerate(self._skills, start=1):
                agents = ", ".join(at.value for at in skill.agent_types) if skill.agent_types else "all"
                keywords = ", ".join(skill.keywords[:8]) if skill.keywords else "all"
                if len(skill.keywords) > 8:
                    keywords += ", ..."
                lines.extend([
                    f"{i}. {skill.name}",
                    f"    agent: {agents}",
                    f"    keywords: {keywords}",
                    f"    path: {skill.path}",
                ])
        else:
            lines.append("No any skill")

        if self._errors:
            lines.append("errors:")
            lines.extend(f"    - {error}" for error in self._errors)

        lines.append("⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ Loaded AssIstant Skills ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆\n")
        logger.info("\n".join(lines))








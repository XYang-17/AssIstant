from typing import Any, Iterable, List, Optional
from pathlib import Path

from AssIstant.core.boost.base.skill import Skill, SKILL_SUFFIXES
from AssIstant.common.utils import (split_front_matter, first_heading, strip_first_heading,
                                    as_list, as_bool)
def chunkify(text: str, chunk_size: int = 500) -> List[str]:
    """
    将完整的一段text划分为若干chunk
    :param text:
    :param chunk_size:
    :return: [chunk, ...]
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    from re import split

    chunks = []
    chunk = ""
    sentences = split(r'[。.\r\n]', text)
    for st in sentences:
        st = st.strip()
        if not st: continue
        if len(chunk) + len(st) + 1 > chunk_size:
            if chunk: chunks.append(chunk)
            chunk = st
        else:
            chunk = f"{chunk}. {st}" if chunk else st
    if chunk:
        chunks.append(chunk)
    return chunks


def discover_skills(root_dir: Path) -> Iterable[Path]:
    """
    以生成器形式搜索Skill文档
    :param root_dir:
    :return:
    """
    skill_md_files = sorted(root_dir.rglob("SKILL.md"))
    yielded = {path.resolve() for path in skill_md_files}
    for path in skill_md_files:
        yield path

    for path in sorted(root_dir.rglob("*")):
        resolved = path.resolve()
        if (resolved in yielded
                or not path.is_file()
                or path.name.startswith(".")
                or path.name.upper() == "README.MD"):
            continue
        if path.suffix.lower() in SKILL_SUFFIXES:
            yield  path


def load_skill_json(path: Path) -> Optional[Skill]:
    """
    加载Skill JSON文档
    :param path:
    :return:
    """
    from json import loads
    raw = loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f"Dict format is required in skill JSON file, path: {path}")

    content = str(raw.get("content") or raw.get("instructions") or "") .strip()
    if not content:
        raise ValueError(f"Missing content or instructions in skill JSON file, path: {path}")

    return Skill(
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        content=content,
        path=str(path),
        keywords=as_list(raw.get("keywords")),
        agents=[item.lower() for item in as_list(raw.get("agents"))],
        enabled=as_bool(raw.get("enabled"), default=True),
    )


def load_skill_text(path: Path) -> Optional[Skill]:
    """
    加载Skill 文本文档
    :param path:
    :return:
    """
    raw = path.read_text(encoding='utf-8')
    meta, body = split_front_matter(raw)
    body = body.strip()
    if not body:
        return None

    default_name = path.parent.name if "SKILL.md" == path.name else path.stem
    name = str(meta.get("name") or first_heading(body) or default_name)
    body = strip_first_heading(body, name)

    return Skill(
        name=name,
        description=str(meta.get("description") or ""),
        content=body,
        path=str(path),
        keywords=as_list(meta.get("keywords")),
        agents=[item.lower() for item in as_list(meta.get("agents"))],
        enabled=as_bool(meta.get("enabled"), default=True),
    )


def load_skill(path: Path) -> Optional[Skill]:
    """
    加载Skill文档
    :param path:
    :return:
    """
    if path.suffix.lower() == '.json':
        return load_skill_json(path)
    return load_skill_text(path)

from typing import Iterable, Any, List
from hashlib import md5

def extract_text_content(content: Iterable[Any]) -> str:
    """
    从LLM回复的结构中提取内容
    :param content:
    :return:
    """
    texts = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, 'type', None)
        text = getattr(block_type, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in ("text", None)):
            texts.append(text)

    return "\n".join(t for t in texts if t)


def embedding(text: str, dims: int = 256) -> List[float]:
    """
    文本embedding，n-gram
    :param text:
    :param dims:
    :return:
    """
    text = text.strip()
    vec = [0.0] * dims
    tokens = set()
    for n in range(1, 4):
        if len(text) >= n:
            tokens.update(text[i:i+n] for i in range(len(text) - n + 1))
    if not tokens:
        tokens.add(text)

    for token in tokens:
        digest = md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if 0 == digest[4] % 2 else -1.0
        vec[idx] += sign
    return vec


if __name__ == "__main__":
    print(embedding("yanx"))

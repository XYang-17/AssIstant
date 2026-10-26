from typing import List, Any, Dict, Optional, Sequence, Set, Iterable, Callable

def as_list(value: Any, delimiter: str = ",") -> List[str]:
    if value is None or "" == value:
        return []
    if isinstance(value, list):
        return [tmp for item in value if (tmp := str(item).strip())]
    return [tmp for item in str(value).split(delimiter) if (tmp := item.strip())]


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None or "" == value:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "none", "off", "disabled")


def extract_value(data: Any, key: Any) -> Any:
    if isinstance(data, Dict):
        return data.get(key, None)
    return getattr(data, key, None)

def extract(data: Any, key: Any, predicate: Callable = lambda : True, result_type: type = list) -> Any:
    if isinstance(data, Dict):
        data = data.get(key, None)
    else:
        data = getattr(data, key, None)
    if not data: return result_type()
    while hasattr(data, "__getitem__") and isinstance(data[0], result_type):
        data = data[0]
    if not isinstance(data, Iterable):
        data = (data,)
    return result_type(d for d in data if predicate(d))

# def unique(data: Sequence) -> Any:
#     if not isinstance(data, Sequence): return data
#     seen = set()
#     res = list()
#     for d in data:
#         if isinstance(d, str):
#             d = d.strip()
#         if not d or d in seen:
#             continue
#         seen.add(d)
#         res.append(d)
#     return type(data)(res)


def clean_text(text: Any, encoding: str = "utf-8") -> str:
    """
    文本清洗
    :param text:
    :param encoding:
    :return:
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode(encoding, errors="ignore").decode(encoding)

def recursive_clean_text(value: Any, encoding: str = "utf-8") -> Any:
    if isinstance(value, str):
        return clean_text(value, encoding)
    if isinstance(value, Dict):
        return {clean_text(k, encoding): recursive_clean_text(v, encoding) for k, v in value.items()}
    if isinstance(value, Sequence):
        return type(value)(recursive_clean_text(v, encoding) for v in value)
    return value


def truncate_text(text: str, size: int = 10) -> str:
    """
    截断文本内容
    :param text:
    :param size:
    :return:
    """
    if not isinstance(text, str):
        data = str(text)
    else:
        data = text
    data = data.rstrip()

    if len(data) > size - 3:
        return data[:size-3]+"..."
    return data

def unique(data: Sequence[Any]) -> Sequence[Any]:
    if isinstance(data, (Dict, Set)):
        return type(data)(
            stripped for d in data
            if d and (stripped := (d.strip() if isinstance(d, str) else d))
        )

    return type(data)(dict.fromkeys(
        stripped for d in data
        if d and (stripped := (d.strip() if isinstance(d, str) else d))
    ))


def split_front_matter(raw: str) -> tuple[Dict[str, Any], str]:
    """
    分离Markdown文档头部front matter与正文body
    :param raw: Markdown文档，格式为
        ---\n
        front matter
        ---\n
        body
    :return: (front matter, body)
    """
    text = raw.strip()
    if not text.startswith("---"):
        return dict(), raw # 没有front matter，返回原始字符串

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return dict(), raw # 没有front matter，返回原始字符串

    meta = dict()
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip("\"'")

    if end is None:
        return dict(), raw

    return meta, "\n".join(lines[end+1:])


def first_heading(body: str) -> Optional[str]:
    """
    提取Markdown body的首个标题(#开头的行)
    :param body:
    :return:
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def strip_first_heading(body: str, name: str) -> str:
    """
    当Markdown body的首行为与name相同的标题，移除该行
    :param body:
    :param name:
    :return:
    """
    lines = body.splitlines()
    if not lines:
        return body
    first = lines[0].strip()
    if first.startswith("#") and first.lstrip("#").strip() == name:
        return "\n". join(lines[1:]).strip()
    return body

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

if __name__ == "__main__":
    # a = [1, 2, 2]
    # b = unique(a)
    # print(b)
    # print(type(b))

    a = list(({1: "  323 "}, 2, 3))
    b = recursive_clean_text(a)
    print(b)

    ...
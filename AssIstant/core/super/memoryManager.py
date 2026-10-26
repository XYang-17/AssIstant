import chromadb, asyncio
from typing import Optional
from anthropic import AsyncAnthropic
from redis import asyncio as aredis
from json import loads
from functools import wraps
from hashlib import md5

from base.memory import *
from AssIstant.core.agent.utils import extract_text_content
from AssIstant.common.logger import FormatLog
from AssIstant.common.utils import recursive_clean_text, extract, unique
from AssIstant.common.prompt import PROMPTS
from AssIstant.common.config import (MAX_WORKING_MEMORY, COMPRESS_THS, RETAIN_NUM, HISTORY_TOP_K,
                                     MAX_SUMMARY_CHARS, PORTRAIT_PREFIX, REDIS_EXPIRE)


def _clean_text(*, n):
    async def decorator(func):
        @wraps(func)
        async def wrapper(*args):
            new_args = (clean_text(arg).strip() if i < n else arg for i, arg in enumerate(args))
            return await func(*new_args)
        return wrapper
    return decorator


class MemoryManager(FormatLog):
    def __init__(
        self,
        redis_url: str,
        chroma_host: str,
        chroma_port: int,
        chroma_path: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        name: str = "memoryManager"
    ):
        FormatLog.__init__(self, name)
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._redis = aredis.from_url(redis_url, decode_responses=True)

        try:
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            self.info(f"Successfully connected ChromaDB: {chroma_host}:{chroma_port}")
        except Exception:
            self.warning(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        self._episodic = chroma.get_or_create_collection("episodic")
        self._portrait  = chroma.get_or_create_collection("portrait")

    @_clean_text(n=2)
    async def remember(
        self,
        user_id: str,
        conv_id: str,
        role: Role,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        存储对话记忆，自动压缩
        :param user_id:
        :param conv_id:
        :param role:
        :param content:
        :param metadata:
        :return:
        """
        # user_id = clean_text(user_id)
        # conv_id = clean_text(conv_id)
        metadata = {
            clean_text(k): recursive_clean_text(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=clean_text(content), metadata=metadata)

        working_key = self._working_memory_key(user_id, conv_id)
        await self._redis.lpush(working_key, dumps({
            "role": msg.role.value,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        }))
        await self._redis.expire(working_key, REDIS_EXPIRE)


        if await self._redis.llen(working_key) >= COMPRESS_THS:
            await self._compress(user_id, conv_id)

    @_clean_text(n=3)
    async def get_context(
        self,
        user_id: str,
        conv_id: str,
        query: str = ""
    ) -> Context:
        recent = await self._working_memory(user_id, conv_id)
        history = await self._search_episodic(user_id, conv_id, query or (recent[-1].content if recent else ""))
        portrait = await self._user_portrait(user_id)
        summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return Context(
            recent_message=recent,
            related_history=history,
            portrait=portrait,
            summary=summary
        )

    @_clean_text(n=2)
    async def update_portrait(self, user_id: str, conv_id: str):
        """
        更新用户画像
        :param user_id:
        :param conv_id:
        :return:
        """
        # user_id = clean_text(user_id)
        # conv_id = clean_text(conv_id)
        msg = await self._working_memory(user_id, conv_id)
        if not msg: return

        current_portrait = await self._user_portrait(user_id)
        conversation = clean_text("\n".join(f"{m.role.value}: {m.content}" for m in msg[-10:]))
        portrait = dumps(current_portrait, ensure_ascii=False) if current_portrait else"{}"
        prompt = PROMPTS["update_portrait"].format(portrait, conversation)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}")+1
            portrait = loads(raw[s:e])

            doc_id = self._portrait_doc_id(user_id)
            doc_text = clean_text(dumps(portrait, ensure_ascii=False))
            try:
                await asyncio.to_thread(self._portrait.delete, ids=[doc_id])
            except Exception:
                ...
            await asyncio.to_thread(
                self._portrait.add,
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "updated_at": datetime.now().isoformat()
                }]
            )
            self.info(f"successfully updated user portrait (user_id={user_id})")
        except Exception as e:
            self.exception(f"failed to update user portrait (user_id={user_id}) because of exception: {e}")

    async def close(self):
        await self._redis.close()

    async def _compress(self, user_id: str, conv_id: str):
        """
        压缩记忆
        :param user_id:
        :param conv_id:
        :return:
        """
        msg = await self._working_memory(user_id, conv_id)
        if len(msg) < COMPRESS_THS: return

        compressed = msg[:-RETAIN_NUM]
        retain = msg[-RETAIN_NUM:]

        conversation = clean_text('\n'.join(f"{m.role.value}: {m.content}" for m in compressed))
        prompt = PROMPTS["compress"].format(conversation)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = clean_text(extract_text_content(resp.content)).strip()
        except Exception as e:
            self.exception(f"failed to compress because of exception: {e}\n"
                           f"the compressed conversations as followed: \n{conversation}")
            summary = f"压缩{len(compressed)}条对话内容失败"

        summary_key = self._summary_key(user_id, conv_id)
        old_summary = await self._redis.get(summary_key) or ""
        new_summary = await self._merge_summary(old_summary, summary)
        await self._redis.setex(summary_key, REDIS_EXPIRE, new_summary)

        await self._store_episodic(user_id, conv_id, conversation, summary)

        working_key = self._working_memory_key(user_id, conv_id)
        await self._redis. delete(working_key)
        for m in reversed(retain):
            await self._redis.lpush(working_key, dumps({
                "role": m.role.value,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "metadata": m.metadata
            }))
        await self._redis.expire(working_key, REDIS_EXPIRE)
        self.info(f"successfully compressed working memory(user_id: {user_id}, conv_id: {conv_id}), {len(summary)} characters")

    @staticmethod
    def _working_memory_key(user_id: str, conv_id: str) -> str:
        return f"working:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _portrait_doc_id(user_id: str) -> str:
        return f"{PORTRAIT_PREFIX}{user_id}"

    async def _working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key = self._working_memory_key(user_id, conv_id)
        raw = await self._redis.lrange(key, 0, MAX_WORKING_MEMORY-1)
        msgs = []
        for r in reversed(raw):
            t = loads(r)
            msgs.append(Message(
                role=Role(t["role"]),
                content=t["content"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                metadata=t.get("metadata", {})
            ))
        return msgs

    async def _search_episodic(self, user_id: str, conv_id: str, query: str) -> List[str]:
        query = clean_text(query).strip()
        if not query: return []

        try:
            res = await asyncio.to_thread(
                self._episodic.query,
                query_texts=[query],
                n_results=HISTORY_TOP_K,
                where={"user_id": user_id, "conv_id": conv_id}
            )
            docs = self._extract_document(res)
            if len(docs) < HISTORY_TOP_K:
                fallback = await asyncio.to_thread(
                    self._episodic.query,
                    query_texts=[query],
                    n_results=HISTORY_TOP_K,
                    where={"user_id": user_id}
                )
                docs.extend(self._extract_document(fallback))
            return unique(docs)[:HISTORY_TOP_K]
        except Exception as e:
            self.exception(f"failed to search episodic because of exception: {e}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str):
        try:
            doc_id = md5(f"{user_id}{conv_id}{datetime.now().time()}".encode()).hexdigest()
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "timestamp": datetime.now().isoformat(),
                    "full_text": clean_text(text[:500])
                }]
            )
        except Exception as e:
            self.exception(f"failed to store episodic because of exception: {e}")

    async def _user_portrait(self, user_id: str) -> Dict[str, Any]:
        try:
            doc_id = self._portrait_doc_id(user_id)
            direct = await asyncio.to_thread(self._portrait.get, ids=[doc_id])
            if direct.get("documents"):
                return loads(direct["documents"][0])
            res = await asyncio.to_thread(self._portrait.get, where={"user_id": user_id})
            return self._lastest_portrait(res)
        except Exception:
            ...
        return {}

    @staticmethod
    def _extract_document(data: Any) -> List[str]:
        return extract(data, "documents", lambda x: isinstance(x, str) and x.strip(), list)

    @_clean_text(n=2)
    async def _merge_summary(self, old: str, new: str) -> str:
        if not old: return new[:MAX_SUMMARY_CHARS]
        if not new: return old[:MAX_SUMMARY_CHARS]

        prompt = PROMPTS["merge_summary"].format(MAX_SUMMARY_CHARS, old, new)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            merged = clean_text(extract_text_content(resp.content)).strip()
            if merged:  return merged[:MAX_SUMMARY_CHARS]
        except Exception as e:
            self.exception(f"failed to merge summary because of exception: {e}")

        merged = clean_text(f"{old}\n{new}").strip()
        return merged[-MAX_SUMMARY_CHARS:]

    @staticmethod
    def _lastest_portrait(res: Dict[str, Any]) -> Dict[str, Any]:
        documents = res.get("documents") or []
        if not documents: return {}
        metadatas = res.get("metadatas") or []

        candidates = []
        for i, doc in enumerate(documents):
            if not doc: continue
            metadata = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], Dict) else {}
            timestamp = str(metadata.get("updated_at") or metadata.get("timestamp") or "")
            candidates.append((timestamp, metadata, doc))

        if not candidates: return {}
        candidates.sort(key=lambda x: x[0], reverse=True)
        lastest_doc = candidates[0][2]
        try:
            return loads(lastest_doc)
        except Exception:
            return {}
import asyncio
from anthropic import AsyncAnthropic
from time import monotonic
from json import loads

from AssIstant.core.agent.base.intent import *
from AssIstant.core.agent.utils import extract_text_content, embedding
from AssIstant.common.utils import cosine, truncate_text, unique
from AssIstant.common.logger import FormatLog

class IntentInterpreter(FormatLog):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        confidence_threshold: float = 0.5,
        name: str = "intentInterpreter"
    ):
        FormatLog.__init__(self, name)
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._confidence_threshold = confidence_threshold

        self._embedding_enable = True
        self._template_embedding: Dict[IntentBase, List[List[float]]] = dict()
        self._cache = IntentCache(1000, 200)
        self._cache_hits = 0
        self._cache_miss = 0

    @property
    def cache_state(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_miss
        return {
            "used": len(self._cache),
            "hits_rate": self._cache_hits / total if total else 0.0,
            "hits": self._cache_hits,
            "miss": self._cache_miss
        }

    async def interpret(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> IntentResult:
        cache_key = self._cache.cache_key(message, history)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache.get(cache_key)
        self._cache_miss += 1

        time_b = monotonic()
        llm_task = asyncio.create_task(self._llm_interpret(message, history))
        pat_task = asyncio.create_task(self._pattern_interpret(message))
        if self._embedding_enable:
            emb_task = asyncio.create_task(self._embedding_interpret(message))
            llm_res, emb_res, pat_res = await asyncio.gather(llm_task, emb_task, pat_task)
        else:
            llm_res, pat_res = await asyncio.gather(llm_task, pat_task)
            emb_res = {"intent": OtherIntent, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm_res, emb_res, pat_res)
        entities = self._extract_entities(message)
        ugc = urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=ugc,
            intent_group=intent.group,
            entities=entities,
            reasoning=llm_res.get("reasoning", ""),
            latency=(monotonic() - time_b) * 1000,
            source_scores=source_scores
        )

        self._cache.set(cache_key, result)
        return result

    def learn(self, message: str, correct: IntentBase):
        """
        意图学习
        :param message:
        :param correct:
        :return:
        """
        if message not in correct.template:
            correct.template.append(message)
            self._template_embedding.pop(correct, None)
            self._cache.clear()
            self.info(f"learned new example {{{type(correct).__name__}: {truncate_text(message)}}}")

    async  def _llm_interpret(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """
        基于LLM，结合历史上下文理解意图
        :param message:
        :param history:
        :return:
        """
        from AssIstant.common.prompt import PROMPTS

        message = clean_text(message)
        context = ""
        if history:
            context = PROMPTS["recent_conversations"].format(
                "\n".join(
                    f"  {clean_text(his.get('role', 'user'))}: {clean_text(his.get('content', ''))}"
                    for his in history[-3:]
                )
            )

        prompt = PROMPTS["interpret"].format(
            ", ".join(sorted(IntentBase.names())),
            context, message
        )
        prompt = clean_text(prompt)
        self.info(f"prompt for LLM to interpret intent:\n{prompt}")

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find('{'), raw.rfind("}")+1
            data = loads(raw[s:e])
            data["intent"] = IntentBase.intent(data["intent"])
            if data["intent"] is None:
                data["intent"] = OtherIntent
            return data
        except Exception as e:
            self.exception(f"failed to interpret intent by LLM because of exception: {e}")
            return {"intent": OtherIntent, "confidence": 0.0, "reasoning": "LLM interpretation failure", "failed": True}

    async def _embedding_interpret(self, message: str) -> Dict[str, Any]:
        """
        基于Embedding匹配意图
        :param message:
        :return:
        """
        try:
            await self._load_template_embedding()
            vec = await self._remote_embedding(message)

            best_cat, score = OtherIntent, 0.0
            for cat, vecs in self._template_embedding.items():
                sc = max(cosine(vec, v) for v in vecs)
                if sc > score:
                    best_cat, score = cat, sc

            return {"intent": best_cat, "confidence": score}
        except Exception as e:
            self.exception(f"failed to interpret intent through embedding vector because of exception: {e}")
            return {"intent": OtherIntent, "confidence": 0.0, "reasoning": "LLM interpretation failure", "failed": True}

    async def _pattern_interpret(self, message: str) -> Dict[str, Any]:
        """
        基于关键词匹配意图
        :param message:
        :return:
        """
        msg = message.lower()
        best_cat, score = pattern_match(msg, {intent: intent.keywords
                                              for intent in SpecificIntentBase.intents()
                                              })
        if OtherIntent == best_cat:
            best_cat, score = pattern_match(msg, {intent: intent.keywords
                                                  for intent in GenericIntentBase.intents()
                                                  })
        return {"intent": best_cat, "confidence": score}

    def _vote(
        self,
        llm: Dict[str, Any],
        embedding: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> tuple[IntentBase, float, Dict[str, float]]:
        """
        对不同的意图解释打分加权头片，综合判断意图
        :param llm:
        :param embedding:
        :param pattern:
        :return:
        """
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(embedding.get("confidence", 0.0) or 0.0),
            "pattern": float(pattern.get("confidence", 0.0) or 0.0)
        }
        if llm.get("failed"):
            if embedding.get("intent") != OtherIntent and embedding.get("confidence", 0.0) > 0:
                return embedding["intent"], source_scores["embedding"], source_scores
            if pattern.get("intent") != OtherIntent and pattern.get("confidence", 0.0) > 0:
                return pattern["intent"], source_scores["pattern"], source_scores
            return OtherIntent, 0.0, source_scores

        if self._embedding_enable:
            W = [(llm, 0.7), (embedding, 0.2), (pattern, 0.1)]
        else:
            W = [(llm, 0.85), (pattern, 0.15)]
        scores: Dict[IntentBase, float] = dict()
        for source, w in W:
            cat = source.get("intent", OtherIntent)
            conf = source.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)
        best_score = scores[best]
        pattern_intent = pattern.get("intent", OtherIntent)
        pattern_conf = float(pattern.get("confidence", 0.0) or 0.0)
        if best in GenericIntentBase.intents() \
                and pattern_intent in SpecificIntentBase.intents() \
                and pattern_conf >= 0.5 \
                and best_score < 0.8:
            source_scores["refined_by_pattern"] = pattern_conf
            return pattern_intent, max(best_score, pattern_conf), source_scores
        if best_score < self._confidence_threshold:
            return OtherIntent, best_score, source_scores
        return best, best_score, source_scores

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """
        提取某些重要的字段
        :param message:
        :return:
        """
        from re import findall, I
        message = clean_text(message)
        return {
            "id": unique(findall(r"(?<![A-Za-z0-9_])(?:\w*_id|id)\s*[:：=]?\s*([A-Za-z0-9](?:[A-Za-z0-9_-]{2,30})[A-Za-z0-9])(?![A-Za-z0-9_])", message, I)),
            "date": unique(findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            # "amount": unique(findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", content, I)),
            "error_code": unique(
                findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, I)
                + findall(r"\b([45]\d{2})\b", message)
            )
        }

    async def _load_template_embedding(self):
        """
        加载意图的TEMPLATE，并向量化
        :return:
        """
        missing = [intent for intent in IntentBase.intents() if intent not in self._template_embedding]
        if not missing: return

        vecs = await asyncio.gather(*(self._remote_embedding(tplt) for intent in missing for tplt in intent.template))
        idx = 0
        for tplt in missing:
            n = len(tplt.template)
            self._template_embedding[tplt] = vecs[idx: idx+n]
            idx += n

    async def _remote_embedding(self, text: str) -> List[float]:
        """
        调用模型执行文本embedding，如果失败退回本地embedding
        :param text:
        :return:
        """
        embeddings = getattr(self._client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as e:
                self.exception(f"remote embedding failure because exception: {e}")
        return embedding(text)



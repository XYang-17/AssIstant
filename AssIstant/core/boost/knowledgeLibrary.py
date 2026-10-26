import chromadb, asyncio
from hashlib import md5
from typing import List

from base.document import *
from AssIstant.core.boost.utils import chunkify
from AssIstant.common.logger import logger
from AssIstant.common.utils import truncate_text

class KnowledgeLibrary:

    def __init__(
        self,
        collection: str,
        host: str = "localhost",
        port: int = 8728,
        path: str = './data/chroma'
    ):
        self._use_server = False
        try:
            self._client = chromadb.HttpClient(
                host=host, port=port,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"Successfully connected ChromaDB({host}:{port})")
        except Exception:
            logger.warn(f"Failed to connect ChromaDB({host}:{port})")
            self._client = chromadb.PersistentClient(
                path=path,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            logger.info(f"Successfully connected ChromaDB({path})")

        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"description": "RAG for AssIstant"}
        )

        # if 0 == self._collection.count():
        #     self._load_default_docs()
        self._name = collection

    def import_documents(self, documents: List[Document]) -> int:
        """
        将一批文档导入知识库
        :param documents:
        :return:
        """
        ids, docs, metas = [], [], []
        for doc in documents:
            chunks = chunkify(doc.content, chunk_size=500)
            for i, chunk in enumerate(chunks):
                ids.append(md5(f"{doc.title}_{i}_{truncate_text(chunk)}".encode()).hexdigest())
                docs.append(chunk)
                metas.append({
                    "title": doc.title,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

        if ids:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"Add {len(ids)} documents(chunk) into {self._name}")
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        从数据库中搜索文档
        :param query:
        :param top_k:
        :return:
        """
        res = self._collection.query(
            query_texts=[query],
            n_results=top_k
        )

        items = []
        if res["documents"] and res["documents"][0]:
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0]
            ):
                items.append(SearchResult(
                    title=meta["title"],
                    content=doc,
                    score=round(1.0-dist, 4),
                    chunk=meta["chunk_index"]
                ))
        return items

    async def aimport_documents(self, documents: List[Document]) -> int:
        return await asyncio.to_thread(self.import_documents, documents)

    async def asearch(self, query: str, top_k: int = 5) -> List[SearchResult]:
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def count(self) -> int:
        """
        数据库记录数
        :return:
        """
        return self._collection.count()

    async def acount(self) -> int:
        return await asyncio.to_thread(self._collection.count)

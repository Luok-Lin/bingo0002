"""RAG Subsystem.

核心改进:
1) 向量库增量写入，不再每次重建覆盖；
2) 文档默认带 ticker 元数据，检索支持 ticker 强过滤；
3) 保留日期过滤，避免未来函数。
"""
import datetime
import hashlib
import os
import re

from rag.embedding_factory import get_embedding_function, project_base_dir, resolve_embedding_model


def _load_chroma_class():
    try:
        from langchain_chroma import Chroma
    except Exception as exc:
        raise RuntimeError(
            "langchain-chroma 未能导入。请先安装依赖: pip install -r requirements.txt"
        ) from exc
    return Chroma


def _build_document(page_content: str, metadata: dict):
    try:
        from langchain_core.documents import Document
    except Exception as exc:
        raise RuntimeError("langchain-core 未能导入。请先安装依赖: pip install -r requirements.txt") from exc
    return Document(page_content=page_content, metadata=metadata)


def resolve_chroma_persist_dir(base_dir: str | None = None) -> str:
    base = base_dir or project_base_dir()
    configured = str(os.getenv("CHROMA_PERSIST_DIR", "") or "").strip()
    if configured:
        return configured if os.path.isabs(configured) else os.path.abspath(os.path.join(base, configured))
    return os.path.join(base, "data", "vector_db", "chroma_db")


class SimpleRAG:
    def __init__(self, data_sources=None):
        self.kb = data_sources or []
        self.memory_docs = []
        self.vectorstore = None
        self.degraded_reason = ""
        self._init_vectorstore()
        if self.kb:
            self.ingest(self.kb)

    def _init_vectorstore(self):
        print("[RAG Subsystem] 正在初始化 HuggingFace Embeddings 与 ChromaDB...")
        try:
            base_dir = project_base_dir()
            model_name_or_path, _ = resolve_embedding_model()
            print(f"[RAG Subsystem] Embedding 模型: {model_name_or_path}")
            embeddings = get_embedding_function()
            Chroma = _load_chroma_class()
            self.vectorstore = Chroma(
                collection_name="market_news_reports",
                embedding_function=embeddings,
                persist_directory=resolve_chroma_persist_dir(base_dir),
            )
        except Exception as exc:
            self.vectorstore = None
            self.degraded_reason = str(exc)
            print(f"[RAG Subsystem][WARN] 向量库不可用，启用内存关键词检索: {exc}")

    @staticmethod
    def _normalize_item(item):
        if isinstance(item, dict):
            page_content = str(item.get("page_content", "")).strip()
            metadata = dict(item.get("metadata", {}) or {})
        else:
            page_content = str(item).strip()
            metadata = {}

        if not page_content:
            return None, None

        metadata.setdefault("date_int", 20991231)
        metadata["ticker"] = str(metadata.get("ticker", "UNKNOWN")).strip().zfill(6) if str(
            metadata.get("ticker", "UNKNOWN")
        ).strip().isdigit() else str(metadata.get("ticker", "UNKNOWN")).strip()
        metadata.setdefault("source", "news_or_report")
        metadata["is_fallback"] = str(metadata.get("source", "")).lower() == "fallback"
        return page_content, metadata

    @staticmethod
    def _build_doc_id(page_content: str, metadata: dict) -> str:
        raw_key = f"{metadata.get('ticker','UNKNOWN')}|{metadata.get('date_int',20991231)}|{metadata.get('source','news_or_report')}|{page_content}"
        return "doc_" + hashlib.sha1(raw_key.encode("utf-8")).hexdigest()

    def ingest(self, data_sources):
        if not data_sources:
            return 0

        normalized_items = []
        for item in data_sources:
            page_content, metadata = self._normalize_item(item)
            if page_content:
                normalized_items.append((page_content, metadata))

        if not self.vectorstore:
            existing_ids = {doc["id"] for doc in self.memory_docs}
            added = 0
            for page_content, metadata in normalized_items:
                doc_id = self._build_doc_id(page_content, metadata)
                if doc_id in existing_ids:
                    continue
                self.memory_docs.append({"id": doc_id, "page_content": page_content, "metadata": metadata})
                existing_ids.add(doc_id)
                added += 1
            print(f"[RAG Subsystem] 内存知识库入库完成: 新增 {added} 条。")
            return added

        documents, ids = [], []
        for page_content, metadata in normalized_items:
            documents.append(_build_document(page_content, metadata))
            ids.append(self._build_doc_id(page_content, metadata))

        if not documents:
            return 0

        # 增量入库：跳过已有 ID，避免重复灌库。
        existing = self.vectorstore.get(ids=ids)
        existing_ids = set(existing.get("ids", []) if existing else [])
        new_docs = [doc for doc, doc_id in zip(documents, ids) if doc_id not in existing_ids]
        new_ids = [doc_id for doc_id in ids if doc_id not in existing_ids]
        if new_docs:
            self.vectorstore.add_documents(new_docs, ids=new_ids)
        print(f"[RAG Subsystem] 增量入库完成: 新增 {len(new_docs)} 条, 跳过重复 {len(documents) - len(new_docs)} 条。")
        return len(new_docs)

    def retrieve(self, query: str, target_date: str = None, ticker: str = None, top_k: int = 3):
        print(
            f"[RAG Subsystem] 检索词 \"{query}\" | ticker={ticker} | 截止日期={target_date} "
            "(防未来函数 & 30天保鲜期)..."
        )
        if not self.vectorstore:
            return self._memory_retrieve(query, target_date=target_date, ticker=ticker, top_k=top_k)

        filter_conditions = []
        if ticker:
            filter_conditions.append({"ticker": {"$eq": str(ticker).strip().zfill(6)}})

        if target_date:
            try:
                dt_obj = datetime.datetime.strptime(str(target_date)[:10], "%Y-%m-%d")
                end_int = int(dt_obj.strftime("%Y%m%d"))
                start_int = int((dt_obj - datetime.timedelta(days=30)).strftime("%Y%m%d"))
                filter_conditions.append({"date_int": {"$lte": end_int}})
                filter_conditions.append({"date_int": {"$gte": start_int}})
            except Exception:
                pass

        filter_dict = self._build_filter(filter_conditions)
        results = self.vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
        if not results and target_date:
            fallback_conditions = [{"is_fallback": {"$eq": True}}]
            if ticker:
                fallback_conditions.insert(0, {"ticker": {"$eq": str(ticker).strip().zfill(6)}})
            results = self.vectorstore.similarity_search(
                query,
                k=top_k,
                filter=self._build_filter(fallback_conditions),
            )
        return [doc.page_content for doc in results]

    def _memory_retrieve(self, query: str, target_date: str = None, ticker: str = None, top_k: int = 3):
        end_int = None
        start_int = None
        if target_date:
            try:
                dt_obj = datetime.datetime.strptime(str(target_date)[:10], "%Y-%m-%d")
                end_int = int(dt_obj.strftime("%Y%m%d"))
                start_int = int((dt_obj - datetime.timedelta(days=30)).strftime("%Y%m%d"))
            except Exception:
                pass

        query_tokens = [
            token.lower()
            for token in re.split(r"[\s,，。；;:：|/\\()\[\]{}<>《》\"']+", str(query or ""))
            if token.strip()
        ]
        scored = []
        fallback_docs = []
        ticker_norm = str(ticker).strip().zfill(6) if ticker else ""
        for doc in self.memory_docs:
            metadata = doc.get("metadata", {}) or {}
            if ticker_norm and str(metadata.get("ticker", "")).strip().zfill(6) != ticker_norm:
                continue
            date_int = int(metadata.get("date_int", 0) or 0)
            is_fallback = bool(metadata.get("is_fallback"))
            if is_fallback:
                fallback_docs.append(doc)
            if end_int is not None and date_int and date_int > end_int:
                continue
            if start_int is not None and date_int and not is_fallback and date_int < start_int:
                continue
            text = str(doc.get("page_content", ""))
            lowered = text.lower()
            score = sum(1 for token in query_tokens if token and token in lowered)
            if ticker_norm and ticker_norm in lowered:
                score += 2
            if score <= 0 and query_tokens:
                score = 0.1
            scored.append((score, date_int, text))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        docs = [text for score, _, text in scored[:top_k] if score > 0]
        if not docs and fallback_docs:
            docs = [str(doc.get("page_content", "")) for doc in fallback_docs[:top_k]]
        return docs

    @staticmethod
    def _build_filter(filter_conditions: list[dict]) -> dict | None:
        if not filter_conditions:
            return None
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        return {"$and": filter_conditions}

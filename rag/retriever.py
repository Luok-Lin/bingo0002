"""RAG Subsystem.

核心改进:
1) 向量库增量写入，不再每次重建覆盖；
2) 文档默认带 ticker 元数据，检索支持 ticker 强过滤；
3) 保留日期过滤，避免未来函数。
"""
import datetime
import hashlib
import os

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
        self.vectorstore = None
        self._init_vectorstore()
        if self.kb:
            self.ingest(self.kb)

    def _init_vectorstore(self):
        print("[RAG Subsystem] 正在初始化 HuggingFace Embeddings 与 ChromaDB...")
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
        if not self.vectorstore or not data_sources:
            return 0

        documents, ids = [], []
        for item in data_sources:
            page_content, metadata = self._normalize_item(item)
            if not page_content:
                continue
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
            return []

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

    @staticmethod
    def _build_filter(filter_conditions: list[dict]) -> dict | None:
        if not filter_conditions:
            return None
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        return {"$and": filter_conditions}

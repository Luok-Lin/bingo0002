"""RAG Subsystem.

核心改进:
1) 向量库增量写入，不再每次重建覆盖；
2) 文档默认带 ticker 元数据，检索支持 ticker 强过滤；
3) 保留日期过滤，避免未来函数。
"""
import datetime
import hashlib
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = None
            print(
                "[RAG Subsystem] WARNING: HuggingFaceEmbeddings import failed.\n"
                "建议安装/更新依赖: pip install -U langchain-huggingface"
            )


class SimpleRAG:
    def __init__(self, data_sources=None):
        self.kb = data_sources or []
        self.vectorstore = None
        self._init_vectorstore()
        if self.kb:
            self.ingest(self.kb)

    def _init_vectorstore(self):
        print("[RAG Subsystem] 正在初始化 HuggingFace Embeddings 与 ChromaDB...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        if HuggingFaceEmbeddings is None:
            raise RuntimeError("HuggingFaceEmbeddings 未能导入，请运行: pip install -U langchain-huggingface")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_base = os.path.join(
            base_dir,
            "data",
            "models",
            "all-MiniLM-L6-v2",
            "models--sentence-transformers--all-MiniLM-L6-v2",
            "snapshots",
        )
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {}
        if os.path.exists(local_model_base):
            snapshots = [d for d in os.listdir(local_model_base) if os.path.isdir(os.path.join(local_model_base, d))]
            if snapshots:
                model_name_or_path = os.path.join(local_model_base, snapshots[0])
                model_kwargs = {"local_files_only": True}
                print(f"[RAG Subsystem] 使用本地模型: {model_name_or_path}")

        embeddings = HuggingFaceEmbeddings(model_name=model_name_or_path, model_kwargs=model_kwargs)
        self.vectorstore = Chroma(
            collection_name="market_news_reports",
            embedding_function=embeddings,
            persist_directory=os.path.join(base_dir, "data", "vector_db", "chroma_db"),
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
            documents.append(Document(page_content=page_content, metadata=metadata))
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

        if not filter_conditions:
            filter_dict = None
        elif len(filter_conditions) == 1:
            filter_dict = filter_conditions[0]
        else:
            filter_dict = {"$and": filter_conditions}

        results = self.vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
        return [doc.page_content for doc in results]


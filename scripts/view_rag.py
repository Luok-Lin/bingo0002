import os
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

if HuggingFaceEmbeddings is None:
    print("[view_rag] WARNING: HuggingFaceEmbeddings 未安装，若需启用向量化功能请运行: pip install -U langchain-huggingface")
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 与主流程保持一致：优先使用本地模型快照，避免访问 huggingface.co
    local_model_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
            print(f"[view_rag] 使用本地模型: {model_name_or_path}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name_or_path, model_kwargs=model_kwargs)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "vector_db", "chroma_db")

vectorstore = Chroma(
    collection_name="market_news_reports",
    persist_directory=db_path,
    embedding_function=embeddings,
)

collections = vectorstore.get()
docs = collections.get('documents', [])
metadatas = collections.get('metadatas', [])
print(f"RAG库中共找到 {len(docs)} 条记录：\n")
for i, doc in enumerate(docs, 1):
    md = metadatas[i - 1] if i - 1 < len(metadatas) else {}
    print(f"[{i}] ticker={md.get('ticker', 'UNKNOWN')} | date={md.get('date_int', 'NA')} | source={md.get('source', 'NA')}")
    print(f"    {doc}\n")

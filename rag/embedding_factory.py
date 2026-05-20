from __future__ import annotations

import os
import threading
from functools import lru_cache


_EMBEDDING_LOCK = threading.Lock()


def project_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_embedding_model() -> tuple[str, dict]:
    base_dir = project_base_dir()
    explicit_model = str(os.getenv("EMBEDDING_MODEL_NAME", "") or "").strip()
    if explicit_model:
        return explicit_model, {}

    local_model_base = os.path.join(
        base_dir,
        "data",
        "models",
        "all-MiniLM-L6-v2",
        "models--sentence-transformers--all-MiniLM-L6-v2",
        "snapshots",
    )
    if os.path.exists(local_model_base):
        snapshots = [
            item
            for item in os.listdir(local_model_base)
            if os.path.isdir(os.path.join(local_model_base, item))
        ]
        if snapshots:
            snapshots.sort()
            return os.path.join(local_model_base, snapshots[0]), {"local_files_only": True}

    return "sentence-transformers/all-MiniLM-L6-v2", {}


def _load_huggingface_embeddings_class():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as exc:
            raise RuntimeError(
                "HuggingFaceEmbeddings 未能导入，请运行: pip install -U langchain-huggingface"
            ) from exc
    return HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def _cached_embedding_function(model_name_or_path: str, model_kwargs_items: tuple[tuple[str, object], ...]):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    HuggingFaceEmbeddings = _load_huggingface_embeddings_class()
    model_kwargs = dict(model_kwargs_items)
    return HuggingFaceEmbeddings(model_name=model_name_or_path, model_kwargs=model_kwargs)


def get_embedding_function():
    model_name_or_path, model_kwargs = resolve_embedding_model()
    model_kwargs_items = tuple(sorted(model_kwargs.items()))
    with _EMBEDDING_LOCK:
        return _cached_embedding_function(model_name_or_path, model_kwargs_items)

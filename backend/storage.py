from __future__ import annotations

import json
import os
import threading
from typing import Any

from .config import WEB_INDEX_DIR

_LOCK = threading.Lock()


def ensure_web_index_dir() -> None:
    os.makedirs(WEB_INDEX_DIR, exist_ok=True)


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: str, payload: Any) -> None:
    ensure_web_index_dir()
    tmp_path = f"{path}.tmp"
    with _LOCK:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)


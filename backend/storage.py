from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime
from typing import Any

from .config import WEB_INDEX_DIR

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

_LOCK = threading.RLock()
_MISSING = object()


def ensure_web_index_dir() -> None:
    os.makedirs(WEB_INDEX_DIR, exist_ok=True)


def _state_db_path() -> str:
    return os.path.join(WEB_INDEX_DIR, "state.sqlite3")


def _is_web_index_path(path: str) -> bool:
    try:
        return os.path.commonpath([WEB_INDEX_DIR, os.path.abspath(path)]) == WEB_INDEX_DIR
    except ValueError:
        return False


def _document_key(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), WEB_INDEX_DIR)


def _connect_state_db() -> sqlite3.Connection:
    ensure_web_index_dir()
    conn = sqlite3.connect(_state_db_path(), timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS json_documents (
            path TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _read_from_state_db(path: str) -> Any:
    if not _is_web_index_path(path):
        return _MISSING
    key = _document_key(path)
    try:
        with _connect_state_db() as conn:
            row = conn.execute("SELECT payload FROM json_documents WHERE path = ?", (key,)).fetchone()
        if row is None:
            return _MISSING
        return json.loads(row[0])
    except Exception as exc:
        print(f"[storage] SQLite 状态读取失败: {path}; error={exc}")
        return _MISSING


def _write_to_state_db(path: str, payload: Any) -> None:
    if not _is_web_index_path(path):
        return
    key = _document_key(path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect_state_db() as conn:
        conn.execute(
            """
            INSERT INTO json_documents(path, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (key, serialized, now),
        )


class _FileLock:
    def __init__(self, path: str) -> None:
        self.lock_path = f"{path}.lock"
        self._handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
        self._handle = open(self.lock_path, "a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _backup_corrupt_json(path: str, error: Exception) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.corrupt_{timestamp}"
    try:
        shutil.copy2(path, backup_path)
        print(f"[storage] JSON 读取失败，已备份损坏文件: {backup_path}; error={error}")
    except Exception as backup_error:
        print(f"[storage] JSON 读取失败且备份失败: {path}; error={error}; backup_error={backup_error}")


def read_json(path: str, default: Any) -> Any:
    with _LOCK:
        state_payload = _read_from_state_db(path)
        if state_payload is not _MISSING:
            return state_payload

    if not os.path.exists(path):
        return default
    with _LOCK:
        with _FileLock(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                _write_to_state_db(path, payload)
                return payload
            except json.JSONDecodeError as exc:
                _backup_corrupt_json(path, exc)
                return default
            except Exception as exc:
                print(f"[storage] JSON 读取失败: {path}; error={exc}")
                return default


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with _LOCK:
        _write_to_state_db(path, payload)
        with _FileLock(path):
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

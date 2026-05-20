from __future__ import annotations

import threading
import uuid
import os
from datetime import datetime
from typing import Callable

from .config import TASKS_PATH
from .storage import read_json, write_json

TaskCallable = Callable[[], dict]


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running_keys: set[str] = set()

    def _load_tasks(self) -> list[dict]:
        tasks = read_json(TASKS_PATH, [])
        return tasks if isinstance(tasks, list) else []

    def _save_tasks(self, tasks: list[dict]) -> None:
        try:
            history_limit = max(50, int(str(os.getenv("TASK_HISTORY_LIMIT", "500")).strip() or "500"))
        except ValueError:
            history_limit = 500
        tasks = sorted(tasks, key=lambda x: x.get("created_at", ""))[-history_limit:]
        write_json(TASKS_PATH, tasks)

    def _append_task(self, task: dict) -> None:
        with self._lock:
            tasks = self._load_tasks()
            tasks.append(task)
            self._save_tasks(tasks)

    def _update_task(self, task_id: str, **kwargs) -> None:
        with self._lock:
            tasks = self._load_tasks()
            for task in tasks:
                if task.get("task_id") == task_id:
                    task.update(kwargs)
                    break
            self._save_tasks(tasks)

    def list_tasks(self, limit: int = 50) -> list[dict]:
        tasks = self._load_tasks()
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks[:limit]

    def get_task(self, task_id: str) -> dict | None:
        for task in self._load_tasks():
            if task.get("task_id") == task_id:
                return task
        return None

    def recover_interrupted_tasks(self, message: str = "服务重启导致任务中断，请重新生成。") -> int:
        """Mark orphaned running/queued tasks as failed after API restart."""
        recovered = 0
        with self._lock:
            tasks = self._load_tasks()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for task in tasks:
                if str(task.get("status", "")).lower() not in {"running", "queued"}:
                    continue
                task["status"] = "failed"
                task["ended_at"] = now
                task["message"] = message
                lock_key = str(task.get("lock_key", "") or "")
                if lock_key:
                    self._running_keys.discard(lock_key)
                recovered += 1
            if recovered:
                self._save_tasks(tasks)
        return recovered

    def run_background(self, task_type: str, fn: TaskCallable, lock_key: str | None = None) -> str:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "type": task_type,
            "status": "queued",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "ended_at": None,
            "message": "",
            "result": None,
            "lock_key": lock_key or "",
        }
        self._append_task(task)

        def _runner() -> None:
            should_skip_due_to_lock = False
            if lock_key:
                with self._lock:
                    if lock_key in self._running_keys:
                        should_skip_due_to_lock = True
                    else:
                        self._running_keys.add(lock_key)

            if should_skip_due_to_lock:
                self._update_task(
                    task_id,
                    status="skipped",
                    started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ended_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    message=f"Task with lock_key={lock_key} is already running.",
                )
                return

            self._update_task(task_id, status="running", started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                result = fn()
                self._update_task(
                    task_id,
                    status="done",
                    ended_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    result=result,
                    message="ok",
                )
            except Exception as e:
                self._update_task(
                    task_id,
                    status="failed",
                    ended_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    message=str(e),
                )
            finally:
                if lock_key:
                    with self._lock:
                        self._running_keys.discard(lock_key)

        threading.Thread(target=_runner, daemon=True).start()
        return task_id

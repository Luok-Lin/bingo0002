from __future__ import annotations

import os
import re
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    BASE_DIR,
    EVOLUTION_HISTORY_PATH,
    INDEX_SUMMARY_PATH,
    REFLECTIONS_PATH,
    SESSIONS_PATH,
    TASKS_PATH,
    TOP_HOLDINGS_CSV,
    TRAIN_UPLOAD_DIR,
    USER_ADVICE_HISTORY_PATH,
    USERS_PATH,
    WATCHLIST_PATH,
)
from .indexer import build_dashboard_summary
from .services import (
    get_latest_advice_for_ticker,
    list_recent_backtest_summaries,
    run_daily_evolution,
    settle_advice_experience,
    run_initial_training,
    run_investment_advice,
)
from .storage import ensure_web_index_dir, read_json, write_json
from .tasks import TaskManager

app = FastAPI(title="TradingAgents Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_manager = TaskManager()
scheduler = BackgroundScheduler(timezone="Asia/Shanghai")


class TrainingRequest(BaseModel):
    top_n: int = Field(default=10, ge=1, le=50)
    days: int = Field(default=22, ge=1, le=120)
    debate_depth: int = Field(default=2, ge=1, le=6)
    skip_auto_tune: bool = True
    csv_path: str | None = None


class AdviceRequest(BaseModel):
    ticker: str
    debate_depth: int = Field(default=2, ge=1, le=6)
    human_comment: str = ""
    human_decision: str | None = None


class EvolutionRequest(BaseModel):
    tickers: list[str] | None = None
    debate_depth: int = Field(default=2, ge=1, le=6)
    mode: Literal["backtest_update", "advice_only"] = "backtest_update"


class AdviceSettlementRequest(BaseModel):
    max_items: int = Field(default=2000, ge=100, le=10000)


class RegisterRequest(BaseModel):
    email: str
    phone: str
    password: str = Field(min_length=6, max_length=128)
    nickname: str = Field(default="", max_length=32)


class LoginRequest(BaseModel):
    account: str
    password: str = Field(min_length=6, max_length=128)


class LogoutRequest(BaseModel):
    token: str | None = None


class AdviceHistoryRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, max_length=500)


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _normalize_phone(phone: str) -> str:
    return re.sub(r"\s+", "", str(phone or "").strip())


def _validate_register_payload(req: RegisterRequest) -> tuple[str, str]:
    email = _normalize_email(req.email)
    phone = _normalize_phone(req.phone)
    if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确。")
    if not re.match(r"^\+?\d{6,20}$", phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确。")
    return email, phone


def _hash_password(raw_password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{raw_password}".encode("utf-8")).hexdigest()


def _resolve_train_csv_path(csv_path: str | None) -> str:
    candidate = str(csv_path or "").strip()
    if not candidate:
        return TOP_HOLDINGS_CSV
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    return os.path.abspath(candidate)


def _parse_time(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def _load_users() -> list[dict]:
    users = read_json(USERS_PATH, [])
    return users if isinstance(users, list) else []


def _load_sessions() -> list[dict]:
    sessions = read_json(SESSIONS_PATH, [])
    return sessions if isinstance(sessions, list) else []


def _write_sessions(sessions: list[dict]) -> None:
    write_json(SESSIONS_PATH, sessions)


def _normalize_ticker(value: str) -> str:
    digits = re.sub(r"\D+", "", str(value or ""))
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _load_user_advice_history_map() -> dict[str, list[str]]:
    payload = read_json(USER_ADVICE_HISTORY_PATH, {})
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for user_id, raw in payload.items():
        uid = str(user_id or "").strip()
        if not uid:
            continue
        if not isinstance(raw, list):
            continue
        tickers = []
        for item in raw:
            ticker = _normalize_ticker(str(item or ""))
            if ticker:
                tickers.append(ticker)
        out[uid] = list(dict.fromkeys(tickers))[-500:]
    return out


def _save_user_advice_history_map(history_map: dict[str, list[str]]) -> None:
    normalized: dict[str, list[str]] = {}
    for user_id, tickers in history_map.items():
        uid = str(user_id or "").strip()
        if not uid:
            continue
        if not isinstance(tickers, list):
            continue
        vals = []
        for item in tickers:
            ticker = _normalize_ticker(str(item or ""))
            if ticker:
                vals.append(ticker)
        normalized[uid] = list(dict.fromkeys(vals))[-500:]
    write_json(USER_ADVICE_HISTORY_PATH, normalized)


def _get_user_advice_tickers(user_id: str) -> list[str]:
    history_map = _load_user_advice_history_map()
    return history_map.get(str(user_id), [])


def _merge_user_advice_tickers(user_id: str, tickers: list[str]) -> list[str]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    incoming = [_normalize_ticker(x) for x in tickers]
    incoming = [x for x in incoming if x]
    if not incoming:
        return _get_user_advice_tickers(uid)
    history_map = _load_user_advice_history_map()
    current = history_map.get(uid, [])
    history_map[uid] = list(dict.fromkeys(current + incoming))[-500:]
    _save_user_advice_history_map(history_map)
    return history_map[uid]


def _build_user_advice_evaluation(user_id: str) -> dict:
    tickers = set(_get_user_advice_tickers(user_id))
    if not tickers:
        return {
            "tracked_tickers": 0,
            "settled_samples": 0,
            "directional_samples": 0,
            "directional_win_rate": None,
            "positive_ratio": None,
            "avg_pnl_percent": None,
            "avg_reward": None,
            "last_settled_at": "",
        }

    reflections = read_json(REFLECTIONS_PATH, [])
    if not isinstance(reflections, list):
        reflections = []

    rows = []
    for row in reflections:
        if str(row.get("source", "")).strip() != "advice_settlement":
            continue
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if ticker not in tickers:
            continue
        rows.append(row)

    if not rows:
        return {
            "tracked_tickers": len(tickers),
            "settled_samples": 0,
            "directional_samples": 0,
            "directional_win_rate": None,
            "positive_ratio": None,
            "avg_pnl_percent": None,
            "avg_reward": None,
            "last_settled_at": "",
        }

    pnl_values = []
    reward_values = []
    positive_count = 0
    directional_total = 0
    directional_win = 0
    last_settled_at = ""

    for row in rows:
        pnl = None
        reward = None
        try:
            pnl = float(row.get("pnl_percent"))
            pnl_values.append(pnl)
            if pnl > 0:
                positive_count += 1
        except Exception:
            pnl = None
        try:
            reward = float(row.get("reward_score"))
            reward_values.append(reward)
        except Exception:
            reward = None

        decision = str(row.get("decision", "")).upper()
        if decision in {"BUY", "SELL"} and pnl is not None:
            directional_total += 1
            if pnl > 0:
                directional_win += 1

        settled_at = str(row.get("settled_at", "")).strip()
        if settled_at and settled_at > last_settled_at:
            last_settled_at = settled_at

    return {
        "tracked_tickers": len(tickers),
        "settled_samples": len(rows),
        "directional_samples": directional_total,
        "directional_win_rate": round(directional_win / directional_total, 4) if directional_total else None,
        "positive_ratio": round(positive_count / len(rows), 4) if rows else None,
        "avg_pnl_percent": round(sum(pnl_values) / len(pnl_values), 4) if pnl_values else None,
        "avg_reward": round(sum(reward_values) / len(reward_values), 4) if reward_values else None,
        "last_settled_at": last_settled_at,
    }


def _to_public_user(user: dict) -> dict:
    return {
        "user_id": user.get("user_id"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "nickname": user.get("nickname"),
        "created_at": user.get("created_at"),
    }


def _cleanup_expired_sessions(sessions: list[dict]) -> list[dict]:
    now = datetime.now()
    out = []
    for session in sessions:
        try:
            if _parse_time(str(session.get("expires_at", ""))) > now:
                out.append(session)
        except Exception:
            continue
    return out


def _create_session_for_user(user_id: str, ttl_days: int = 7) -> dict:
    now = datetime.now()
    expires_at = now + timedelta(days=ttl_days)
    return {
        "token": secrets.token_urlsafe(32),
        "user_id": user_id,
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _extract_bearer_token(authorization: str | None) -> str:
    auth = str(authorization or "").strip()
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


def _get_session_user(authorization: str | None) -> tuple[dict, dict]:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="缺少有效会话令牌。")
    sessions = _cleanup_expired_sessions(_load_sessions())
    users = _load_users()
    session = next((s for s in sessions if str(s.get("token", "")) == token), None)
    if not session:
        _write_sessions(sessions)
        raise HTTPException(status_code=401, detail="会话已失效，请重新登录。")
    user = next((u for u in users if str(u.get("user_id", "")) == str(session.get("user_id", ""))), None)
    if not user:
        raise HTTPException(status_code=401, detail="会话对应用户不存在。")
    _write_sessions(sessions)
    return session, user


@app.on_event("startup")
def _startup() -> None:
    ensure_web_index_dir()
    # 每天 18:30 自动进化，且支持手动补跑。
    if not scheduler.running:
        scheduler.add_job(
            lambda: task_manager.run_background(
                "daily_evolution_auto",
                lambda: run_daily_evolution(base_dir=BASE_DIR, tickers=None, debate_depth=2, mode="backtest_update"),
                lock_key="daily_evolution",
            ),
            "cron",
            hour=18,
            minute=30,
            id="daily_evolution_job",
            replace_existing=True,
        )
        scheduler.add_job(
            lambda: task_manager.run_background(
                "advice_settlement_auto",
                lambda: settle_advice_experience(base_dir=BASE_DIR, max_items=2000),
                lock_key="advice_settlement",
            ),
            "cron",
            hour=15,
            minute=10,
            id="advice_settlement_job",
            replace_existing=True,
        )
        scheduler.start()
    build_dashboard_summary()


@app.on_event("shutdown")
def _shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/api/auth/register")
def auth_register(req: RegisterRequest) -> dict:
    email, phone = _validate_register_payload(req)
    users = read_json(USERS_PATH, [])
    if not isinstance(users, list):
        users = []
    for user in users:
        if _normalize_email(user.get("email", "")) == email:
            raise HTTPException(status_code=409, detail="该邮箱已注册。")
        if _normalize_phone(user.get("phone", "")) == phone:
            raise HTTPException(status_code=409, detail="该手机号已注册。")

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = f"u_{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
    salt = secrets.token_hex(8)
    record = {
        "user_id": user_id,
        "email": email,
        "phone": phone,
        "nickname": (req.nickname or "").strip() or f"用户{phone[-4:]}",
        "password_hash": _hash_password(req.password, salt),
        "password_salt": salt,
        "created_at": created_at,
    }
    users.append(record)
    write_json(USERS_PATH, users)
    sessions = _cleanup_expired_sessions(_load_sessions())
    session = _create_session_for_user(user_id=user_id)
    sessions.append(session)
    _write_sessions(sessions)
    return {
        "status": "ok",
        "user": _to_public_user(record),
        "access_token": session["token"],
        "expires_at": session["expires_at"],
    }


@app.post("/api/auth/login")
def auth_login(req: LoginRequest) -> dict:
    account = str(req.account or "").strip()
    if not account:
        raise HTTPException(status_code=400, detail="请输入邮箱或手机号。")
    users = _load_users()
    email = _normalize_email(account)
    phone = _normalize_phone(account)
    user = next(
        (
            u
            for u in users
            if _normalize_email(u.get("email", "")) == email or _normalize_phone(u.get("phone", "")) == phone
        ),
        None,
    )
    if not user:
        raise HTTPException(status_code=401, detail="账号或密码错误。")
    salt = str(user.get("password_salt", ""))
    expected = str(user.get("password_hash", ""))
    if not salt or _hash_password(req.password, salt) != expected:
        raise HTTPException(status_code=401, detail="账号或密码错误。")

    sessions = _cleanup_expired_sessions(_load_sessions())
    session = _create_session_for_user(user_id=str(user.get("user_id", "")))
    sessions.append(session)
    _write_sessions(sessions)
    return {
        "status": "ok",
        "user": _to_public_user(user),
        "access_token": session["token"],
        "expires_at": session["expires_at"],
    }


@app.post("/api/auth/logout")
def auth_logout(req: LogoutRequest, authorization: str | None = Header(default=None)) -> dict:
    token = str(req.token or "").strip() or _extract_bearer_token(authorization)
    if not token:
        return {"status": "ok"}
    sessions = _cleanup_expired_sessions(_load_sessions())
    sessions = [s for s in sessions if str(s.get("token", "")) != token]
    _write_sessions(sessions)
    return {"status": "ok"}


@app.get("/api/auth/session")
def auth_session(authorization: str | None = Header(default=None)) -> dict:
    session, user = _get_session_user(authorization)
    return {
        "status": "ok",
        "user": _to_public_user(user),
        "expires_at": session.get("expires_at"),
    }


@app.get("/api/user/advice-history")
def user_advice_history(authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    user_id = str(user.get("user_id", ""))
    return {
        "status": "ok",
        "tickers": _get_user_advice_tickers(user_id),
    }


@app.post("/api/user/advice-history")
def user_advice_history_sync(req: AdviceHistoryRequest, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    user_id = str(user.get("user_id", ""))
    tickers = _merge_user_advice_tickers(user_id, req.tickers or [])
    return {
        "status": "ok",
        "tickers": tickers,
        "count": len(tickers),
    }


@app.get("/api/user/advice-evaluation")
def user_advice_evaluation(authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    user_id = str(user.get("user_id", ""))
    return {
        "status": "ok",
        "evaluation": _build_user_advice_evaluation(user_id),
    }


@app.get("/api/admin/users")
def admin_users(x_admin_password: str | None = Header(default=None, alias="X-Admin-Password")) -> dict:
    required = str(os.getenv("ADMIN_PASSWORD", "123456"))
    if str(x_admin_password or "") != required:
        raise HTTPException(status_code=403, detail="管理员认证失败。")
    users = _load_users()
    rows = sorted(users, key=lambda x: x.get("created_at", ""), reverse=True)
    return {"users": [_to_public_user(u) for u in rows], "count": len(rows)}


@app.post("/api/train/init")
def train_init(req: TrainingRequest) -> dict:
    train_csv_path = _resolve_train_csv_path(req.csv_path)
    if not os.path.exists(train_csv_path):
        raise HTTPException(status_code=400, detail=f"训练CSV不存在: {train_csv_path}")
    task_id = task_manager.run_background(
        "initial_training",
        lambda: run_initial_training(
            base_dir=BASE_DIR,
            top_n=req.top_n,
            days=req.days,
            debate_depth=req.debate_depth,
            skip_auto_tune=req.skip_auto_tune,
            csv_path=train_csv_path,
        ),
        lock_key="initial_training",
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/api/train/upload-csv")
async def train_upload_csv(file: UploadFile = File(...)) -> dict:
    filename = str(file.filename or "").strip()
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持上传 CSV 文件。")
    os.makedirs(TRAIN_UPLOAD_DIR, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:120]
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}_{safe_name}"
    saved_path = os.path.join(TRAIN_UPLOAD_DIR, saved_name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    with open(saved_path, "wb") as f:
        f.write(content)
    return {
        "status": "ok",
        "csv_path": os.path.abspath(saved_path),
        "filename": filename,
    }


@app.post("/api/advice/run")
def advice_run(req: AdviceRequest, authorization: str | None = Header(default=None)) -> dict:
    payload = run_investment_advice(
        base_dir=BASE_DIR,
        ticker=req.ticker,
        debate_depth=req.debate_depth,
        human_comment=req.human_comment,
        human_decision=req.human_decision,
    )
    token = _extract_bearer_token(authorization)
    if token:
        try:
            _, user = _get_session_user(authorization)
            user_id = str(user.get("user_id", ""))
            ticker = _normalize_ticker(str(payload.get("ticker", "") or req.ticker))
            if user_id and ticker:
                _merge_user_advice_tickers(user_id, [ticker])
        except Exception:
            # Advice generation itself should not fail due to history recording.
            pass
    return payload


@app.get("/api/advice/latest/{ticker}")
def advice_latest(ticker: str) -> dict:
    payload = get_latest_advice_for_ticker(ticker)
    if not payload:
        raise HTTPException(status_code=404, detail="No advice found for ticker.")
    return payload


@app.get("/api/agent-trace/{ticker}")
def agent_trace(ticker: str) -> dict:
    payload = get_latest_advice_for_ticker(ticker)
    if not payload:
        raise HTTPException(status_code=404, detail="No advice found for ticker.")
    return {
        "ticker": payload.get("ticker"),
        "generated_at": payload.get("generated_at"),
        "analyst_cases": payload.get("analyst_cases", {}),
        "referee": payload.get("referee", {}),
        "risk": payload.get("risk", {}),
        "recommendation": payload.get("recommendation", {}),
    }


@app.post("/api/evolution/run")
def evolution_run(req: EvolutionRequest) -> dict:
    task_id = task_manager.run_background(
        "daily_evolution_manual",
        lambda: run_daily_evolution(
            base_dir=BASE_DIR,
            tickers=req.tickers,
            debate_depth=req.debate_depth,
            mode=req.mode,
        ),
        lock_key="daily_evolution",
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/api/evolution/settle-advice")
def evolution_settle_advice(req: AdviceSettlementRequest) -> dict:
    task_id = task_manager.run_background(
        "advice_settlement_manual",
        lambda: settle_advice_experience(base_dir=BASE_DIR, max_items=req.max_items),
        lock_key="advice_settlement",
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/tasks")
def list_tasks(limit: int = 50) -> dict:
    return {"tasks": task_manager.list_tasks(limit=limit)}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.get("/api/dashboard/summary")
def dashboard_summary(refresh: bool = False) -> dict:
    if refresh:
        return build_dashboard_summary()
    return read_json(INDEX_SUMMARY_PATH, build_dashboard_summary())


@app.get("/api/dashboard/backtests")
def dashboard_backtests(limit: int = 20) -> dict:
    return {"runs": list_recent_backtest_summaries(limit=limit)}


@app.get("/api/evolution/history")
def evolution_history(limit: int = 20) -> dict:
    history = read_json(EVOLUTION_HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history = sorted(history, key=lambda x: x.get("run_at", ""), reverse=True)
    return {"history": history[:limit]}


@app.get("/api/system/files")
def system_files() -> dict:
    return {
        "base_dir": BASE_DIR,
        "tasks_path": TASKS_PATH,
        "watchlist_path": WATCHLIST_PATH,
        "index_summary_path": INDEX_SUMMARY_PATH,
        "evolution_history_path": EVOLUTION_HISTORY_PATH,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)


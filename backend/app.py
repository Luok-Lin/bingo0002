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
from fastapi.staticfiles import StaticFiles
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
    get_market_ohlc_bars,
    list_recent_backtest_summaries,
    resolve_evolution_tickers,
    run_daily_evolution,
    settle_advice_experience,
    run_initial_training,
    run_investment_advice,
    _load_tickers_from_csv,
)
from .storage import ensure_web_index_dir, read_json, write_json
from .tasks import TaskManager

_PASSWORD_ITERATIONS = 210_000
_DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:8000,"
    "http://localhost:8000,"
    "http://127.0.0.1:5173,"
    "http://localhost:5173"
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, lower: int = 1) -> int:
    try:
        return max(lower, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except ValueError:
        return default


def _parse_allowed_origins() -> list[str]:
    raw = str(os.getenv("ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS)).strip()
    if raw == "*":
        return ["*"]
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return origins or _DEFAULT_ALLOWED_ORIGINS.split(",")


_ALLOWED_ORIGINS = _parse_allowed_origins()

app = FastAPI(title="TradingAgents Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials="*" not in _ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

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
    csv_path: str | None = None
    csv_top_n: int | None = Field(default=None, ge=1, le=500)
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


def _legacy_hash_password(raw_password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{raw_password}".encode("utf-8")).hexdigest()


def _hash_password(raw_password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt.encode("utf-8"),
        _PASSWORD_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${_PASSWORD_ITERATIONS}${digest}"


def _verify_password(raw_password: str, salt: str, expected_hash: str) -> bool:
    expected = str(expected_hash or "")
    if expected.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_raw, digest = expected.split("$", 2)
            iterations = int(iterations_raw)
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                raw_password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations,
            ).hex()
            return secrets.compare_digest(actual, digest)
        except Exception:
            return False
    return secrets.compare_digest(_legacy_hash_password(raw_password, salt), expected)


def _is_path_under_base_dir(path: str) -> bool:
    if _env_flag("ALLOW_EXTERNAL_CSV_PATHS", default=False):
        return True
    try:
        return os.path.commonpath([BASE_DIR, os.path.abspath(path)]) == BASE_DIR
    except ValueError:
        return False


def _resolve_train_csv_path(csv_path: str | None) -> str:
    candidate = str(csv_path or "").strip()
    if not candidate:
        return TOP_HOLDINGS_CSV
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    resolved = os.path.abspath(candidate)
    if not _is_path_under_base_dir(resolved):
        raise HTTPException(status_code=400, detail="CSV 路径必须位于项目目录内。")
    return resolved


def _resolve_upload_csv_path(csv_path: str) -> str:
    candidate = str(csv_path or "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="CSV 路径为空。")
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    resolved = os.path.abspath(candidate)
    if not _is_path_under_base_dir(resolved):
        raise HTTPException(status_code=400, detail="CSV 路径必须位于项目目录内。")
    if not os.path.exists(resolved):
        raise HTTPException(status_code=400, detail=f"CSV 文件不存在: {resolved}")
    return resolved


async def _save_uploaded_csv(file: UploadFile) -> tuple[str, str]:
    filename = str(file.filename or "").strip()
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持上传 CSV 文件。")
    os.makedirs(TRAIN_UPLOAD_DIR, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:120]
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}_{safe_name}"
    saved_path = os.path.join(TRAIN_UPLOAD_DIR, saved_name)
    max_upload_bytes = _env_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024)
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"CSV 文件过大，最大允许 {max_upload_bytes} 字节。")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    with open(saved_path, "wb") as f:
        f.write(bytes(content))
    return os.path.abspath(saved_path), filename


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


def _require_admin_password(raw_password: str | None) -> None:
    required = str(os.getenv("ADMIN_PASSWORD", "") or "").strip()
    if not required or required in {"123456", "admin", "password", "change_me_before_running"}:
        raise HTTPException(status_code=503, detail="管理员密码未配置，请设置 ADMIN_PASSWORD。")
    if not secrets.compare_digest(str(raw_password or ""), required):
        raise HTTPException(status_code=403, detail="管理员认证失败。")


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
    recovered = task_manager.recover_interrupted_tasks()
    if recovered:
        print(f"[startup] recovered {recovered} interrupted task(s)")
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
    if not salt or not _verify_password(req.password, salt, expected):
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
    _require_admin_password(x_admin_password)
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
    saved_path, filename = await _save_uploaded_csv(file)
    return {
        "status": "ok",
        "csv_path": saved_path,
        "filename": filename,
    }


@app.post("/api/evolution/upload-csv")
async def evolution_upload_csv(file: UploadFile = File(...), top_n: int | None = None) -> dict:
    saved_path, filename = await _save_uploaded_csv(file)
    limit = top_n if top_n and top_n > 0 else None
    tickers = _load_tickers_from_csv(saved_path, top_n=limit)
    if not tickers:
        raise HTTPException(status_code=400, detail="CSV 中未找到有效股票代码（需包含「股票代码」列）。")
    return {
        "status": "ok",
        "csv_path": saved_path,
        "filename": filename,
        "tickers": tickers,
        "count": len(tickers),
    }


@app.post("/api/advice/run")
def advice_run(req: AdviceRequest, authorization: str | None = Header(default=None)) -> dict:
    user_id = ""
    token = _extract_bearer_token(authorization)
    if token:
        try:
            _, user = _get_session_user(authorization)
            user_id = str(user.get("user_id", ""))
        except Exception:
            pass

    captured_user_id = user_id
    captured_ticker = req.ticker

    def _do_advice() -> dict:
        payload = run_investment_advice(
            base_dir=BASE_DIR,
            ticker=captured_ticker,
            debate_depth=req.debate_depth,
            human_comment=req.human_comment,
            human_decision=req.human_decision,
        )
        if captured_user_id:
            ticker_val = _normalize_ticker(str(payload.get("ticker", "") or captured_ticker))
            if ticker_val:
                _merge_user_advice_tickers(captured_user_id, [ticker_val])
        return payload

    task_id = task_manager.run_background(
        "investment_advice",
        _do_advice,
        lock_key=f"advice_{_normalize_ticker(req.ticker)}",
    )
    return {"task_id": task_id, "status": "queued"}


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
    evolution_csv_path = None
    if req.csv_path:
        evolution_csv_path = _resolve_upload_csv_path(req.csv_path)
    try:
        preview_tickers = resolve_evolution_tickers(
            tickers=req.tickers,
            csv_path=evolution_csv_path,
            csv_top_n=req.csv_top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not preview_tickers:
        raise HTTPException(status_code=400, detail="未指定有效股票列表。")

    captured_tickers = req.tickers
    captured_csv_path = evolution_csv_path
    captured_csv_top_n = req.csv_top_n
    captured_depth = req.debate_depth
    captured_mode = req.mode

    task_id = task_manager.run_background(
        "daily_evolution_manual",
        lambda: run_daily_evolution(
            base_dir=BASE_DIR,
            tickers=captured_tickers,
            debate_depth=captured_depth,
            mode=captured_mode,
            csv_path=captured_csv_path,
            csv_top_n=captured_csv_top_n,
        ),
        lock_key="daily_evolution",
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "ticker_count": len(preview_tickers),
        "tickers_preview": preview_tickers[:20],
    }


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
    cached = read_json(INDEX_SUMMARY_PATH, None)
    return cached if isinstance(cached, dict) else build_dashboard_summary()


@app.get("/api/dashboard/backtests")
def dashboard_backtests(limit: int = 20) -> dict:
    return {"runs": list_recent_backtest_summaries(limit=limit)}


@app.get("/api/market/ohlc/{ticker}")
def market_ohlc(ticker: str, limit: int = 40) -> dict:
    return get_market_ohlc_bars(ticker=ticker, limit=limit)


@app.get("/api/evolution/history")
def evolution_history(limit: int = 20) -> dict:
    history = read_json(EVOLUTION_HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history = sorted(history, key=lambda x: x.get("run_at", ""), reverse=True)
    return {"history": history[:limit]}


@app.get("/api/system/files")
def system_files(x_admin_password: str | None = Header(default=None, alias="X-Admin-Password")) -> dict:
    _require_admin_password(x_admin_password)
    return {
        "base_dir": BASE_DIR,
        "tasks_path": TASKS_PATH,
        "watchlist_path": WATCHLIST_PATH,
        "index_summary_path": INDEX_SUMMARY_PATH,
        "evolution_history_path": EVOLUTION_HISTORY_PATH,
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

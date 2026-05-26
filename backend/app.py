from __future__ import annotations

import os
import re
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Literal
from urllib.parse import quote

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.llm_client import LLMClient

from .config import (
    BASE_DIR,
    EVOLUTION_HISTORY_PATH,
    INDEX_SUMMARY_PATH,
    REFLECTIONS_PATH,
    SESSIONS_PATH,
    TASKS_PATH,
    TOP_HOLDINGS_CSV,
    TRAIN_UPLOAD_DIR,
    USER_AVATAR_DIR,
    USER_ADVICE_HISTORY_PATH,
    USER_STOCK_PERSONALIZATION_PATH,
    USERS_PATH,
    WATCHLIST_PATH,
)
from .indexer import build_dashboard_summary
from .reports import build_advice_pdf_bytes
from .strategy_rules import group_strategy_rules_for_display
from .services import (
    get_advice_for_report,
    get_latest_advice_for_ticker,
    get_market_ohlc_bars,
    list_recent_backtest_summaries,
    resolve_evolution_tickers,
    run_daily_evolution,
    ask_advice_question,
    build_advice_quality_ranking,
    build_advice_settlement_evaluation,
    build_portfolio_advice,
    load_strategy_rules,
    simulate_advice_counterfactual,
    settle_advice_experience,
    run_initial_training,
    run_investment_advice,
    _load_tickers_from_csv,
    _normalize_personalization_preferences,
    _normalize_user_profile,
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
os.makedirs(USER_AVATAR_DIR, exist_ok=True)

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
    user_profile: dict | None = None
    personalization: dict | None = None


class EvolutionRequest(BaseModel):
    tickers: list[str] | None = None
    csv_path: str | None = None
    csv_top_n: int | None = Field(default=None, ge=1, le=500)
    debate_depth: int = Field(default=2, ge=1, le=6)
    mode: Literal["backtest_update", "advice_only"] = "backtest_update"


class AdviceSettlementRequest(BaseModel):
    max_items: int = Field(default=2000, ge=100, le=10000)
    horizons: list[int] | None = None


class RegisterRequest(BaseModel):
    email: str
    phone: str
    password: str = Field(min_length=6, max_length=128)
    nickname: str = Field(default="", max_length=32)


class LoginRequest(BaseModel):
    account: str
    password: str = Field(min_length=6, max_length=128)


class UserAccountRequest(BaseModel):
    email: str
    phone: str
    nickname: str = Field(default="", max_length=32)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=6, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


class LogoutRequest(BaseModel):
    token: str | None = None


class AdviceHistoryRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list, max_length=500)


class AdviceQuestionRequest(BaseModel):
    ticker: str
    question: str = Field(min_length=2, max_length=500)
    include_latest_advice: bool = True
    include_rag: bool = False


class AdviceSimulationRequest(BaseModel):
    ticker: str
    scenario: dict = Field(default_factory=dict)
    user_profile: dict | None = None


class PortfolioAdviceRequest(BaseModel):
    holdings: list[dict] = Field(default_factory=list, max_length=100)
    cash_percent: float = Field(default=0.0, ge=0, le=100)
    objective: str = Field(default="balanced", max_length=32)
    user_profile: dict | None = None


class UserProfileRequest(BaseModel):
    risk_profile: str = "balanced"
    holding_period: str = "swing"
    max_position_per_stock: float = Field(default=20.0, ge=1, le=100)
    already_holding: bool = False
    current_position_percent: float = Field(default=0.0, ge=0, le=100)
    cost_price: float = Field(default=0.0, ge=0)
    prefer_stop_loss: bool = True


class StockPersonalizationRequest(BaseModel):
    ticker: str
    profile: dict | None = None
    preferences: dict | None = None


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _normalize_phone(phone: str) -> str:
    return re.sub(r"\s+", "", str(phone or "").strip())


def _validate_register_payload(req: RegisterRequest) -> tuple[str, str]:
    email = _normalize_email(req.email)
    phone = _normalize_phone(req.phone)
    _validate_email_phone(email, phone)
    return email, phone


def _validate_email_phone(email: str, phone: str) -> None:
    if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确。")
    if not re.match(r"^\+?\d{6,20}$", phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确。")


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


def _write_users(users: list[dict]) -> None:
    write_json(USERS_PATH, users)


def _detect_avatar_extension(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    return ""


async def _read_avatar_upload(file: UploadFile) -> tuple[bytes, str]:
    max_upload_bytes = _env_int("MAX_AVATAR_BYTES", 2 * 1024 * 1024)
    content = bytearray()
    while True:
        chunk = await file.read(256 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"头像文件过大，最大允许 {max_upload_bytes} 字节。")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    ext = _detect_avatar_extension(bytes(content[:32]))
    if not ext:
        raise HTTPException(status_code=400, detail="仅支持 PNG、JPG、WEBP 或 GIF 图片。")
    return bytes(content), ext


def _update_user_avatar(user_id: str, avatar_url: str) -> dict:
    uid = str(user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="缺少用户身份。")
    users = _load_users()
    target = next((u for u in users if str(u.get("user_id", "")) == uid), None)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在。")
    target["avatar_url"] = str(avatar_url or "").strip()
    target["avatar_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_users(users)
    return _to_public_user(target)


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


def _get_user_profile(user: dict) -> dict:
    raw = user.get("user_profile") if isinstance(user, dict) else {}
    return _normalize_user_profile(raw if isinstance(raw, dict) else {})


def _load_user_stock_personalization_map() -> dict[str, dict]:
    payload = read_json(USER_STOCK_PERSONALIZATION_PATH, {})
    return payload if isinstance(payload, dict) else {}


def _save_user_stock_personalization_map(payload: dict[str, dict]) -> None:
    normalized: dict[str, dict] = {}
    for user_id, raw_items in (payload or {}).items():
        uid = str(user_id or "").strip()
        if not uid or not isinstance(raw_items, dict):
            continue
        user_items: dict[str, dict] = {}
        for raw_ticker, raw_config in raw_items.items():
            ticker = _normalize_ticker(str(raw_ticker or ""))
            if not ticker or not isinstance(raw_config, dict):
                continue
            profile = _normalize_user_profile(raw_config.get("profile") if isinstance(raw_config.get("profile"), dict) else {})
            preferences = _normalize_personalization_preferences(
                raw_config.get("preferences") if isinstance(raw_config.get("preferences"), dict) else {}
            )
            updated_at = str(raw_config.get("updated_at", "") or "").strip()
            user_items[ticker] = {
                "ticker": ticker,
                "profile": profile,
                "preferences": preferences,
                "updated_at": updated_at,
            }
        normalized[uid] = user_items
    write_json(USER_STOCK_PERSONALIZATION_PATH, normalized)


def _get_user_stock_personalization(user_id: str, ticker: str, default_profile: dict | None = None) -> dict:
    uid = str(user_id or "").strip()
    normalized_ticker = _normalize_ticker(ticker)
    base_profile = _normalize_user_profile(default_profile or {})
    if not uid or not normalized_ticker:
        return {
            "ticker": normalized_ticker,
            "profile": base_profile,
            "preferences": _normalize_personalization_preferences({}),
            "updated_at": "",
        }
    all_items = _load_user_stock_personalization_map()
    user_items = all_items.get(uid) if isinstance(all_items.get(uid), dict) else {}
    raw = user_items.get(normalized_ticker) if isinstance(user_items, dict) else None
    if not isinstance(raw, dict):
        return {
            "ticker": normalized_ticker,
            "profile": base_profile,
            "preferences": _normalize_personalization_preferences({}),
            "updated_at": "",
        }
    raw_profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else {}
    merged_profile = _normalize_user_profile({**base_profile, **raw_profile})
    return {
        "ticker": normalized_ticker,
        "profile": merged_profile,
        "preferences": _normalize_personalization_preferences(raw.get("preferences") if isinstance(raw.get("preferences"), dict) else {}),
        "updated_at": str(raw.get("updated_at", "") or ""),
    }


def _update_user_stock_personalization(user_id: str, ticker: str, profile: dict | None, preferences: dict | None) -> dict:
    uid = str(user_id or "").strip()
    normalized_ticker = _normalize_ticker(ticker)
    if not uid:
        raise HTTPException(status_code=401, detail="缺少用户身份。")
    if not normalized_ticker:
        raise HTTPException(status_code=400, detail="股票代码无效。")
    current_user = next((u for u in _load_users() if str(u.get("user_id", "")) == uid), None)
    if not current_user:
        raise HTTPException(status_code=404, detail="用户不存在。")
    base_profile = _get_user_profile(current_user)
    merged_profile = _normalize_user_profile({**base_profile, **(profile or {})})
    normalized_preferences = _normalize_personalization_preferences(preferences or {})
    all_items = _load_user_stock_personalization_map()
    user_items = all_items.get(uid) if isinstance(all_items.get(uid), dict) else {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_items[normalized_ticker] = {
        "ticker": normalized_ticker,
        "profile": merged_profile,
        "preferences": normalized_preferences,
        "updated_at": now,
    }
    all_items[uid] = user_items
    _save_user_stock_personalization_map(all_items)
    return user_items[normalized_ticker]


def _build_effective_user_profile_for_ticker(user: dict, ticker: str, request_personalization: dict | None = None) -> dict:
    account_profile = _get_user_profile(user)
    user_id = str(user.get("user_id", "") or "")
    stock_config = _get_user_stock_personalization(user_id, ticker, default_profile=account_profile)
    profile = stock_config.get("profile") if isinstance(stock_config.get("profile"), dict) else account_profile
    preferences = stock_config.get("preferences") if isinstance(stock_config.get("preferences"), dict) else {}
    if isinstance(request_personalization, dict):
        if isinstance(request_personalization.get("profile"), dict):
            profile = _normalize_user_profile({**profile, **request_personalization["profile"]})
        preferences = _normalize_personalization_preferences(
            {**preferences, **(request_personalization.get("preferences") if isinstance(request_personalization.get("preferences"), dict) else request_personalization)}
        )
    effective = dict(profile)
    effective["personalization_scope"] = "stock" if stock_config.get("updated_at") else "account"
    effective["personalization_ticker"] = _normalize_ticker(ticker)
    effective["stock_personalization_updated_at"] = stock_config.get("updated_at", "")
    effective["personalization_preferences"] = _normalize_personalization_preferences(preferences)
    return effective


def _update_user_profile(user_id: str, profile: dict) -> dict:
    uid = str(user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="缺少用户身份。")
    normalized = _normalize_user_profile(profile)
    users = _load_users()
    updated = False
    for user in users:
        if str(user.get("user_id", "")) == uid:
            user["user_profile"] = normalized
            user["profile_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="用户不存在。")
    _write_users(users)
    return normalized


def _update_user_account(user_id: str, *, email: str, phone: str, nickname: str = "") -> dict:
    uid = str(user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="缺少用户身份。")
    normalized_email = _normalize_email(email)
    normalized_phone = _normalize_phone(phone)
    _validate_email_phone(normalized_email, normalized_phone)
    clean_nickname = str(nickname or "").strip()[:32]

    users = _load_users()
    target = None
    for user in users:
        current_id = str(user.get("user_id", ""))
        if current_id == uid:
            target = user
            continue
        if _normalize_email(user.get("email", "")) == normalized_email:
            raise HTTPException(status_code=409, detail="该邮箱已被其他账号使用。")
        if _normalize_phone(user.get("phone", "")) == normalized_phone:
            raise HTTPException(status_code=409, detail="该手机号已被其他账号使用。")
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在。")

    target["email"] = normalized_email
    target["phone"] = normalized_phone
    target["nickname"] = clean_nickname or f"用户{normalized_phone[-4:]}"
    target["account_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_users(users)
    return _to_public_user(target)


def _change_user_password(user_id: str, current_password: str, new_password: str) -> None:
    uid = str(user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="缺少用户身份。")
    users = _load_users()
    target = next((u for u in users if str(u.get("user_id", "")) == uid), None)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在。")
    salt = str(target.get("password_salt", ""))
    expected = str(target.get("password_hash", ""))
    if not salt or not _verify_password(current_password, salt, expected):
        raise HTTPException(status_code=403, detail="当前密码不正确。")
    new_salt = secrets.token_hex(8)
    target["password_salt"] = new_salt
    target["password_hash"] = _hash_password(new_password, new_salt)
    target["password_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_users(users)


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
            "horizon_evaluation": {},
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
            "horizon_evaluation": {},
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
        "horizon_evaluation": build_advice_settlement_evaluation(rows),
        "last_settled_at": last_settled_at,
    }


def _to_public_user(user: dict) -> dict:
    return {
        "user_id": user.get("user_id"),
        "role": user.get("role", "user"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "nickname": user.get("nickname"),
        "avatar_url": user.get("avatar_url", ""),
        "avatar_updated_at": user.get("avatar_updated_at", ""),
        "user_profile": _get_user_profile(user),
        "profile_updated_at": user.get("profile_updated_at", ""),
        "account_updated_at": user.get("account_updated_at", ""),
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


def _clear_startup_sessions_if_needed() -> bool:
    if _env_flag("PERSIST_WEB_SESSIONS", default=False):
        return False
    _write_sessions([])
    return True


@app.on_event("startup")
def _startup() -> None:
    ensure_web_index_dir()
    _clear_startup_sessions_if_needed()
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


@app.get("/api/llm/health")
def llm_health() -> dict:
    return LLMClient().check_connection()


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


@app.get("/api/user/profile")
def user_profile_get(authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    return {
        "status": "ok",
        "user_profile": _get_user_profile(user),
        "profile_updated_at": user.get("profile_updated_at", ""),
    }


@app.post("/api/user/profile")
def user_profile_update(req: UserProfileRequest, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    profile = _update_user_profile(str(user.get("user_id", "")), req.dict())
    return {
        "status": "ok",
        "user_profile": profile,
    }


@app.get("/api/user/stock-personalization/{ticker}")
def user_stock_personalization_get(ticker: str, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    profile = _get_user_profile(user)
    config = _get_user_stock_personalization(str(user.get("user_id", "")), ticker, default_profile=profile)
    return {
        "status": "ok",
        **config,
        "account_profile": profile,
    }


@app.post("/api/user/stock-personalization")
def user_stock_personalization_update(req: StockPersonalizationRequest, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    config = _update_user_stock_personalization(
        str(user.get("user_id", "")),
        req.ticker,
        req.profile if isinstance(req.profile, dict) else {},
        req.preferences if isinstance(req.preferences, dict) else {},
    )
    return {
        "status": "ok",
        **config,
    }


@app.post("/api/user/account")
def user_account_update(req: UserAccountRequest, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    public_user = _update_user_account(
        str(user.get("user_id", "")),
        email=req.email,
        phone=req.phone,
        nickname=req.nickname,
    )
    return {
        "status": "ok",
        "user": public_user,
    }


@app.post("/api/user/avatar")
async def user_avatar_upload(file: UploadFile = File(...), authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    content, ext = await _read_avatar_upload(file)
    user_id = str(user.get("user_id", "")).strip()
    safe_user_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id) or "user"
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{safe_user_id}_{stamp}_{secrets.token_urlsafe(8)}.{ext}"
    saved_path = os.path.join(USER_AVATAR_DIR, filename)
    with open(saved_path, "wb") as f:
        f.write(content)
    avatar_url = f"/avatars/{filename}"
    return {
        "status": "ok",
        "user": _update_user_avatar(user_id, avatar_url),
        "avatar_url": avatar_url,
    }


@app.post("/api/user/password")
def user_password_update(req: PasswordChangeRequest, authorization: str | None = Header(default=None)) -> dict:
    _, user = _get_session_user(authorization)
    _change_user_password(str(user.get("user_id", "")), req.current_password, req.new_password)
    return {"status": "ok"}


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
    captured_profile = req.user_profile
    if captured_profile is None and user_id:
        try:
            _, captured_user = _get_session_user(authorization)
            captured_profile = _build_effective_user_profile_for_ticker(captured_user, captured_ticker, req.personalization)
        except Exception:
            captured_profile = None
    elif captured_profile is not None and isinstance(req.personalization, dict):
        explicit_profile = req.personalization.get("profile") if isinstance(req.personalization.get("profile"), dict) else {}
        explicit_preferences = (
            req.personalization.get("preferences") if isinstance(req.personalization.get("preferences"), dict) else req.personalization
        )
        captured_profile = {
            **_normalize_user_profile({**captured_profile, **explicit_profile}),
            "personalization_preferences": _normalize_personalization_preferences(explicit_preferences),
        }

    def _do_advice(progress=None) -> dict:
        payload = run_investment_advice(
            base_dir=BASE_DIR,
            ticker=captured_ticker,
            debate_depth=req.debate_depth,
            human_comment=req.human_comment,
            human_decision=req.human_decision,
            user_profile=captured_profile,
            progress_callback=progress,
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


@app.get("/api/advice/report/{ticker}")
def advice_report(
    ticker: str,
    advice_id: str | None = None,
    generated_at: str | None = None,
    use_llm: bool = True,
) -> Response:
    try:
        payload, file_id = get_advice_for_report(ticker, advice_id=advice_id, generated_at=generated_at)
        pdf_bytes = build_advice_pdf_bytes(payload, use_llm_narrative=use_llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    filename = f"{file_id}_investment_advice_report.pdf"
    encoded = quote(filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )


@app.post("/api/advice/ask")
def advice_ask(req: AdviceQuestionRequest, authorization: str | None = Header(default=None)) -> dict:
    user_id = ""
    token = _extract_bearer_token(authorization)
    if token:
        try:
            _, user = _get_session_user(authorization)
            user_id = str(user.get("user_id", ""))
        except Exception:
            user_id = ""
    try:
        payload = ask_advice_question(
            ticker=req.ticker,
            question=req.question,
            include_latest_advice=req.include_latest_advice,
            include_rag=req.include_rag,
            user_id=user_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if user_id:
        ticker_val = _normalize_ticker(req.ticker)
        if ticker_val:
            _merge_user_advice_tickers(user_id, [ticker_val])
    return payload


@app.post("/api/advice/simulate")
def advice_simulate(req: AdviceSimulationRequest, authorization: str | None = Header(default=None)) -> dict:
    profile = req.user_profile
    token = _extract_bearer_token(authorization)
    if profile is None and token:
        try:
            _, user = _get_session_user(authorization)
            profile = _get_user_profile(user)
        except Exception:
            profile = None
    try:
        return simulate_advice_counterfactual(
            ticker=req.ticker,
            scenario=req.scenario or {},
            user_profile=profile,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/portfolio/advice")
def portfolio_advice(req: PortfolioAdviceRequest, authorization: str | None = Header(default=None)) -> dict:
    profile = req.user_profile
    token = _extract_bearer_token(authorization)
    if profile is None and token:
        try:
            _, user = _get_session_user(authorization)
            profile = _get_user_profile(user)
        except Exception:
            profile = None
    try:
        return build_portfolio_advice(
            holdings=req.holdings,
            user_profile=profile,
            cash_percent=req.cash_percent,
            objective=req.objective,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategy-rules")
def strategy_rules(limit: int = 50, status: str | None = None) -> dict:
    payload = load_strategy_rules()
    rules = payload.get("rules", [])
    if status:
        status_value = str(status).strip().lower()
        rules = [rule for rule in rules if str(rule.get("status", "")).lower() == status_value]
    raw_count = len(rules)
    rules = group_strategy_rules_for_display(rules)
    payload["rules"] = rules[: max(1, min(200, int(limit)))]
    payload["count"] = len(payload["rules"])
    payload["raw_count"] = raw_count
    payload["grouped"] = True
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
        lambda: settle_advice_experience(base_dir=BASE_DIR, max_items=req.max_items, horizons=req.horizons),
        lock_key="advice_settlement",
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/advice/quality-ranking")
def advice_quality_ranking(limit: int = 20, horizon: str | None = None, min_samples: int = 1) -> dict:
    return build_advice_quality_ranking(limit=limit, horizon=horizon, min_samples=min_samples)


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


app.mount("/avatars", StaticFiles(directory=USER_AVATAR_DIR), name="avatars")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

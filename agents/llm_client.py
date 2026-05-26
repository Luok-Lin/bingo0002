from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(DOTENV_PATH)


def env_first(names: list[str], default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return default


def looks_unconfigured_secret(value: str) -> bool:
    normalized = str(value or "").strip().strip('"').strip("'").lower()
    return normalized in {
        "",
        "your_api_key_here",
        "your_siliconflow_or_deepseek_api_key_here",
        "sk-xxx",
        "changeme",
    }


def env_int(name: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip() or str(default))
    except ValueError:
        value = default
    return max(lower, min(upper, value))


def env_float(name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(str(os.getenv(name, str(default))).strip() or str(default))
    except ValueError:
        value = default
    return max(lower, min(upper, value))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def extract_schema_keys(raw_prompt: str) -> list[str]:
    keys: list[str] = []
    for key in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', str(raw_prompt or "")):
        if key not in keys:
            keys.append(key)
    return keys


def extract_json_object(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()
    return text


def repair_json_text(raw_text: str) -> str:
    text = extract_json_object(raw_text)
    if not text:
        return text
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("：", ":")
    )
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def is_valid_json_payload(raw_text: str, schema_keys: list[str]) -> bool:
    cleaned = repair_json_text(raw_text)
    if not cleaned:
        return False
    try:
        obj = json.loads(cleaned)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    if not schema_keys:
        return True
    present = sum(1 for key in schema_keys if key in obj)
    return present >= min(2, len(schema_keys))


def rule_based_fallback(raw_prompt: str, error_msg: str = "") -> str:
    text = str(raw_prompt or "").lower()
    positive_keywords = [
        "positive", "buy", "上涨", "突破", "增持", "超预期", "净流入", "吸筹", "利好", "看多", "修复", "回升"
    ]
    negative_keywords = [
        "negative", "sell", "下跌", "回落", "减持", "透支", "净流出", "派发", "风险", "看空", "高估", "承压"
    ]
    neutral_keywords = ["neutral", "震荡", "观望", "中性", "分歧", "不确定"]

    pos_hit = [key for key in positive_keywords if key in text]
    neg_hit = [key for key in negative_keywords if key in text]
    neu_hit = [key for key in neutral_keywords if key in text]

    pos_score = len(pos_hit)
    neg_score = len(neg_hit)
    if pos_score >= neg_score + 2:
        sentiment = "positive"
    elif neg_score >= pos_score + 2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    signal_strength = max(pos_score, neg_score, 1)
    confidence = min(0.78, round(0.48 + 0.06 * signal_strength + 0.03 * abs(pos_score - neg_score), 2))
    hits = pos_hit if sentiment == "positive" else neg_hit if sentiment == "negative" else neu_hit
    key_text = "、".join(hits[:4]) if hits else "可用信号不足"
    thought = "依据提示词中的多空关键词密度估计方向，供临时参考。"
    if error_msg:
        thought = f"{thought} 触发原因: {error_msg[:120]}"
    payload = {
        "sentiment": sentiment,
        "confidence": confidence,
        "reasoning": (
            f"[规则降级] 外部LLM暂不可用，改用关键词规则推断；"
            f"识别信号: {key_text}；当前倾向: {sentiment}。"
        ),
        "thought_process": thought,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_https_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


@dataclass(frozen=True)
class LLMClientConfig:
    api_key: str
    base_url: str
    model_name: str
    stable_mode: bool
    json_retry: int
    timeout_seconds: int
    temperature: float
    top_p: float
    max_tokens: int = 1024
    response_format_json: bool = False


class LLMClient:
    def __init__(self, config: LLMClientConfig | None = None) -> None:
        self.config = config or self.from_env()

    @staticmethod
    def from_env() -> LLMClientConfig:
        stable_mode = str(os.getenv("STABLE_ADVICE_MODE", "1")).strip() != "0"
        if stable_mode:
            temperature = env_float("LLM_TEMPERATURE", 0.1, 0.0, 0.2)
            top_p = env_float("LLM_TOP_P", 0.2, 0.05, 0.9)
        else:
            temperature = env_float("LLM_TEMPERATURE", 0.7, 0.0, 2.0)
            top_p = env_float("LLM_TOP_P", 0.7, 0.01, 1.0)

        return LLMClientConfig(
            api_key=env_first(["API_KEY", "OPENAI_API_KEY"]),
            base_url=env_first(["API_BASE", "OPENAI_BASE_URL"], "https://api.deepseek.com/v1").rstrip("/"),
            model_name=env_first(["MODEL_NAME", "OPENAI_MODEL"], "deepseek-chat"),
            stable_mode=stable_mode,
            json_retry=env_int("LLM_JSON_RETRY", 3, 1, 5),
            timeout_seconds=env_int("LLM_TIMEOUT_SECONDS", 120, 5, 300),
            temperature=temperature,
            top_p=top_p,
            max_tokens=env_int("LLM_MAX_TOKENS", 1024, 128, 4096),
            response_format_json=env_bool("LLM_RESPONSE_FORMAT_JSON", False),
        )

    def _post_chat_completion(self, payload: dict) -> dict:
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "X-Failover-Enabled": "true",
            },
            timeout=self.config.timeout_seconds,
            proxies={"http": None, "https": None, "all": None},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"http {response.status_code}: {response.text[:300]}")
        return response.json()

    def public_config(self) -> dict:
        return {
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "stable_mode": self.config.stable_mode,
            "timeout_seconds": self.config.timeout_seconds,
            "max_tokens": self.config.max_tokens,
            "response_format_json": self.config.response_format_json,
            "api_key_configured": not looks_unconfigured_secret(self.config.api_key),
        }

    def check_connection(self) -> dict:
        config = self.public_config()
        if not config["api_key_configured"]:
            return {
                **config,
                "ok": False,
                "status": "unconfigured",
                "error": "LLM API key is not configured; set API_KEY or OPENAI_API_KEY in .env",
            }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "Return a tiny JSON object."},
                {"role": "user", "content": '{"ok": true}'},
            ],
            "max_tokens": 16,
            "temperature": 0,
        }
        try:
            data = self._post_chat_completion(payload)
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            return {
                **config,
                "ok": False,
                "status": "http_error",
                "error": f"http {exc.code}: {error_body[:300]}",
            }
        except urllib.error.URLError as exc:
            return {
                **config,
                "ok": False,
                "status": "url_error",
                "error": f"url error: {exc}",
            }
        except Exception as exc:
            return {
                **config,
                "ok": False,
                "status": "request_error",
                "error": f"request/response parse failed: {exc}",
            }

        if "error" in data:
            return {
                **config,
                "ok": False,
                "status": "provider_error",
                "error": str(data.get("error"))[:300],
            }

        choices = data.get("choices", [])
        if not choices:
            return {
                **config,
                "ok": False,
                "status": "empty_response",
                "error": "provider returned no choices",
            }
        return {
            **config,
            "ok": True,
            "status": "ok",
            "error": "",
        }

    def query(self, prompt: str, role: str) -> str:
        if looks_unconfigured_secret(self.config.api_key):
            return rule_based_fallback(prompt, "LLM API key is not configured; set API_KEY or OPENAI_API_KEY in .env")

        prompt_lower = str(prompt or "").lower()
        require_json = self.config.stable_mode and (
            "json" in prompt_lower or "输出纯" in prompt_lower or "合法" in prompt_lower
        )
        schema_keys = extract_schema_keys(prompt) if require_json else []
        attempts = self.config.json_retry if require_json else 1
        last_error = ""

        for attempt in range(1, attempts + 1):
            user_prompt = prompt
            if require_json:
                user_prompt = (
                    f"{prompt}\n\n"
                    "【输出约束】仅输出一个合法 JSON 对象，不要包含解释、代码块标记或额外文字。"
                    "必须使用英文双引号；布尔值必须使用 true/false；不要使用尾逗号；"
                    "若字段无法确定，也必须给出保守默认值。"
                )
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"你是一个专业的金融量化系统中的 {role}。请简明扼要、客观理性地回答。"
                            "当用户要求 JSON 时，你只能返回可被 json.loads 解析的 JSON 对象。"
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": 1,
            }
            if require_json and self.config.response_format_json and attempt == 1:
                payload["response_format"] = {"type": "json_object"}
            try:
                data = self._post_chat_completion(payload)
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                last_error = f"http {exc.code}: {error_body[:300]}"
                continue
            except urllib.error.URLError as exc:
                last_error = f"url error: {exc}"
                continue
            except Exception as exc:
                last_error = f"request/response parse failed: {exc}"
                continue

            if "error" in data:
                last_error = str(data.get("error"))
                continue

            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            if content is None:
                content = message.get("reasoning_content", "")
                if content is None:
                    content = ""
            content = str(content).strip()

            if require_json and not is_valid_json_payload(content, schema_keys):
                last_error = "model output invalid json/schema"
                continue

            return content

        return rule_based_fallback(prompt, last_error or "json retry exhausted")

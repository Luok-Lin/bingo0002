from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


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


def is_valid_json_payload(raw_text: str, schema_keys: list[str]) -> bool:
    cleaned = extract_json_object(raw_text)
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
            timeout_seconds=env_int("LLM_TIMEOUT_SECONDS", 90, 5, 180),
            temperature=temperature,
            top_p=top_p,
        )

    def _post_chat_completion(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "X-Failover-Enabled": "true",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)

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

        for _ in range(1, attempts + 1):
            user_prompt = prompt
            if require_json:
                user_prompt = (
                    f"{prompt}\n\n"
                    "【输出约束】仅输出一个合法 JSON 对象，不要包含解释、代码块标记或额外文字。"
                )
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": f"你是一个专业的金融量化系统中的 {role}。请简明扼要、客观理性地回答。"},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1024,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": 1,
            }
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

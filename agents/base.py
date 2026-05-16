# Trading Agent Base System

import os
from dotenv import load_dotenv
import json
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(DOTENV_PATH)

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.scratchpad = []

    def log(self, message: str):
        print(f"[{self.name} - {self.role}] {message}")

    def query_llm(self, prompt: str) -> str:
        """调用真实的 LLM API 进行推理"""
        api_key = str(os.getenv("API_KEY", "your_siliconflow_or_deepseek_api_key_here")).strip()
        base_url = str(os.getenv("API_BASE", "https://api.deepseek.com/v1")).strip()
        model_name = str(os.getenv("MODEL_NAME", "deepseek-chat")).strip()
        stable_mode = str(os.getenv("STABLE_ADVICE_MODE", "1")).strip() != "0"
        json_retry = max(1, min(5, int(str(os.getenv("LLM_JSON_RETRY", "3")).strip() or "3")))

        if stable_mode:
            temperature = float(str(os.getenv("LLM_TEMPERATURE", "0.1")).strip() or "0.1")
            top_p = float(str(os.getenv("LLM_TOP_P", "0.2")).strip() or "0.2")
            temperature = max(0.0, min(0.2, temperature))
            top_p = max(0.05, min(0.9, top_p))
        else:
            temperature = float(str(os.getenv("LLM_TEMPERATURE", "0.7")).strip() or "0.7")
            top_p = float(str(os.getenv("LLM_TOP_P", "0.7")).strip() or "0.7")

        def _extract_schema_keys(raw_prompt: str):
            # 从提示词中的 JSON 示例抽取关键字段名，作为最小 schema 校验。
            keys = []
            for key in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', str(raw_prompt or "")):
                if key not in keys:
                    keys.append(key)
            return keys

        def _extract_json_object(raw_text: str):
            text = str(raw_text or "").strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            # 允许模型返回前后解释文本，提取首个 JSON 对象。
            if not text.startswith("{"):
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    text = m.group(0).strip()
            return text

        def _is_valid_json_payload(raw_text: str, schema_keys: list[str]) -> bool:
            cleaned = _extract_json_object(raw_text)
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
            present = sum(1 for k in schema_keys if k in obj)
            # 至少命中 2 个 schema key，避免“伪 JSON”。
            return present >= min(2, len(schema_keys))
        
        def _rule_based_fallback(raw_prompt: str, error_msg: str = "") -> str:
            text = str(raw_prompt or "").lower()
            positive_keywords = [
                "positive", "buy", "上涨", "突破", "增持", "超预期", "净流入", "吸筹", "利好", "看多", "修复", "回升"
            ]
            negative_keywords = [
                "negative", "sell", "下跌", "回落", "减持", "透支", "净流出", "派发", "风险", "看空", "高估", "承压"
            ]
            neutral_keywords = ["neutral", "震荡", "观望", "中性", "分歧", "不确定"]

            pos_hit = [k for k in positive_keywords if k in text]
            neg_hit = [k for k in negative_keywords if k in text]
            neu_hit = [k for k in neutral_keywords if k in text]

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
            fallback_reason = (
                f"[规则降级] 外部LLM暂不可用，改用关键词规则推断；"
                f"识别信号: {key_text}；当前倾向: {sentiment}。"
            )
            thought = "依据提示词中的多空关键词密度估计方向，供临时参考。"
            if error_msg:
                thought = f"{thought} 触发原因: {error_msg[:120]}"
            payload = {
                "sentiment": sentiment,
                "confidence": confidence,
                "reasoning": fallback_reason,
                "thought_process": thought,
            }
            return json.dumps(payload, ensure_ascii=False)

        try:
            # 为了规避 requests/httpx 在某些本地环境（如代理、证书等）下出现长挂起的问题，以及规避部分模型过长的响应时间，改用 curl+subprocess 做极致底层的系统调用
            import subprocess
            prompt_lower = str(prompt or "").lower()
            require_json = stable_mode and ("json" in prompt_lower or "输出纯" in prompt_lower or "合法" in prompt_lower)
            schema_keys = _extract_schema_keys(prompt) if require_json else []
            attempts = json_retry if require_json else 1
            last_error = ""

            for attempt in range(1, attempts + 1):
                user_prompt = prompt
                if require_json:
                    user_prompt = (
                        f"{prompt}\n\n"
                        "【输出约束】仅输出一个合法 JSON 对象，不要包含解释、代码块标记或额外文字。"
                    )
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": f"你是一个专业的金融量化系统中的 {self.role}。请简明扼要、客观理性地回答。"},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": 1,
                    "extra_body": {"top_k": 50}
                }
                curl_cmd = [
                    "curl", "-s", "-k", "-X", "POST",
                    f"{base_url}/chat/completions",
                    "-H", "Content-Type: application/json",
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "X-Failover-Enabled: true",
                    "-d", json.dumps(payload, ensure_ascii=False)
                ]

                result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=90)
                if result.returncode != 0:
                    last_error = f"curl returncode={result.returncode}"
                    continue
                try:
                    data = json.loads(result.stdout)
                except Exception as e:
                    last_error = f"response json parse failed: {e}"
                    continue

                if "error" in data:
                    last_error = str(data.get("error"))
                    continue

                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content")

                # 兼容带有深度思考的模型（如GLM-5等），可能先返回 reasoning_content 而 content 为 null
                if content is None:
                    content = message.get("reasoning_content", "")
                    if content is None:
                        content = ""
                content = str(content).strip()

                if require_json and not _is_valid_json_payload(content, schema_keys):
                    last_error = "model output invalid json/schema"
                    continue

                return content

            return _rule_based_fallback(prompt, last_error or "json retry exhausted")
        except Exception as e:
            self.log(f"⚠️ LLM API 调用失败: {e}")
            return _rule_based_fallback(prompt, str(e))

    def step(self, task: str) -> str:
        raise NotImplementedError("Each agent must implement its own step logic.")

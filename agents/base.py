# Trading Agent Base System

import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.scratchpad = []

    def log(self, message: str):
        print(f"[{self.name} - {self.role}] {message}")

    def query_llm(self, prompt: str) -> str:
        """调用真实的 LLM API 进行推理"""
        api_key = os.getenv("API_KEY", "your_siliconflow_or_deepseek_api_key_here")
        base_url = os.getenv("API_BASE", "https://api.deepseek.com/v1") # 默认给以DeepSeek为例的通用API兼容层
        model_name = os.getenv("MODEL_NAME", "deepseek-chat")
        
        try:
            # 为了规避 requests/httpx 在某些本地环境（如代理、证书等）下出现长挂起的问题，以及规避部分模型过长的响应时间，改用 curl+subprocess 做极致底层的系统调用
            import subprocess
            import json
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": f"你是一个专业的金融量化系统中的 {self.role}。请简明扼要、客观理性地回答。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.7,
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
            data = json.loads(result.stdout)
            
            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            
            # 兼容带有深度思考的模型（如GLM-5等），可能先返回 reasoning_content 而 content 为 null
            if content is None:
                content = message.get("reasoning_content", "")
                if content is None:
                    content = ""
                    
            return content.strip()
        except Exception as e:
            self.log(f"⚠️ LLM API 调用失败: {e}")
            return "[模拟降级] 基本面向好，近期发布了若干利好政策。"

    def step(self, task: str) -> str:
        raise NotImplementedError("Each agent must implement its own step logic.")

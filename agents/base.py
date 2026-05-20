# Trading Agent Base System

import os
from dotenv import load_dotenv

from agents.llm_client import LLMClient, rule_based_fallback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(DOTENV_PATH)


class BaseAgent:
    _llm_client: LLMClient | None = None

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.scratchpad = []

    def log(self, message: str):
        print(f"[{self.name} - {self.role}] {message}")

    def query_llm(self, prompt: str) -> str:
        """调用真实的 LLM API 进行推理"""
        try:
            if BaseAgent._llm_client is None:
                BaseAgent._llm_client = LLMClient()
            return BaseAgent._llm_client.query(prompt, role=self.role)
        except Exception as e:
            self.log(f"⚠️ LLM API 调用失败: {e}")
            return rule_based_fallback(prompt, str(e))

    def step(self, task: str) -> str:
        raise NotImplementedError("Each agent must implement its own step logic.")

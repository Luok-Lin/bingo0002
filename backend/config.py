from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEB_INDEX_DIR = os.path.join(DATA_DIR, "monitoring", "web_index")
TASKS_PATH = os.path.join(WEB_INDEX_DIR, "tasks.json")
EVOLUTION_HISTORY_PATH = os.path.join(WEB_INDEX_DIR, "evolution_history.json")
WATCHLIST_PATH = os.path.join(WEB_INDEX_DIR, "watchlist.json")
INDEX_SUMMARY_PATH = os.path.join(WEB_INDEX_DIR, "dashboard_summary.json")
USERS_PATH = os.path.join(WEB_INDEX_DIR, "users.json")
SESSIONS_PATH = os.path.join(WEB_INDEX_DIR, "sessions.json")
USER_AVATAR_DIR = os.path.join(WEB_INDEX_DIR, "avatars")
ADVICE_SETTLEMENT_PATH = os.path.join(WEB_INDEX_DIR, "advice_settlements.json")
USER_ADVICE_HISTORY_PATH = os.path.join(WEB_INDEX_DIR, "user_advice_history.json")
USER_STOCK_PERSONALIZATION_PATH = os.path.join(WEB_INDEX_DIR, "user_stock_personalization.json")
ADVICE_QA_HISTORY_PATH = os.path.join(WEB_INDEX_DIR, "advice_qa_history.json")
STRATEGY_RULES_PATH = os.path.join(WEB_INDEX_DIR, "strategy_rules.json")

TOP_HOLDINGS_CSV = os.path.join(BASE_DIR, "个人_持股排名.csv")
ADVICE_DIR = os.path.join(DATA_DIR, "investment_advice")
BACKTEST_SUMMARY_DIR = os.path.join(DATA_DIR, "monitoring", "backtest_runs")
TRAIN_UPLOAD_DIR = os.path.join(DATA_DIR, "monitoring", "uploads")
REFLECTIONS_PATH = os.path.join(DATA_DIR, "json", "reflections.json")

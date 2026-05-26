# TradingAgents_RAG_DL

一个结合 `RAG`、`Deep Learning`、`Reward` 反馈和自动调参的 A 股多智能体投研系统。

## 你现在可以做什么

- `main.py`：单标的历史回测
- `scripts/auto_tune_reward.py`：基于回测日志自动调优 `rl/reward_config.json`
- `batch_run.py`：批量并发回测
- `webui.py`：结果与知识库可视化

## 一键回测 + 自动调参

回测结束后自动运行 reward 调参，并生成图表报告：

```bash
python3 main.py 600519 120 --auto-tune
```

常用参数：

```bash
python3 main.py 600519 120 --auto-tune --tune-window 120 --tune-samples 80
python3 main.py 600519 120 --auto-tune --tune-grid
python3 main.py 600519 120 --auto-tune --tune-dry-run
python3 main.py 600519 120 --auto-tune --tune-report-dir data/reward_tuning_reports/custom_run
```

## 单独运行 reward 调参

随机搜索：

```bash
python3 scripts/auto_tune_reward.py --window 120 --samples 80
```

Optuna 自动搜索：

```bash
python3 scripts/auto_tune_reward.py --window 120 --optuna-trials 40
```

只预览不写回配置：

```bash
python3 scripts/auto_tune_reward.py --window 120 --samples 80 --dry-run
```

## 输出目录

- 回测日志：`data/json/reflections.json`
- Reward 配置：`rl/reward_config.json`
- 图表报告：`data/reward_tuning_reports/<timestamp>/`
- 回测监控摘要：`data/monitoring/backtest_runs/<timestamp>.json`

## 依赖

核心依赖见 `requirements.txt`，其中 reward 调参图表和 Optuna 调参分别依赖 `matplotlib` 和 `optuna`。
# TradingAgents_RAG_DL

## Reward 自动调参

你可以用 `scripts/auto_tune_reward.py` 基于最近的回测日志自动搜索更优的 `rl/reward_config.json`。

### 示例

```bash
python3 scripts/auto_tune_reward.py --window 120 --samples 80
```

只做演练、不写回配置：

```bash
python3 scripts/auto_tune_reward.py --window 120 --samples 80 --dry-run
```

使用小型网格搜索：

```bash
python3 scripts/auto_tune_reward.py --window 90 --grid
```

### 输入输出

- 输入：`data/json/reflections.json`
- 输出：`rl/reward_config.json`
- 报告：终端打印基线、最优参数和指标差异# TradingAgents-RAG-DL: Multi-Agent System for Quantitative Trading

🎓 **面向金融量化研究的大语言模型多智能体协作框架**

本项目基于大型语言模型 (LLM) 架构，实现了一个融合了 **RAG (检索增强生成)**、**Deep Learning (深度学习序列预测)** 与 **强化学习 (RL 经验反馈机制)** 的复杂多智能体量化投研系统。灵感来源于并扩展自开源投研智能体系统 [KylinMountain/TradingAgents-AShare](https://github.com/KylinMountain/TradingAgents-AShare)。

本项目专为计算机科学、金融工程等交叉学科的学术研究与课程实践设计，提供了一个高度可扩展的底层框架，开箱即用。

---

## 🚀 快速开始 (Quick Start - 一键傻瓜式运行)

为了方便复现和测试，我们提供了跨平台的一键启动脚本。**克隆本项目后，你只需要 2 个步骤即可运行：**

### 💻 Windows 用户
1. 双击运行目录下的 `start.bat`。*(脚本会自动创建虚拟环境并配置清华源下载依赖)*
2. 脚本会在根目录自动生成一个 `.env` 文件。**请用记事本打开 `.env` 文件，填入你的大模型 API 密钥** (如 OpenAI / DeepSeek 等的 Key)。
3. 再次双击 `start.bat`，即可启动主程序！

### 🍎 MacOS / Linux 用户
2. 脚本会自动创建虚拟环境和 `.env` 配置文件。**使用编辑器打开 `.env` 并填入您的 API 密钥**。
3. 再次运行 `bash start.sh`，即可启动主程序！

> **🔔 注意事项**: 请勿将填好真实密钥的 `.env` 文件 `git push` 到公共仓库。系统已预设了严格的 `.gitignore` 保护您的隐私。

---

## 🔬 核心研究架构 (Architecture)

本框架采用 **大模型感知-决策-执行** 的链路范式，由专门的数字员工团队进行市场博弈：

1. **基本面分析师团队 (Fundamental Analysts)**：结合 **RAG** 技术，从海量历史研报、财报、新闻库中动态检索外挂知识，消除大语言模型的“幻觉”，从而进行精准的基本面与市场情绪分析。
2. **量化研究员团队 (Quantitative Researchers)**：内置了基于 **PyTorch 构建的 Deep Learning 基座**，用于处理股价连续 K 线数据和高频时间序列特征（如在 `dl/predictor.py` 中实现了基于 LSTM 的涨跌预测网络），用于为 LLM 提供精确的数学支撑网络，解决 LLM 在纯文本推理时数学较弱的短板。
3. **基金主理人 (Portfolio Manager / Trader)**：兼顾分析师研报和量化指标，应用基于博弈论的综合研判进行投资组合的买卖调度。
4. **反思与强化学习模块 (Memory/RL Feedback)**：包含长短期记忆库与打分网络雏形。历史交易盈利会给予系统正反馈打分 (Reward)，亏损给予负反馈，引导 Agent 自动修复历史认知偏误。

## 📂 目录结构说明

```text
TradingAgents_RAG_DL/
├── agents/             # Agent 角色定义（基类、资深分析师、量化架构师、基金主理人）
├── data/               # 运行时数据归档（记录反射JSON、SQLite记忆表及 ChromaDB 本地向量库）
├── models/             # 深度学习预训练模型权重保存目录 (.pth)
├── rag/                # 向量数据库检索模块（外挂知识基座与文档解析）
├── dl/                 # 深度学习算法验证模块（基于 PyTorch 的 LSTM等系列预测器）
├── memory/             # 基于长期、短期记忆的外带重构中心，支持经验反馈与打分网络
├── dataflows/          # 金融数据管道（支持 AKShare、Tushare 等本地及 API 接入）
├── scripts/            # 辅助调试脚本与视图工具 (如：知识库检索观测脚本)
├── start.sh / start.bat# 各平台一键启动脚本
├── main.py             # 核心投研系统执行主入口（执行单只或传参回测）
├── batch_run.py        # 基于切片的全市场批量回测运行脚本
└── requirements.txt    # 核心运行依赖清单
```

## 🛠 开发与学术二次拓展指南

1. **更换大模型与接口**: 你可以在修改 `.env` 文件的同时，在对应配置处将 `base_url` 改为国内大模型（例如智谱、Moonshot、Deepseek）的中转接口进行低成本测试。
2. **修改 RAG 数据库源**: 将你关注的证券研报放入本地目录，并在 `rag/retriever.py` 中引入对应的 PDF/TXT 文本解析器重新建立 ChromaDB 向量本地库。
3. **结合强化学习 (RL)**: 在 `main.py` 的架构预留层，引入 `Stable-Baselines3` 或 `FinRL` 将当前策略验证与真实的 PPO/DQN 算法对接。

---

## Web 版投研系统（前后端分离）

本仓库已新增前后端分离版本：

- 后端 API：`backend/app.py`（FastAPI）
- 前端页面：`frontend/index.html`（独立静态前端）

### 1) 启动后端 API

在项目根目录执行：

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

主要接口：

- `POST /api/train/init`：初始化训练（前十股票回测一个月）
- `POST /api/advice/run`：生成结构化投资建议
- `GET /api/agent-trace/{ticker}`：获取多Agent思考链路
- `POST /api/evolution/run`：手动触发自进化
- `GET /api/dashboard/summary`：总览数据
- `GET /api/tasks`：任务状态

### 2) 打开前端

推荐直接访问后端托管的页面（同源，避免 `Failed to fetch`）：

- 浏览器打开：`http://127.0.0.1:8000`

也可一键启动（默认不启用 `--reload`，避免长任务被中断）：

```bash
bash start_web.sh
```

如需独立静态前端（可选）：

```bash
cd frontend
python -m http.server 5173
```

然后浏览器打开：`http://127.0.0.1:5173`（API 默认连 `http://127.0.0.1:8000`）

### 2.1) 允许非局域网用户访问

如果只是把服务给同一台机器访问，默认 `127.0.0.1` 即可；如果要让其他电脑访问，需要让后端监听所有网卡：

```bash
PUBLIC_WEB=1 bash start_web.sh
```

如果部署在有公网 IP 的服务器上：

```bash
PUBLIC_WEB=1 PUBLIC_URL="https://你的域名" bash start_web.sh
```

公网访问需要满足以下条件：

- 服务器有公网 IP，或已经配置内网穿透/反向代理地址。
- 云服务器安全组、防火墙或路由器端口转发已放行访问端口。
- 生产环境建议用 Nginx/Caddy 做 HTTPS 反向代理，再转发到本机 `127.0.0.1:8000`。
- `.env` 中必须设置强 `ADMIN_PASSWORD`，不要使用示例值。
- 如果前端和后端不是同源部署，把公开前端地址加入 `ALLOWED_ORIGINS`，多个地址用英文逗号分隔。

常见方式：

- 云服务器：把项目部署到服务器，开放 80/443，经 Nginx/Caddy 转发到 `127.0.0.1:8000`。
- 内网穿透：本机运行后端，再用 Cloudflare Tunnel、frp、ngrok 等工具暴露一个 HTTPS 地址。
- 仅局域网共享：运行 `PUBLIC_WEB=1 bash start_web.sh` 后，同一网络内访问 `http://你的局域网IP:8000`。

本机已配置 ngrok 时，可以直接执行：

```bash
bash start_public_ngrok.sh
```

脚本会把当前本地后端 `http://127.0.0.1:8000` 暴露为一个公网 HTTPS 地址，并在终端输出可分享的 `https://...ngrok-free.app` 链接。停止公网访问时可结束 ngrok 进程，或执行：

```bash
bash stop_public_ngrok.sh
```

### 3) 每日自进化

后端内置每日自动任务（默认每天 `18:30`），执行“经验增量更新”流程；同时保留手动触发接口，支持补跑与重试。

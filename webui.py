import streamlit as st
import subprocess
import json
import pandas as pd
import os

st.set_page_config(page_title="多智能体量化推演系统", layout="wide", page_icon="📈")

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    
    /* 渐变全息大标题 */
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: 1px;
    }
    
    /* 弱化二级标题颜色让其在暗色更通透 */
    h2, h3 {
        color: #A0AEC0;
        font-weight: 700;
    }
    
    /* 深空呼吸感操作按钮 */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        border: none;
        color: #0B0F19 !important;
        border-radius: 30px;
        height: 3.5em;
        width: 100%;
        font-weight: 800;
        font-size: 16px;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.6);
        color: #000 !important;
    }
    
    /* 磨砂玻璃质感的关键数据指标卡片 */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border: 1px solid #00C9FF;
        box-shadow: 0 8px 30px rgba(0, 201, 255, 0.15);
    }
    
    /* 放大数据卡片的值字号 */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #E2E8F0 !important;
    }
    
    /* 自定义精美日志的滚动容器 */
    .log-container {
        background-color: #1e1e2e;
        color: #cdd6f4;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        height: 500px;
        overflow-y: auto;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .log-line { margin: 4px 0; line-height: 1.5; font-size: 14px; }
    .log-date { color: #f38ba8; font-weight: bold; font-size: 16px; margin-top: 15px; margin-bottom: 5px;}
    .log-sys { color: #a6adc8; font-style: italic; }
    .log-analyst { color: #89b4fa; }
    .log-rag { color: #cba6f7; font-weight: bold; }
    .log-debate { color: #fab387; }
    .log-referee { color: #f9e2af; font-weight: bold; }
    .log-risk { color: #a6e3a1; font-weight: bold; }
    .log-success { color: #a6e3a1; }
    .log-normal { color: #bac2de; }
</style>
""", unsafe_allow_html=True)

st.title("📈 A股智能投研多智能体系统")
st.markdown("##### 基于 **Agentic RAG**、**深度学习 (LSTM)** 与 **多智能体博弈架构** 的量化回测与实盘推演平台。")
st.markdown("---")

# 侧边栏
st.sidebar.header("⚙️ 推演参数配置")
ticker = st.sidebar.text_input("股票代码", value="000001", help="输入A股代码，如000001或600519")
days = st.sidebar.number_input("推演天数", min_value=1, max_value=750, value=30, help="向前推演运行的交易日天数")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #A0AEC0; font-size: 0.8rem; margin-bottom: 2rem;'>
<span style='font-size: 1.5rem;'>🤖</span><br>
Powered by <strong>Agentic RAG</strong> & <strong>PyTorch</strong><br>
<em>Quant Architecture Lab</em>
</div>
""", unsafe_allow_html=True)

def render_agent_card(name, icon, text, color, height="145px"):
    import html
    safe_text = html.escape(str(text)) if text else "..."
    return f"""
    <div style="background: linear-gradient(180deg, rgba(26,28,36,0.9) 0%, rgba(11,15,25,0.9) 100%); 
                border-top: 4px solid {color}; padding: 15px; border-radius: 12px; margin-bottom: 15px; 
                height: {height}; box-shadow: 0 8px 16px rgba(0,0,0,0.6); overflow-y: auto;">
        <div style="color: {color}; font-weight: 800; font-size: 15px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 20px;">{icon}</span> {name}
        </div>
        <div style="color: #CBD5E1; font-size: 13px; line-height: 1.5; font-family: 'Courier New', Courier, monospace;">
            {safe_text}
        </div>
    </div>
    """

if st.sidebar.button("▶️ 启动深度推演", type="primary"):
    st.toast(f"正在对 {ticker} 展开 {days} 天的多智能体兵棋推演...", icon="🔥")
    
    st.markdown("---")
    st.markdown("### 📡 实时态势感知沙盘 (Live Agent Radar)")
    
    # 顶部时间轴
    date_ph = st.empty()
    date_ph.markdown("<br><h3 style='text-align: center; color: #92FE9D;'>⏳ 正在初始化时空模拟器...</h3><br>", unsafe_allow_html=True)
    
    # 架构版面：分为三大板块
    st.markdown("#### 🧠 1. 前线信息搜集兵团 (Intelligence Gatherers)")
    c1, c2, c3 = st.columns(3)
    ph_tech = c1.empty()
    ph_fund = c2.empty()
    ph_macro = c3.empty()
    
    c4, c5, c6 = st.columns(3)
    ph_sent = c4.empty()
    ph_smart = c5.empty()
    ph_rag = c6.empty()
    
    st.markdown("#### 🤺 2. 多空博弈枢纽 (Command & C-Examine)")
    d1, d2 = st.columns(2)
    ph_bull = d1.empty()
    ph_bear = d2.empty()
    
    st.markdown("#### ⚖️ 3. 最终决策与战略反思 (Execution & Memory)")
    c_ref, c_mem = st.columns(2)
    ph_referee = c_ref.empty()
    ph_memory = c_mem.empty()
    
    # 隐藏的原始日志后台，供极客查看
    raw_expander = st.expander("🛠️ 原始终端数据流 (Raw Console Log)")
    raw_ph = raw_expander.empty()
    
    state = {
        "tech": "💤 挂起", "fund": "💤 挂起", "macro": "💤 挂起",
        "sent": "💤 挂起", "smart": "💤 挂起", "rag": "💤 挂起",
        "bull": "🥊 等待分析报告", "bear": "🥊 等待分析报告",
        "referee": "⚖️ 数据截断等待中", "memory": "🧠 记忆突触沉睡"
    }
    
    def refresh_ui():
        ph_tech.markdown(render_agent_card("量价技术分析", "📈", state["tech"], "#38BDF8"), unsafe_allow_html=True)
        ph_fund.markdown(render_agent_card("价值基本面分析", "🏢", state["fund"], "#A78BFA"), unsafe_allow_html=True)
        ph_macro.markdown(render_agent_card("宏观趋势预警", "🌍", state["macro"], "#F472B6"), unsafe_allow_html=True)
        ph_sent.markdown(render_agent_card("网络散户情绪", "💬", state["sent"], "#34D399"), unsafe_allow_html=True)
        ph_smart.markdown(render_agent_card("游资主力追踪", "💸", state["smart"], "#FB923C"), unsafe_allow_html=True)
        ph_rag.markdown(render_agent_card("深度研报特工 (RAG)", "🔎", state["rag"], "#FBBF24"), unsafe_allow_html=True)
        
        ph_bull.markdown(render_agent_card("多方先锋辩手 (Bull)", "🐂", state["bull"], "#EF4444", "170px"), unsafe_allow_html=True)
        ph_bear.markdown(render_agent_card("空方看空辩手 (Bear)", "🐻", state["bear"], "#10B981", "170px"), unsafe_allow_html=True)
        
        ph_referee.markdown(render_agent_card("风控裁判官 (Referee)", "🛡️", state["referee"], "#FDE047", "160px"), unsafe_allow_html=True)
        ph_memory.markdown(render_agent_card("量化反射中枢 (Memory)", "✨", state["memory"], "#2DD4BF", "160px"), unsafe_allow_html=True)

    refresh_ui()
    
    with st.spinner("🚀 系统指挥部已下达演算指令，智囊团正在高频运转..."):
        process = subprocess.Popen(
            ["python", "-u", "main.py", str(ticker), str(days)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, HTTP_PROXY="", HTTPS_PROXY="", http_proxy="", https_proxy="", ALL_PROXY="", all_proxy="") # 绕过系统代理防止爬虫被阻断
        )
        
        raw_logs = []
        for line in process.stdout:
            line_str = line.strip()
            if not line_str: continue
            
            raw_logs.append(line_str)
            if len(raw_logs) > 30: raw_logs.pop(0)
            raw_ph.code("\n".join(raw_logs), language="bash")
            
            content = line_str.split("] ")[-1] if "]" in line_str else line_str
            
            if "【时间游标滑动】" in line_str:
                d_str = line_str.split("当前日期:")[-1].replace("========", "").strip()
                date_ph.markdown(f"""
                <div style='text-align:center; padding:15px; background: rgba(0, 201, 255, 0.08); border: 1px solid #00C9FF; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0, 201, 255, 0.2);'>
                    <h2 style='margin:0; color:#00C9FF; font-weight:800;'>⏱️ 战略推演日节点：{d_str}</h2>
                </div>
                """, unsafe_allow_html=True)
                for k in state: state[k] = "🔍 情报收集中..."
                refresh_ui()
            
            elif "技术面" in line_str:
                state["tech"] = content; refresh_ui()
            elif "基本面" in line_str:
                state["fund"] = content; refresh_ui()
            elif "宏观" in line_str:
                state["macro"] = content; refresh_ui()
            elif "舆情" in line_str:
                state["sent"] = content; refresh_ui()
            elif "主力资金" in line_str:
                state["smart"] = content; refresh_ui()
            elif "RAG" in line_str or "研报" in line_str:
                state["rag"] = content; refresh_ui()
            elif "多方" in line_str or "看多" in line_str:
                state["bull"] = content; refresh_ui()
            elif "空方" in line_str or "看空" in line_str:
                state["bear"] = content; refresh_ui()
            elif "裁判" in line_str or "风控" in line_str or "交易接口" in line_str:
                state["referee"] = content; refresh_ui()
            elif "量化策略" in line_str or "经验" in line_str or "公理" in line_str or "实际账户盈亏" in line_str:
                state["memory"] = content; refresh_ui()
                
        process.wait()
    
    st.success("🏁 战略沙盘推演完成！战场与情报数据已全部沉淀入分布式微观知识库。")

# --- 战报可视化 ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
reflections_path = os.path.join(data_dir, "json", "reflections.json")
principles_path = os.path.join(data_dir, "json", "principles.json")

st.markdown("---")
st.subheader("📊 智能投研数据中心")

# 创建 Tab 页签来分离内容
tab_overview, tab_history, tab_principles = st.tabs(["🚀 策略总览 (Dashboard)", "📝 详细交易日志", "💎 知识图谱与核心公理"])

with tab_overview:
    if os.path.exists(reflections_path):
        try:
            with open(reflections_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            
            if records:
                df = pd.DataFrame(records)
                if 'pnl_percent' in df.columns:
                    df['pnl_ratio'] = df['pnl_percent'] / 100.0
                    initial_capital = 100000.0
                    equity = [initial_capital]
                    for pnl in df['pnl_ratio']:
                        equity.append(equity[-1] * (1 + pnl))
                    
                    df['equity'] = equity[1:]
                    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1.0

                    # 计算指标
                    total_return = (df['equity'].iloc[-1] / initial_capital - 1) * 100
                    max_drawdown = df['drawdown'].min() * 100
                    valid_trades = df[df['pnl_ratio'] != 0]
                    win_rate = len(df[df['pnl_ratio'] > 0]) / len(valid_trades) * 100 if len(valid_trades) > 0 else 0
                    trade_count = len(df[df['decision'] != 'HOLD'])
                    
                    # 使用卡片化布局顶部指标
                    st.markdown("### 🏆 核心绩效指标 (Key Performance Indicators)")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with st.container():
                        m1.metric(label="📈 累计复利收益率", value=f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
                        m2.metric(label="📉 最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta=f"{max_drawdown:.2f}%", delta_color="inverse")
                        m3.metric(label="🎯 产生交易胜率", value=f"{win_rate:.2f}%")
                        m4.metric(label="🔄 总策略捕捉单数", value=f"{trade_count} 笔")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📈 策略复利净值曲线 (Equity Curve)")
                    st.area_chart(df['equity'], use_container_width=True, color="#4CAF50")
            else:
                st.info("💡 暂无有效交易数据，请先点击侧边栏运行一次推演。")
        except Exception as e:
            st.error(f"读取记录日志失败, error: {e}")
    else:
        st.warning("⚠️ 暂无日志文件 (reflections.json)，请执行推演系统初始化数据。")

with tab_history:
    st.markdown("### 🔍 详细历史交易日志矩阵")
    if os.path.exists(reflections_path):
        try:
            with open(reflections_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True, height=600)
            else:
                st.info("数据为空。")
        except:
            st.error("解析文件失败。")
    else:
        st.warning("无历史日志。")

with tab_principles:
    st.markdown("### 🧠 Agent 底层公理引擎")
    st.markdown("这里集中展示从过去的错误和获利经历中提取出的、由 **自我迭代大模型结晶出的高维交易原则**。通过向量空间持久化指导未来的决策。")
    if os.path.exists(principles_path):
        try:
            with open(principles_path, "r", encoding="utf-8") as f:
                pr = json.load(f)
            if pr:
                st.dataframe(pd.DataFrame(pr), use_container_width=True, height=500)
            else:
                st.info("知识库暂为空原则数据。")
        except:
            st.error("解析历史反思准则发生错误。")
    else:
        st.warning("无知识库。")

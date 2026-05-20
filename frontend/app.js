const resolveSameOriginApiBase = () => {
  const { protocol, hostname, port } = window.location;
  const host = String(hostname || "").trim();
  if (!host) return "http://127.0.0.1:8000";
  return `${protocol}//${host}${port ? `:${port}` : ""}`;
};

const resolveDefaultApiBase = () => {
  const { hostname, port } = window.location;
  const host = String(hostname || "").trim();
  if (!host) return "http://127.0.0.1:8000";
  // 后端已托管 frontend/ 时走同源，避免跨域与错误端口
  if (port === "" || /^800\d$/.test(String(port || ""))) {
    return resolveSameOriginApiBase();
  }
  return `${window.location.protocol}//${host}:8000`;
};

const resolveApiBase = () => {
  const stored = String(window.localStorage.getItem("apiBase") || "").trim();
  const sameOrigin = resolveSameOriginApiBase();
  if (!stored) return resolveDefaultApiBase();
  const staleLocal8000 =
    /^https?:\/\/(127\.0\.0\.1|localhost):8000$/i.test(stored) &&
    /^https?:\/\/(127\.0\.0\.1|localhost):800\d$/i.test(sameOrigin) &&
    stored !== sameOrigin;
  return staleLocal8000 ? sameOrigin : stored;
};

let API_BASE = resolveApiBase();
const API_TIMEOUT_MS = 15000;
const ADVICE_API_TIMEOUT_MS = 60000;
const ADVICE_POLL_MAX_WAIT_MS = 600000;
const ADMIN_PASSWORD = window.localStorage.getItem("adminPassword") || "123456";
const MODE_USER = "user";
const MODE_ADMIN = "admin";
let currentMode = MODE_USER;
let pendingMode = null;
let lastOverviewSignature = "";
const USER_ADVICE_HISTORY_KEY = "userAdviceHistoryKeys";
const CURRENT_USER_KEY = "currentUserProfile";
const AUTH_TOKEN_KEY = "authToken";
const DEFAULT_TRAIN_SOURCE_TEXT = "默认文件：个人_持股排名.csv";
let currentUser = null;
let authToken = "";
let selectedTrainCsvPath = "";
let selectedEvolveCsvPath = "";
let settleAdviceSubmitting = false;
let settleAdviceTaskActive = false;
let settleTaskPollTimer = null;
let globalStatusTimer = null;
let userAdviceTickersCache = [];

const inferStatusType = (text = "") => {
  const raw = String(text || "").toLowerCase();
  if (/错误|失败|异常|超时|timeout|error|failed/.test(raw)) return "error";
  if (/成功|已提交|已更新|已加载|已退出|已读取|完成|ok/.test(raw)) return "success";
  return "info";
};

const renderGlobalStatus = (text, type = "info") => {
  const bar = document.getElementById("globalStatusBar");
  if (!bar) return;
  const nextType = ["info", "success", "error"].includes(type) ? type : "info";
  bar.textContent = String(text || "");
  bar.classList.remove("hidden", "info", "success", "error");
  bar.classList.add(nextType);

  if (globalStatusTimer) {
    window.clearTimeout(globalStatusTimer);
    globalStatusTimer = null;
  }
  if (nextType !== "error") {
    globalStatusTimer = window.setTimeout(() => {
      bar.classList.add("hidden");
      globalStatusTimer = null;
    }, 3500);
  }
};

const setStatus = (text, type = "") => {
  const normalizedText = String(text || "");
  const normalizedType = type || inferStatusType(normalizedText);
  renderGlobalStatus(normalizedText, normalizedType);
  // 保留控制台日志，方便排查
  // eslint-disable-next-line no-console
  console.log(`[TradingAgents UI][${normalizedType}] ${normalizedText}`);
};

const actionTag = (action = "") => {
  const a = String(action).toUpperCase();
  if (a === "BUY") return `<span class="tag tag-buy">BUY</span>`;
  if (a === "SELL") return `<span class="tag tag-sell">SELL</span>`;
  return `<span class="tag tag-hold">HOLD</span>`;
};

const normalizeAction = (value = "") => {
  const s = String(value || "").toUpperCase();
  if (s.includes("BUY")) return "BUY";
  if (s.includes("SELL")) return "SELL";
  return "HOLD";
};

const actionFromSignal = (value = "") => {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "HOLD";
  if (/(positive|bull|buy|看多|做多|增持)/i.test(raw)) return "BUY";
  if (/(negative|bear|sell|看空|做空|减持)/i.test(raw)) return "SELL";
  if (/(neutral|hold|观望|中性)/i.test(raw)) return "HOLD";
  return normalizeAction(value);
};

const actionExplain = (action = "HOLD", positionPercent = 0) => {
  const pos = Number(positionPercent || 0);
  const a = normalizeAction(action);
  if (a === "BUY") return pos > 20 ? "主动加仓" : "试探性买入";
  if (a === "SELL") return pos > 0 ? "减仓防守" : "空仓观望";
  return pos > 0 ? "持仓观察" : "空仓观望";
};

const getRecAction = (recommendation = {}) =>
  normalizeAction(recommendation.action || recommendation.decision || "HOLD");

const getRiskAction = (risk = {}, recommendation = {}) =>
  normalizeAction(risk.final_action || risk.action || risk.decision || recommendation.action || "HOLD");

const getExecutionAction = (recommendation = {}, risk = {}) =>
  normalizeAction(recommendation.execution_action || risk.execution_action || risk.final_action || risk.action || recommendation.action || "HOLD");

const normalizeReasonToken = (text = "") =>
  cleanText(text)
    .toLowerCase()
    .replace(/[\s，。、“”"':：;；,.!?！？()（）\[\]\-_/|]+/g, "");

const splitReasonParts = (text = "") =>
  cleanText(text)
    .split(/[|；;\n]+/)
    .map((x) => cleanText(x).replace(/^风控补充[:：]\s*/i, ""))
    .filter(Boolean);

const uniqueReasonText = (...texts) => {
  const seen = [];
  const out = [];
  texts.forEach((text) => {
    splitReasonParts(text).forEach((part) => {
      const key = normalizeReasonToken(part);
      if (!key) return;
      const duplicated = seen.some((prev) => {
        if (prev === key) return true;
        if (key.length >= 16 && prev.length >= 16) {
          return prev.includes(key) || key.includes(prev);
        }
        return false;
      });
      if (duplicated) return;
      seen.push(key);
      out.push(part);
    });
  });
  return out.join("；");
};

const buildRoundNote = (judge = {}, sideA = {}, sideB = {}) => {
  const fusion = String(judge.fusion_strategy || "").trim();
  const layers = judge.consensus_layers || {};
  const layerBits = [
    layers.argument_clash,
    layers.constraint_binding,
    layers.weight_balance,
  ]
    .map((x) => cleanText(x))
    .filter(Boolean);
  if (fusion || layerBits.length) {
    const head = fusion ? `【${fusion}】` : "【合作博弈共识】";
    const body = layerBits.length ? layerBits.join("；") : "";
    return toBriefReason(`${head}${body}`, "", 110);
  }

  const judgeReason = uniqueReasonText(judge.reason || judge.reasoning || judge.commentary || judge.note || "");
  const judgeBrief = toBriefReason(judgeReason, "", 96);
  if (judgeBrief && judgeBrief !== "-") return judgeBrief;

  const aAction = actionFromSignal(sideA.sentiment || sideA.decision);
  const bAction = actionFromSignal(sideB.sentiment || sideB.decision);
  const finalAction = actionFromSignal(judge.sentiment || judge.decision);
  return `A侧倾向 ${aAction}，B侧倾向 ${bAction}，裁判最终给出 ${finalAction}。`;
};

const buildDecisionConsistencyNote = (recommendation = {}, risk = {}) => {
  const recAction = getRecAction(recommendation);
  const riskAction = getRiskAction(risk, recommendation);
  const execAction = getExecutionAction(recommendation, risk);
  const pos = recommendation.position_percent ?? risk.position_percent;
  if (recAction !== riskAction) {
    return `策略结论为 ${recAction}，但风控约束下执行层调整为 ${riskAction}，当前按 ${execAction} 执行，建议仓位 ${fmtPercent(pos)}。`;
  }
  if (execAction !== recAction) {
    return `策略层与风控层方向一致（${recAction}），执行层细化为 ${execAction}，建议仓位 ${fmtPercent(pos)}。`;
  }
  return `策略层、风控层与执行层动作一致（${execAction}），建议仓位 ${fmtPercent(pos)}。`;
};

const taskStatusTag = (status = "") => {
  const s = String(status).toLowerCase();
  if (s === "done") return `<span class="tag tag-buy">DONE</span>`;
  if (s === "failed") return `<span class="tag tag-sell">FAILED</span>`;
  if (s === "running") return `<span class="tag tag-hold">RUNNING</span>`;
  return `<span class="tag tag-hold">${String(status || "QUEUED").toUpperCase()}</span>`;
};

const isTaskInProgress = (status = "") => {
  const s = String(status || "").toLowerCase();
  return s === "queued" || s === "running";
};

const parseTaskTime = (raw = "") => {
  const text = String(raw || "").trim();
  if (!text) return null;
  const matched = text.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$/);
  if (matched) {
    const [, y, m, d, hh, mm, ss] = matched;
    const parsed = new Date(
      Number(y),
      Number(m) - 1,
      Number(d),
      Number(hh),
      Number(mm),
      Number(ss)
    );
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  const parsed = new Date(text);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const isTaskStale = (task, staleMinutes = 15) => {
  const baseTime = parseTaskTime(task?.started_at || task?.created_at || "");
  if (!baseTime) return false;
  return Date.now() - baseTime.getTime() > staleMinutes * 60 * 1000;
};

function updateSettleAdviceButtonState() {
  const btn = document.getElementById("settleAdviceBtn");
  if (!btn) return;
  const busy = settleAdviceSubmitting || settleAdviceTaskActive;
  btn.disabled = busy;
  if (settleAdviceSubmitting) {
    btn.textContent = "提交中...";
    return;
  }
  btn.textContent = settleAdviceTaskActive ? "兑现任务执行中..." : "立即兑现一次";
}

function syncSettlementTaskPolling() {
  if (settleAdviceTaskActive) {
    if (settleTaskPollTimer) return;
    settleTaskPollTimer = window.setInterval(() => {
      if (document.hidden) return;
      refreshTasks().catch((e) => setStatus(`错误: ${e.message}`));
    }, 10000);
    return;
  }
  if (!settleTaskPollTimer) return;
  window.clearInterval(settleTaskPollTimer);
  settleTaskPollTimer = null;
}

const toLabel = (key = "") =>
  String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const escapeHtml = (value = "") =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const renderKvRows = (obj, keys) => {
  const rows = keys
    .map((k) => [k, obj ? obj[k] : undefined])
    .filter(([, v]) => v !== undefined && v !== null && typeof v !== "object");
  if (!rows.length) return `<div class="empty-tip">暂无结构化字段</div>`;
  return `<div class="kv-list">${rows
    .map(
      ([k, v]) =>
        `<div class="kv-row"><span class="kv-key">${toLabel(k)}</span><span class="kv-val">${String(v)}</span></div>`
    )
    .join("")}</div>`;
};

const renderKvRowsWithExpandableReason = (obj, keys, emptyReason = "暂无文字说明") => {
  const rows = keys
    .map((k) => [k, obj ? obj[k] : undefined])
    .filter(([, v]) => v !== undefined && v !== null && typeof v !== "object");
  if (!rows.length) return `<div class="empty-tip">暂无结构化字段</div>`;
  return `<div class="kv-list">${rows
    .map(([k, v]) => {
      if (k !== "reason" && k !== "risk_reason" && k !== "commentary") {
        return `<div class="kv-row"><span class="kv-key">${toLabel(k)}</span><span class="kv-val">${String(v)}</span></div>`;
      }
      const full = uniqueReasonText(String(v || ""));
      const summary = toBriefReason(full, emptyReason, 72);
      return `
        <div class="kv-row kv-row-reason">
          <span class="kv-key">${toLabel(k)}</span>
          <span class="kv-val">
            <details class="reason-details">
              <summary>${escapeHtml(summary)}</summary>
              <div class="reason-full">${escapeHtml(full || emptyReason)}</div>
            </details>
          </span>
        </div>
      `;
    })
    .join("")}</div>`;
};

const truncateText = (text, maxLen = 88) => {
  const s = String(text || "").trim();
  if (!s) return "-";
  return s.length > maxLen ? `${s.slice(0, maxLen)}...` : s;
};

const cleanText = (text) => String(text || "").trim();

const pickFirstSentence = (text) => {
  const s = cleanText(text);
  if (!s) return "";
  const segs = s.split(/(?<=[。！？!?])/);
  return cleanText(segs.find((x) => cleanText(x)) || s);
};

const toBriefReason = (text, fallback = "-", maxLen = 68) => {
  const base = pickFirstSentence(text) || cleanText(text) || fallback;
  return base.length > maxLen ? `${base.slice(0, maxLen)}...` : base;
};

const padTicker = (value) => String(value || "").trim().padStart(6, "0");
const toNumOrNull = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const fmtPercent = (v, digits = 2) => {
  const n = toNumOrNull(v);
  return n === null ? "-" : `${n.toFixed(digits)}%`;
};
const fmtNumber = (v, digits = 2) => {
  const n = toNumOrNull(v);
  return n === null ? "-" : n.toFixed(digits);
};

const resolveAdvicePayload = (payload) => {
  if (!payload || typeof payload !== "object") return {};
  if (payload.advice && typeof payload.advice === "object") return payload.advice;
  return payload;
};

function setAdviceFeedback(type, text) {
  const el = document.getElementById("adviceFeedback");
  if (!el) return;
  el.className = `inline-feedback ${type}`;
  el.textContent = text;
  el.classList.remove("hidden");
}

const adviceProgressFromElapsed = (elapsedMs) => {
  const ratio = Math.max(0, elapsedMs) / ADVICE_POLL_MAX_WAIT_MS;
  return Math.min(92, Math.round((1 - Math.exp(-3.2 * ratio)) * 92));
};

function setAdviceProgress(type, message, percent = null) {
  const el = document.getElementById("adviceFeedback");
  if (!el) return;
  const pct = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : null;
  const label = escapeHtml(String(message || "正在处理…"));
  const indeterminate = pct === null;
  const width = indeterminate ? "35%" : `${pct}%`;
  const pctLabel = indeterminate ? "进行中" : `${Math.round(pct)}%`;
  el.className = `inline-feedback advice-progress-feedback ${type}`;
  el.innerHTML = `
    <div class="advice-progress-head">
      <span class="advice-progress-label">${label}</span>
      <span class="advice-progress-pct">${pctLabel}</span>
    </div>
    <div class="advice-progress-track${indeterminate ? " is-indeterminate" : ""}">
      <div class="advice-progress-fill" style="width:${width}"></div>
    </div>
  `;
  el.classList.remove("hidden");
}

function setAdviceBusy(busy, runText = "生成建议", loadText = "读取最新建议") {
  const runBtn = document.getElementById("runAdviceBtn");
  const loadBtn = document.getElementById("loadLatestAdviceBtn");
  if (!runBtn || !loadBtn) return;
  runBtn.disabled = busy;
  loadBtn.disabled = busy;
  runBtn.classList.toggle("is-loading", busy);
  runBtn.textContent = busy ? runText : "生成建议";
  loadBtn.textContent = busy ? loadText : "读取最新建议";
}

function setOverviewFeedback(type, text) {
  const el = document.getElementById("overviewFeedback");
  if (!el) return;
  el.className = `inline-feedback ${type}`;
  el.textContent = text;
  el.classList.remove("hidden");
}

function setOverviewBusy(busy) {
  const btn = document.getElementById("refreshOverviewBtn");
  if (!btn) return;
  btn.disabled = busy;
  btn.classList.toggle("is-loading", busy);
  btn.textContent = busy ? "刷新中..." : "刷新总览";
}

function updateTrainSourceText() {
  const el = document.getElementById("trainSourceText");
  if (!el) return;
  if (!selectedTrainCsvPath) {
    el.textContent = DEFAULT_TRAIN_SOURCE_TEXT;
    return;
  }
  el.textContent = `当前文件：${selectedTrainCsvPath}`;
}

function updateEvolveSourceText(count = null) {
  const el = document.getElementById("evolveSourceText");
  if (!el) return;
  if (!selectedEvolveCsvPath) {
    el.textContent = "可手动输入逗号分隔代码，或导入 CSV（需含「股票代码」列）";
    return;
  }
  const suffix = count === null ? "" : `，共 ${count} 只股票`;
  el.textContent = `当前文件：${selectedEvolveCsvPath}${suffix}`;
}

function parseEvolutionTickersInput(raw = "") {
  return Array.from(
    new Set(
      String(raw || "")
        .split(/[,，;\s\n]+/)
        .map((x) => normalizeTicker(x))
        .filter(Boolean)
    )
  );
}

function buildEvolutionPayload() {
  const tickers = parseEvolutionTickersInput(document.getElementById("evolveTickers")?.value || "");
  const topNRaw = String(document.getElementById("evolveCsvTopN")?.value || "").trim();
  const csvTopN = topNRaw ? Number(topNRaw) : null;
  return {
    tickers: tickers.length ? tickers : null,
    csv_path: !tickers.length && selectedEvolveCsvPath ? selectedEvolveCsvPath : null,
    csv_top_n: csvTopN && Number.isFinite(csvTopN) && csvTopN > 0 ? csvTopN : null,
    debate_depth: Number(document.getElementById("evolveDepth")?.value || 2),
    mode: document.getElementById("evolveMode")?.value || "backtest_update",
  };
}

function renderSettlementStats(stats = {}) {
  const el = document.getElementById("settlementStatsText");
  if (!el) return;
  const settled = Number(stats.settled_count || 0);
  const pending = Number(stats.pending_count || 0);
  const updatedAt = String(stats.updated_at || "").trim();
  el.textContent = `已兑现：${settled} | 待兑现：${pending}${updatedAt ? ` | 更新时间：${updatedAt}` : ""}`;
}

function getUserScopedHistoryKey() {
  const userId = currentUser && currentUser.user_id ? String(currentUser.user_id) : "guest";
  return `${USER_ADVICE_HISTORY_KEY}:${userId}`;
}

function updateUserBadge() {
  const badge = document.getElementById("currentUserBadge");
  const registerBtn = document.getElementById("registerBtn");
  const loginBtn = document.getElementById("loginBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  if (!badge || !registerBtn || !loginBtn || !logoutBtn) return;
  if (currentUser && currentUser.user_id) {
    badge.classList.remove("hidden");
    badge.textContent = `用户：${currentUser.nickname || currentUser.email || currentUser.phone || "已注册"}`;
    registerBtn.textContent = "已注册";
    registerBtn.disabled = true;
    loginBtn.classList.add("hidden");
    logoutBtn.classList.remove("hidden");
    return;
  }
  badge.classList.add("hidden");
  badge.textContent = "未登录";
  registerBtn.textContent = "注册";
  registerBtn.disabled = false;
  loginBtn.classList.remove("hidden");
  logoutBtn.classList.add("hidden");
}

function loadCurrentUser() {
  try {
    const raw = window.localStorage.getItem(CURRENT_USER_KEY);
    if (!raw) {
      currentUser = null;
      return;
    }
    const parsed = JSON.parse(raw);
    currentUser = parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    currentUser = null;
  }
}

function loadAuthToken() {
  authToken = String(window.localStorage.getItem(AUTH_TOKEN_KEY) || "");
}

function saveCurrentUser(user) {
  currentUser = user && typeof user === "object" ? user : null;
  if (currentUser) {
    window.localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(currentUser));
  } else {
    window.localStorage.removeItem(CURRENT_USER_KEY);
  }
  updateUserBadge();
}

function saveAuthToken(token) {
  authToken = String(token || "");
  if (authToken) {
    window.localStorage.setItem(AUTH_TOKEN_KEY, authToken);
  } else {
    window.localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

function clearSessionState() {
  saveAuthToken("");
  saveCurrentUser(null);
}

function setAuthRequiredView(loggedIn) {
  const gate = document.getElementById("authGate");
  const appMain = document.getElementById("appMain");
  const appHeader = document.getElementById("appHeader");
  const modeSwitchBtn = document.getElementById("modeSwitchBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const loginBtn = document.getElementById("loginBtn");
  const registerBtn = document.getElementById("registerBtn");
  const isLoggedIn = !!(loggedIn && currentUser && currentUser.user_id);
  if (gate) gate.classList.toggle("hidden", isLoggedIn);
  if (appMain) appMain.classList.toggle("hidden", !isLoggedIn);
  if (appHeader) appHeader.classList.toggle("hidden", !isLoggedIn);
  if (modeSwitchBtn) modeSwitchBtn.classList.toggle("hidden", !isLoggedIn);
  if (!isLoggedIn) {
    currentMode = MODE_USER;
    if (logoutBtn) logoutBtn.classList.add("hidden");
    if (loginBtn) loginBtn.classList.remove("hidden");
    if (registerBtn) {
      registerBtn.disabled = false;
      registerBtn.textContent = "注册";
    }
  }
}

function openRegisterModal() {
  document.getElementById("registerEmailInput").value = "";
  document.getElementById("registerPhoneInput").value = "";
  document.getElementById("registerNicknameInput").value = "";
  document.getElementById("registerPasswordInput").value = "";
  document.getElementById("registerError").textContent = "";
  document.getElementById("registerModal").classList.remove("hidden");
}

function closeRegisterModal() {
  document.getElementById("registerModal").classList.add("hidden");
}

function openLoginModal() {
  document.getElementById("loginAccountInput").value = "";
  document.getElementById("loginPasswordInput").value = "";
  document.getElementById("loginError").textContent = "";
  document.getElementById("loginModal").classList.remove("hidden");
}

function closeLoginModal() {
  document.getElementById("loginModal").classList.add("hidden");
}

function normalizeTicker(value = "") {
  const digits = String(value || "").replace(/\D+/g, "");
  if (!digits) return "";
  return digits.slice(-6).padStart(6, "0");
}

function readUserAdviceHistoryTickers() {
  try {
    const raw = window.localStorage.getItem(getUserScopedHistoryKey()) || "[]";
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    const tickers = arr
      .map((x) => {
        const text = String(x || "");
        if (text.includes("__")) return normalizeTicker(text.split("__")[0] || "");
        return normalizeTicker(text);
      })
      .filter(Boolean);
    return Array.from(new Set(tickers)).slice(-500);
  } catch {
    return [];
  }
}

function writeUserAdviceHistoryTickers(tickers) {
  try {
    const normalized = Array.from(new Set((tickers || []).map((x) => normalizeTicker(x)).filter(Boolean))).slice(-500);
    window.localStorage.setItem(getUserScopedHistoryKey(), JSON.stringify(normalized));
  } catch {
    // ignore localStorage failures
  }
}

function getKnownUserAdviceTickers() {
  const local = readUserAdviceHistoryTickers();
  const remote = Array.isArray(userAdviceTickersCache) ? userAdviceTickersCache : [];
  return Array.from(new Set([...local, ...remote].map((x) => normalizeTicker(x)).filter(Boolean)));
}

async function loadUserAdviceHistoryFromServer() {
  if (!authToken || !currentUser || !currentUser.user_id) {
    userAdviceTickersCache = [];
    return;
  }
  try {
    const localTickers = readUserAdviceHistoryTickers();
    const data = await api("/api/user/advice-history");
    const tickers = Array.isArray(data.tickers) ? data.tickers.map((x) => normalizeTicker(x)).filter(Boolean) : [];
    userAdviceTickersCache = Array.from(new Set(tickers)).slice(-500);
    const merged = Array.from(new Set([...localTickers, ...userAdviceTickersCache]));
    writeUserAdviceHistoryTickers(merged);
    // 将旧浏览器缓存里的历史补传到服务端，完成跨设备迁移
    await syncUserAdviceHistoryToServer(localTickers);
  } catch {
    // 历史同步失败不应阻塞主流程
  }
}

async function syncUserAdviceHistoryToServer(tickers = []) {
  if (!authToken || !currentUser || !currentUser.user_id) return;
  const normalized = Array.from(new Set((tickers || []).map((x) => normalizeTicker(x)).filter(Boolean)));
  if (!normalized.length) return;
  try {
    const data = await api("/api/user/advice-history", {
      method: "POST",
      body: JSON.stringify({ tickers: normalized }),
    });
    const serverTickers = Array.isArray(data.tickers) ? data.tickers.map((x) => normalizeTicker(x)).filter(Boolean) : [];
    if (serverTickers.length) {
      userAdviceTickersCache = Array.from(new Set(serverTickers)).slice(-500);
      const merged = Array.from(new Set([...readUserAdviceHistoryTickers(), ...userAdviceTickersCache]));
      writeUserAdviceHistoryTickers(merged);
    }
  } catch {
    // 同步失败不打断用户操作
  }
}

function rememberUserAdvice(advice = {}) {
  const ticker = normalizeTicker(advice.ticker || "");
  if (!ticker) return;
  const merged = Array.from(new Set([...readUserAdviceHistoryTickers(), ticker])).slice(-500);
  writeUserAdviceHistoryTickers(merged);
  userAdviceTickersCache = Array.from(new Set([...(userAdviceTickersCache || []), ticker])).slice(-500);
  syncUserAdviceHistoryToServer([ticker]);
}

function buildOverviewSignature(data = {}) {
  const rowsAll = Array.isArray(data.latest_advice) ? data.latest_advice : [];
  const knownTickers = getKnownUserAdviceTickers();
  const rows =
    currentMode === MODE_USER
      ? rowsAll.filter((x) => knownTickers.includes(normalizeTicker(x.ticker || "")))
      : rowsAll;
  const top = rows[0] || {};
  const reflection = data.reflection_stats || {};
  return JSON.stringify({
    mode: currentMode,
    advice_total_count: data.advice_total_count ?? data.latest_advice_count ?? 0,
    latest_advice_count: data.latest_advice_count || 0,
    latest_ticker: top.ticker || "",
    latest_generated_at: top.generated_at || "",
    latest_action: top.action || "",
    advice_action_distribution: data.advice_action_distribution || {},
    reflection_rows: reflection.rows || 0,
    reward_avg: reflection.reward_avg ?? null,
    reward_neg_ratio: reflection.reward_neg_ratio ?? null,
  });
}

function buildRiskReason(risk, recommendation) {
  const direct = cleanText(risk.reason || risk.risk_reason || risk.commentary);
  if (direct) return uniqueReasonText(direct);
  const action = getRiskAction(risk, recommendation);
  const position = risk.position_percent ?? recommendation.position_percent;
  const riskLevel = risk.risk_level || risk.level || "normal";
  return `基于当前风险预算与交易约束，风控建议动作为 ${action}，建议仓位 ${position ?? "-"}%，风险等级评估为 ${riskLevel}。`;
}

function buildFinalReason(recommendation, referee, analysts, riskReason) {
  const raw = cleanText(recommendation.reason || recommendation.thesis || recommendation.rationale);
  const sentiment = cleanText(referee.sentiment || referee.decision || recommendation.sentiment || "neutral");
  const confidence = recommendation.confidence ?? referee.confidence;
  const technical = analysts.technical_flow || {};
  const fundamental = analysts.fundamental_news || {};
  const tBias = cleanText(technical.sentiment || "neutral");
  const fBias = cleanText(fundamental.sentiment || "neutral");
  const fallback = `综合技术面(${tBias})与基本面(${fBias})信号，裁判倾向为 ${sentiment}，形成当前交易建议（置信度 ${confidence ?? "-"}）。`;
  const merged = uniqueReasonText(raw || fallback, riskReason ? `风控补充：${riskReason}` : "");
  return merged || fallback;
}

function renderFlowChart(data) {
  const analysts = data.analyst_cases || {};
  const technical = analysts.technical_flow || {};
  const fundamental = analysts.fundamental_news || {};
  const referee = data.referee || {};
  const risk = data.risk || {};
  const recommendation = data.recommendation || {};
  const riskReason = buildRiskReason(risk, recommendation);
  const finalReason = buildFinalReason(recommendation, referee, analysts, riskReason);
  const decisionNote = buildDecisionConsistencyNote(recommendation, risk);
  const recAction = getRecAction(recommendation);
  const riskAction = getRiskAction(risk, recommendation);
  const execAction = getExecutionAction(recommendation, risk);
  const riskReasonBrief = toBriefReason(
    riskReason,
    "风控侧确认动作与仓位满足当前风险约束。",
    62
  );

  const trace = referee.debate_trace || [];
  const rounds = trace.length
    ? trace
        .map((x) => {
          const j = x.judge || {};
          const a = actionFromSignal(j.sentiment || j.decision);
          const aSide = actionFromSignal((x.case_a || {}).sentiment || (x.case_a || {}).decision);
          const bSide = actionFromSignal((x.case_b || {}).sentiment || (x.case_b || {}).decision);
          return `<span class="round-chip">R${x.round}: ${a} / A:${aSide} B:${bSide} / conf ${j.confidence ?? "-"}</span>`;
        })
        .join("")
    : `<span class="round-chip">无多轮博弈（直接一致）</span>`;

  document.getElementById("traceFlow").innerHTML = `
    <div class="flow-wrapper">
      <div class="flow-row">
        <div class="flow-node">
          <div class="node-title">技术分析师</div>
          <div class="node-meta">${actionTag(actionFromSignal(technical.sentiment || technical.decision))} | 置信度 ${Number(technical.confidence || 0).toFixed(2)}</div>
          <div class="node-text">${truncateText(technical.reasoning, 96)}</div>
        </div>
        <div class="flow-node">
          <div class="node-title">基本面分析师</div>
          <div class="node-meta">${actionTag(actionFromSignal(fundamental.sentiment || fundamental.decision))} | 置信度 ${Number(fundamental.confidence || 0).toFixed(2)}</div>
          <div class="node-text">${truncateText(fundamental.reasoning, 96)}</div>
        </div>
      </div>
      <div class="flow-row">
        <div class="flow-arrow">↓</div>
      </div>
      <div class="flow-row decision-row">
        <div class="flow-node">
          <div class="node-title">裁判整合</div>
          <div class="node-meta">${actionTag(actionFromSignal(referee.sentiment || referee.decision))} | 置信度 ${referee.confidence ?? "-"}</div>
          <div class="node-text">${truncateText(referee.reason || referee.commentary || "融合双方观点并形成中间结论", 110)}</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
          <div class="node-title">风控决策</div>
          <div class="node-meta">${actionTag(riskAction)} | ${actionExplain(riskAction, risk.position_percent ?? recommendation.position_percent)} | 仓位 ${fmtPercent(risk.position_percent ?? recommendation.position_percent)}</div>
          <div class="node-text">${riskReasonBrief}</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
          <div class="node-title">最终建议</div>
          <div class="node-meta">${actionTag(recAction)} | ${actionExplain(recAction, recommendation.position_percent)} | 执行: ${execAction} | 仓位 ${fmtPercent(recommendation.position_percent)}</div>
          <div class="node-text node-text-long">${finalReason}</div>
        </div>
      </div>
      <div class="flow-row">
        <div class="flow-node" style="width:100%;">
          <div class="node-title">执行一致性说明</div>
          <div class="node-text">${decisionNote}</div>
        </div>
      </div>
      <div class="flow-rounds">
        <div class="flow-rounds-title">博弈轮次快照</div>
        <div class="round-chip-list">${rounds}</div>
      </div>
    </div>
  `;
}

const formatFetchError = (error, path = "") => {
  const raw = String((error && error.message) || error || "");
  if (error && error.name === "AbortError") {
    return `请求超时：${path || "API"}。建议生成耗时较长，请确认后端仍在运行后重试。`;
  }
  if (/failed to fetch|networkerror|load failed|network request failed/i.test(raw)) {
    return `无法连接后端 API（${API_BASE}）。请先执行：uvicorn backend.app:app --host 127.0.0.1 --port 8000，并打开 http://127.0.0.1:8000`;
  }
  return raw || "未知网络错误";
};

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

async function pollAdviceTask(taskId) {
  const startTime = Date.now();
  let consecutiveNetworkErrors = 0;
  while (Date.now() - startTime < ADVICE_POLL_MAX_WAIT_MS) {
    await sleep(3000);
    try {
      const task = await api(`/api/tasks/${taskId}`, { timeout_ms: ADVICE_API_TIMEOUT_MS });
      consecutiveNetworkErrors = 0;
      if (task.status === "done" || task.status === "failed" || task.status === "skipped") {
        return task;
      }
      const elapsedMs = Date.now() - startTime;
      const elapsedSec = Math.round(elapsedMs / 1000);
      const pct = adviceProgressFromElapsed(elapsedMs);
      setAdviceProgress("info", `建议生成中（约 ${elapsedSec}s）`, pct);
    } catch (e) {
      if (/failed to fetch|network|超时|timeout/i.test(String(e.message || ""))) {
        consecutiveNetworkErrors += 1;
        if (consecutiveNetworkErrors <= 10) {
          setAdviceProgress(
            "info",
            `后端短暂不可达，正在重试（${consecutiveNetworkErrors}/10）`,
            null
          );
          continue;
        }
      }
      throw e;
    }
  }
  return null;
}

async function api(path, options = {}) {
  const timeoutMs =
    Number.isFinite(Number(options.timeout_ms)) && Number(options.timeout_ms) > 0
      ? Number(options.timeout_ms)
      : API_TIMEOUT_MS;
  const hasExternalSignal = !!options.signal;
  const controller =
    !hasExternalSignal && typeof AbortController !== "undefined" ? new AbortController() : null;
  let timeoutId = null;
  if (controller && timeoutMs > 0) {
    timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  }

  const hasFormDataBody = typeof FormData !== "undefined" && options.body instanceof FormData;
  const mergedHeaders = {
    ...(hasFormDataBody ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {}),
  };
  if (authToken) {
    mergedHeaders.Authorization = `Bearer ${authToken}`;
  }
  const fetchOptions = {
    ...options,
    headers: mergedHeaders,
  };
  delete fetchOptions.timeout_ms;
  if (controller) fetchOptions.signal = controller.signal;

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, fetchOptions);
  } catch (e) {
    throw new Error(formatFetchError(e, path));
  } finally {
    if (timeoutId !== null) window.clearTimeout(timeoutId);
  }

  if (!res.ok) {
    if (res.status === 401) {
      clearSessionState();
      setAuthRequiredView(false);
    }
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function activateTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
}

function openAdminModal(mode = MODE_ADMIN) {
  pendingMode = mode;
  document.getElementById("adminPasswordInput").value = "";
  document.getElementById("adminError").textContent = "";
  document.getElementById("adminModalText").textContent = "切换到管理者模式需要管理员密码。";
  document.getElementById("adminModal").classList.remove("hidden");
}

function closeAdminModal() {
  document.getElementById("adminModal").classList.add("hidden");
  pendingMode = null;
}

function setModeSwitchText() {
  const btn = document.getElementById("modeSwitchBtn");
  if (!btn) return;
  btn.textContent = currentMode === MODE_ADMIN ? "切换到用户模式" : "切换到管理者模式";
}

function setOverviewLabelsByMode() {
  const primary = document.getElementById("metricTitlePrimary");
  const secondary = document.getElementById("metricTitleSecondary");
  const third = document.getElementById("metricTitleThird");
  const fourth = document.getElementById("metricTitleFourth");
  const secondaryCardTitle = document.getElementById("overviewSecondaryCardTitle");
  if (currentMode === MODE_ADMIN) {
    if (primary) primary.textContent = "建议总量";
    if (secondary) secondary.textContent = "经验样本数";
    if (third) third.textContent = "Reward均值";
    if (fourth) fourth.textContent = "负Reward占比";
    if (secondaryCardTitle) secondaryCardTitle.textContent = "经验统计";
    return;
  }
  if (primary) primary.textContent = "我的建议数";
  if (secondary) secondary.textContent = "已查询标的数";
  if (third) third.textContent = "平均建议仓位";
  if (fourth) fourth.textContent = "平均置信度";
  if (secondaryCardTitle) secondaryCardTitle.textContent = "股票板块分布";
}

function applyMode(nextMode, preferredTab = "") {
  currentMode = nextMode === MODE_ADMIN ? MODE_ADMIN : MODE_USER;

  const userSidebar = document.getElementById("sidebarUser");
  const adminSidebar = document.getElementById("sidebarAdmin");
  if (userSidebar) userSidebar.classList.toggle("hidden", currentMode !== MODE_USER);
  if (adminSidebar) adminSidebar.classList.toggle("hidden", currentMode !== MODE_ADMIN);
  const overviewTitle = document.getElementById("overviewTitle");
  if (overviewTitle) {
    overviewTitle.textContent = currentMode === MODE_ADMIN ? "系统总览" : "历史投资建议总览";
  }
  const adminUsersCard = document.getElementById("adminUsersCard");
  if (adminUsersCard) adminUsersCard.classList.toggle("hidden", currentMode !== MODE_ADMIN);
  const userAdviceEvalCard = document.getElementById("userAdviceEvalCard");
  if (userAdviceEvalCard) userAdviceEvalCard.classList.toggle("hidden", currentMode !== MODE_USER);
  setOverviewLabelsByMode();
  setModeSwitchText();

  const allowedTabs = currentMode === MODE_ADMIN ? ["training", "evolution", "overview"] : ["advice", "overview"];
  const activePanel = document.querySelector(".tab-panel.active");
  const activeTabId = activePanel ? activePanel.id : "";
  const targetTab = allowedTabs.includes(preferredTab)
    ? preferredTab
    : allowedTabs.includes(activeTabId)
      ? activeTabId
      : allowedTabs[0];
  activateTab(targetTab);
}

async function refreshAdminUsers() {
  if (currentMode !== MODE_ADMIN) return;
  const tableHost = document.getElementById("adminUsersTable");
  const countHost = document.getElementById("usersCountText");
  if (!tableHost || !countHost) return;
  try {
    const adminPassword = window.localStorage.getItem("adminPassword") || ADMIN_PASSWORD;
    const data = await api("/api/admin/users", {
      method: "GET",
      headers: {
        "X-Admin-Password": adminPassword,
      },
    });
    const users = Array.isArray(data.users) ? data.users : [];
    countHost.textContent = `用户数：${data.count ?? users.length}`;
    if (!users.length) {
      tableHost.innerHTML = `<div class="empty-tip">暂无注册用户</div>`;
      return;
    }
    tableHost.innerHTML = `
      <table>
        <thead><tr><th>昵称</th><th>邮箱</th><th>手机号</th><th>注册时间</th></tr></thead>
        <tbody>
          ${users
            .map(
              (u) =>
                `<tr><td>${u.nickname || "-"}</td><td>${u.email || "-"}</td><td>${u.phone || "-"}</td><td>${u.created_at || "-"}</td></tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
  } catch (e) {
    tableHost.innerHTML = `<div class="empty-tip">用户列表加载失败：${e.message}</div>`;
  }
}

function renderUserAdviceEvaluation(evaluation = {}) {
  const host = document.getElementById("userAdviceEval");
  if (!host) return;
  const tracked = Number(evaluation.tracked_tickers || 0);
  const settled = Number(evaluation.settled_samples || 0);
  const directional = Number(evaluation.directional_samples || 0);
  const winRate = evaluation.directional_win_rate;
  const positiveRatio = evaluation.positive_ratio;
  const avgPnl = evaluation.avg_pnl_percent;
  const avgReward = evaluation.avg_reward;
  const lastSettledAt = String(evaluation.last_settled_at || "").trim();

  if (!tracked) {
    host.innerHTML = `<div class="empty-tip">暂无可评估标的，请先生成建议。</div>`;
    return;
  }
  if (!settled) {
    host.innerHTML = `<div class="empty-tip">已跟踪 ${tracked} 个标的，暂无已兑现样本（T+1结算后会自动出现）。</div>`;
    return;
  }

  host.innerHTML = `
    <div class="stats-grid">
      <div class="stat-box"><div class="label">跟踪标的数</div><div class="value">${tracked}</div></div>
      <div class="stat-box"><div class="label">已兑现样本</div><div class="value">${settled}</div></div>
      <div class="stat-box"><div class="label">方向命中率</div><div class="value">${winRate === null || winRate === undefined ? "-" : `${(Number(winRate) * 100).toFixed(1)}%`}</div></div>
      <div class="stat-box"><div class="label">正收益占比</div><div class="value">${positiveRatio === null || positiveRatio === undefined ? "-" : `${(Number(positiveRatio) * 100).toFixed(1)}%`}</div></div>
      <div class="stat-box"><div class="label">平均收益(%)</div><div class="value">${avgPnl === null || avgPnl === undefined ? "-" : Number(avgPnl).toFixed(2)}</div></div>
      <div class="stat-box"><div class="label">平均Reward</div><div class="value">${avgReward === null || avgReward === undefined ? "-" : Number(avgReward).toFixed(3)}</div></div>
      <div class="stat-box"><div class="label">方向样本数</div><div class="value">${directional}</div></div>
      <div class="stat-box"><div class="label">最近结算时间</div><div class="value">${lastSettledAt || "-"}</div></div>
    </div>
  `;
}

async function refreshUserAdviceEvaluation() {
  const host = document.getElementById("userAdviceEval");
  if (!host) return;
  if (currentMode !== MODE_USER) {
    host.innerHTML = "-";
    return;
  }
  host.innerHTML = `<div class="empty-tip">正在评估历史建议表现...</div>`;
  try {
    const data = await api("/api/user/advice-evaluation");
    renderUserAdviceEvaluation(data.evaluation || {});
  } catch (e) {
    host.innerHTML = `<div class="empty-tip">评估加载失败：${e.message}</div>`;
  }
}

function verifyAdmin() {
  const input = document.getElementById("adminPasswordInput").value || "";
  if (input === ADMIN_PASSWORD) {
    if (pendingMode === MODE_ADMIN) {
      applyMode(MODE_ADMIN, "training");
      refreshOverview().catch((e) => setStatus(`错误: ${e.message}`));
    }
    closeAdminModal();
  } else {
    document.getElementById("adminError").textContent = "密码错误，请重试。";
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tabId = btn.dataset.tab;
    const allowedTabs = currentMode === MODE_ADMIN ? ["training", "evolution", "overview"] : ["advice", "overview"];
    if (!allowedTabs.includes(tabId)) {
      return;
    }
    activateTab(tabId);
  });
});

function renderAdviceDistribution(adviceDist) {
  const host = document.getElementById("overviewAdviceDistCards");
  const keys = ["BUY", "HOLD", "SELL"];
  const total = keys.reduce((s, k) => s + Number(adviceDist[k] || 0), 0) || 1;
  const colorMap = {
    BUY: "linear-gradient(90deg,#10b981,#34d399)",
    HOLD: "linear-gradient(90deg,#f59e0b,#fbbf24)",
    SELL: "linear-gradient(90deg,#ef4444,#f87171)",
  };
  host.innerHTML = keys
    .map((k) => {
      const count = Number(adviceDist[k] || 0);
      const pct = (count / total) * 100;
      return `
        <div class="dist-row">
          <div class="dist-row-head">
            <span>${actionTag(k)} ${k}</span>
            <span>${count} (${pct.toFixed(1)}%)</span>
          </div>
          <div class="dist-bar">
            <div class="dist-bar-fill" style="width:${pct.toFixed(1)}%;background:${colorMap[k]};"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderExperienceStats(stats) {
  const host = document.getElementById("overviewReflectionStatsCards");
  const actionDist = stats.action_distribution || {};
  host.innerHTML = `
    <div class="stats-grid">
      <div class="stat-box"><div class="label">样本总量</div><div class="value">${stats.rows ?? "-"}</div></div>
      <div class="stat-box"><div class="label">Reward均值</div><div class="value">${stats.reward_avg ?? "-"}</div></div>
      <div class="stat-box"><div class="label">负Reward占比</div><div class="value">${stats.reward_neg_ratio !== undefined ? (Number(stats.reward_neg_ratio) * 100).toFixed(1) + "%" : "-"}</div></div>
      <div class="stat-box"><div class="label">动作分布</div><div class="value">B:${actionDist.BUY || 0} H:${actionDist.HOLD || 0} S:${actionDist.SELL || 0}</div></div>
    </div>
  `;
}

function renderUserSupplementStats(rows) {
  const host = document.getElementById("overviewReflectionStatsCards");
  const list = Array.isArray(rows) ? rows : [];
  if (!list.length) {
    host.innerHTML = `<div class="empty-tip">暂无你主动查询/生成的建议记录</div>`;
    return;
  }

  const boardOfTicker = (ticker = "") => {
    const t = padTicker(ticker);
    if (/^(688|689)/.test(t)) return "科创板";
    if (/^(300|301)/.test(t)) return "创业板";
    if (/^(8|4)/.test(t)) return "北交所";
    if (/^(600|601|603|605)/.test(t)) return "沪市主板";
    if (/^(000|001|002|003)/.test(t)) return "深市主板";
    return "其他";
  };

  const boardCounter = {};
  list.forEach((row) => {
    const board = boardOfTicker(row.ticker || "");
    boardCounter[board] = Number(boardCounter[board] || 0) + 1;
  });
  const entries = Object.entries(boardCounter).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, c]) => s + Number(c || 0), 0) || 1;

  const colorMap = {
    科创板: "#3b82f6",
    创业板: "#10b981",
    北交所: "#f59e0b",
    沪市主板: "#6366f1",
    深市主板: "#ec4899",
    其他: "#94a3b8",
  };

  let cursor = 0;
  const gradientParts = entries.map(([board, count]) => {
    const start = cursor;
    const pct = (Number(count || 0) / total) * 100;
    cursor += pct;
    const color = colorMap[board] || "#94a3b8";
    return `${color} ${start.toFixed(2)}% ${cursor.toFixed(2)}%`;
  });
  const gradient = gradientParts.join(", ");

  host.innerHTML = `
    <div class="board-pie-wrap">
      <div class="board-pie" style="background: conic-gradient(${gradient});"></div>
      <div class="board-legend">
        ${entries
          .map(([board, count]) => {
            const pct = ((Number(count || 0) / total) * 100).toFixed(1);
            const color = colorMap[board] || "#94a3b8";
            return `<div class="board-legend-item"><span class="board-dot" style="background:${color};"></span><span>${board}</span><span>${count} (${pct}%)</span></div>`;
          })
          .join("")}
      </div>
    </div>
  `;
}

async function refreshOverview() {
  setStatus("刷新总览中...");
  setOverviewBusy(true);
  setOverviewFeedback("info", "正在拉取总览数据...");
  try {
    const data = await api("/api/dashboard/summary?refresh=true");
    const allRows = Array.isArray(data.latest_advice) ? data.latest_advice : [];
    const knownTickers = getKnownUserAdviceTickers();
    const userRows = allRows.filter((x) => knownTickers.includes(normalizeTicker(x.ticker || "")));
    const rows = currentMode === MODE_USER ? userRows : allRows;
    const adviceDist =
      currentMode === MODE_USER
        ? rows.reduce(
            (acc, r) => {
              const a = normalizeAction(r.action || "HOLD");
              acc[a] = Number(acc[a] || 0) + 1;
              return acc;
            },
            { BUY: 0, HOLD: 0, SELL: 0 }
          )
        : (data.advice_action_distribution || {});
    const reflectionStats = data.reflection_stats || {};
    const settlementStats = data.advice_settlement_stats || {};
    renderAdviceDistribution(adviceDist);
    renderSettlementStats(settlementStats);
    if (currentMode === MODE_ADMIN) {
      renderExperienceStats(reflectionStats);
      await refreshAdminUsers();
    } else {
      renderUserSupplementStats(rows);
      await refreshUserAdviceEvaluation();
    }
    document.getElementById("metricAdviceCount").textContent = String(
      currentMode === MODE_USER ? rows.length : (data.advice_total_count ?? data.latest_advice_count ?? 0)
    );
    if (currentMode === MODE_ADMIN) {
      document.getElementById("metricReflectionRows").textContent = String(reflectionStats.rows || 0);
      document.getElementById("metricRewardAvg").textContent = String(reflectionStats.reward_avg ?? "-");
      const neg = reflectionStats.reward_neg_ratio;
      document.getElementById("metricNegRatio").textContent = neg === undefined ? "-" : `${(Number(neg) * 100).toFixed(1)}%`;
    } else {
      const uniqueTickers = new Set(rows.map((x) => padTicker(x.ticker || ""))).size;
      const avgPosition = rows.length
        ? rows.reduce((s, x) => s + Number(x.position_percent || 0), 0) / rows.length
        : null;
      const avgConfidence = rows.length
        ? rows.reduce((s, x) => s + Number(x.confidence || 0), 0) / rows.length
        : null;
      document.getElementById("metricReflectionRows").textContent = String(uniqueTickers);
      document.getElementById("metricRewardAvg").textContent = avgPosition === null ? "-" : `${avgPosition.toFixed(2)}%`;
      document.getElementById("metricNegRatio").textContent = avgConfidence === null ? "-" : avgConfidence.toFixed(2);
    }
    const tableHtml = `
      <table>
        <thead><tr><th>Ticker</th><th>Action</th><th>Position%</th><th>Confidence</th><th>Time</th></tr></thead>
        <tbody>
          ${rows
            .map(
              (r) =>
                `<tr><td>${r.ticker}</td><td>${actionTag(r.action)}</td><td>${Number(r.position_percent || 0).toFixed(2)}</td><td>${Number(r.confidence || 0).toFixed(2)}</td><td>${r.generated_at || ""}</td></tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
    document.getElementById("overviewAdviceTable").innerHTML =
      rows.length > 0
        ? tableHtml
        : `<div class="empty-tip">${currentMode === MODE_USER ? "暂无你主动查询/生成的建议记录" : "暂无建议记录"}</div>`;

    const updatedAt = data.updated_at || new Date().toLocaleString("zh-CN", { hour12: false });
    const updatedAtEl = document.getElementById("overviewUpdatedAt");
    if (updatedAtEl) updatedAtEl.textContent = `最后刷新：${updatedAt}`;
    const signature = buildOverviewSignature(data);
    const unchanged = lastOverviewSignature && lastOverviewSignature === signature;
    lastOverviewSignature = signature;
    setOverviewFeedback(
      unchanged ? "info" : "success",
      unchanged
        ? `总览已刷新（${updatedAt}），但暂无新增数据。${currentMode === MODE_USER ? "（仅展示你主动查询/生成的建议）" : ""}`
        : `总览刷新成功（${updatedAt}）${currentMode === MODE_USER ? "（仅展示你主动查询/生成的建议）" : ""}`
    );
    setStatus("总览已更新");
  } catch (e) {
    setOverviewFeedback("error", `总览刷新失败：${e.message}`);
    setStatus(`刷新总览失败: ${e.message}`);
    throw e;
  } finally {
    setOverviewBusy(false);
  }
}

async function refreshTasks() {
  setStatus("拉取任务列表...");
  const data = await api("/api/tasks?limit=50");
  const tasks = data.tasks || [];
  settleAdviceTaskActive = tasks.some(
    (t) => t.type === "advice_settlement_manual" && isTaskInProgress(t.status) && !isTaskStale(t, 15)
  );
  updateSettleAdviceButtonState();
  syncSettlementTaskPolling();
  const html = tasks
    .map(
      (t) =>
        `<div class="timeline-item"><b>${t.type}</b> ${taskStatusTag(t.status)}<br/>id=${t.task_id}<br/>${t.created_at} ~ ${t.ended_at || "-"}</div>`
    )
    .join("");
  document.getElementById("taskList").innerHTML = html || "-";
  setStatus("任务列表已更新");
}

function drawRadar(canvas, labels, values) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) * 0.34;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(100,116,139,0.45)";
  ctx.lineWidth = 1;
  for (let level = 1; level <= 4; level++) {
    const rr = (r * level) / 4;
    ctx.beginPath();
    for (let i = 0; i < labels.length; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / labels.length;
      const x = cx + Math.cos(a) * rr;
      const y = cy + Math.sin(a) * rr;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(37,99,235,0.85)";
  ctx.fillStyle = "rgba(37,99,235,0.2)";
  ctx.beginPath();
  for (let i = 0; i < labels.length; i++) {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / labels.length;
    const v = Math.max(0, Math.min(1, Number(values[i] || 0)));
    const x = cx + Math.cos(a) * r * v;
    const y = cy + Math.sin(a) * r * v;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#334155";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  labels.forEach((label, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / labels.length;
    const x = cx + Math.cos(a) * (r + 20);
    const y = cy + Math.sin(a) * (r + 20);
    ctx.fillText(label, x - 24, y + 4);
  });
}

function buildMiniCandles(rows) {
  const list = Array.isArray(rows) ? rows : [];
  if (!list.length) return [];
  const hasOhlc = list.some((x) => Number.isFinite(Number(x.open ?? x.high ?? x.low ?? x.close)));
  if (hasOhlc) {
    const out = list
      .map((x, idx) => {
        const close = Number(x.close ?? x.c ?? x.price_close);
        const open = Number(x.open ?? x.o ?? x.price_open ?? close);
        const high = Number(x.high ?? x.h ?? x.price_high ?? Math.max(open, close));
        const low = Number(x.low ?? x.l ?? x.price_low ?? Math.min(open, close));
        if (![open, high, low, close].every((n) => Number.isFinite(n))) return null;
        return { open, high, low, close, label: x.date || x.next_date || String(idx + 1) };
      })
      .filter(Boolean);
    if (out.length >= 2) return out;
  }
  let prev = 100;
  return list.map((x, idx) => {
    const realRet = Number(x.real_pnl ?? x.effective_pnl ?? 0) / 100;
    const effRet = Number(x.effective_pnl ?? 0) / 100;
    const open = prev;
    const close = Math.max(1, open * (1 + realRet));
    const spanBase = Math.abs(close - open);
    const wick = Math.max(spanBase * 0.4, open * Math.min(0.035, Math.abs(effRet) * 0.45 + 0.004));
    const high = Math.max(open, close) + wick;
    const low = Math.max(0.1, Math.min(open, close) - wick);
    prev = close;
    return { open, high, low, close, label: x.date || x.next_date || String(idx + 1) };
  });
}

function drawMiniKline(svg, candles) {
  if (!svg) return;
  const width = 360;
  const height = 220;
  const data = Array.isArray(candles) ? candles : [];
  if (data.length < 2) {
    svg.innerHTML = `<text x="18" y="30" fill="#64748b" font-size="12">暂无足够K线数据</text>`;
    return;
  }
  const min = Math.min(...data.map((x) => x.low));
  const max = Math.max(...data.map((x) => x.high));
  const range = max - min || 1;
  const left = 12;
  const right = width - 12;
  const top = 12;
  const bottom = height - 20;
  const plotH = bottom - top;
  const plotW = right - left;
  const step = plotW / data.length;
  const candleW = Math.max(2, step * 0.55);
  const y = (v) => bottom - ((v - min) / range) * plotH;
  const wicks = data
    .map((c, i) => {
      const cx = left + i * step + step / 2;
      const color = c.close >= c.open ? "#10b981" : "#ef4444";
      return `<line x1="${cx}" y1="${y(c.high)}" x2="${cx}" y2="${y(c.low)}" stroke="${color}" stroke-width="1.2"/>`;
    })
    .join("");
  const bodies = data
    .map((c, i) => {
      const cx = left + i * step + step / 2;
      const yOpen = y(c.open);
      const yClose = y(c.close);
      const bodyTop = Math.min(yOpen, yClose);
      const bodyH = Math.max(1.4, Math.abs(yOpen - yClose));
      const color = c.close >= c.open ? "#10b981" : "#ef4444";
      const fill = c.close >= c.open ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)";
      return `<rect x="${cx - candleW / 2}" y="${bodyTop}" width="${candleW}" height="${bodyH}" fill="${fill}" stroke="${color}" stroke-width="1.2" rx="1"/>`;
    })
    .join("");
  const last = data[data.length - 1];
  svg.innerHTML = `
    <line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="rgba(100,116,139,0.25)"/>
    <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="rgba(100,116,139,0.25)"/>
    ${wicks}
    ${bodies}
    <text x="14" y="16" fill="#64748b" font-size="12">L ${min.toFixed(2)}</text>
    <text x="${width - 88}" y="16" fill="#64748b" font-size="12">H ${max.toFixed(2)}</text>
    <text x="${width - 160}" y="${height - 4}" fill="#64748b" font-size="11">最后: O ${last.open.toFixed(2)} / C ${last.close.toFixed(2)}</text>
  `;
}

function renderStabilityPanel(data) {
  const host = document.getElementById("stabilityPanel");
  if (!host) return;
  if (!data || typeof data !== "object") {
    host.textContent = "稳定度：-";
    return;
  }
  const scoreRaw = Number(data.stability_score);
  const score = Number.isFinite(scoreRaw) ? scoreRaw : null;
  const level = String(data.stability_level || "").toLowerCase();
  const note = String(data.stability_note || "暂无稳定度说明");
  const diagnostics = data.stability_diagnostics || {};
  const levelText = level === "high" ? "高" : level === "medium" ? "中" : level === "low" ? "低" : "-";
  const levelClass =
    level === "high"
      ? "stability-level-high"
      : level === "medium"
        ? "stability-level-medium"
        : level === "low"
          ? "stability-level-low"
          : "";
  const parseFails = diagnostics.parse_fail_count ?? "-";
  const fallbackCount = diagnostics.rule_fallback_count ?? "-";
  const emptyCount = diagnostics.empty_reason_count ?? "-";
  const parseFailures = Array.isArray(diagnostics.parse_failures) ? diagnostics.parse_failures : [];
  const parseFailDetail = parseFailures.length
    ? `<div class="stability-fail-list">${parseFailures
        .map((x) => `<span>${escapeHtml(x.agent || x.path || "unknown")}</span>`)
        .join("")}</div>`
    : "";

  host.innerHTML = `
    <div class="stability-head">
      <span class="stability-title">建议稳定度</span>
      <span class="stability-score">${score === null ? "-" : score.toFixed(2)}</span>
    </div>
    <div class="stability-head">
      <span class="stability-title">等级</span>
      <span class="stability-level ${levelClass}">${levelText}</span>
    </div>
    <div class="stability-note">${note}</div>
    <div class="stability-note">解析失败:${parseFails} | 规则降级:${fallbackCount} | 空理由:${emptyCount}</div>
    ${parseFailDetail}
  `;
}

function qualityLevelText(level = "") {
  const value = String(level || "").toLowerCase();
  if (value === "high") return "高";
  if (value === "medium") return "中";
  if (value === "low") return "低";
  return "-";
}

function renderScoreBar(score, className = "") {
  const n = Number(score);
  const safe = Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;
  return `
    <div class="quality-score-track ${className}">
      <div class="quality-score-fill" style="width:${Math.round(safe * 100)}%"></div>
    </div>
  `;
}

function renderDataQualityPanel(data) {
  const host = document.getElementById("dataQualityPanel");
  if (!host) return;
  const quality = data && typeof data === "object" ? data.data_quality || {} : {};
  if (!quality || !Object.keys(quality).length) {
    host.textContent = "数据质量：-";
    return;
  }
  const score = Number(quality.score);
  const level = String(quality.level || "").toLowerCase();
  const market = quality.market || {};
  const rag = quality.rag || {};
  const analyst = quality.analyst || {};
  const components = quality.components || {};
  const diagnostics = Array.isArray(quality.diagnostics) ? quality.diagnostics : [];
  const sourceTypes = rag.source_types && typeof rag.source_types === "object" ? rag.source_types : {};
  const sourceTypeText = Object.entries(sourceTypes)
    .map(([k, v]) => `${k}:${v}`)
    .join(" / ") || "-";
  const levelClass =
    level === "high"
      ? "quality-level-high"
      : level === "medium"
        ? "quality-level-medium"
        : level === "low"
          ? "quality-level-low"
          : "";

  host.innerHTML = `
    <div class="quality-head">
      <span class="quality-title">综合质量</span>
      <span class="quality-score">${Number.isFinite(score) ? score.toFixed(2) : "-"}</span>
    </div>
    ${renderScoreBar(score)}
    <div class="quality-head">
      <span class="quality-title">等级</span>
      <span class="quality-level ${levelClass}">${qualityLevelText(level)}</span>
    </div>
    <div class="quality-note">${escapeHtml(quality.note || "暂无质量说明")}</div>
    <div class="quality-mini-grid">
      <div><span>行情</span><b>${escapeHtml(market.source || "-")}</b></div>
      <div><span>样本</span><b>${market.rows ?? "-"}</b></div>
      <div><span>日期</span><b>${escapeHtml(market.as_of_date || "-")}</b></div>
      <div><span>RAG</span><b>${rag.documents ?? "-"}</b></div>
      <div><span>来源</span><b>${escapeHtml(sourceTypeText)}</b></div>
      <div><span>解析失败</span><b>${analyst.parse_fail_count ?? "-"}</b></div>
    </div>
    <div class="quality-components">
      <div><span>行情</span>${renderScoreBar(components.market_score, "quality-bar-compact")}</div>
      <div><span>研报</span>${renderScoreBar(components.rag_score, "quality-bar-compact")}</div>
      <div><span>分析</span>${renderScoreBar(components.analyst_score, "quality-bar-compact")}</div>
    </div>
    <div class="quality-chips">
      ${diagnostics.map((x) => `<span>${escapeHtml(x)}</span>`).join("")}
    </div>
  `;
}

function clearAdvicePanelsForGenerating() {
  setAdviceProgress("info", "正在生成建议…", 15);
  document.getElementById("signalAction").innerHTML = "-";
  document.getElementById("signalPosition").textContent = "-";
  document.getElementById("signalConfidence").textContent = "-";
  document.getElementById("adviceSummary").innerHTML = `<div class="empty-tip">正在生成建议...</div>`;
  document.getElementById("traceFlow").innerHTML = `<div class="empty-tip">正在生成建议...</div>`;
  document.getElementById("traceAnalysts").innerHTML = `<div class="empty-tip">正在生成建议...</div>`;
  document.getElementById("traceReferee").innerHTML = `<div class="empty-tip">正在生成建议...</div>`;
  document.getElementById("traceTimeline").innerHTML = `<div class="empty-tip">正在生成建议...</div>`;
  drawRadar(
    document.getElementById("agentRadarCanvas"),
    ["技术", "基本面", "裁判", "趋势", "仓位"],
    [0, 0, 0, 0, 0]
  );
  drawMiniKline(document.getElementById("miniPnlChart"), []);
  renderStabilityPanel(null);
  renderDataQualityPanel(null);
}

function renderAdviceSummary(data) {
  const rec = data.recommendation || {};
  const analysts = data.analyst_cases || {};
  const referee = data.referee || {};
  const risk = data.risk || {};
  const recAction = getRecAction(rec);
  const execAction = getExecutionAction(rec, risk);
  const decisionNote = buildDecisionConsistencyNote(rec, risk);
  const riskReason = buildRiskReason(risk, rec);
  const fullReason = buildFinalReason(rec, referee, analysts, riskReason);
  const dataQuality = data.data_quality || {};
  const briefReason = toBriefReason(
    fullReason,
    "综合多Agent观点后给出当前执行建议。",
    86
  );
  document.getElementById("adviceSummary").innerHTML = `
    <div class="advice-summary">
      <div class="summary-card">
        <div class="summary-title">建议动作</div>
        <div class="summary-value">${actionTag(recAction)} ${actionExplain(recAction, rec.position_percent)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-title">执行动作</div>
        <div class="summary-value">${actionTag(execAction)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-title">建议仓位</div>
        <div class="summary-value">${fmtPercent(rec.position_percent)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-title">置信度</div>
        <div class="summary-value">${fmtNumber(rec.confidence)}</div>
      </div>
      <div class="summary-card">
        <div class="summary-title">数据质量</div>
        <div class="summary-value">${Number.isFinite(Number(dataQuality.score)) ? Number(dataQuality.score).toFixed(2) : "-"} / ${qualityLevelText(dataQuality.level)}</div>
      </div>
      <div class="summary-card full-span">
        <div class="summary-title">结论摘要</div>
        <div class="summary-value long-text">${briefReason}</div>
      </div>
      <div class="summary-card full-span">
        <div class="summary-title">证据校准</div>
        <div class="summary-value long-text">${escapeHtml(rec.data_quality_note || dataQuality.note || "暂无数据质量校准说明")}</div>
      </div>
      <div class="summary-card full-span">
        <div class="summary-title">执行说明</div>
        <div class="summary-value long-text">${decisionNote}</div>
      </div>
    </div>
  `;
}

async function renderTraceFromAdvice(data) {
  setStatus("渲染多Agent链路...");
  const ticker = (data.ticker || document.getElementById("adviceTicker").value || "").trim();
  renderFlowChart(data);
  const analysts = data.analyst_cases || {};
  const analystEntries = Object.entries(analysts);
  document.getElementById("traceAnalysts").innerHTML = analystEntries.length
    ? `<div class="analyst-list">${analystEntries
        .map(([k, v]) => {
          const sentimentAction = v.sentiment === "positive" ? "BUY" : v.sentiment === "negative" ? "SELL" : "HOLD";
          return `
            <div class="analyst-card">
              <div class="analyst-head">
                <div class="analyst-name">${toLabel(k)}</div>
                <div>${actionTag(sentimentAction)}</div>
              </div>
              <div class="analyst-meta">置信度: ${Number(v.confidence || 0).toFixed(2)}</div>
              <div class="analyst-reason">${v.reasoning || "暂无文字说明"}</div>
            </div>
          `;
        })
        .join("")}</div>`
    : `<div class="empty-tip">暂无分析师观点</div>`;

  const referee = data.referee || {};
  const risk = data.risk || {};
  const refereeKeys = ["decision", "sentiment", "confidence", "agreement", "trend_strength", "position_percent", "reason"];
  const riskKeys = ["decision", "final_action", "action", "position_percent", "confidence", "risk_level", "reason"];
  document.getElementById("traceReferee").innerHTML = `
    <div class="decision-grid">
      <div class="decision-card">
        <div class="decision-title">裁判结论</div>
        ${renderKvRowsWithExpandableReason(referee, refereeKeys, "暂无裁判说明")}
      </div>
      <div class="decision-card">
        <div class="decision-title">风控结论</div>
        ${renderKvRowsWithExpandableReason(risk, riskKeys, "暂无风控说明")}
      </div>
    </div>
  `;

  const trace = (data.referee && data.referee.debate_trace) || [];
  document.getElementById("traceTimeline").innerHTML = trace.length
    ? `<div class="debate-list">${trace
        .map((x) => {
          const judge = x.judge || {};
          const sideA = x.case_a || {};
          const sideB = x.case_b || {};
          const sentimentAction = actionFromSignal(judge.sentiment || judge.decision);
          const aAction = actionFromSignal(sideA.sentiment || sideA.decision);
          const bAction = actionFromSignal(sideB.sentiment || sideB.decision);
          const aConf = fmtNumber(sideA.confidence);
          const bConf = fmtNumber(sideB.confidence);
          const note = buildRoundNote(judge, sideA, sideB);
          return `
            <div class="debate-item">
              <div class="debate-head">
                <span class="round-badge">Round ${x.round}</span>
                <span>${actionTag(sentimentAction)}</span>
              </div>
              <div class="analyst-meta">A: ${aAction} (${aConf}) | B: ${bAction} (${bConf}) | 裁判置信度: ${judge.confidence ?? "-"}</div>
              <div class="debate-note">${truncateText(note, 110)}</div>
            </div>
          `;
        })
        .join("")}</div>`
    : `<div class="empty-tip">无博弈轮次（可能观点已直接一致）</div>`;

  const rec = data.recommendation || {};
  document.getElementById("signalAction").innerHTML = actionTag(getExecutionAction(rec, data.risk || {}));
  document.getElementById("signalPosition").textContent = fmtPercent(rec.position_percent);
  document.getElementById("signalConfidence").textContent = fmtNumber(rec.confidence);

  const technicalConf = Number(((analysts.technical_flow || {}).confidence) || 0);
  const fundamentalConf = Number(((analysts.fundamental_news || {}).confidence) || 0);
  const refereeConf = Number(((data.referee || {}).confidence) || rec.confidence || 0);
  const trendStrength = Number(((data.referee || {}).trend_strength) || 0);
  const riskPos = Math.min(1, Number(rec.position_percent || 0) / 100);

  drawRadar(
    document.getElementById("agentRadarCanvas"),
    ["技术", "基本面", "裁判", "趋势", "仓位"],
    [technicalConf, fundamentalConf, refereeConf, trendStrength, riskPos]
  );

  if (ticker) {
    let candles = [];
    try {
      const ohlc = await api(`/api/market/ohlc/${padTicker(ticker)}?limit=40`, {
        timeout_ms: ADVICE_API_TIMEOUT_MS,
      });
      candles = buildMiniCandles(ohlc.bars || []);
    } catch {
      candles = [];
    }
    if (candles.length < 2) {
      const backtests = await api("/api/dashboard/backtests?limit=50");
      const runs = backtests.runs || [];
      const targetRun = runs.find((r) => padTicker(r.ticker) === padTicker(ticker));
      candles = buildMiniCandles((targetRun && targetRun.rows ? targetRun.rows : []).slice(-40));
    }
    drawMiniKline(document.getElementById("miniPnlChart"), candles);
  } else {
    drawMiniKline(document.getElementById("miniPnlChart"), []);
  }
  renderStabilityPanel(data);
  renderDataQualityPanel(data);

  setStatus("多Agent链路已加载");
}

async function runTraining() {
  setStatus("提交训练任务...");
  const payload = {
    top_n: Number(document.getElementById("trainTopN").value || 10),
    days: Number(document.getElementById("trainDays").value || 22),
    debate_depth: Number(document.getElementById("trainDebateDepth").value || 2),
    skip_auto_tune: !!document.getElementById("trainSkipTune").checked,
    csv_path: selectedTrainCsvPath || null,
  };
  const data = await api("/api/train/init", { method: "POST", body: JSON.stringify(payload) });
  setStatus(`训练任务已提交: ${data.task_id}`);
  await refreshTasks();
}

async function uploadTrainCsv() {
  const input = document.getElementById("trainCsvFileInput");
  if (!input || !input.files || !input.files.length) {
    setStatus("请先选择CSV文件。");
    return;
  }
  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);
  setStatus("正在上传训练CSV...");
  const data = await api("/api/train/upload-csv", { method: "POST", body: form });
  selectedTrainCsvPath = String(data.csv_path || "");
  updateTrainSourceText();
  setStatus(`CSV上传成功：${data.filename || file.name}`);
}

async function ensureBackendReady() {
  try {
    await api("/api/health", { timeout_ms: 5000 });
    return true;
  } catch {
    return false;
  }
}

async function runAdvice() {
  setStatus("生成建议中...");
  setAdviceBusy(true, "生成中...", "请稍候...");
  setAdviceProgress("info", "正在检查后端连接…", 5);
  const backendOk = await ensureBackendReady();
  if (!backendOk) {
    setAdviceFeedback(
      "error",
      `后端未运行（${API_BASE}）。请在项目目录终端执行：uvicorn backend.app:app --host 127.0.0.1 --port 8000，然后打开 http://127.0.0.1:8000`
    );
    setAdviceBusy(false);
    setStatus("后端未连接", "error");
    return;
  }
  setAdviceProgress("info", "正在提交建议任务…", 12);
  clearAdvicePanelsForGenerating();
  try {
    const payload = {
      ticker: document.getElementById("adviceTicker").value.trim(),
      debate_depth: Number(document.getElementById("adviceDepth").value || 2),
    };
    const submitData = await api("/api/advice/run", {
      method: "POST",
      body: JSON.stringify(payload),
      timeout_ms: ADVICE_API_TIMEOUT_MS,
    });
    const taskId = submitData.task_id;
    if (!taskId) throw new Error("后端未返回 task_id，请检查后端版本。");
    setAdviceProgress("info", `任务已提交，正在生成建议…`, 18);

    const task = await pollAdviceTask(taskId);
    if (!task) {
      throw new Error("建议生成超时（10分钟），请稍后在任务列表中查看结果。");
    }
    if (task.status === "failed") {
      throw new Error(task.message || "建议生成任务失败，请检查后端日志。");
    }
    if (task.status === "skipped") {
      throw new Error("同一股票的建议任务正在运行中，请稍后重试。");
    }

    const data = task.result || {};
    const runResult = data.run_result || {};
    if (runResult.returncode !== undefined && Number(runResult.returncode) !== 0) {
      const errText = cleanText(runResult.stderr || runResult.stdout || "建议脚本执行失败");
      throw new Error(errText);
    }
    const advice = resolveAdvicePayload(data);
    if (!advice.recommendation || !advice.recommendation.action) {
      throw new Error("建议结果为空或字段不完整，请检查后端日志。");
    }
    rememberUserAdvice(advice);
    setAdviceProgress("success", "生成完成，正在渲染结果…", 100);
    renderAdviceSummary(advice);
    await renderTraceFromAdvice(advice);
    setAdviceFeedback("success", `生成成功：${padTicker(advice.ticker || payload.ticker)} 建议已更新。`);
    setStatus("建议生成完成");
  } catch (e) {
    setAdviceFeedback("error", `生成失败：${e.message}`);
    throw e;
  } finally {
    setAdviceBusy(false);
  }
}

async function loadLatestAdvice() {
  setStatus("读取最新建议...");
  setAdviceBusy(true, "请稍候...", "读取中...");
  setAdviceFeedback("info", "正在读取最新建议...");
  try {
    const ticker = document.getElementById("adviceTicker").value.trim();
    const data = await api(`/api/advice/latest/${ticker}`);
    const advice = resolveAdvicePayload(data);
    if (!advice.recommendation || !advice.recommendation.action) {
      throw new Error("最新建议文件缺少 recommendation 字段。");
    }
    rememberUserAdvice(advice);
    renderAdviceSummary(advice);
    await renderTraceFromAdvice(advice);
    setAdviceFeedback("success", `读取成功：${padTicker(advice.ticker || ticker)} 最新建议已加载。`);
    setStatus("已读取最新建议");
  } catch (e) {
    setAdviceFeedback("error", `读取失败：${e.message}`);
    throw e;
  } finally {
    setAdviceBusy(false);
  }
}

async function uploadEvolveCsv() {
  const input = document.getElementById("evolveCsvFileInput");
  if (!input || !input.files || !input.files.length) {
    setStatus("请先选择 CSV 文件。");
    return;
  }
  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);
  const topNRaw = String(document.getElementById("evolveCsvTopN")?.value || "").trim();
  const topN = topNRaw ? Number(topNRaw) : null;
  const query = topN && Number.isFinite(topN) && topN > 0 ? `?top_n=${Math.floor(topN)}` : "";
  setStatus("正在导入进化标的 CSV...");
  const data = await api(`/api/evolution/upload-csv${query}`, { method: "POST", body: form });
  selectedEvolveCsvPath = String(data.csv_path || "");
  const tickers = Array.isArray(data.tickers) ? data.tickers.map((x) => normalizeTicker(x)).filter(Boolean) : [];
  const evolveInput = document.getElementById("evolveTickers");
  if (evolveInput) {
    evolveInput.value = tickers.join(",");
  }
  updateEvolveSourceText(data.count ?? tickers.length);
  setStatus(`CSV 导入成功：${data.count ?? tickers.length} 只股票`);
}

async function runEvolution() {
  setStatus("提交自进化任务...");
  const payload = buildEvolutionPayload();
  const data = await api("/api/evolution/run", { method: "POST", body: JSON.stringify(payload) });
  const countHint = data.ticker_count ? `（${data.ticker_count} 只）` : "";
  setStatus(`自进化任务已提交${countHint}: ${data.task_id}`);
  await refreshTasks();
}

async function refreshEvolutionHistory() {
  setStatus("拉取进化历史...");
  const data = await api("/api/evolution/history?limit=20");
  const html = (data.history || [])
    .map(
      (h) =>
        `<div class="timeline-item"><b>${h.run_at}</b> | mode=${h.mode}<br/>tickers=${(h.tickers || []).join(", ")}<br/>records=${(h.records || []).length}</div>`
    )
    .join("");
  document.getElementById("evolveHistory").innerHTML = html || "-";
  setStatus("进化历史已更新");
}

async function settleAdviceNow() {
  if (settleAdviceSubmitting) return;
  settleAdviceSubmitting = true;
  updateSettleAdviceButtonState();
  setStatus("提交建议兑现任务...");
  const submitGuard = window.setTimeout(() => {
    if (!settleAdviceSubmitting) return;
    settleAdviceSubmitting = false;
    updateSettleAdviceButtonState();
    setStatus("建议兑现提交等待超时，请重试。");
  }, API_TIMEOUT_MS + 2000);
  try {
    await refreshTasks();
    if (settleAdviceTaskActive) {
      setStatus("已有建议兑现任务在执行，请稍后刷新任务列表。");
      return;
    }
    const data = await api("/api/evolution/settle-advice", {
      method: "POST",
      body: JSON.stringify({ max_items: 2000 }),
    });
    setStatus(`建议兑现任务已提交: ${data.task_id}`);
    await refreshTasks();
    await refreshOverview();
  } finally {
    window.clearTimeout(submitGuard);
    settleAdviceSubmitting = false;
    updateSettleAdviceButtonState();
  }
}

async function submitRegister() {
  const email = (document.getElementById("registerEmailInput").value || "").trim();
  const phone = (document.getElementById("registerPhoneInput").value || "").trim();
  const nickname = (document.getElementById("registerNicknameInput").value || "").trim();
  const password = document.getElementById("registerPasswordInput").value || "";
  const errorEl = document.getElementById("registerError");
  if (!email || !phone || !password) {
    errorEl.textContent = "邮箱、手机号、密码为必填项。";
    return;
  }
  if (password.length < 6) {
    errorEl.textContent = "密码长度至少 6 位。";
    return;
  }
  errorEl.textContent = "";
  try {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, phone, nickname, password }),
    });
    const user = data.user || null;
    if (!user || !user.user_id) {
      throw new Error("注册返回数据异常");
    }
    if (!data.access_token) {
      throw new Error("注册返回缺少会话令牌");
    }
    saveAuthToken(data.access_token);
    saveCurrentUser(user);
    setAuthRequiredView(true);
    applyMode(MODE_USER, "advice");
    closeRegisterModal();
    setStatus(`注册成功：${user.nickname || user.email}`);
    await loadUserAdviceHistoryFromServer();
    await refreshOverview();
  } catch (e) {
    errorEl.textContent = `注册失败：${e.message}`;
    throw e;
  }
}

async function submitLogin() {
  const account = (document.getElementById("loginAccountInput").value || "").trim();
  const password = document.getElementById("loginPasswordInput").value || "";
  const errorEl = document.getElementById("loginError");
  if (!account || !password) {
    errorEl.textContent = "账号与密码均为必填项。";
    return;
  }
  errorEl.textContent = "";
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ account, password }),
    });
    const user = data.user || null;
    const token = data.access_token || "";
    if (!user || !user.user_id || !token) {
      throw new Error("登录返回数据异常");
    }
    saveAuthToken(token);
    saveCurrentUser(user);
    setAuthRequiredView(true);
    applyMode(MODE_USER, "advice");
    closeLoginModal();
    setStatus(`登录成功：${user.nickname || user.email || user.phone}`);
    await loadUserAdviceHistoryFromServer();
    await refreshOverview();
  } catch (e) {
    errorEl.textContent = `登录失败：${e.message}`;
    throw e;
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST", body: JSON.stringify({ token: authToken || null }) });
  } catch {
    // 即使后端失败，也执行本地会话清理，避免前端残留登录态
  } finally {
    clearSessionState();
    userAdviceTickersCache = [];
    setAuthRequiredView(false);
    applyMode(MODE_USER, "advice");
    closeAdminModal();
    setStatus("已退出登录");
  }
}

async function validateSession() {
  if (!authToken) {
    userAdviceTickersCache = [];
    if (currentUser && currentUser.user_id) clearSessionState();
    return false;
  }
  try {
    const data = await api("/api/auth/session");
    const user = data.user || null;
    if (!user || !user.user_id) {
      userAdviceTickersCache = [];
      clearSessionState();
      return false;
    }
    saveCurrentUser(user);
    return true;
  } catch {
    userAdviceTickersCache = [];
    clearSessionState();
    return false;
  }
}

const bindEvent = (id, event, handler) => {
  const el = document.getElementById(id);
  if (!el) {
    setStatus(`事件绑定跳过：#${id} 不存在`);
    return;
  }
  el.addEventListener(event, handler);
};

bindEvent("refreshOverviewBtn", "click", () => refreshOverview().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("runTrainingBtn", "click", () => runTraining().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("uploadTrainCsvBtn", "click", () => uploadTrainCsv().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("uploadEvolveCsvBtn", "click", () => uploadEvolveCsv().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("evolveCsvFileInput", "change", () => {
  const input = document.getElementById("evolveCsvFileInput");
  if (!input || !input.files || !input.files.length) return;
  setStatus(`已选择文件：${input.files[0].name}，点击“导入CSV文件”后生效。`);
});
bindEvent("trainCsvFileInput", "change", () => {
  const input = document.getElementById("trainCsvFileInput");
  if (!input || !input.files || !input.files.length) return;
  setStatus(`已选择文件：${input.files[0].name}，点击“导入CSV文件”后生效。`);
});
bindEvent("refreshTasksBtn", "click", () => refreshTasks().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("runAdviceBtn", "click", () => runAdvice().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("loadLatestAdviceBtn", "click", () => loadLatestAdvice().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("runEvolveBtn", "click", () => runEvolution().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("settleAdviceBtn", "click", () => settleAdviceNow().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("refreshEvolveHistoryBtn", "click", () => refreshEvolutionHistory().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("registerBtn", "click", openRegisterModal);
bindEvent("registerSubmitBtn", "click", () => submitRegister().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("registerCancelBtn", "click", closeRegisterModal);
bindEvent("registerPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") submitRegister().catch((err) => setStatus(`错误: ${err.message}`));
});
bindEvent("loginBtn", "click", openLoginModal);
bindEvent("authGateLoginBtn", "click", openLoginModal);
bindEvent("authGateRegisterBtn", "click", openRegisterModal);
bindEvent("loginSubmitBtn", "click", () => submitLogin().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("loginCancelBtn", "click", closeLoginModal);
bindEvent("loginPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") submitLogin().catch((err) => setStatus(`错误: ${err.message}`));
});
bindEvent("logoutBtn", "click", () => logout().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("refreshUsersBtn", "click", () => refreshAdminUsers().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("modeSwitchBtn", "click", () => {
  if (currentMode === MODE_ADMIN) {
    applyMode(MODE_USER, "advice");
    refreshOverview().catch((e) => setStatus(`错误: ${e.message}`));
  } else {
    openAdminModal(MODE_ADMIN);
  }
});
bindEvent("adminSubmitBtn", "click", verifyAdmin);
bindEvent("adminCancelBtn", "click", closeAdminModal);
bindEvent("adminPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") verifyAdmin();
});

async function bootstrap() {
  try {
    updateSettleAdviceButtonState();
    updateTrainSourceText();
    loadCurrentUser();
    loadAuthToken();
    const isLoggedIn = await validateSession();
    updateUserBadge();
    setAuthRequiredView(isLoggedIn);
    if (!isLoggedIn) return;
    await loadUserAdviceHistoryFromServer();
    applyMode(currentMode, currentMode === MODE_ADMIN ? "training" : "advice");
    updateEvolveSourceText();
    await api("/api/health");
    await refreshOverview();
    await refreshTasks();
    await refreshEvolutionHistory();
  } catch (e) {
    setStatus(`连接失败: ${e.message}`);
  }
}

bootstrap();

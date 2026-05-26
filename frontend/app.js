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
const ADVICE_QA_TIMEOUT_MS = 90000;
const ADVICE_POLL_MAX_WAIT_MS = 1800000;
const MODE_USER = "user";
const MODE_ADMIN = "admin";
let currentMode = MODE_USER;
let pendingMode = null;
let lastOverviewSignature = "";
const USER_ADVICE_HISTORY_KEY = "userAdviceHistoryKeys";
const CURRENT_USER_KEY = "currentUserProfile";
const AUTH_TOKEN_KEY = "authToken";
const ADMIN_PASSWORD_KEY = "adminPassword";
const DEFAULT_TRAIN_SOURCE_TEXT = "默认文件：个人_持股排名.csv";
let currentUser = null;
let authToken = "";
let loginMode = MODE_USER;
let selectedTrainCsvPath = "";
let selectedEvolveCsvPath = "";
let settleAdviceSubmitting = false;
let settleAdviceTaskActive = false;
let activeSettleHorizon = "";
let settleTaskPollTimer = null;
let globalStatusTimer = null;
let userAdviceTickersCache = [];
let currentAdvicePayload = null;
let adviceAskSubmitting = false;
let overviewAdviceRowsCache = [];
let overviewAdviceFilter = {
  action: "ALL",
  ticker: "",
};
let currentStockPersonalization = {
  ticker: "",
  profile: null,
  preferences: null,
  updated_at: "",
};
let customConstraintsDraft = [];
let customStrategiesDraft = [];
let stockRuleChoicesCache = [];

const inferStatusType = (text = "") => {
  const raw = String(text || "").toLowerCase();
  if (/错误|失败|异常|超时|timeout|error|failed/.test(raw)) return "error";
  if (/警告|降级|fallback|warning/.test(raw)) return "warning";
  if (/成功|已提交|已更新|已加载|已退出|已读取|完成|ok/.test(raw)) return "success";
  return "info";
};

const renderGlobalStatus = (text, type = "info") => {
  const bar = document.getElementById("globalStatusBar");
  if (!bar) return;
  const nextType = ["info", "success", "warning", "error"].includes(type) ? type : "info";
  bar.textContent = String(text || "");
  bar.classList.remove("hidden", "info", "success", "warning", "error");
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

const normalizeMultiPeriodAdvice = (data = {}) => {
  const rec = data.recommendation || {};
  const risk = data.risk || {};
  const multi = data.multi_period_advice || {};
  const shortTerm = multi.short_term || data.short_term || null;
  const swingTerm = multi.swing_term || data.swing_term || null;
  const riskPlan = multi.risk_plan || data.risk_plan || null;
  const fallbackAction = getRecAction(rec);
  const fallbackReason = cleanText(rec.reason || risk.reason || "综合多Agent观点后保持当前执行建议。");
  return {
    short_term: {
      horizon: "1-3 trading days",
      action: fallbackAction,
      confidence: rec.confidence ?? null,
      reason: fallbackReason,
      ...(shortTerm || {}),
    },
    swing_term: {
      horizon: "2-4 weeks",
      action: fallbackAction,
      confidence: rec.confidence ?? null,
      reason: fallbackReason,
      ...(swingTerm || {}),
    },
    risk_plan: {
      stop_loss: cleanText(risk.stop_loss || risk.reason || "按风控结论控制仓位，跌破关键均线时降仓。"),
      take_profit: cleanText(risk.take_profit || "达到预期收益或出现动量衰竭时分批止盈。"),
      invalid_conditions: Array.isArray(risk.invalid_conditions) && risk.invalid_conditions.length
        ? risk.invalid_conditions
        : ["跌破关键均线", "成交量连续萎缩"],
      ...(riskPlan || {}),
    },
  };
};

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
  const busy = settleAdviceSubmitting || settleAdviceTaskActive;
  const buttons = Array.from(document.querySelectorAll("#settleAdviceBtn, .settle-horizon-btn"));
  buttons.forEach((btn) => {
    btn.disabled = busy;
  });
  const btn = document.getElementById("settleAdviceBtn");
  if (!btn) return;
  if (settleAdviceSubmitting) {
    btn.textContent = activeSettleHorizon ? `提交 T+${activeSettleHorizon}...` : "提交全部周期...";
    return;
  }
  btn.textContent = settleAdviceTaskActive ? "兑现任务执行中..." : "全部周期";
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

const PIPELINE_STAGES = [
  ["start", "任务启动"],
  ["market_data", "行情获取"],
  ["technical_features", "技术特征"],
  ["rag", "RAG证据"],
  ["analysts", "分析师启动"],
  ["technical_analysis", "技术分析"],
  ["fundamental_analysis", "基本面分析"],
  ["debate", "裁判博弈"],
  ["risk", "风控执行"],
  ["data_quality", "数据质量"],
  ["finalize", "最终建议"],
  ["output", "结果保存"],
];

function eventStatusClass(status = "") {
  const raw = String(status || "").toLowerCase();
  if (raw === "warning" || raw === "fallback") return "warning";
  if (raw === "failed" || raw === "error") return "error";
  if (raw === "running") return "running";
  return "done";
}

function formatPipelinePayload(payload = {}) {
  if (!payload || typeof payload !== "object") return "";
  const bits = [];
  if (payload.source) bits.push(`来源 ${payload.source}`);
  if (payload.rows !== undefined) bits.push(`样本 ${payload.rows}`);
  if (payload.as_of_date) bits.push(`日期 ${payload.as_of_date}`);
  if (payload.documents !== undefined) bits.push(`文档 ${payload.documents}`);
  if (payload.sentiment) bits.push(`倾向 ${payload.sentiment}`);
  if (payload.decision) bits.push(`结论 ${payload.decision}`);
  if (payload.action) bits.push(`动作 ${payload.action}`);
  if (payload.position_percent !== undefined) bits.push(`仓位 ${fmtPercent(payload.position_percent)}`);
  if (payload.confidence !== undefined) bits.push(`置信度 ${fmtNumber(payload.confidence)}`);
  if (payload.score !== undefined) bits.push(`评分 ${fmtNumber(payload.score)}`);
  if (payload.level) bits.push(`等级 ${qualityLevelText(payload.level)}`);
  const diagnostics = Array.isArray(payload.diagnostics) ? payload.diagnostics.filter(Boolean).slice(0, 2) : [];
  diagnostics.forEach((x) => bits.push(cleanText(x)));
  const signals = Array.isArray(payload.signals) ? payload.signals.filter(Boolean).slice(0, 2) : [];
  signals.forEach((x) => bits.push(cleanText(x)));
  if (payload.reason) bits.push(toBriefReason(payload.reason, "", 80));
  if (payload.reasoning) bits.push(toBriefReason(payload.reasoning, "", 80));
  return bits.filter(Boolean).slice(0, 6).join(" | ");
}

function latestEventsByStage(events = []) {
  const map = new Map();
  events.forEach((event) => {
    if (!event || typeof event !== "object") return;
    const stage = String(event.stage || "").trim();
    if (!stage) return;
    map.set(stage, event);
  });
  return map;
}

function renderAdvicePipeline(events = [], task = {}) {
  const host = document.getElementById("advicePipeline");
  if (!host) return;
  const safeEvents = Array.isArray(events) ? events : [];
  const eventMap = latestEventsByStage(safeEvents);
  const currentStage = String(task.stage || "").trim();
  const progress = Number(task.progress_percent || 0);
  if (!safeEvents.length && !currentStage) {
    host.classList.add("hidden");
    host.innerHTML = "";
    return;
  }
  host.classList.remove("hidden");
  const customStages = safeEvents
    .map((event) => String(event.stage || "").trim())
    .filter((stage) => stage && !PIPELINE_STAGES.some(([key]) => key === stage));
  const stages = [...PIPELINE_STAGES, ...Array.from(new Set(customStages)).map((stage) => [stage, stage])];
  host.innerHTML = `
    <div class="pipeline-head">
      <div>
        <div class="pipeline-title">实时分析进程</div>
        <div class="pipeline-subtitle">${escapeHtml(task.message || "正在逐步生成结构化建议")}</div>
      </div>
      <span>${Number.isFinite(progress) ? Math.round(progress) : 0}%</span>
    </div>
    <div class="pipeline-list">
      ${stages
        .map(([stage, label]) => {
          const event = eventMap.get(stage);
          const isCurrent = currentStage === stage && !event;
          const status = eventStatusClass(event?.status || (isCurrent ? "running" : ""));
          const summary = event?.summary || (isCurrent ? "进行中..." : "");
          const meta = formatPipelinePayload(event?.payload || {});
          return `
            <div class="pipeline-step ${event ? status : isCurrent ? "running" : "pending"}">
              <div class="pipeline-dot"></div>
              <div class="pipeline-body">
                <div class="pipeline-step-head">
                  <b>${escapeHtml(event?.title || label)}</b>
                  <span>${escapeHtml(event?.created_at || "")}</span>
                </div>
                ${summary ? `<div class="pipeline-summary">${escapeHtml(summary)}</div>` : ""}
                ${meta ? `<div class="pipeline-meta">${escapeHtml(meta)}</div>` : ""}
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function buildAdviceLikeFromTask(task = {}) {
  const partial = task && typeof task.partial_result === "object" ? task.partial_result || {} : {};
  const ticker = normalizeTicker(
    partial.ticker || (currentAdvicePayload && currentAdvicePayload.ticker) || document.getElementById("adviceTicker")?.value || ""
  );
  const dataQuality = { ...(partial.data_quality || {}) };
  if (partial.rag && typeof partial.rag === "object" && !dataQuality.rag) dataQuality.rag = partial.rag;
  if (partial.latest_market && typeof partial.latest_market === "object" && !dataQuality.market) {
    dataQuality.market = {
      source: partial.latest_market.source,
      rows: partial.latest_market.rows,
      as_of_date: partial.latest_market.as_of_date,
    };
  }
  return {
    ticker,
    latest_market: partial.latest_market || {},
    technical_features: partial.technical_features || {},
    data_quality: dataQuality,
    recommendation: partial.recommendation || {},
    risk: partial.risk || {},
    referee: partial.referee || {},
    analyst_cases: {
      technical_flow: partial.technical_case || {},
      fundamental_news: partial.fundamental_case || {},
    },
    multi_period_advice: partial.multi_period_advice || {},
    stability_score: partial.stability_score,
    stability_level: partial.stability_level,
    stability_note: partial.stability_note,
    stability_diagnostics: partial.stability_diagnostics,
  };
}

function hasObjectData(value) {
  return !!value && typeof value === "object" && Object.keys(value).length > 0;
}

function renderTracePanelsFromAdviceLike(data = {}) {
  const analysts = data.analyst_cases || {};
  const technical = analysts.technical_flow || {};
  const fundamental = analysts.fundamental_news || {};
  const analystEntries = Object.entries(analysts).filter(([, v]) => hasObjectData(v));
  if (hasObjectData(technical) || hasObjectData(fundamental) || hasObjectData(data.referee) || hasObjectData(data.risk) || hasObjectData(data.recommendation)) {
    renderFlowChart(data);
  }
  const analystsHost = document.getElementById("traceAnalysts");
  if (analystsHost && analystEntries.length) {
    analystsHost.innerHTML = `<div class="analyst-list">${analystEntries
      .map(([k, v]) => {
        const sentimentAction = v.sentiment === "positive" ? "BUY" : v.sentiment === "negative" ? "SELL" : "HOLD";
        return `
          <div class="analyst-card">
            <div class="analyst-head">
              <div class="analyst-name">${toLabel(k)}</div>
              <div>${actionTag(sentimentAction)}</div>
            </div>
            <div class="analyst-meta">置信度: ${Number(v.confidence || 0).toFixed(2)}</div>
            <div class="analyst-reason">${escapeHtml(v.reasoning || "分析中，等待完整文字说明...")}</div>
          </div>
        `;
      })
      .join("")}</div>`;
  }
  const refereeHost = document.getElementById("traceReferee");
  if (refereeHost && (hasObjectData(data.referee) || hasObjectData(data.risk))) {
    const referee = data.referee || {};
    const risk = data.risk || {};
    const refereeKeys = ["decision", "sentiment", "confidence", "agreement", "trend_strength", "position_percent", "reason"];
    const riskKeys = ["decision", "final_action", "action", "position_percent", "confidence", "risk_level", "reason"];
    refereeHost.innerHTML = `
      <div class="decision-grid">
        <div class="decision-card">
          <div class="decision-title">裁判结论</div>
          ${hasObjectData(referee) ? renderKvRowsWithExpandableReason(referee, refereeKeys, "暂无裁判说明") : `<div class="empty-tip">等待裁判博弈...</div>`}
        </div>
        <div class="decision-card">
          <div class="decision-title">风控结论</div>
          ${hasObjectData(risk) ? renderKvRowsWithExpandableReason(risk, riskKeys, "暂无风控说明") : `<div class="empty-tip">等待风控约束...</div>`}
        </div>
      </div>
    `;
  }
  const trace = (data.referee && data.referee.debate_trace) || [];
  const timelineHost = document.getElementById("traceTimeline");
  if (timelineHost && trace.length) {
    timelineHost.innerHTML = `<div class="debate-list">${trace
      .map((x) => {
        const judge = x.judge || {};
        const sideA = x.case_a || {};
        const sideB = x.case_b || {};
        const sentimentAction = actionFromSignal(judge.sentiment || judge.decision);
        const aAction = actionFromSignal(sideA.sentiment || sideA.decision);
        const bAction = actionFromSignal(sideB.sentiment || sideB.decision);
        const note = buildRoundNote(judge, sideA, sideB);
        return `
          <div class="debate-item">
            <div class="debate-head">
              <span class="round-badge">Round ${x.round}</span>
              <span>${actionTag(sentimentAction)}</span>
            </div>
            <div class="analyst-meta">A: ${aAction} | B: ${bAction} | 裁判置信度: ${judge.confidence ?? "-"}</div>
            <div class="debate-note">${escapeHtml(truncateText(note, 110))}</div>
          </div>
        `;
      })
      .join("")}</div>`;
  }
}

function renderAdvicePartialsIntoCards(task = {}) {
  const data = buildAdviceLikeFromTask(task);
  const hasQuality = hasObjectData(data.data_quality);
  const hasMarketOrTech = hasObjectData(data.latest_market) || hasObjectData(data.technical_features) || hasQuality;
  const hasTrace =
    hasObjectData(data.analyst_cases.technical_flow) ||
    hasObjectData(data.analyst_cases.fundamental_news) ||
    hasObjectData(data.referee) ||
    hasObjectData(data.risk) ||
    hasObjectData(data.recommendation);

  if (hasMarketOrTech) {
    renderDataQualityPanel(data);
    renderAdviceDataDetails(data);
  }
  if (hasObjectData(data.multi_period_advice)) {
    renderMultiPeriodCards(data);
  }
  if (hasObjectData(data.recommendation)) {
    const rec = data.recommendation || {};
    document.getElementById("signalAction").innerHTML = actionTag(getExecutionAction(rec, data.risk || {}));
    document.getElementById("signalPosition").textContent = fmtPercent(rec.position_percent);
    document.getElementById("signalConfidence").textContent = fmtNumber(rec.confidence);
    renderAdviceSummary(data);
  }
  if (hasTrace) {
    renderTracePanelsFromAdviceLike(data);
    const technicalConf = Number(((data.analyst_cases.technical_flow || {}).confidence) || 0);
    const fundamentalConf = Number(((data.analyst_cases.fundamental_news || {}).confidence) || 0);
    const refereeConf = Number(((data.referee || {}).confidence) || ((data.recommendation || {}).confidence) || 0);
    const trendStrength = Number(((data.referee || {}).trend_strength) || 0);
    drawRadar(
      document.getElementById("agentRadarCanvas"),
      ["技术", "基本面", "裁判", "趋势"],
      [technicalConf, fundamentalConf, refereeConf, trendStrength]
    );
  }
  if (data.stability_score !== undefined || data.stability_level || data.stability_diagnostics) {
    renderStabilityPanel(data);
  }
}

function hasCompletedAdvicePayload() {
  return !!(
    currentAdvicePayload &&
    typeof currentAdvicePayload === "object" &&
    currentAdvicePayload.recommendation &&
    currentAdvicePayload.recommendation.action
  );
}

function updateAdviceReportButtonState(busy = false) {
  const reportBtn = document.getElementById("downloadAdviceReportBtn");
  if (!reportBtn) return;
  const ready = hasCompletedAdvicePayload();
  reportBtn.disabled = !!busy || !ready;
  reportBtn.textContent = busy ? "生成PDF中..." : "生成PDF报告";
  reportBtn.title = ready
    ? "使用LLM生成报告叙述；LLM不可用时自动使用结构化兜底"
    : "生成或读取建议完成后可生成PDF报告";
}

function setAdviceBusy(busy, runText = "生成建议", loadText = "读取最新建议") {
  const runBtn = document.getElementById("runAdviceBtn");
  const loadBtn = document.getElementById("loadLatestAdviceBtn");
  if (!runBtn || !loadBtn) return;
  runBtn.disabled = busy;
  loadBtn.disabled = busy;
  updateAdviceReportButtonState(busy);
  runBtn.classList.toggle("is-loading", busy);
  runBtn.textContent = busy ? runText : "生成建议";
  loadBtn.textContent = busy ? loadText : "读取最新建议";
}

function setAdviceQaBusy(busy) {
  adviceAskSubmitting = !!busy;
  const btn = document.getElementById("askAdviceBtn");
  const input = document.getElementById("adviceQuestionInput");
  if (btn) {
    btn.disabled = adviceAskSubmitting;
    btn.classList.toggle("is-loading", adviceAskSubmitting);
    btn.textContent = adviceAskSubmitting ? "回答中..." : "提问";
  }
  if (input) input.disabled = adviceAskSubmitting;
}

function setAdviceQaFeedback(type, text) {
  const el = document.getElementById("adviceQaFeedback");
  if (!el) return;
  el.className = `inline-feedback ${type}`;
  el.textContent = String(text || "");
  el.classList.remove("hidden");
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

function buildUserProfilePayload() {
  const profileControlsExist = !!document.getElementById("profileRisk");
  if (!profileControlsExist) {
    return {
      ...defaultUserProfile(),
      ...((currentUser && currentUser.user_profile && typeof currentUser.user_profile === "object") ? currentUser.user_profile : {}),
    };
  }
  const maxPosition = Number(document.getElementById("profileMaxPosition")?.value || 20);
  const currentPosition = Number(document.getElementById("profileCurrentPosition")?.value || 0);
  const costPrice = Number(document.getElementById("profileCostPrice")?.value || 0);
  return {
    risk_profile: document.getElementById("profileRisk")?.value || "balanced",
    holding_period: document.getElementById("profilePeriod")?.value || "swing",
    max_position_per_stock: Number.isFinite(maxPosition) ? maxPosition : 20,
    already_holding: !!document.getElementById("profileAlreadyHolding")?.checked,
    current_position_percent: Number.isFinite(currentPosition) ? currentPosition : 0,
    cost_price: Number.isFinite(costPrice) ? costPrice : 0,
    prefer_stop_loss: !!document.getElementById("profilePreferStopLoss")?.checked,
  };
}

function normalizePersonalizationPreferences(preferences = {}) {
  const raw = preferences && typeof preferences === "object" ? preferences : {};
  const list = Array.isArray(raw.custom_constraints) ? raw.custom_constraints : [];
  const customRules = Array.isArray(raw.custom_rules) ? raw.custom_rules : [];
  const inferUserStrategyPolicy = (item = {}) => {
    const explicit = String(item?.policy || item?.type || "").trim();
    if (["block_buy", "block_sell", "force_hold", "position_cap", "min_confidence"].includes(explicit)) return explicit;
    const text = cleanText(item?.text || "");
    if (/仓位|仓|不超过|上限|半仓|轻仓|小仓/.test(text)) return "position_cap";
    if (/置信|信心|确定性|把握|概率/.test(text)) return "min_confidence";
    if (/观望|等待|暂不|不操作|空仓|休息/.test(text)) return "force_hold";
    if (/卖|减仓|止盈|止损|清仓/.test(text)) return "block_sell";
    return "block_buy";
  };
  return {
    use_system_rules: raw.use_system_rules !== false,
    selected_rule_ids: Array.isArray(raw.selected_rule_ids) ? raw.selected_rule_ids.map((x) => String(x || "").trim()).filter(Boolean) : [],
    disabled_rule_ids: Array.isArray(raw.disabled_rule_ids) ? raw.disabled_rule_ids.map((x) => String(x || "").trim()).filter(Boolean) : [],
    custom_rules: customRules
      .map((item) => ({
        policy: inferUserStrategyPolicy(item),
        text: cleanText(item?.text || ""),
        value: Number(item?.value ?? 0),
        enabled: item?.enabled !== false,
      }))
      .filter((item) => ["block_buy", "block_sell", "force_hold", "position_cap", "min_confidence"].includes(item.policy)),
    custom_constraints: list
      .map((item) => ({
        type: String(item?.type || "").trim(),
        text: cleanText(item?.text || ""),
        value: Number(item?.value ?? 0),
        enabled: item?.enabled !== false,
      }))
      .filter((item) => ["block_buy", "block_sell", "force_hold", "position_cap", "min_confidence"].includes(item.type)),
  };
}

function buildStockProfilePayload() {
  const base = currentUser && currentUser.user_profile ? currentUser.user_profile : defaultUserProfile();
  const maxPosition = Number(document.getElementById("stockProfileMaxPosition")?.value || base.max_position_per_stock || 20);
  const currentPosition = Number(document.getElementById("stockProfileCurrentPosition")?.value || 0);
  const costPrice = Number(document.getElementById("stockProfileCostPrice")?.value || 0);
  return {
    risk_profile: document.getElementById("stockProfileRisk")?.value || base.risk_profile || "balanced",
    holding_period: document.getElementById("stockProfilePeriod")?.value || base.holding_period || "swing",
    max_position_per_stock: Number.isFinite(maxPosition) ? maxPosition : 20,
    already_holding: !!document.getElementById("stockProfileAlreadyHolding")?.checked,
    current_position_percent: Number.isFinite(currentPosition) ? currentPosition : 0,
    cost_price: Number.isFinite(costPrice) ? costPrice : 0,
    prefer_stop_loss: !!document.getElementById("stockProfilePreferStopLoss")?.checked,
  };
}

function buildStockPreferencesPayload() {
  const disabledRuleIds = Array.from(document.querySelectorAll(".stock-rule-checkbox"))
    .filter((input) => !input.checked)
    .flatMap((input) => String(input.dataset.ruleIds || "").split(","))
    .map((x) => x.trim())
    .filter(Boolean);
  return normalizePersonalizationPreferences({
    ...(currentStockPersonalization.preferences || {}),
    use_system_rules: document.getElementById("stockUseSystemRules")?.value !== "false",
    disabled_rule_ids: disabledRuleIds,
    custom_rules: customStrategiesDraft,
    custom_constraints: customConstraintsDraft,
  });
}

function applyStockPersonalizationToForm(config = {}) {
  const fallbackProfile = currentUser && currentUser.user_profile ? currentUser.user_profile : defaultUserProfile();
  const profile = { ...defaultUserProfile(), ...fallbackProfile, ...(config.profile || {}) };
  const prefs = normalizePersonalizationPreferences(config.preferences || {});
  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.value = String(value);
  };
  const setChecked = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.checked = !!value;
  };
  setValue("stockProfileRisk", profile.risk_profile || "balanced");
  setValue("stockProfilePeriod", profile.holding_period || "swing");
  setValue("stockProfileMaxPosition", profile.max_position_per_stock ?? 20);
  setValue("stockProfileCurrentPosition", profile.current_position_percent ?? 0);
  setValue("stockProfileCostPrice", profile.cost_price ?? 0);
  setValue("stockUseSystemRules", prefs.use_system_rules ? "true" : "false");
  setChecked("stockProfileAlreadyHolding", profile.already_holding);
  setChecked("stockProfilePreferStopLoss", profile.prefer_stop_loss !== false);
  customStrategiesDraft = prefs.custom_rules || [];
  customConstraintsDraft = prefs.custom_constraints || [];
  currentStockPersonalization = {
    ticker: normalizeTicker(config.ticker || document.getElementById("adviceTicker")?.value || ""),
    profile,
    preferences: prefs,
    updated_at: config.updated_at || "",
  };
  renderCustomConstraintList();
  renderCustomStrategyList();
  renderStockRuleChoices();
  updateStockPersonalizationStatus(config.updated_at ? `已加载：${config.updated_at}` : "使用账户默认画像");
}

function updateStockPersonalizationStatus(text = "") {
  const el = document.getElementById("stockPersonalizationStatus");
  if (el) el.textContent = text || "未加载";
}

function renderCustomConstraintList() {
  const host = document.getElementById("customConstraintList");
  if (!host) return;
  if (!customConstraintsDraft.length) {
    host.className = "constraint-list empty-tip";
    host.textContent = "暂无自定义约束";
    return;
  }
  const typeMap = {
    block_buy: "禁止买入",
    block_sell: "禁止卖出",
    force_hold: "强制观望",
    position_cap: "仓位上限",
    min_confidence: "最低置信度",
  };
  host.className = "constraint-list";
  host.innerHTML = customConstraintsDraft
    .map((item, idx) => {
      const numeric = item.type === "position_cap" ? `${Number(item.value || 0).toFixed(1)}%` : item.type === "min_confidence" ? Number(item.value || 0).toFixed(2) : "";
      return `
        <div class="constraint-item">
          <span><b>${escapeHtml(typeMap[item.type] || item.type)}</b>${numeric ? ` ${escapeHtml(numeric)}` : ""}${item.text ? ` | ${escapeHtml(item.text)}` : ""}</span>
          <button type="button" class="constraint-remove-btn" data-constraint-index="${idx}">删除</button>
        </div>
      `;
    })
    .join("");
}

function renderCustomStrategyList() {
  const host = document.getElementById("customStrategyList");
  if (!host) return;
  if (!customStrategiesDraft.length) {
    host.className = "constraint-list empty-tip";
    host.textContent = "暂无用户自定义策略";
    return;
  }
  const policyMap = {
    natural_language: "自动识别",
    block_buy: "买入前必须满足",
    block_sell: "卖出前必须满足",
    force_hold: "优先观望策略",
    position_cap: "仓位控制策略",
    min_confidence: "置信度门槛策略",
  };
  host.className = "constraint-list";
  host.innerHTML = customStrategiesDraft
    .map((item, idx) => {
      const numeric = item.policy === "position_cap" ? `${Number(item.value || 0).toFixed(1)}%` : item.policy === "min_confidence" ? Number(item.value || 0).toFixed(2) : "";
      return `
        <div class="constraint-item">
          <span><b>用户策略</b> ${escapeHtml(policyMap[item.policy] || item.policy)}${numeric ? ` ${escapeHtml(numeric)}` : ""}${item.text ? ` | ${escapeHtml(item.text)}` : ""}</span>
          <button type="button" class="strategy-remove-btn" data-strategy-index="${idx}">删除</button>
        </div>
      `;
    })
    .join("");
}

function addCustomStrategy() {
  const selectedPolicy = document.getElementById("customStrategyPolicy")?.value || "natural_language";
  const valueRaw = Number(document.getElementById("customStrategyValue")?.value || 0);
  const text = cleanText(document.getElementById("customStrategyText")?.value || "");
  if (!text) {
    setStatus("请先填写用户自定义策略内容。", "warning");
    return;
  }
  const item = {
    policy: selectedPolicy === "natural_language" ? "" : selectedPolicy,
    text,
    enabled: true,
  };
  if (selectedPolicy === "position_cap") item.value = Number.isFinite(valueRaw) ? Math.max(0, Math.min(100, valueRaw)) : 0;
  if (selectedPolicy === "min_confidence") item.value = Number.isFinite(valueRaw) ? Math.max(0, Math.min(1, valueRaw)) : 0;
  customStrategiesDraft.push(item);
  document.getElementById("customStrategyText").value = "";
  document.getElementById("customStrategyValue").value = "";
  renderCustomStrategyList();
}

function addCustomConstraint() {
  const type = document.getElementById("customConstraintType")?.value || "block_buy";
  const valueRaw = Number(document.getElementById("customConstraintValue")?.value || 0);
  const text = cleanText(document.getElementById("customConstraintText")?.value || "");
  const item = {
    type,
    text,
    enabled: true,
  };
  if (type === "position_cap") item.value = Number.isFinite(valueRaw) ? Math.max(0, Math.min(100, valueRaw)) : 0;
  if (type === "min_confidence") item.value = Number.isFinite(valueRaw) ? Math.max(0, Math.min(1, valueRaw)) : 0;
  customConstraintsDraft.push(item);
  document.getElementById("customConstraintText").value = "";
  document.getElementById("customConstraintValue").value = "";
  renderCustomConstraintList();
}

async function loadStockRuleChoices() {
  try {
    const data = await api("/api/strategy-rules?limit=80&status=active");
    stockRuleChoicesCache = Array.isArray(data.rules) ? data.rules : [];
  } catch {
    stockRuleChoicesCache = [];
  }
  renderStockRuleChoices();
}

function renderStockRuleChoices() {
  const host = document.getElementById("stockRuleChoiceList");
  if (!host) return;
  const prefs = normalizePersonalizationPreferences(currentStockPersonalization.preferences || {});
  const disabled = new Set(prefs.disabled_rule_ids || []);
  const systemEnabled = document.getElementById("stockUseSystemRules")?.value !== "false";
  if (!systemEnabled) {
    host.className = "stock-rule-choice-list empty-tip";
    host.textContent = "系统策略已停用";
    return;
  }
  if (!stockRuleChoicesCache.length) {
    host.className = "stock-rule-choice-list empty-tip";
    host.textContent = "暂无可选系统策略";
    return;
  }
  host.className = "stock-rule-choice-list";
  host.innerHTML = stockRuleChoicesCache
    .slice(0, 12)
    .map((rule) => {
      const ids = Array.isArray(rule.source_rule_ids) && rule.source_rule_ids.length ? rule.source_rule_ids : [rule.id];
      const checked = ids.some((id) => !disabled.has(String(id || "")));
      return `
        <label class="stock-rule-choice">
          <input class="stock-rule-checkbox" type="checkbox" data-rule-ids="${escapeHtml(ids.filter(Boolean).join(","))}" ${checked ? "checked" : ""} />
          <span>
            <b><span class="strategy-source-badge">系统策略</span>${escapeHtml(rule.mistake_type || "策略规则")}</b>
            <em>${escapeHtml(rule.rule_text || "-")}</em>
          </span>
        </label>
      `;
    })
    .join("");
}

async function loadStockPersonalizationFromServer({ silent = false } = {}) {
  const ticker = normalizeTicker(document.getElementById("adviceTicker")?.value || "");
  if (!ticker) {
    if (!silent) setStatus("请先输入股票代码", "warning");
    return null;
  }
  await loadStockRuleChoices();
  if (!authToken || !currentUser || isAdminCurrentUser()) {
    applyStockPersonalizationToForm({ ticker, profile: currentUser?.user_profile || defaultUserProfile(), preferences: {} });
    updateStockPersonalizationStatus("登录后可保存单票设置");
    return null;
  }
  const data = await api(`/api/user/stock-personalization/${ticker}`);
  applyStockPersonalizationToForm(data);
  if (!silent) setStatus(`已读取 ${ticker} 个性化设置`);
  return data;
}

async function saveStockPersonalizationToServer() {
  const ticker = normalizeTicker(document.getElementById("adviceTicker")?.value || "");
  if (!ticker) {
    setStatus("请先输入股票代码", "warning");
    return;
  }
  if (!authToken || !currentUser || isAdminCurrentUser()) {
    setStatus("请先以用户身份登录后再保存单票设置。", "warning");
    return;
  }
  const payload = {
    ticker,
    profile: buildStockProfilePayload(),
    preferences: buildStockPreferencesPayload(),
  };
  const data = await api("/api/user/stock-personalization", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  applyStockPersonalizationToForm(data);
  setStatus(`${ticker} 个性化设置已保存`);
}

function parsePortfolioHoldings(text = "") {
  return String(text || "")
    .split(/\n+/)
    .map((line) => cleanText(line))
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/[,\s，、]+/).map((x) => cleanText(x)).filter(Boolean);
      const ticker = normalizeTicker(parts[0] || "");
      const weight = Number(parts[1] || 0);
      const cost = Number(parts[2] || 0);
      return {
        ticker,
        weight_percent: Number.isFinite(weight) ? weight : 0,
        cost_price: Number.isFinite(cost) ? cost : 0,
      };
    })
    .filter((row) => row.ticker);
}

function defaultUserProfile() {
  return {
    risk_profile: "balanced",
    holding_period: "swing",
    max_position_per_stock: 20,
    already_holding: false,
    current_position_percent: 0,
    cost_price: 0,
    prefer_stop_loss: true,
  };
}

function applyUserProfileToForm(profile = {}) {
  const normalized = { ...defaultUserProfile(), ...(profile && typeof profile === "object" ? profile : {}) };
  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.value = String(value);
  };
  const setChecked = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.checked = !!value;
  };
  setValue("profileRisk", normalized.risk_profile || "balanced");
  setValue("profilePeriod", normalized.holding_period || "swing");
  setValue("profileMaxPosition", normalized.max_position_per_stock ?? 20);
  setValue("profileCurrentPosition", normalized.current_position_percent ?? 0);
  setValue("profileCostPrice", normalized.cost_price ?? 0);
  setChecked("profileAlreadyHolding", normalized.already_holding);
  setChecked("profilePreferStopLoss", normalized.prefer_stop_loss !== false);
}

function updateProfileStatus(text = "") {
  const el = document.getElementById("profileStatusText");
  if (!el) return;
  el.textContent = text || "登录后自动套用账户默认画像";
}

async function loadUserProfileFromServer() {
  if (isAdminCurrentUser()) {
    applyUserProfileToForm(defaultUserProfile());
    updateProfileStatus("管理员会话不使用用户画像");
    return null;
  }
  if (!authToken || !currentUser || !currentUser.user_id) {
    applyUserProfileToForm(defaultUserProfile());
    updateProfileStatus("登录后自动套用账户默认画像");
    return null;
  }
  try {
    const data = await api("/api/user/profile");
    const profile = data.user_profile || currentUser.user_profile || defaultUserProfile();
    currentUser.user_profile = profile;
    if (data.profile_updated_at !== undefined) currentUser.profile_updated_at = data.profile_updated_at;
    saveCurrentUser(currentUser);
    applyUserProfileToForm(profile);
    updateProfileStatus(data.profile_updated_at ? `账户默认画像已加载：${data.profile_updated_at}` : "账户默认画像已加载");
    return profile;
  } catch (e) {
    const fallback = currentUser.user_profile || defaultUserProfile();
    applyUserProfileToForm(fallback);
    updateProfileStatus(`默认画像读取失败，已使用本地缓存：${e.message}`);
    return fallback;
  }
}

async function saveUserProfileToServer({ silent = false } = {}) {
  if (!authToken || !currentUser || !currentUser.user_id || isAdminCurrentUser()) {
    if (!silent) setStatus("请先登录后再保存账户画像。", "warning");
    return null;
  }
  const profile = buildUserProfilePayload();
  const data = await api("/api/user/profile", {
    method: "POST",
    body: JSON.stringify(profile),
  });
  const saved = data.user_profile || profile;
  currentUser.user_profile = saved;
  saveCurrentUser(currentUser);
  applyUserProfileToForm(saved);
  updateProfileStatus("账户默认画像已保存，后续生成建议会自动套用。");
  if (!silent) setStatus("账户默认画像已保存");
  return saved;
}

async function saveAccountSettings() {
  if (!authToken || isAdminCurrentUser()) {
    setSettingsError("管理员会话不支持修改普通用户资料。");
    return;
  }
  const nickname = (document.getElementById("settingsNicknameInput")?.value || "").trim();
  const email = (document.getElementById("settingsEmailInput")?.value || "").trim();
  const phone = (document.getElementById("settingsPhoneInput")?.value || "").trim();
  const data = await api("/api/user/account", {
    method: "POST",
    body: JSON.stringify({ nickname, email, phone }),
  });
  const user = data.user || null;
  if (!user || !user.user_id) throw new Error("账户资料保存返回异常");
  saveCurrentUser(user);
  setSettingsError("账户资料已保存。", "success");
  setStatus("账户资料已保存");
}

async function uploadAvatarFromSettings() {
  if (!authToken || isAdminCurrentUser()) {
    setSettingsError("管理员会话不支持上传普通用户头像。");
    return;
  }
  const input = document.getElementById("settingsAvatarFileInput");
  if (!input || !input.files || !input.files.length) {
    setSettingsError("请先选择一张头像图片。");
    return;
  }
  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);
  setSettingsError("正在上传头像...", "success");
  const data = await api("/api/user/avatar", {
    method: "POST",
    body: form,
    timeout_ms: ADVICE_API_TIMEOUT_MS,
  });
  const user = data.user || null;
  if (!user || !user.user_id) throw new Error("头像上传返回异常");
  saveCurrentUser(user);
  renderAvatar(document.getElementById("settingsAvatar"));
  input.value = "";
  setSettingsError("头像已更新。", "success");
  setStatus("头像已更新");
}

async function changePasswordFromSettings() {
  if (!authToken || isAdminCurrentUser()) {
    setSettingsError("管理员会话不支持修改普通用户密码。");
    return;
  }
  const currentPassword = document.getElementById("settingsCurrentPasswordInput")?.value || "";
  const newPassword = document.getElementById("settingsNewPasswordInput")?.value || "";
  if (newPassword.length < 6) {
    setSettingsError("新密码长度至少 6 位。");
    return;
  }
  await api("/api/user/password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  document.getElementById("settingsCurrentPasswordInput").value = "";
  document.getElementById("settingsNewPasswordInput").value = "";
  setSettingsError("密码已修改。", "success");
  setStatus("密码已修改");
}

async function saveSettingsProfileToServer() {
  if (!authToken || isAdminCurrentUser()) {
    setSettingsError("管理员会话不支持保存普通用户画像。");
    return;
  }
  const profile = buildSettingsProfilePayload();
  const data = await api("/api/user/profile", {
    method: "POST",
    body: JSON.stringify(profile),
  });
  const saved = data.user_profile || profile;
  currentUser.user_profile = saved;
  saveCurrentUser(currentUser);
  applyUserProfileToForm(saved);
  applySettingsProfileToForm(saved);
  updateProfileStatus("账户默认画像已保存，后续生成建议会自动套用。");
  setSettingsError("用户画像已保存。", "success");
  setStatus("用户画像已保存");
}

function formatUserProfile(profile = {}, personalization = {}) {
  if (!profile || typeof profile !== "object") return "未应用用户画像";
  const riskMap = { conservative: "保守", balanced: "均衡", aggressive: "激进" };
  const periodMap = { intraday: "日内", swing: "波段", mid_term: "中线" };
  const notes = Array.isArray(personalization.notes) ? personalization.notes.filter((x) => cleanText(x)) : [];
  const base = [
    riskMap[profile.risk_profile] || profile.risk_profile || "-",
    periodMap[profile.holding_period] || profile.holding_period || "-",
    `单票上限 ${fmtPercent(profile.max_position_per_stock)}`,
    profile.already_holding ? `已持仓 ${fmtPercent(profile.current_position_percent)}` : "空仓",
  ].join(" / ");
  return notes.length ? `${base}。${notes.join("；")}` : base;
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

function isAdminCurrentUser() {
  return !!(currentUser && currentUser.role === MODE_ADMIN);
}

function getAdminPassword() {
  return String(window.localStorage.getItem(ADMIN_PASSWORD_KEY) || "");
}

function buildAdminUser() {
  return {
    user_id: "admin",
    role: MODE_ADMIN,
    nickname: "管理员",
    email: "",
    phone: "",
    user_profile: defaultUserProfile(),
  };
}

function getUserDisplayName() {
  if (!currentUser) return "";
  if (isAdminCurrentUser()) return "管理员";
  return currentUser.nickname || currentUser.email || currentUser.phone || "用户";
}

function getAvatarInitials() {
  const name = getUserDisplayName();
  if (!name) return "TA";
  if (isAdminCurrentUser()) return "管";
  const clean = String(name).trim();
  if (/[\u4e00-\u9fff]/.test(clean)) return clean.slice(0, 1);
  return clean.slice(0, 2).toUpperCase();
}

function getAvatarUrl() {
  if (!currentUser || isAdminCurrentUser()) return "";
  const raw = String(currentUser.avatar_url || "").trim();
  if (!raw) return "";
  if (/^(https?:|data:|blob:)/i.test(raw)) return raw;
  if (raw.startsWith("/")) return `${API_BASE}${raw}`;
  return raw;
}

function renderAvatar(el) {
  if (!el) return;
  const url = getAvatarUrl();
  if (url) {
    el.classList.add("has-avatar-image");
    el.innerHTML = `<img src="${escapeHtml(url)}" alt="" />`;
    return;
  }
  el.classList.remove("has-avatar-image");
  el.textContent = getAvatarInitials();
}

function updateUserBadge() {
  const badge = document.getElementById("currentUserBadge");
  const registerBtn = document.getElementById("registerBtn");
  const loginBtn = document.getElementById("loginBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const avatarBtn = document.getElementById("userAvatarBtn");
  if (!badge || !registerBtn || !loginBtn || !logoutBtn || !avatarBtn) return;
  if (currentUser && currentUser.user_id) {
    badge.classList.add("hidden");
    badge.textContent = "";
    registerBtn.classList.add("hidden");
    loginBtn.classList.add("hidden");
    logoutBtn.classList.add("hidden");
    avatarBtn.classList.remove("hidden");
    renderAvatar(avatarBtn);
    avatarBtn.title = `${getUserDisplayName()} - 系统设置`;
    return;
  }
  badge.classList.add("hidden");
  badge.textContent = "未登录";
  registerBtn.textContent = "注册";
  registerBtn.disabled = false;
  registerBtn.classList.remove("hidden");
  loginBtn.classList.remove("hidden");
  logoutBtn.classList.add("hidden");
  avatarBtn.classList.add("hidden");
  avatarBtn.classList.remove("has-avatar-image");
  avatarBtn.textContent = "TA";
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
  window.localStorage.removeItem(ADMIN_PASSWORD_KEY);
  applyUserProfileToForm(defaultUserProfile());
  applyStockPersonalizationToForm({ ticker: document.getElementById("adviceTicker")?.value || "", profile: defaultUserProfile(), preferences: {} });
  updateProfileStatus("登录后自动套用账户默认画像");
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
  if (modeSwitchBtn) modeSwitchBtn.classList.add("hidden");
  if (!isLoggedIn) {
    currentMode = MODE_USER;
    if (logoutBtn) logoutBtn.classList.add("hidden");
    if (loginBtn) loginBtn.classList.remove("hidden");
    if (registerBtn) {
      registerBtn.disabled = false;
      registerBtn.textContent = "注册";
      registerBtn.classList.remove("hidden");
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
  setLoginMode(MODE_USER);
  document.getElementById("loginAccountInput").value = "";
  document.getElementById("loginPasswordInput").value = "";
  document.getElementById("loginAdminPasswordInput").value = "";
  document.getElementById("loginError").textContent = "";
  document.getElementById("loginModal").classList.remove("hidden");
}

function closeLoginModal() {
  document.getElementById("loginModal").classList.add("hidden");
}

function setLoginMode(mode = MODE_USER) {
  loginMode = mode === MODE_ADMIN ? MODE_ADMIN : MODE_USER;
  const userFields = document.getElementById("loginUserFields");
  const adminFields = document.getElementById("loginAdminFields");
  const userBtn = document.getElementById("loginModeUserBtn");
  const adminBtn = document.getElementById("loginModeAdminBtn");
  if (userFields) userFields.classList.toggle("hidden", loginMode !== MODE_USER);
  if (adminFields) adminFields.classList.toggle("hidden", loginMode !== MODE_ADMIN);
  if (userBtn) userBtn.classList.toggle("active", loginMode === MODE_USER);
  if (adminBtn) adminBtn.classList.toggle("active", loginMode === MODE_ADMIN);
}

function applySettingsProfileToForm(profile = {}) {
  const normalized = { ...defaultUserProfile(), ...(profile && typeof profile === "object" ? profile : {}) };
  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.value = String(value);
  };
  const setChecked = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.checked = !!value;
  };
  setValue("settingsProfileRisk", normalized.risk_profile || "balanced");
  setValue("settingsProfilePeriod", normalized.holding_period || "swing");
  setValue("settingsProfileMaxPosition", normalized.max_position_per_stock ?? 20);
  setValue("settingsProfileCurrentPosition", normalized.current_position_percent ?? 0);
  setValue("settingsProfileCostPrice", normalized.cost_price ?? 0);
  setChecked("settingsProfileAlreadyHolding", normalized.already_holding);
  setChecked("settingsProfilePreferStopLoss", normalized.prefer_stop_loss !== false);
}

function buildSettingsProfilePayload() {
  const maxPosition = Number(document.getElementById("settingsProfileMaxPosition")?.value || 20);
  const currentPosition = Number(document.getElementById("settingsProfileCurrentPosition")?.value || 0);
  const costPrice = Number(document.getElementById("settingsProfileCostPrice")?.value || 0);
  return {
    risk_profile: document.getElementById("settingsProfileRisk")?.value || "balanced",
    holding_period: document.getElementById("settingsProfilePeriod")?.value || "swing",
    max_position_per_stock: Number.isFinite(maxPosition) ? maxPosition : 20,
    already_holding: !!document.getElementById("settingsProfileAlreadyHolding")?.checked,
    current_position_percent: Number.isFinite(currentPosition) ? currentPosition : 0,
    cost_price: Number.isFinite(costPrice) ? costPrice : 0,
    prefer_stop_loss: !!document.getElementById("settingsProfilePreferStopLoss")?.checked,
  };
}

function setSettingsError(text = "", type = "error") {
  const el = document.getElementById("settingsError");
  if (!el) return;
  el.textContent = String(text || "");
  el.classList.toggle("success-text", type === "success");
}

function openSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (!modal || !currentUser) return;
  const avatar = document.getElementById("settingsAvatar");
  const subtitle = document.getElementById("settingsSubtitle");
  const adminNotice = document.getElementById("settingsAdminNotice");
  const userContent = document.getElementById("settingsUserContent");
  renderAvatar(avatar);
  if (subtitle) subtitle.textContent = isAdminCurrentUser() ? "管理后台会话" : "管理账户资料、密码与默认用户画像。";
  if (adminNotice) adminNotice.classList.toggle("hidden", !isAdminCurrentUser());
  if (userContent) userContent.classList.toggle("hidden", isAdminCurrentUser());
  if (!isAdminCurrentUser()) {
    const setValue = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.value = String(value || "");
    };
    setValue("settingsNicknameInput", currentUser.nickname || "");
    setValue("settingsEmailInput", currentUser.email || "");
    setValue("settingsPhoneInput", currentUser.phone || "");
    applySettingsProfileToForm(currentUser.user_profile || defaultUserProfile());
    ["settingsCurrentPasswordInput", "settingsNewPasswordInput"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.value = "";
    });
    const avatarInput = document.getElementById("settingsAvatarFileInput");
    if (avatarInput) avatarInput.value = "";
  }
  setSettingsError("");
  modal.classList.remove("hidden");
}

function closeSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.classList.add("hidden");
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
    latest_settled_return_pct: top.settled_return_pct ?? null,
    latest_settled_date: top.settled_date || "",
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
    return `无法连接后端 API（${API_BASE}）。请先在项目目录执行 bash start_web.sh；公网访问时使用 PUBLIC_WEB=1 并配置公网域名或穿透地址。`;
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
      renderAdvicePartialsIntoCards(task);
      if (Number.isFinite(Number(task.progress_percent))) {
        setAdviceProgress("info", task.message || "建议生成中...", Number(task.progress_percent));
      }
      if (task.status === "done" || task.status === "failed" || task.status === "skipped") {
        return task;
      }
      const elapsedMs = Date.now() - startTime;
      const elapsedSec = Math.round(elapsedMs / 1000);
      const pct = adviceProgressFromElapsed(elapsedMs);
      if (!Number.isFinite(Number(task.progress_percent))) {
        setAdviceProgress("info", `建议生成中（约 ${elapsedSec}s）`, pct);
      }
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
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      detail = parsed.detail || parsed.message || text;
    } catch {
      // keep raw text
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

const filenameFromDisposition = (header = "", fallback = "investment_advice_report.pdf") => {
  const value = String(header || "");
  const utf8Match = value.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1] || fallback;
    }
  }
  const plainMatch = value.match(/filename="?([^";]+)"?/i);
  return plainMatch ? plainMatch[1] : fallback;
};

async function downloadAdviceReport({ ticker = "", generatedAt = "", adviceId = "", useLlm = true } = {}) {
  if (!hasCompletedAdvicePayload() && !adviceId && !generatedAt) {
    setAdviceFeedback("warning", "请先生成或读取完整建议，再生成PDF报告。");
    updateAdviceReportButtonState(false);
    return;
  }
  const targetTicker = normalizeTicker(
    ticker || (currentAdvicePayload && currentAdvicePayload.ticker) || document.getElementById("adviceTicker")?.value || ""
  );
  if (!targetTicker) {
    setAdviceFeedback("warning", "请先输入股票代码或加载一条建议。");
    return;
  }
  const params = new URLSearchParams();
  if (adviceId) params.set("advice_id", adviceId);
  if (generatedAt) params.set("generated_at", generatedAt);
  params.set("use_llm", useLlm ? "true" : "false");
  const query = params.toString() ? `?${params.toString()}` : "";
  const headers = {};
  if (authToken) headers.Authorization = `Bearer ${authToken}`;
  updateAdviceReportButtonState(true);
  setAdviceFeedback("info", `正在生成 ${targetTicker} PDF报告，默认使用LLM撰写叙述层...`);
  setStatus(`正在生成 ${targetTicker} PDF报告...`);
  let res;
  try {
    res = await fetch(`${API_BASE}/api/advice/report/${targetTicker}${query}`, { headers });
  } catch (e) {
    updateAdviceReportButtonState(false);
    throw new Error(formatFetchError(e, `/api/advice/report/${targetTicker}`));
  }
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      detail = parsed.detail || parsed.message || text;
    } catch {
      // keep raw text
    }
    updateAdviceReportButtonState(false);
    throw new Error(`${res.status} ${detail}`);
  }
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filenameFromDisposition(
    res.headers.get("Content-Disposition"),
    `${targetTicker}_investment_advice_report.pdf`
  );
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => window.URL.revokeObjectURL(url), 1500);
  setAdviceFeedback("success", `PDF报告已生成：${targetTicker}。LLM不可用时已自动使用结构化兜底。`);
  setStatus("PDF报告已开始下载");
  updateAdviceReportButtonState(false);
}

function activateTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
  if (tabId === "strategy-rules") refreshStrategyRules().catch((e) => setStatus(`错误: ${e.message}`));
  if (tabId === "admin-users") refreshAdminUsers().catch((e) => setStatus(`错误: ${e.message}`));
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

  const allowedTabs =
    currentMode === MODE_ADMIN
      ? ["overview", "training", "evolution", "strategy-rules", "admin-users"]
      : ["overview", "advice", "portfolio", "advisor-tools", "decision-trace"];
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
    const adminPassword = getAdminPassword();
    if (!adminPassword) {
      countHost.textContent = "用户数：-";
      tableHost.innerHTML = `<div class="empty-tip">请先完成管理员验证。</div>`;
      return;
    }
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
    countHost.textContent = "用户数：-";
    tableHost.innerHTML = `<div class="empty-tip">用户列表加载失败：${escapeHtml(e.message)}</div>`;
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
  const horizonEval = evaluation.horizon_evaluation || {};
  const horizons = Array.isArray(horizonEval.horizons) ? horizonEval.horizons : [];
  const calibration = Array.isArray(horizonEval.confidence_calibration) ? horizonEval.confidence_calibration : [];

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
    ${
      horizons.length
        ? `
          <div class="eval-section">
            <h4>多周期表现</h4>
            <table>
              <thead><tr><th>周期</th><th>样本</th><th>方向样本</th><th>命中率</th><th>平均收益(%)</th><th>最大回撤(%)</th></tr></thead>
              <tbody>
                ${horizons
                  .map(
                    (row) => `
                      <tr>
                        <td>${escapeHtml(row.horizon || "-")}</td>
                        <td>${row.samples ?? "-"}</td>
                        <td>${row.directional_samples ?? "-"}</td>
                        <td>${row.hit_rate === null || row.hit_rate === undefined ? "-" : `${(Number(row.hit_rate) * 100).toFixed(1)}%`}</td>
                        <td>${row.avg_return === null || row.avg_return === undefined ? "-" : Number(row.avg_return).toFixed(2)}</td>
                        <td>${row.max_drawdown === null || row.max_drawdown === undefined ? "-" : Number(row.max_drawdown).toFixed(2)}</td>
                      </tr>
                    `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `
        : ""
    }
    ${
      calibration.length
        ? `
          <div class="eval-section">
            <h4>置信度校准</h4>
            <table>
              <thead><tr><th>周期</th><th>置信区间</th><th>样本</th><th>平均置信</th><th>实际命中</th><th>校准误差</th></tr></thead>
              <tbody>
                ${calibration
                  .slice(0, 12)
                  .map(
                    (row) => `
                      <tr>
                        <td>${escapeHtml(row.horizon || "-")}</td>
                        <td>${escapeHtml(row.confidence_bucket || "-")}</td>
                        <td>${row.samples ?? "-"}</td>
                        <td>${row.avg_confidence === null || row.avg_confidence === undefined ? "-" : `${(Number(row.avg_confidence) * 100).toFixed(1)}%`}</td>
                        <td>${row.hit_rate === null || row.hit_rate === undefined ? "-" : `${(Number(row.hit_rate) * 100).toFixed(1)}%`}</td>
                        <td>${row.calibration_error === null || row.calibration_error === undefined ? "-" : `${(Number(row.calibration_error) * 100).toFixed(1)}%`}</td>
                      </tr>
                    `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `
        : ""
    }
  `;
}

function renderQualityRanking(data = {}) {
  const host = document.getElementById("qualityRankingPanel");
  const meta = document.getElementById("qualityRankingMeta");
  if (!host) return;
  const ranking = Array.isArray(data.ranking) ? data.ranking : [];
  const baselines = Array.isArray(data.baseline_comparison) ? data.baseline_comparison : [];
  const worstCases = Array.isArray(data.worst_cases) ? data.worst_cases : [];
  const mistakes = data.mistake_distribution || {};
  if (meta) {
    meta.textContent = `样本：${data.samples ?? 0}${data.inferred_attribution_count ? ` | 补归因：${data.inferred_attribution_count}` : ""}${data.updated_at ? ` | 更新时间：${data.updated_at}` : ""}`;
  }
  if (!ranking.length && !baselines.length) {
    host.innerHTML = `<div class="empty-tip">暂无已兑现样本。先在自进化中心执行多周期建议兑现。</div>`;
    return;
  }
  const mistakeText = Object.entries(mistakes)
    .slice(0, 6)
    .map(([k, v]) => `<span>${escapeHtml(k)}: ${v}</span>`)
    .join("");
  host.innerHTML = `
    ${
      ranking.length
        ? `
          <div class="eval-section">
            <h4>质量排行榜</h4>
            <table>
              <thead><tr><th>标的</th><th>质量分</th><th>样本</th><th>命中率</th><th>平均收益</th><th>超额(最佳基准)</th><th>主要错误</th></tr></thead>
              <tbody>
                ${ranking
                  .slice(0, 12)
                  .map(
                    (row) => `
                      <tr>
                        <td>${escapeHtml(row.ticker || "-")}</td>
                        <td>${Number(row.quality_score || 0).toFixed(1)}</td>
                        <td>${row.samples ?? "-"}</td>
                        <td>${row.hit_rate === null || row.hit_rate === undefined ? "-" : `${(Number(row.hit_rate) * 100).toFixed(1)}%`}</td>
                        <td>${row.avg_return === null || row.avg_return === undefined ? "-" : Number(row.avg_return).toFixed(2)}</td>
                        <td>${row.avg_excess_vs_best_baseline === null || row.avg_excess_vs_best_baseline === undefined ? "-" : Number(row.avg_excess_vs_best_baseline).toFixed(2)}</td>
                        <td>${escapeHtml(row.dominant_mistake_type || "-")}</td>
                      </tr>
                    `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `
        : ""
    }
    ${
      baselines.length
        ? `
          <div class="eval-section">
            <h4>基准策略对照</h4>
            <table>
              <thead><tr><th>周期</th><th>样本</th><th>平均收益</th><th>超额买持</th><th>超额最佳基准</th><th>最大回撤</th></tr></thead>
              <tbody>
                ${baselines
                  .map(
                    (row) => `
                      <tr>
                        <td>${escapeHtml(row.horizon || "-")}</td>
                        <td>${row.samples ?? "-"}</td>
                        <td>${row.avg_return === null || row.avg_return === undefined ? "-" : Number(row.avg_return).toFixed(2)}</td>
                        <td>${row.avg_excess_vs_buy_hold === null || row.avg_excess_vs_buy_hold === undefined ? "-" : Number(row.avg_excess_vs_buy_hold).toFixed(2)}</td>
                        <td>${row.avg_excess_vs_best_baseline === null || row.avg_excess_vs_best_baseline === undefined ? "-" : Number(row.avg_excess_vs_best_baseline).toFixed(2)}</td>
                        <td>${row.max_drawdown === null || row.max_drawdown === undefined ? "-" : Number(row.max_drawdown).toFixed(2)}</td>
                      </tr>
                    `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `
        : ""
    }
    ${mistakeText ? `<div class="quality-chips">${mistakeText}</div>` : ""}
    ${
      worstCases.length
        ? `
          <div class="eval-section">
            <h4>错误案例归因</h4>
            <div class="mistake-list">
              ${worstCases
                .slice(0, 6)
                .map((row) => {
                  const attr = row.mistake_attribution || {};
                  return `
                    <div class="mistake-item">
                      <b>${escapeHtml(row.ticker || "-")} ${escapeHtml(row.horizon || "")} ${actionTag(row.decision || "HOLD")}</b>
                      <span>收益 ${Number(row.pnl_percent || 0).toFixed(2)}%，超额最佳基准 ${Number(row.excess_return_vs_best_baseline || 0).toFixed(2)}%</span>
                      <span>${escapeHtml(attr.mistake_type || "-")}：${escapeHtml(attr.root_cause || "")}</span>
                    </div>
                  `;
                })
                .join("")}
            </div>
          </div>
        `
        : ""
    }
  `;
}

async function refreshQualityRanking() {
  const host = document.getElementById("qualityRankingPanel");
  if (host) host.innerHTML = `<div class="empty-tip">正在计算质量排行榜...</div>`;
  const data = await api("/api/advice/quality-ranking?limit=20&min_samples=1");
  renderQualityRanking(data);
}

function renderStrategyRules(data = {}) {
  const host = document.getElementById("strategyRulesPanel");
  const meta = document.getElementById("strategyRulesMeta");
  if (!host) return;
  const rules = Array.isArray(data.rules) ? data.rules : [];
  const summary = data.summary || {};
  if (meta) {
    const rawRuleText = data.grouped && data.raw_count ? ` | 合并展示：${rules.length}` : "";
    meta.textContent = `规则：${summary.total_rules ?? data.raw_count ?? data.count ?? rules.length}${rawRuleText} | 生效：${summary.active_rules ?? "-"}${data.updated_at ? ` | 更新时间：${data.updated_at}` : ""}`;
  }
  if (!rules.length) {
    host.innerHTML = `<div class="empty-tip">暂无策略规则。运行建议兑现后，错误归因会自动沉淀为规则。</div>`;
    return;
  }
  host.innerHTML = `
    <div class="stats-grid">
      <div class="stat-box"><div class="label">规则总数</div><div class="value">${summary.total_rules ?? rules.length}</div></div>
      <div class="stat-box"><div class="label">生效规则</div><div class="value">${summary.active_rules ?? "-"}</div></div>
      <div class="stat-box"><div class="label">观察规则</div><div class="value">${summary.watch_rules ?? "-"}</div></div>
      <div class="stat-box"><div class="label">合并展示</div><div class="value">${rules.length}</div></div>
    </div>
    <div class="strategy-rule-list">
      ${rules
        .map((rule) => {
          const examples = Array.isArray(rule.source_examples) ? rule.source_examples : [];
          return `
            <div class="strategy-rule-item">
              <div class="strategy-rule-head">
                <b>${escapeHtml(rule.mistake_type || "unknown")}</b>
                <span>${escapeHtml(rule.status || "-")} | ${escapeHtml(rule.action || "HOLD")} | 支撑 ${rule.support_count ?? 0}${rule.source_rule_count > 1 ? ` | 合并 ${rule.source_rule_count}` : ""}</span>
              </div>
              <div class="strategy-rule-text">${escapeHtml(rule.rule_text || "-")}</div>
              <div class="strategy-rule-meta">
                <span>严重度 ${rule.severity_score ?? "-"}</span>
                <span>平均收益 ${fmtNumber(rule.avg_pnl_percent)}%</span>
                <span>跑输最佳基准 ${fmtNumber(rule.avg_excess_vs_best_baseline)}%</span>
                <span>${escapeHtml(rule.stock_category || "all")}</span>
                ${Array.isArray(rule.horizons) && rule.horizons.length ? `<span>${escapeHtml(rule.horizons.filter(Boolean).slice(0, 4).join(" / "))}</span>` : ""}
              </div>
              ${
                examples.length
                  ? `<div class="qa-limitations">样本：${examples
                      .slice(-3)
                      .map((x) => `${escapeHtml(x.ticker || "-")} ${escapeHtml(x.horizon || "")} ${fmtNumber(x.pnl_percent)}%`)
                      .join("；")}</div>`
                  : ""
              }
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

async function refreshStrategyRules() {
  const host = document.getElementById("strategyRulesPanel");
  if (host) host.innerHTML = `<div class="empty-tip">正在读取策略规则库...</div>`;
  const data = await api("/api/strategy-rules?limit=80");
  renderStrategyRules(data);
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

async function verifyAdmin() {
  const input = document.getElementById("adminPasswordInput").value || "";
  const errorEl = document.getElementById("adminError");
  if (!input.trim()) {
    errorEl.textContent = "请输入管理员密码。";
    return;
  }
  errorEl.textContent = "正在验证...";
  try {
    await api("/api/admin/users", {
      method: "GET",
      headers: {
        "X-Admin-Password": input,
      },
    });
    window.localStorage.setItem(ADMIN_PASSWORD_KEY, input);
    if (pendingMode === MODE_ADMIN) {
      applyMode(MODE_ADMIN, "overview");
      refreshOverview().catch((e) => setStatus(`错误: ${e.message}`));
    }
    closeAdminModal();
  } catch (e) {
    errorEl.textContent = e.message || "管理员验证失败，请重试。";
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tabId = btn.dataset.tab;
    const allowedTabs =
      currentMode === MODE_ADMIN
        ? ["overview", "training", "evolution", "strategy-rules", "admin-users"]
        : ["overview", "advice", "portfolio", "advisor-tools", "decision-trace"];
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
    科创板: "#9f1239",
    创业板: "#c2410c",
    北交所: "#b7791f",
    沪市主板: "#7f1d1d",
    深市主板: "#be123c",
    其他: "#8a7a72",
  };

  let cursor = 0;
  const gradientParts = entries.map(([board, count]) => {
    const start = cursor;
    const pct = (Number(count || 0) / total) * 100;
    cursor += pct;
    const color = colorMap[board] || "#9a8a83";
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
            const color = colorMap[board] || "#9a8a83";
            return `<div class="board-legend-item"><span class="board-dot" style="background:${color};"></span><span>${board}</span><span>${count} (${pct}%)</span></div>`;
          })
          .join("")}
      </div>
    </div>
  `;
}

function normalizeTickerSearch(text = "") {
  return String(text || "").replace(/\D/g, "").slice(0, 6);
}

function updateAdviceFilterControls() {
  document.querySelectorAll(".advice-filter-btn").forEach((btn) => {
    btn.classList.toggle("active", String(btn.dataset.adviceAction || "ALL") === overviewAdviceFilter.action);
  });
  const input = document.getElementById("adviceTickerSearchInput");
  if (input && input.value !== overviewAdviceFilter.ticker) input.value = overviewAdviceFilter.ticker;
}

function getFilteredOverviewAdviceRows() {
  const rows = Array.isArray(overviewAdviceRowsCache) ? overviewAdviceRowsCache : [];
  const actionFilter = String(overviewAdviceFilter.action || "ALL").toUpperCase();
  const tickerFilter = normalizeTickerSearch(overviewAdviceFilter.ticker);
  return rows.filter((row) => {
    const action = normalizeAction(row.action || "HOLD");
    const ticker = padTicker(row.ticker || "");
    if (actionFilter !== "ALL" && action !== actionFilter) return false;
    if (tickerFilter && !ticker.includes(tickerFilter)) return false;
    return true;
  });
}

function formatSettlementReturnCell(row = {}) {
  if (!row.settled) return `<span class="settlement-return settlement-pending">待兑现</span>`;
  const value = toNumOrNull(row.settled_return_pct);
  if (value === null) return `<span class="settlement-return settlement-pending">已兑现</span>`;
  const horizon = String(row.settled_horizon || (row.settlement || {}).horizon || "T+1");
  const signed = `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
  const tone = value > 0 ? "positive" : value < 0 ? "negative" : "flat";
  const date = row.settled_date ? `兑现日 ${row.settled_date}` : "";
  const excess = toNumOrNull(row.settled_excess_best_pct);
  const excessText = excess === null ? "" : `；超额最佳基准 ${excess > 0 ? "+" : ""}${excess.toFixed(2)}%`;
  const title = `${horizon}${date ? `；${date}` : ""}${excessText}`;
  return `
    <span class="settlement-return settlement-${tone}" title="${escapeHtml(title)}">
      <b>${escapeHtml(horizon)}</b>
      <span>${signed}</span>
    </span>
  `;
}

function renderOverviewAdviceTable() {
  const host = document.getElementById("overviewAdviceTable");
  if (!host) return;
  updateAdviceFilterControls();
  const rows = getFilteredOverviewAdviceRows();
  const total = overviewAdviceRowsCache.length;
  const meta = document.getElementById("adviceFilterMeta");
  if (meta) meta.textContent = `显示：${rows.length} / ${total}`;

  if (!total) {
    host.innerHTML = `<div class="empty-tip">${currentMode === MODE_USER ? "暂无你主动查询/生成的建议记录" : "暂无建议记录"}</div>`;
    return;
  }
  if (!rows.length) {
    host.innerHTML = `<div class="empty-tip">没有匹配当前筛选条件的建议记录。</div>`;
    return;
  }

  host.innerHTML = `
    <table>
      <thead><tr><th>Ticker</th><th>Action</th><th>Position%</th><th>Confidence</th><th>兑现收益</th><th>Time</th><th>报告</th></tr></thead>
      <tbody>
        ${rows
          .map(
            (r) =>
              `<tr><td>${escapeHtml(padTicker(r.ticker || ""))}</td><td>${actionTag(r.action)}</td><td>${Number(r.position_percent || 0).toFixed(2)}</td><td>${Number(r.confidence || 0).toFixed(2)}</td><td>${formatSettlementReturnCell(r)}</td><td>${escapeHtml(r.generated_at || "")}</td><td><button class="table-action-btn download-advice-report-btn" data-ticker="${escapeHtml(padTicker(r.ticker || ""))}" data-generated-at="${escapeHtml(r.generated_at || "")}">PDF</button></td></tr>`
          )
          .join("")}
      </tbody>
    </table>
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
    await refreshQualityRanking();
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
    overviewAdviceRowsCache = rows;
    renderOverviewAdviceTable();

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

  ctx.strokeStyle = "rgba(122,92,82,0.42)";
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

  ctx.strokeStyle = "rgba(159,18,57,0.86)";
  ctx.fillStyle = "rgba(159,18,57,0.18)";
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

  ctx.fillStyle = "#4a2f2a";
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
    svg.innerHTML = `<text x="18" y="30" fill="#7a5c52" font-size="12">暂无足够K线数据</text>`;
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
    <text x="14" y="16" fill="#7a5c52" font-size="12">L ${min.toFixed(2)}</text>
    <text x="${width - 88}" y="16" fill="#7a5c52" font-size="12">H ${max.toFixed(2)}</text>
    <text x="${width - 160}" y="${height - 4}" fill="#7a5c52" font-size="11">最后: O ${last.open.toFixed(2)} / C ${last.close.toFixed(2)}</text>
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

const SOURCE_TYPE_LABELS = {
  news: "个股新闻",
  market_news: "市场要闻",
  report: "研报",
  announcement: "公告",
  financial_abstract: "财务摘要",
  financial_indicator: "财务指标",
  income_statement: "利润表",
  balance_sheet: "资产负债表",
  cash_flow: "现金流",
  fallback: "兜底文本",
};

const SOURCE_TYPE_GROUPS = [
  { title: "舆情", keys: ["news", "market_news"] },
  { title: "公告研报", keys: ["announcement", "report"] },
  { title: "基本面", keys: ["financial_abstract", "financial_indicator", "income_statement", "balance_sheet", "cash_flow"] },
];

const SOURCE_LINK_DEFS = {
  news: { label: "东方财富个股新闻", url: (ticker) => `https://so.eastmoney.com/news/s?keyword=${encodeURIComponent(ticker)}` },
  market_news: { label: "财新市场要闻", url: (ticker) => `https://www.caixin.com/search/${encodeURIComponent(ticker)}.html` },
  report: { label: "东方财富研报", url: (ticker) => `https://data.eastmoney.com/report/${encodeURIComponent(ticker)}.html` },
  announcement: { label: "巨潮资讯公告", url: (ticker) => `https://www.cninfo.com.cn/new/fulltextSearch?notautosubmit=&keyWord=${encodeURIComponent(ticker)}` },
  financial_abstract: { label: "东方财富财务摘要", url: (_ticker, exchangeTicker) => `https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=${encodeURIComponent(exchangeTicker)}#/cwfx` },
  financial_indicator: { label: "东方财富财务指标", url: (_ticker, exchangeTicker) => `https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=${encodeURIComponent(exchangeTicker)}#/cwfx` },
  income_statement: { label: "东方财富利润表", url: (_ticker, exchangeTicker) => `https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=${encodeURIComponent(exchangeTicker)}#/cwfx` },
  balance_sheet: { label: "东方财富资产负债表", url: (_ticker, exchangeTicker) => `https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=${encodeURIComponent(exchangeTicker)}#/cwfx` },
  cash_flow: { label: "东方财富现金流量表", url: (_ticker, exchangeTicker) => `https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=${encodeURIComponent(exchangeTicker)}#/cwfx` },
};

function normalizeTickerForLinks(ticker = "") {
  return String(ticker || "").replace(/\D/g, "").padStart(6, "0").slice(-6);
}

function exchangeTickerForLinks(ticker = "") {
  const code = normalizeTickerForLinks(ticker);
  return `${code.startsWith("6") ? "SH" : "SZ"}${code}`;
}

function deriveSourceLinks(sourceTypes = {}, ticker = "") {
  const code = normalizeTickerForLinks(ticker || currentAdvicePayload?.ticker || "");
  const exchangeTicker = exchangeTickerForLinks(code);
  return Object.entries(sourceTypes || {})
    .map(([source, count]) => ({ source, count: Number(count) || 0 }))
    .filter((item) => item.count > 0 && SOURCE_LINK_DEFS[item.source])
    .sort((a, b) => b.count - a.count || a.source.localeCompare(b.source))
    .map((item) => {
      const spec = SOURCE_LINK_DEFS[item.source];
      return {
        ...item,
        label: spec.label,
        url: spec.url(code, exchangeTicker),
      };
    });
}

function normalizeSourceLinks(rag = {}, sourceTypes = {}, ticker = "") {
  const links = Array.isArray(rag?.source_links) ? rag.source_links : [];
  const normalized = links
    .map((item) => ({
      source: item?.source || "",
      label: item?.label || SOURCE_TYPE_LABELS[item?.source] || item?.source || "数据来源",
      url: item?.url || "",
      count: Number(item?.count) || Number(sourceTypes?.[item?.source]) || 0,
    }))
    .filter((item) => item.url);
  return normalized.length ? normalized : deriveSourceLinks(sourceTypes, ticker);
}

function sourceUrlForType(source = "", sourceLinks = []) {
  return (sourceLinks || []).find((item) => item.source === source)?.url || "";
}

function renderSourceLinkList(sourceLinks = []) {
  const links = (sourceLinks || []).filter((item) => item?.url).slice(0, 8);
  if (!links.length) return "";
  return `
    <div class="data-detail-section">
      <div class="data-detail-title">数据来源链接</div>
      <div class="data-source-link-list">
        ${links
          .map(
            (item) => `
              <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">
                <span class="source-link-label">
                  <span class="source-link-icon" aria-hidden="true"></span>
                  <b>${escapeHtml(item.label || SOURCE_TYPE_LABELS[item.source] || item.source || "数据来源")}</b>
                </span>
                <span>${item.count ? `${Number(item.count)}条` : "打开来源"}</span>
              </a>
            `
          )
          .join("")}
      </div>
    </div>
  `;
}

function renderSourceBreakdown(sourceTypes = {}, sourceLinks = []) {
  const entries = Object.entries(sourceTypes)
    .map(([key, value]) => [key, Number(value)])
    .filter(([, value]) => Number.isFinite(value) && value > 0);
  if (!entries.length) {
    return `
      <div class="source-breakdown">
        <div class="source-breakdown-head">
          <span>数据来源</span>
          <b>0</b>
        </div>
        <div class="source-empty">暂无来源统计</div>
      </div>
    `;
  }

  const entryMap = new Map(entries);
  const usedKeys = new Set();
  const groups = SOURCE_TYPE_GROUPS.map((group) => {
    const items = group.keys
      .filter((key) => entryMap.has(key))
      .map((key) => {
        usedKeys.add(key);
        return [key, entryMap.get(key)];
      });
    return { ...group, items };
  }).filter((group) => group.items.length);

  const otherItems = entries.filter(([key]) => !usedKeys.has(key));
  if (otherItems.length) groups.push({ title: "其他", items: otherItems });

  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  return `
    <div class="source-breakdown">
      <div class="source-breakdown-head">
        <span>数据来源</span>
        <b>${total} 条</b>
      </div>
      <div class="source-group-list">
        ${groups
          .map(
            (group) => `
              <div class="source-group">
                <span class="source-group-title">${escapeHtml(group.title)}</span>
                <div class="source-chip-row">
                  ${group.items
                    .map(
                      ([key, value]) => `
                        <${sourceUrlForType(key, sourceLinks) ? "a" : "span"} class="source-chip" ${
                          sourceUrlForType(key, sourceLinks)
                            ? `href="${escapeHtml(sourceUrlForType(key, sourceLinks))}" target="_blank" rel="noopener noreferrer" title="打开${escapeHtml(SOURCE_LINK_DEFS[key]?.label || SOURCE_TYPE_LABELS[key] || key)}"`
                            : ""
                        }>
                          <b>${escapeHtml(SOURCE_TYPE_LABELS[key] || key)}</b>
                          <em>${value}</em>
                        </${sourceUrlForType(key, sourceLinks) ? "a" : "span"}>
                      `
                    )
                    .join("")}
                </div>
              </div>
            `
          )
          .join("")}
      </div>
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
  const sourceLinks = normalizeSourceLinks(rag, sourceTypes, data.ticker);
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
      <div><span>RAG</span><b>${rag.vectorstore_degraded ? "关键词" : "向量"} / ${rag.documents ?? "-"}</b></div>
      <div><span>解析失败</span><b>${analyst.parse_fail_count ?? "-"}</b></div>
    </div>
    ${renderSourceBreakdown(sourceTypes, sourceLinks)}
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

function fmtDataValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (Number.isFinite(n)) {
    const digits = Math.abs(n) >= 1000 ? 0 : 2;
    return `${n.toFixed(digits)}${suffix}`;
  }
  return `${escapeHtml(String(value))}${suffix}`;
}

function renderDataKv(title, rows = []) {
  const items = rows.filter((row) => row && row.label);
  if (!items.length) return "";
  return `
    <div class="data-detail-section">
      <div class="data-detail-title">${escapeHtml(title)}</div>
      <div class="data-kv-grid">
        ${items
          .map(
            (row) => `
              <div class="data-kv-item">
                <span>${escapeHtml(row.label)}</span>
                <b>${row.html ? row.value : escapeHtml(String(row.value ?? "-"))}</b>
              </div>
            `
          )
          .join("")}
      </div>
    </div>
  `;
}

function collectSourceReports(analysts = {}) {
  const rows = [];
  Object.entries(analysts || {}).forEach(([agentKey, item]) => {
    const reports = Array.isArray(item?.source_reports) ? item.source_reports : [];
    reports.forEach((report) => {
      rows.push({
        agent: report.agent || agentKey,
        sentiment: report.sentiment || item.sentiment || "-",
        confidence: report.confidence ?? item.confidence,
        reasoning: report.reasoning || report.thought_process || "",
      });
    });
  });
  return rows;
}

function renderAdviceDataDetails(data) {
  const host = document.getElementById("adviceDataDetails");
  if (!host) return;
  if (!data || typeof data !== "object") {
    host.className = "data-detail-panel empty-tip";
    host.textContent = "生成建议后显示本次建议使用的行情、技术指标与证据来源。";
    return;
  }

  const latest = data.latest_market || {};
  const features = data.technical_features || {};
  const trend = features.trend || {};
  const momentum = features.momentum || {};
  const risk = features.risk || {};
  const volume = features.volume || {};
  const quality = data.data_quality || {};
  const market = quality.market || {};
  const rag = quality.rag || {};
  const analysts = data.analyst_cases || {};
  const sourceReports = collectSourceReports(analysts).slice(0, 8);
  const signals = Array.isArray(features.signals) ? features.signals.filter((x) => cleanText(x)).slice(0, 8) : [];
  const sourceTypes = rag.source_types && typeof rag.source_types === "object" ? rag.source_types : {};
  const sourceLinks = normalizeSourceLinks(rag, sourceTypes, data.ticker);
  const sourceRows = Object.entries(sourceTypes)
    .filter(([, value]) => Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  host.className = "data-detail-panel";
  host.innerHTML = `
    <div class="data-detail-grid">
      ${renderDataKv("行情快照", [
        { label: "数据日期", value: data.as_of_date || market.as_of_date || "-" },
        { label: "收盘价", value: fmtDataValue(latest.close), html: true },
        { label: "涨跌幅", value: fmtDataValue(latest.pct_change, "%"), html: true },
        { label: "换手率", value: fmtDataValue(latest.turnover, "%"), html: true },
        { label: "成交量", value: fmtDataValue(latest.volume), html: true },
        { label: "行情来源", value: market.source || "-" },
        { label: "样本行数", value: market.rows ?? features.lookback_rows ?? "-" },
      ])}
      ${renderDataKv("趋势与动量", [
        { label: "MA5", value: fmtDataValue(trend.ma5), html: true },
        { label: "MA10", value: fmtDataValue(trend.ma10), html: true },
        { label: "MA20", value: fmtDataValue(trend.ma20), html: true },
        { label: "MA60", value: fmtDataValue(trend.ma60), html: true },
        { label: "价距MA20", value: fmtDataValue(trend.price_vs_ma20_pct, "%"), html: true },
        { label: "5日收益", value: fmtDataValue(momentum.return_5d_pct, "%"), html: true },
        { label: "20日收益", value: fmtDataValue(momentum.return_20d_pct, "%"), html: true },
        { label: "RSI14", value: fmtDataValue(momentum.rsi14), html: true },
        { label: "MACD", value: fmtDataValue(momentum.macd), html: true },
        { label: "MACD柱", value: fmtDataValue(momentum.macd_hist), html: true },
      ])}
      ${renderDataKv("风险与成交", [
        { label: "ATR14占比", value: fmtDataValue(risk.atr14_pct, "%"), html: true },
        { label: "20日年化波动", value: fmtDataValue(risk.volatility20_annual_pct, "%"), html: true },
        { label: "60日最大回撤", value: fmtDataValue(risk.max_drawdown60_pct, "%"), html: true },
        { label: "5/20日量比", value: fmtDataValue(volume.volume_ratio_5_20), html: true },
        { label: "成交额比", value: fmtDataValue(volume.amount_ratio_5_20), html: true },
        { label: "换手分位", value: fmtDataValue(volume.turnover_percentile_60), html: true },
      ])}
      ${renderDataKv("RAG证据", [
        { label: "检索方式", value: rag.vectorstore_degraded ? "关键词检索" : "向量检索" },
        { label: "文档数量", value: rag.documents ?? "-" },
        { label: "最新证据日期", value: rag.latest_date || "-" },
        { label: "新鲜度", value: rag.freshness_days === null || rag.freshness_days === undefined ? "-" : `${rag.freshness_days}天` },
        { label: "数据质量", value: `${fmtDataValue(quality.score)} / ${qualityLevelText(quality.level)}`, html: true },
      ])}
    </div>
    ${
      sourceRows.length
        ? `<div class="data-detail-section">
            <div class="data-detail-title">证据来源分布</div>
            <div class="data-source-chip-list">
              ${sourceRows
                .map(([key, value]) => {
                  const url = sourceUrlForType(key, sourceLinks);
                  const tag = url ? "a" : "span";
                  const attrs = url ? ` href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" title="打开${escapeHtml(SOURCE_LINK_DEFS[key]?.label || SOURCE_TYPE_LABELS[key] || key)}"` : "";
                  return `<${tag}${attrs}><b>${escapeHtml(SOURCE_TYPE_LABELS[key] || key)}</b><em>${Number(value)}</em></${tag}>`;
                })
                .join("")}
            </div>
          </div>`
        : ""
    }
    ${renderSourceLinkList(sourceLinks)}
    ${
      signals.length
        ? `<div class="data-detail-section">
            <div class="data-detail-title">技术信号摘要</div>
            <div class="data-signal-list">${signals.map((x) => `<span>${escapeHtml(x)}</span>`).join("")}</div>
          </div>`
        : ""
    }
    ${
      sourceReports.length
        ? `<div class="data-detail-section">
            <div class="data-detail-title">分析师引用证据</div>
            <div class="data-report-list">
              ${sourceReports
                .map(
                  (report) => `
                    <div class="data-report-item">
                      <div><b>${escapeHtml(report.agent || "证据")}</b><span>${escapeHtml(report.sentiment || "-")} / ${fmtNumber(report.confidence)}</span></div>
                      <p>${escapeHtml(toBriefReason(report.reasoning || "", "暂无证据摘要", 120))}</p>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>`
        : ""
    }
  `;
}

function clearAdvicePanelsForGenerating() {
  currentAdvicePayload = null;
  updateAdviceReportButtonState(true);
  setAdviceProgress("info", "正在生成建议…", 15);
  const pipeline = document.getElementById("advicePipeline");
  if (pipeline) {
    pipeline.classList.add("hidden");
    pipeline.innerHTML = "";
  }
  document.getElementById("signalAction").innerHTML = "-";
  document.getElementById("signalPosition").textContent = "-";
  document.getElementById("signalConfidence").textContent = "-";
  renderMultiPeriodCards(null);
  document.getElementById("adviceSummary").innerHTML = `<div class="empty-tip">等待最终建议与风控结果...</div>`;
  document.getElementById("traceFlow").innerHTML = `<div class="empty-tip">等待技术分析、基本面分析、裁判与风控结果...</div>`;
  document.getElementById("traceAnalysts").innerHTML = `<div class="empty-tip">等待分析师观点...</div>`;
  document.getElementById("traceReferee").innerHTML = `<div class="empty-tip">等待裁判与风控结果...</div>`;
  document.getElementById("traceTimeline").innerHTML = `<div class="empty-tip">等待博弈轮次...</div>`;
  drawRadar(
    document.getElementById("agentRadarCanvas"),
    ["技术", "基本面", "裁判", "趋势"],
    [0, 0, 0, 0]
  );
  drawMiniKline(document.getElementById("miniPnlChart"), []);
  renderStabilityPanel(null);
  renderDataQualityPanel(null);
  renderAdviceDataDetails(null);
  const dataDetails = document.getElementById("adviceDataDetails");
  if (dataDetails) dataDetails.textContent = "正在获取行情、技术指标与证据来源...";
  const qaAnswer = document.getElementById("adviceQaAnswer");
  if (qaAnswer) {
    qaAnswer.className = "qa-answer empty-tip";
    qaAnswer.textContent = "建议生成中，完成后可以继续追问。";
  }
}

function renderAdviceSummary(data) {
  currentAdvicePayload = data && typeof data === "object" ? data : null;
  updateAdviceReportButtonState(false);
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
  const personalizationText = data.user_profile
    ? formatUserProfile(data.user_profile, data.personalization || {})
    : "未应用用户画像";
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
      <div class="summary-card">
        <div class="summary-title">个性化约束</div>
        <div class="summary-value">${escapeHtml(personalizationText)}</div>
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

function renderPortfolioAdvice(data = {}) {
  const host = document.getElementById("portfolioAdviceResult");
  if (!host) return;
  const holdings = Array.isArray(data.holdings) ? data.holdings : [];
  const warnings = Array.isArray(data.warnings) ? data.warnings.filter((x) => cleanText(x)) : [];
  if (!holdings.length) {
    host.className = "portfolio-result empty-tip";
    host.textContent = "暂无组合建议。";
    return;
  }
  host.className = "portfolio-result";
  host.innerHTML = `
    <div class="portfolio-head">
      <span>目标现金 ${fmtPercent(data.target_cash_percent)}</span>
      <span>目标持仓合计 ${fmtPercent(data.target_total_weight_percent)}</span>
      <span>单票上限 ${fmtPercent(data.effective_max_position_per_stock)}</span>
      <span>规则库 ${data.strategy_rule_awareness?.active_rules ?? 0} 条</span>
    </div>
    ${
      warnings.length
        ? `<div class="portfolio-warnings">${warnings.map((x) => `<span>${escapeHtml(x)}</span>`).join("")}</div>`
        : ""
    }
    <table>
      <thead><tr><th>标的</th><th>建议</th><th>当前</th><th>目标</th><th>调整</th><th>依据</th></tr></thead>
      <tbody>
        ${holdings
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.ticker || "-")}</td>
                <td>${actionTag(row.action || "HOLD")}</td>
                <td>${fmtPercent(row.current_weight_percent)}</td>
                <td>${fmtPercent(row.target_weight_percent)}</td>
                <td>${escapeHtml(row.plan || "-")}</td>
                <td>${escapeHtml(toBriefReason(row.reason || "", "暂无个股依据", 80))}${row.strategy_rule_adjusted ? "；已应用错误规则库校准" : ""}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
    <div class="qa-limitations">${escapeHtml(data.limitations || "")}</div>
  `;
}

function renderMultiPeriodCards(data) {
  const host = document.getElementById("multiPeriodCards");
  if (!host) return;
  if (!data || typeof data !== "object") {
    host.innerHTML = `<div class="empty-tip">暂无多周期建议</div>`;
    return;
  }
  const multi = normalizeMultiPeriodAdvice(data);
  const shortTerm = multi.short_term || {};
  const swingTerm = multi.swing_term || {};
  const riskPlan = multi.risk_plan || {};
  const invalidConditions = Array.isArray(riskPlan.invalid_conditions)
    ? riskPlan.invalid_conditions.filter((x) => cleanText(x))
    : [];

  host.innerHTML = `
    <div class="multi-period-card period-${normalizeAction(shortTerm.action).toLowerCase()}">
      <div class="period-card-head">
        <div>
          <div class="period-title">短线</div>
          <div class="period-horizon">${escapeHtml(shortTerm.horizon || "1-3 trading days")}</div>
        </div>
        ${actionTag(shortTerm.action)}
      </div>
      <div class="period-confidence">
        <span>置信度</span>
        <b>${fmtNumber(shortTerm.confidence)}</b>
      </div>
      <div class="period-reason">${escapeHtml(shortTerm.reason || "暂无短线理由")}</div>
    </div>
    <div class="multi-period-card period-${normalizeAction(swingTerm.action).toLowerCase()}">
      <div class="period-card-head">
        <div>
          <div class="period-title">中线</div>
          <div class="period-horizon">${escapeHtml(swingTerm.horizon || "2-4 weeks")}</div>
        </div>
        ${actionTag(swingTerm.action)}
      </div>
      <div class="period-confidence">
        <span>置信度</span>
        <b>${fmtNumber(swingTerm.confidence)}</b>
      </div>
      <div class="period-reason">${escapeHtml(swingTerm.reason || "暂无中线理由")}</div>
    </div>
    <div class="multi-period-card risk-period-card">
      <div class="period-card-head">
        <div>
          <div class="period-title">风控</div>
          <div class="period-horizon">risk plan</div>
        </div>
        <span class="tag tag-hold">PLAN</span>
      </div>
      <div class="risk-plan-row">
        <span>止损</span>
        <b>${escapeHtml(riskPlan.stop_loss || "暂无止损计划")}</b>
      </div>
      <div class="risk-plan-row">
        <span>止盈</span>
        <b>${escapeHtml(riskPlan.take_profit || "暂无止盈计划")}</b>
      </div>
      <div class="invalid-list">
        ${
          invalidConditions.length
            ? invalidConditions.map((x) => `<span>${escapeHtml(x)}</span>`).join("")
            : `<span>暂无失效条件</span>`
        }
      </div>
    </div>
  `;
}

function renderAdviceQaAnswer(data = {}) {
  const host = document.getElementById("adviceQaAnswer");
  if (!host) return;
  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  const followUps = Array.isArray(data.follow_up_questions) ? data.follow_up_questions : [];
  const confidence = Number(data.confidence);
  const rag = data.rag_status || {};
  const ragText = rag.enabled
    ? `RAG: ${rag.status || "ok"} / ${rag.documents ?? 0}条`
    : "RAG: 未启用";
  const answerMode =
    data.answer_mode === "local_fast" ? "本地快答" : data.llm_fallback ? "本地兜底" : "LLM辅助";
  host.className = "qa-answer";
  host.innerHTML = `
    <div class="qa-answer-head">
      <span>${escapeHtml(data.question || "追问")}</span>
      <span>置信度 ${Number.isFinite(confidence) ? confidence.toFixed(2) : "-"}</span>
    </div>
    <div class="qa-answer-body">${escapeHtml(data.answer || "暂无回答")}</div>
    <div class="qa-meta">${escapeHtml(data.generated_at || "")} | ${escapeHtml(ragText)} | ${escapeHtml(answerMode)}</div>
    ${
      evidence.length
        ? `<div class="qa-section-title">引用证据</div>
           <div class="qa-evidence-list">
             ${evidence
               .map(
                 (item) => `
                   <div class="qa-evidence-item">
                     <b>${escapeHtml(item.source || "证据")}</b>
                     <span>${escapeHtml(item.text || "")}</span>
                   </div>
                 `
               )
               .join("")}
           </div>`
        : ""
    }
    ${
      followUps.length
        ? `<div class="qa-section-title">可以继续问</div>
           <div class="qa-followups">
             ${followUps
               .map((q) => `<button class="qa-followup-btn" data-question="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
               .join("")}
           </div>`
        : ""
    }
    ${data.limitations ? `<div class="qa-limitations">${escapeHtml(data.limitations)}</div>` : ""}
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

  drawRadar(
    document.getElementById("agentRadarCanvas"),
    ["技术", "基本面", "裁判", "趋势"],
    [technicalConf, fundamentalConf, refereeConf, trendStrength]
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

async function checkLlmHealth() {
  try {
    return await api("/api/llm/health", { timeout_ms: 12000 });
  } catch (e) {
    return {
      ok: false,
      status: "api_error",
      error: e.message || "LLM健康检查接口不可用",
    };
  }
}

function formatLlmHealthMessage(health = {}) {
  const model = health.model_name ? `模型 ${health.model_name}` : "当前模型";
  const base = health.base_url ? `接口 ${health.base_url}` : "当前接口";
  const err = cleanText(health.error || health.status || "未知错误");
  return `${model} / ${base} 连接检查未通过：${err}`;
}

function getLlmFallbackCount(advice = {}) {
  const diagnostics = advice.stability_diagnostics || {};
  const count = Number(diagnostics.rule_fallback_count || 0);
  return Number.isFinite(count) ? count : 0;
}

async function runAdvice() {
  setStatus("生成建议中...");
  setAdviceBusy(true, "生成中...", "请稍候...");
  setAdviceProgress("info", "正在检查后端连接…", 5);
  const backendOk = await ensureBackendReady();
  if (!backendOk) {
    setAdviceFeedback(
      "error",
      `后端未运行（${API_BASE}）。请在项目目录终端执行 bash start_web.sh；公网访问时使用 PUBLIC_WEB=1 并配置公网域名或穿透地址。`
    );
    setAdviceBusy(false);
    setStatus("后端未连接", "error");
    return;
  }
  setAdviceProgress("info", "正在检查 LLM 连接…", 8);
  const llmHealth = await checkLlmHealth();
  if (!llmHealth.ok) {
    setStatus(`LLM连接警告：${formatLlmHealthMessage(llmHealth)}`, "warning");
  }
  setAdviceProgress("info", "正在提交建议任务…", 12);
  clearAdvicePanelsForGenerating();
  try {
    const payload = {
      ticker: document.getElementById("adviceTicker").value.trim(),
      debate_depth: Number(document.getElementById("adviceDepth").value || 2),
      user_profile: buildStockProfilePayload(),
      personalization: {
        profile: buildStockProfilePayload(),
        preferences: buildStockPreferencesPayload(),
      },
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
    renderMultiPeriodCards(advice);
    renderAdviceSummary(advice);
    renderAdviceDataDetails(advice);
    await renderTraceFromAdvice(advice);
    const fallbackCount = getLlmFallbackCount(advice);
    if (fallbackCount > 0) {
      setAdviceFeedback(
        "warning",
        `生成完成，但 LLM 连接发生降级（${fallbackCount}处规则兜底）。请检查 /api/llm/health 后再采信。`
      );
      setStatus("建议生成完成，但 LLM 已降级", "warning");
    } else {
      setAdviceFeedback("success", `生成成功：${padTicker(advice.ticker || payload.ticker)} 建议已更新。`);
      setStatus("建议生成完成");
    }
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
    renderMultiPeriodCards(advice);
    renderAdviceSummary(advice);
    renderAdviceDataDetails(advice);
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

async function askAdviceQuestion(questionOverride = "") {
  if (adviceAskSubmitting) return;
  const input = document.getElementById("adviceQuestionInput");
  const rawQuestion = questionOverride || (input ? input.value : "");
  const question = cleanText(rawQuestion);
  const ticker = normalizeTicker(
    (currentAdvicePayload && currentAdvicePayload.ticker) || document.getElementById("adviceTicker")?.value || ""
  );
  if (!ticker) {
    setAdviceQaFeedback("warning", "请先输入股票代码。");
    return;
  }
  if (!question) {
    setAdviceQaFeedback("warning", "请输入想追问的问题。");
    return;
  }
  setAdviceQaBusy(true);
  setAdviceQaFeedback("info", "正在结合最新建议回答...");
  try {
    const data = await api("/api/advice/ask", {
      method: "POST",
      body: JSON.stringify({
        ticker,
        question,
        include_latest_advice: true,
        include_rag: false,
      }),
      timeout_ms: ADVICE_QA_TIMEOUT_MS,
    });
    renderAdviceQaAnswer(data);
    if (input && !questionOverride) input.value = "";
    setAdviceQaFeedback("success", `已回答：${ticker}`);
    setStatus("追问回答已生成");
  } catch (e) {
    setAdviceQaFeedback("error", `回答失败：${e.message}`);
    throw e;
  } finally {
    setAdviceQaBusy(false);
  }
}

async function runCounterfactualSimulation() {
  const ticker = normalizeTicker(
    (currentAdvicePayload && currentAdvicePayload.ticker) || document.getElementById("adviceTicker")?.value || ""
  );
  const host = document.getElementById("simulationResult");
  if (!ticker) {
    if (host) host.textContent = "请先输入股票代码。";
    return;
  }
  if (host) {
    host.className = "simulation-result empty-tip";
    host.textContent = "正在推演情景...";
  }
  const scenario = {
    price_change_pct: Number(document.getElementById("simPriceChange")?.value || 0),
    market_index_change_pct: Number(document.getElementById("simMarketChange")?.value || 0),
    volume_change: document.getElementById("simVolumeChange")?.value || "",
    news_sentiment: document.getElementById("simNewsSentiment")?.value || "",
    user_position_percent: Number(document.getElementById("simUserPosition")?.value || 0),
    breaks_ma20: !!document.getElementById("simBreaksMa20")?.checked,
    breaks_stop_loss: !!document.getElementById("simBreaksStopLoss")?.checked,
  };
  const data = await api("/api/advice/simulate", {
    method: "POST",
      body: JSON.stringify({
        ticker,
        scenario,
      }),
    timeout_ms: ADVICE_API_TIMEOUT_MS,
  });
  if (!host) return;
  host.className = "simulation-result";
  host.innerHTML = `
    <div class="simulation-head">
      <span>原建议 ${actionTag(data.original_action)} ${fmtPercent(data.original_position_percent)}</span>
      <span>推演后 ${actionTag(data.simulated_action)} ${fmtPercent(data.simulated_position_percent)}</span>
    </div>
    <div class="simulation-body">${escapeHtml(data.reason || "")}</div>
    <div class="qa-meta">变化：${data.changed ? "是" : "否"} | 推演置信度 ${fmtNumber(data.simulated_confidence)} | 分数变化 ${fmtNumber(data.scenario_score_delta, 3)}</div>
    <div class="qa-limitations">${escapeHtml(data.risk_note || "")}</div>
  `;
  setStatus("反事实推演完成");
}

function fillCurrentTickerPortfolio() {
  const ticker = normalizeTicker(
    (currentAdvicePayload && currentAdvicePayload.ticker) || document.getElementById("adviceTicker")?.value || ""
  );
  if (!ticker) {
    setStatus("请先输入当前标的。", "warning");
    return;
  }
  const profile = buildUserProfilePayload();
  const currentPosition = Number(profile.current_position_percent || 0);
  const costPrice = Number(profile.cost_price || 0);
  const input = document.getElementById("portfolioHoldingsInput");
  if (!input) return;
  const nextLine = `${ticker}, ${Number.isFinite(currentPosition) ? currentPosition : 0}, ${Number.isFinite(costPrice) ? costPrice : 0}`;
  input.value = input.value.trim() ? `${input.value.trim()}\n${nextLine}` : nextLine;
  setStatus("已填入当前标的");
}

async function runPortfolioAdvice() {
  const host = document.getElementById("portfolioAdviceResult");
  const input = document.getElementById("portfolioHoldingsInput");
  const holdings = parsePortfolioHoldings(input ? input.value : "");
  if (!holdings.length) {
    if (host) {
      host.className = "portfolio-result empty-tip";
      host.textContent = "请至少输入一只股票代码和当前权重。";
    }
    setStatus("组合建议缺少持仓输入", "warning");
    return;
  }
  if (host) {
    host.className = "portfolio-result empty-tip";
    host.textContent = "正在生成组合建议...";
  }
  const cashPercent = Number(document.getElementById("portfolioCashPercent")?.value || 0);
  const payload = {
    holdings,
    cash_percent: Number.isFinite(cashPercent) ? cashPercent : 0,
    objective: document.getElementById("portfolioObjective")?.value || "balanced",
  };
  const data = await api("/api/portfolio/advice", {
    method: "POST",
    body: JSON.stringify(payload),
    timeout_ms: ADVICE_API_TIMEOUT_MS,
  });
  renderPortfolioAdvice(data);
  setStatus("组合建议已生成");
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

async function settleAdviceNow(horizon = "") {
  if (settleAdviceSubmitting) return;
  const horizonValue = String(horizon || "").trim();
  const horizonDays = horizonValue ? Number(horizonValue) : null;
  activeSettleHorizon = horizonValue;
  settleAdviceSubmitting = true;
  updateSettleAdviceButtonState();
  setStatus(horizonDays ? `提交 T+${horizonDays} 建议兑现任务...` : "提交全部周期建议兑现任务...");
  const submitGuard = window.setTimeout(() => {
    if (!settleAdviceSubmitting) return;
    settleAdviceSubmitting = false;
    activeSettleHorizon = "";
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
      body: JSON.stringify({
        max_items: 2000,
        horizons: horizonDays ? [horizonDays] : null,
      }),
    });
    setStatus(`${horizonDays ? `T+${horizonDays}` : "全部周期"}建议兑现任务已提交: ${data.task_id}`);
    await refreshTasks();
    await refreshOverview();
  } finally {
    window.clearTimeout(submitGuard);
    settleAdviceSubmitting = false;
    activeSettleHorizon = "";
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
    await loadUserProfileFromServer();
    await loadUserAdviceHistoryFromServer();
    await refreshOverview();
  } catch (e) {
    errorEl.textContent = `注册失败：${e.message}`;
    throw e;
  }
}

async function submitLogin() {
  if (loginMode === MODE_ADMIN) {
    const password = document.getElementById("loginAdminPasswordInput").value || "";
    const errorEl = document.getElementById("loginError");
    if (!password.trim()) {
      errorEl.textContent = "请输入管理员密码。";
      return;
    }
    errorEl.textContent = "";
    try {
      await api("/api/admin/users", {
        method: "GET",
        headers: {
          "X-Admin-Password": password,
        },
      });
      window.localStorage.setItem(ADMIN_PASSWORD_KEY, password);
      saveAuthToken("");
      saveCurrentUser(buildAdminUser());
      setAuthRequiredView(true);
      applyMode(MODE_ADMIN, "overview");
      closeLoginModal();
      setStatus("管理员登录成功");
      await refreshOverview();
    } catch (e) {
      errorEl.textContent = `管理员登录失败：${e.message}`;
      throw e;
    }
    return;
  }

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
    await loadUserProfileFromServer();
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
  if (isAdminCurrentUser()) {
    clearSessionState();
    return false;
  }
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
    await loadUserProfileFromServer();
    await loadStockPersonalizationFromServer({ silent: true }).catch(() => {});
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
bindEvent("refreshQualityRankingBtn", "click", () => refreshQualityRanking().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("refreshStrategyRulesBtn", "click", () => refreshStrategyRules().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("adviceTickerSearchInput", "input", (e) => {
  overviewAdviceFilter.ticker = normalizeTickerSearch(e.target.value);
  renderOverviewAdviceTable();
});
bindEvent("clearAdviceFiltersBtn", "click", () => {
  overviewAdviceFilter = { action: "ALL", ticker: "" };
  renderOverviewAdviceTable();
});
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
bindEvent("loadStockPersonalizationBtn", "click", () => loadStockPersonalizationFromServer().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("saveStockPersonalizationBtn", "click", () => saveStockPersonalizationToServer().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("addCustomStrategyBtn", "click", () => addCustomStrategy());
bindEvent("addCustomConstraintBtn", "click", () => addCustomConstraint());
bindEvent("adviceTicker", "change", () => loadStockPersonalizationFromServer({ silent: true }).catch(() => {}));
bindEvent("stockUseSystemRules", "change", () => renderStockRuleChoices());
bindEvent("downloadAdviceReportBtn", "click", () =>
  downloadAdviceReport({ generatedAt: (currentAdvicePayload && currentAdvicePayload.generated_at) || "", useLlm: true }).catch((e) =>
    setStatus(`错误: ${e.message}`)
  )
);
bindEvent("askAdviceBtn", "click", () => askAdviceQuestion().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("runSimulationBtn", "click", () => runCounterfactualSimulation().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("runPortfolioAdviceBtn", "click", () => runPortfolioAdvice().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("fillCurrentTickerPortfolioBtn", "click", () => fillCurrentTickerPortfolio());
bindEvent("adviceQuestionInput", "keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    askAdviceQuestion().catch((err) => setStatus(`错误: ${err.message}`));
  }
});
bindEvent("runEvolveBtn", "click", () => runEvolution().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("settleAdviceBtn", "click", () => settleAdviceNow("").catch((e) => setStatus(`错误: ${e.message}`)));
document.querySelectorAll(".settle-horizon-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    settleAdviceNow(btn.dataset.horizon || "").catch((e) => setStatus(`错误: ${e.message}`));
  });
});
document.addEventListener("click", (e) => {
  const btn = e.target && e.target.closest ? e.target.closest(".download-advice-report-btn") : null;
  if (!btn) return;
  downloadAdviceReport({
    ticker: btn.dataset.ticker || "",
    generatedAt: btn.dataset.generatedAt || "",
    useLlm: true,
  }).catch((err) => setStatus(`错误: ${err.message}`));
});
document.addEventListener("click", (e) => {
  const btn = e.target && e.target.closest ? e.target.closest(".constraint-remove-btn") : null;
  if (!btn) return;
  const idx = Number(btn.dataset.constraintIndex);
  if (!Number.isInteger(idx)) return;
  customConstraintsDraft.splice(idx, 1);
  renderCustomConstraintList();
});
document.addEventListener("click", (e) => {
  const btn = e.target && e.target.closest ? e.target.closest(".strategy-remove-btn") : null;
  if (!btn) return;
  const idx = Number(btn.dataset.strategyIndex);
  if (!Number.isInteger(idx)) return;
  customStrategiesDraft.splice(idx, 1);
  renderCustomStrategyList();
});
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
bindEvent("loginModeUserBtn", "click", () => setLoginMode(MODE_USER));
bindEvent("loginModeAdminBtn", "click", () => setLoginMode(MODE_ADMIN));
bindEvent("loginSubmitBtn", "click", () => submitLogin().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("loginCancelBtn", "click", closeLoginModal);
bindEvent("loginPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") submitLogin().catch((err) => setStatus(`错误: ${err.message}`));
});
bindEvent("loginAdminPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") submitLogin().catch((err) => setStatus(`错误: ${err.message}`));
});
bindEvent("logoutBtn", "click", () => logout().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("userAvatarBtn", "click", openSettingsModal);
bindEvent("settingsCloseBtn", "click", closeSettingsModal);
bindEvent("settingsLogoutBtn", "click", () => logout().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("saveAccountSettingsBtn", "click", () => saveAccountSettings().catch((e) => setSettingsError(`保存失败：${e.message}`)));
bindEvent("uploadAvatarBtn", "click", () => uploadAvatarFromSettings().catch((e) => setSettingsError(`上传失败：${e.message}`)));
bindEvent("changePasswordBtn", "click", () => changePasswordFromSettings().catch((e) => setSettingsError(`修改失败：${e.message}`)));
bindEvent("saveSettingsProfileBtn", "click", () => saveSettingsProfileToServer().catch((e) => setSettingsError(`保存失败：${e.message}`)));
bindEvent("refreshUsersBtn", "click", () => refreshAdminUsers().catch((e) => setStatus(`错误: ${e.message}`)));
bindEvent("modeSwitchBtn", "click", () => {
  openSettingsModal();
});
bindEvent("adminSubmitBtn", "click", verifyAdmin);
bindEvent("adminCancelBtn", "click", closeAdminModal);
bindEvent("adminPasswordInput", "keydown", (e) => {
  if (e.key === "Enter") verifyAdmin();
});

document.querySelectorAll(".advice-filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    overviewAdviceFilter.action = String(btn.dataset.adviceAction || "ALL").toUpperCase();
    renderOverviewAdviceTable();
  });
});

document.querySelectorAll(".qa-quick-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const question = String(btn.dataset.question || btn.textContent || "").trim();
    const input = document.getElementById("adviceQuestionInput");
    if (input) input.value = question;
    askAdviceQuestion(question).catch((e) => setStatus(`错误: ${e.message}`));
  });
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement) || !target.classList.contains("qa-followup-btn")) return;
  const question = String(target.dataset.question || target.textContent || "").trim();
  const input = document.getElementById("adviceQuestionInput");
  if (input) input.value = question;
  askAdviceQuestion(question).catch((e) => setStatus(`错误: ${e.message}`));
});

async function bootstrap() {
  try {
    updateSettleAdviceButtonState();
    updateTrainSourceText();
    loadCurrentUser();
    loadAuthToken();
    applyUserProfileToForm((currentUser && currentUser.user_profile) || defaultUserProfile());
    applyStockPersonalizationToForm({
      ticker: document.getElementById("adviceTicker")?.value || "",
      profile: (currentUser && currentUser.user_profile) || defaultUserProfile(),
      preferences: {},
    });
    await loadStockRuleChoices();
    const isLoggedIn = await validateSession();
    updateUserBadge();
    setAuthRequiredView(isLoggedIn);
    if (!isLoggedIn) {
      applyUserProfileToForm(defaultUserProfile());
      applyStockPersonalizationToForm({ ticker: document.getElementById("adviceTicker")?.value || "", profile: defaultUserProfile(), preferences: {} });
      updateProfileStatus("登录后自动套用账户默认画像");
      return;
    }
    if (isAdminCurrentUser()) {
      applyMode(MODE_ADMIN, "overview");
    } else {
      await loadUserAdviceHistoryFromServer();
      applyMode(MODE_USER, "advice");
    }
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

from .base import BaseDataProvider
import akshare as ak
import datetime
import json
import os
import requests
import time


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINANCIAL_CACHE_PATH = os.path.join(BASE_DIR, "data", "json", "financial_cache.json")
SMART_MONEY_CACHE_PATH = os.path.join(BASE_DIR, "data", "json", "smart_money_cache.json")


def _env_int(name: str, default: int, lower: int = 1) -> int:
    try:
        return max(lower, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except ValueError:
        return default


AKSHARE_TIMEOUT_SECONDS = _env_int("AKSHARE_TIMEOUT_SECONDS", 8, lower=3)
_ORIGINAL_REQUEST = requests.sessions.Session.request


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", AKSHARE_TIMEOUT_SECONDS)
    kwargs.setdefault("proxies", {"http": None, "https": None, "all": None})
    return _ORIGINAL_REQUEST(self, method, url, **kwargs)


if not getattr(requests.sessions.Session.request, "_tradingagents_timeout", False):
    _request_with_timeout._tradingagents_timeout = True
    requests.sessions.Session.request = _request_with_timeout


def _load_financial_cache() -> dict:
    try:
        if os.path.exists(FINANCIAL_CACHE_PATH):
            with open(FINANCIAL_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_financial_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(FINANCIAL_CACHE_PATH), exist_ok=True)
        with open(FINANCIAL_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_json_cache(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    return {}


def _save_json_cache(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _retry_call(func, retries: int = 1, delay: float = 0.6):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
        if attempt < retries:
            time.sleep(delay * (attempt + 1))
    raise last_error


def _date_to_int(value, default: int = 0) -> int:
    try:
        text = str(value or "")[:10]
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 8:
            return int(digits[:8])
    except Exception:
        pass
    return default


def _exchange_symbol(ticker: str) -> str:
    ticker = str(ticker).strip().zfill(6)
    return ("SH" if ticker.startswith("6") else "SZ") + ticker


def _clean_value(value) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _first_existing(row, names: tuple[str, ...]) -> str:
    for name in names:
        if name in row:
            value = _clean_value(row.get(name))
            if value:
                return value
    return ""


def _latest_rows(df, limit: int = 3):
    if df is None or getattr(df, "empty", True):
        return []
    try:
        return df.tail(limit).to_dict("records")
    except Exception:
        return []


def _find_column(columns, candidates: tuple[str, ...]) -> str:
    names = [str(col) for col in columns]
    for candidate in candidates:
        if candidate in names:
            return candidate
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found:
            return found
    return ""


def _safe_float(value, default: float = 0.0) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if not text or text.lower() in {"nan", "none", "nat"}:
            return default
        return float(text)
    except Exception:
        return default


def _compact_row(row: dict, *, max_fields: int = 7) -> str:
    if not isinstance(row, dict):
        return ""
    preferred = [
        "报告期",
        "公告日期",
        "日期",
        "营业收入",
        "营业总收入",
        "归母净利润",
        "净利润",
        "扣非净利润",
        "经营现金流量净额",
        "资产负债率",
        "净资产收益率",
        "每股收益",
        "每股净资产",
        "毛利率",
        "净利率",
    ]
    pairs = []
    used = set()
    for key in preferred:
        if key in row:
            value = _clean_value(row.get(key))
            if value:
                pairs.append(f"{key}={value}")
                used.add(key)
        if len(pairs) >= max_fields:
            break
    if len(pairs) < max_fields:
        for key, value in row.items():
            if key in used:
                continue
            clean = _clean_value(value)
            if clean:
                pairs.append(f"{key}={clean}")
            if len(pairs) >= max_fields:
                break
    return "，".join(pairs)


def _filter_by_cutoff(df, cutoff_date: str | None, date_columns: tuple[str, ...]):
    if df is None or getattr(df, "empty", True) or not cutoff_date:
        return df
    cutoff_int = _date_to_int(cutoff_date, default=99991231)
    try:
        for col in date_columns:
            if col in df.columns:
                return df[df[col].apply(lambda x: _date_to_int(x) <= cutoff_int)]
    except Exception:
        return df
    return df


def _find_ticker_row(df, ticker: str):
    if df is None or getattr(df, "empty", True):
        return None
    code_col = _find_column(df.columns, ("代码", "股票代码", "证券代码", "symbol", "code"))
    if not code_col:
        return None
    target = str(ticker).strip().zfill(6)
    try:
        rows = df[df[code_col].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6) == target]
        if rows is not None and not rows.empty:
            return rows.iloc[0].to_dict()
    except Exception:
        return None
    return None


def _smart_money_rank_summary(row: dict, ticker: str, source: str) -> str:
    name = _first_existing(row, ("名称", "股票简称", "简称", "name"))
    latest = _first_existing(row, ("最新价", "收盘价", "价格", "close"))
    pct = _first_existing(row, ("涨跌幅", "涨跌幅%", "今日涨跌幅", "pct_change"))
    main_col = _find_column(
        row.keys(),
        (
            "主力净流入",
            "主力净流入-净额",
            "今日主力净流入",
            "5日主力净流入",
            "10日主力净流入",
            "净额",
            "主力净额",
        ),
    )
    main_pct_col = _find_column(
        row.keys(),
        ("主力净占比", "主力净流入净占比", "今日主力净占比", "5日主力净占比", "10日主力净占比"),
    )
    rank = _first_existing(row, ("序号", "排名", "rank"))
    main_value = row.get(main_col, "") if main_col else ""
    main_pct = row.get(main_pct_col, "") if main_pct_col else ""
    net = _safe_float(main_value)
    direction = "净流入" if net > 0 else "净流出" if net < 0 else "均衡"
    return (
        f"数据来源: {source}。股票={ticker}{'/' + name if name else ''}，"
        f"主力资金{direction}，主力净流入={main_value or '未知'}，主力净占比={main_pct or '未知'}，"
        f"最新价={latest or '未知'}，涨跌幅={pct or '未知'}，排名={rank or '未知'}。"
    )


def _limit_join(items: list[str], limit: int = 6) -> str:
    cleaned = [x for x in (_clean_value(item) for item in items) if x]
    return " | ".join(cleaned[:limit])


def _call_ak(name: str, *args, **kwargs):
    func = getattr(ak, name, None)
    if func is None:
        raise RuntimeError(f"akshare missing {name}")
    return _retry_call(lambda: func(*args, **kwargs))


def _build_financial_summary(ticker: str, cutoff_date: str | None = None) -> tuple[list[str], list[str], dict]:
    ticker = str(ticker).strip().zfill(6)
    source_notes: list[str] = []
    errors: list[str] = []
    cache_fields: dict = {}

    try:
        info_df = _call_ak("stock_individual_info_em", symbol=ticker)
        info_dict = dict(zip(info_df["item"], info_df["value"])) if info_df is not None and not info_df.empty else {}
        industry = _clean_value(info_dict.get("行业"))
        market_cap = _clean_value(info_dict.get("总市值"))
        if industry or market_cap:
            source_notes.append(f"个股资料: 所属行业={industry or '未知'}，总市值={market_cap or '未知'}")
            cache_fields.update({"industry": industry or "未知", "market_cap": market_cap or "未知"})
    except Exception as exc:
        errors.append(f"个股资料失败: {exc}")

    valuation_parts = []
    for indicator, key in [("市盈率(TTM)", "pe"), ("市净率", "pb")]:
        try:
            df = _call_ak("stock_zh_valuation_baidu", symbol=ticker, indicator=indicator, period="近一年")
            value = ""
            if df is not None and not df.empty:
                df = _filter_by_cutoff(df, cutoff_date, ("date", "日期"))
                if df is not None and not df.empty:
                    value = _clean_value(df["value"].iloc[-1]) if "value" in df.columns else _compact_row(df.iloc[-1].to_dict(), max_fields=2)
            if value:
                valuation_parts.append(f"{indicator}={value}")
                cache_fields[key] = value
        except Exception as exc:
            errors.append(f"{indicator}失败: {exc}")
    if valuation_parts:
        source_notes.append("估值: " + "，".join(valuation_parts))

    financial_sources = [
        ("财务摘要", "stock_financial_abstract", {"symbol": ticker}, ("报告期", "截止日期", "公告日期", "日期")),
        ("财务指标", "stock_financial_analysis_indicator", {"symbol": ticker, "start_year": str(datetime.datetime.now().year - 5)}, ("日期", "报告期", "截止日期")),
        ("利润表", "stock_profit_sheet_by_report_em", {"symbol": _exchange_symbol(ticker)}, ("公告日期", "REPORT_DATE", "报告期", "日期")),
        ("资产负债表", "stock_balance_sheet_by_report_em", {"symbol": _exchange_symbol(ticker)}, ("公告日期", "REPORT_DATE", "报告期", "日期")),
        ("现金流量表", "stock_cash_flow_sheet_by_report_em", {"symbol": _exchange_symbol(ticker)}, ("公告日期", "REPORT_DATE", "报告期", "日期")),
    ]
    for label, func_name, kwargs, date_cols in financial_sources:
        try:
            df = _call_ak(func_name, **kwargs)
            df = _filter_by_cutoff(df, cutoff_date, date_cols)
            rows = _latest_rows(df, limit=1)
            summary = _compact_row(rows[-1], max_fields=7) if rows else ""
            if summary:
                source_notes.append(f"{label}: {summary}")
                cache_fields[label] = summary
        except Exception as exc:
            errors.append(f"{label}失败: {exc}")

    return source_notes, errors, cache_fields


def _build_sentiment_summary(ticker: str, cutoff_date: str | None = None) -> tuple[list[str], list[str]]:
    ticker = str(ticker).strip().zfill(6)
    source_notes: list[str] = []
    errors: list[str] = []

    try:
        news_df = _call_ak("stock_news_em", symbol=ticker)
        news_df = _filter_by_cutoff(news_df, cutoff_date, ("发布时间", "日期", "date"))
        titles = []
        if news_df is not None and not news_df.empty:
            for _, row in news_df.head(5).iterrows():
                title = _first_existing(row, ("新闻标题", "标题", "title"))
                if title:
                    titles.append(title)
        if titles:
            source_notes.append(f"东方财富新闻: {_limit_join(titles, 5)}")
    except Exception as exc:
        errors.append(f"东方财富新闻失败: {exc}")

    try:
        report_df = _call_ak("stock_research_report_em", symbol=ticker)
        report_df = _filter_by_cutoff(report_df, cutoff_date, ("日期", "发布时间", "公告日期"))
        reports = []
        if report_df is not None and not report_df.empty:
            for _, row in report_df.head(4).iterrows():
                name = _first_existing(row, ("报告名称", "标题", "研报名称"))
                rating = _first_existing(row, ("东财评级", "评级", "机构评级"))
                org = _first_existing(row, ("机构", "研究机构", "org"))
                if name:
                    reports.append(f"{org or '机构'} {rating or '未评级'} {name}")
        if reports:
            source_notes.append(f"券商研报: {_limit_join(reports, 4)}")
    except Exception as exc:
        errors.append(f"券商研报失败: {exc}")

    try:
        end_date = str(cutoff_date or datetime.datetime.now().strftime("%Y-%m-%d"))[:10].replace("-", "")
        start_dt = datetime.datetime.strptime(end_date, "%Y%m%d") - datetime.timedelta(days=120)
        notice_df = _call_ak(
            "stock_zh_a_disclosure_report_cninfo",
            symbol=ticker,
            market="沪深京",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_date,
        )
        notice_df = _filter_by_cutoff(notice_df, cutoff_date, ("公告时间", "公告日期", "日期"))
        notices = []
        if notice_df is not None and not notice_df.empty:
            for _, row in notice_df.head(5).iterrows():
                title = _first_existing(row, ("公告标题", "标题", "公告名称", "简称"))
                if title:
                    notices.append(title)
        if notices:
            source_notes.append(f"交易所公告: {_limit_join(notices, 5)}")
    except Exception as exc:
        errors.append(f"交易所公告失败: {exc}")

    try:
        main_df = _call_ak("stock_news_main_cx")
        main_df = _filter_by_cutoff(main_df, cutoff_date, ("发布时间", "时间", "日期"))
        market_news = []
        if main_df is not None and not main_df.empty:
            for _, row in main_df.head(10).iterrows():
                title = _first_existing(row, ("标题", "新闻标题", "内容"))
                if title:
                    market_news.append(title)
        if market_news:
            source_notes.append(f"市场要闻: {_limit_join(market_news, 3)}")
    except Exception as exc:
        errors.append(f"市场要闻失败: {exc}")

    return source_notes, errors

class AkShareProvider(BaseDataProvider):
    def fetch_sentiment_data(self, ticker: str, cutoff_date: str = None):
        """返回截止到 cutoff_date 的多源舆情摘要（若 cutoff_date=None 则返回最新）。"""
        source_notes, errors = _build_sentiment_summary(ticker, cutoff_date=cutoff_date)
        if source_notes:
            error_note = f"；部分来源异常: {'; '.join(errors[:3])}" if errors else ""
            return f"数据来源: 多源舆情/公告/研报{error_note}。" + " || ".join(source_notes)
        return (
            "[舆情降级] 新闻、研报、公告等多源接口均暂不可用或无匹配记录；"
            f"本轮舆情证据不足，应降低新闻面权重。错误摘要: {'; '.join(errors[:3]) or '无'}"
        )

    def fetch_fundamental_data(self, ticker: str, cutoff_date: str = None):
        cache = _load_financial_cache()
        cached = cache.get(str(ticker), {})
        if cached and str(os.getenv("REFRESH_FINANCIAL_LIVE", "0")).strip() != "1":
            source_count = int(cached.get("source_count", 0) or 0)
            summaries = cached.get("summaries") if isinstance(cached.get("summaries"), list) else []
            if summaries:
                return f"数据来源: 本地多源缓存（{source_count or len(summaries)}类）。" + " || ".join(summaries[:8])
            return (
                "数据来源: 本地缓存。"
                f"所属行业: {cached.get('industry', '未知')}, "
                f"总市值: {cached.get('market_cap', '未知')}, "
                f"动态市盈率(PE): {cached.get('pe', '未知')}, 市净率(PB): {cached.get('pb', '未知')}"
            )

        summaries, errors, cache_fields = _build_financial_summary(ticker, cutoff_date=cutoff_date)
        if summaries:
            cache[str(ticker)] = {
                **cached,
                **cache_fields,
                "summaries": summaries,
                "source_count": len(summaries),
                "updated_cutoff": str(cutoff_date)[:10] if cutoff_date else "latest",
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_financial_cache(cache)
            error_note = f"，部分来源异常: {'; '.join(errors[:3])}" if errors else ""
            return f"数据来源: 实时多源基本面（{len(summaries)}类）{error_note}。" + " || ".join(summaries[:8])

        if cached:
            return (
                f"数据来源: 本地缓存；实时多源接口暂不可用: {'; '.join(errors[:3])}。"
                f"所属行业: {cached.get('industry', '未知')}, 总市值: {cached.get('market_cap', '未知')}, "
                f"动态市盈率(PE): {cached.get('pe', '未知')}, 市净率(PB): {cached.get('pb', '未知')}"
            )

        return (
            "[基本面降级] 个股资料、估值、财务摘要、利润表、资产负债表、现金流表等多源接口均暂不可用；"
            f"本轮基本面证据不足，应降低基本面权重并倾向 neutral。错误摘要: {'; '.join(errors[:4]) or '无'}"
        )

    def fetch_macro_data(self, cutoff_date: str = None):
        """返回截止到 cutoff_date 的上证指数近3个交易日表现。"""
        try:
            sh_df = ak.stock_zh_index_daily(symbol="sh000001")
            if cutoff_date:
                try:
                    cutoff_dt = cutoff_date[:10]
                    sh_df['date_str'] = sh_df['date'].astype(str)
                    sh_df = sh_df[sh_df['date_str'] <= cutoff_dt]
                except Exception:
                    pass

            recent_sh = sh_df.tail(3)[['date', 'close', 'volume']].to_dict('records')
            return f"上证指数最近3个交易日表现: {recent_sh}"
        except Exception as e:
            return f"上证大盘数据抓取报错: {e}"

    def fetch_smart_money_data(self, ticker: str, cutoff_date: str = None):
        ticker = str(ticker).strip().zfill(6)
        cache = _load_json_cache(SMART_MONEY_CACHE_PATH)
        cached = cache.get(ticker, {})
        errors: list[str] = []

        try:
            market = "sh" if str(ticker).startswith("6") else "sz"
            fund_df = _call_ak("stock_individual_fund_flow", stock=ticker, market=market)
            if cutoff_date:
                try:
                    cutoff_dt = cutoff_date[:10]
                    date_col = _find_column(fund_df.columns, ("date", "日期"))
                    if date_col:
                        fund_df["date_str"] = fund_df[date_col].astype(str)
                        fund_df = fund_df[fund_df["date_str"] <= cutoff_dt]
                except Exception:
                    pass

            if fund_df is not None and not fund_df.empty:
                close_col = _find_column(fund_df.columns, ("收盘价", "最新价", "close"))
                main_col = _find_column(
                    fund_df.columns,
                    ("主力净流入-净额", "主力净流入净额", "主力净流入", "主力净额", "主力资金净流入", "main_net_inflow"),
                )
                pct_col = _find_column(fund_df.columns, ("涨跌幅", "涨跌幅%", "pct_change"))
                date_col = _find_column(fund_df.columns, ("date", "日期"))
                wanted = [col for col in (date_col, close_col, main_col, pct_col) if col]
                if main_col and wanted:
                    recent_fund = fund_df.tail(3)[wanted].to_dict("records")
                    net_values = [_safe_float(row.get(main_col)) for row in recent_fund]
                    net_sum = round(sum(net_values), 2)
                    direction = "净流入" if net_sum > 0 else "净流出" if net_sum < 0 else "均衡"
                    summary = (
                        f"数据来源: akshare.stock_individual_fund_flow。"
                        f"近{len(recent_fund)}个交易日主力资金合计{direction}约{net_sum}元，明细: {recent_fund}"
                    )
                    cache[ticker] = {
                        "summary": summary,
                        "updated_cutoff": str(cutoff_date)[:10] if cutoff_date else "latest",
                        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _save_json_cache(SMART_MONEY_CACHE_PATH, cache)
                    return summary
                errors.append(f"主力资金字段缺失: columns={list(fund_df.columns)[:12]}")
            else:
                errors.append("主力资金接口返回空表")
        except Exception as exc:
            errors.append(f"stock_individual_fund_flow失败: {exc}")

        if cached.get("summary") and str(os.getenv("REFRESH_SMART_MONEY_LIVE", "0")).strip() != "1":
            return (
                f"数据来源: 本地主力资金缓存（{cached.get('updated_cutoff', 'latest')}）。"
                f"{cached.get('summary')}；实时接口异常: {'; '.join(errors[:2])}"
            )

        backup_sources = [
            ("stock_main_fund_flow", {"symbol": "全部股票"}, "akshare.stock_main_fund_flow"),
            ("stock_individual_fund_flow_rank", {"indicator": "今日"}, "akshare.stock_individual_fund_flow_rank(今日)"),
            ("stock_individual_fund_flow_rank", {"indicator": "5日"}, "akshare.stock_individual_fund_flow_rank(5日)"),
            ("stock_individual_fund_flow_rank", {"indicator": "10日"}, "akshare.stock_individual_fund_flow_rank(10日)"),
        ]
        for func_name, kwargs, source_name in backup_sources:
            try:
                rank_df = _call_ak(func_name, **kwargs)
                row = _find_ticker_row(rank_df, ticker)
                if row:
                    summary = _smart_money_rank_summary(row, ticker, source_name)
                    cache[ticker] = {
                        "summary": summary,
                        "updated_cutoff": str(cutoff_date)[:10] if cutoff_date else "latest",
                        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _save_json_cache(SMART_MONEY_CACHE_PATH, cache)
                    return summary + f"；主接口异常: {'; '.join(errors[:1])}"
                errors.append(f"{source_name}未找到{ticker}")
            except Exception as exc:
                errors.append(f"{source_name}失败: {exc}")

        try:
            end_date = str(cutoff_date or datetime.datetime.now().strftime("%Y-%m-%d"))[:10].replace("-", "")
            start_dt = datetime.datetime.strptime(end_date, "%Y%m%d") - datetime.timedelta(days=45)
            hist_df = _call_ak(
                "stock_zh_a_hist",
                symbol=ticker,
                period="daily",
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_date,
                adjust="",
            )
            hist_df = _filter_by_cutoff(hist_df, cutoff_date, ("日期", "date"))
            if hist_df is not None and not hist_df.empty:
                volume_col = _find_column(hist_df.columns, ("成交量", "volume"))
                amount_col = _find_column(hist_df.columns, ("成交额", "amount"))
                turnover_col = _find_column(hist_df.columns, ("换手率", "turnover"))
                pct_col = _find_column(hist_df.columns, ("涨跌幅", "pct_change"))
                close_col = _find_column(hist_df.columns, ("收盘", "收盘价", "close"))
                rows = hist_df.tail(20)
                last = rows.iloc[-1].to_dict()
                vol_values = [_safe_float(x) for x in rows[volume_col].tail(20)] if volume_col else []
                amount_values = [_safe_float(x) for x in rows[amount_col].tail(20)] if amount_col else []
                vol_ma5 = sum(vol_values[-5:]) / max(len(vol_values[-5:]), 1) if vol_values else 0.0
                vol_ma20 = sum(vol_values) / max(len(vol_values), 1) if vol_values else 0.0
                amount_ma5 = sum(amount_values[-5:]) / max(len(amount_values[-5:]), 1) if amount_values else 0.0
                amount_ma20 = sum(amount_values) / max(len(amount_values), 1) if amount_values else 0.0
                proxy = {
                    "收盘": last.get(close_col, "") if close_col else "",
                    "涨跌幅": last.get(pct_col, "") if pct_col else "",
                    "换手率": last.get(turnover_col, "") if turnover_col else "",
                    "5/20日量比": round(vol_ma5 / vol_ma20, 3) if vol_ma20 else "",
                    "5/20日成交额比": round(amount_ma5 / amount_ma20, 3) if amount_ma20 else "",
                }
                return (
                    "[主力资金代理] 实时主力资金接口不可用，改用近20日量价活跃度作为资金参与度代理。"
                    f"代理指标: {proxy}。接口错误: {'; '.join(errors[:2])}"
                )
        except Exception as exc:
            errors.append(f"stock_zh_a_hist代理失败: {exc}")

        return (
            "[主力资金降级] 主力资金实时接口、缓存与量价代理均不可用；"
            f"本轮资金面证据不足，应降低主力资金权重并保持 neutral。错误摘要: {'; '.join(errors[:3]) or '无'}"
        )

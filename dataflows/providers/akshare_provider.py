from .base import BaseDataProvider
import akshare as ak
import json
import os
import time


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINANCIAL_CACHE_PATH = os.path.join(BASE_DIR, "data", "json", "financial_cache.json")


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


def _retry_call(func, retries: int = 2, delay: float = 0.8):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
    raise last_error

class AkShareProvider(BaseDataProvider):
    def fetch_sentiment_data(self, ticker: str, cutoff_date: str = None):
        """返回截止到 cutoff_date 的新闻摘要（若 cutoff_date=None 则返回最新）。"""
        try:
            news_df = ak.stock_news_em(symbol=ticker)
            if news_df is None or news_df.empty:
                return "近期无明显新闻热点。"

            # 如果提供了 cutoff_date（格式 YYYY-MM-DD），则过滤掉未来的新闻
            if cutoff_date:
                try:
                    cutoff_int = int(str(cutoff_date)[:10].replace('-', ''))
                    def _to_int(d):
                        try:
                            s = str(d)[:10]
                            return int(s.replace('-', ''))
                        except Exception:
                            return 0
                    news_df = news_df[news_df['发布时间'].apply(lambda x: _to_int(x) <= cutoff_int)]
                except Exception:
                    pass

            if not news_df.empty:
                return " | ".join(news_df['新闻标题'].head(3).tolist())
            return "近期无明显新闻热点。"
        except Exception as e:
            return f"实时新闻数据查询受限: {e}"

    def fetch_fundamental_data(self, ticker: str, cutoff_date: str = None):
        cache = _load_financial_cache()
        cached = cache.get(str(ticker), {})
        data_quality = "live"
        errors = []

        try:
            info_df = _retry_call(lambda: ak.stock_individual_info_em(symbol=ticker))
            info_dict = dict(zip(info_df['item'], info_df['value']))
            industry = info_dict.get('行业', '未知')
            market_cap = info_dict.get('总市值', '未知')
        except Exception as e:
            errors.append(f"个股基础信息接口失败: {e}")
            industry = cached.get("industry", "未知")
            market_cap = cached.get("market_cap", "未知")
            data_quality = "cached" if cached else "limited"

        # 由于东方财富基础接口不再包含动态市盈率和市净率，这里使用百度估值接口获取最新的财务估值数据
        try:
            pe_df = _retry_call(lambda: ak.stock_zh_valuation_baidu(symbol=ticker, indicator='市盈率(TTM)', period='近一年'))
            pe = pe_df['value'].iloc[-1] if pe_df is not None and not pe_df.empty else cached.get("pe", "未知")
        except Exception as e:
            errors.append(f"PE估值接口失败: {e}")
            pe = cached.get("pe", "未知")
            data_quality = "cached" if cached else "limited"

        try:
            pb_df = _retry_call(lambda: ak.stock_zh_valuation_baidu(symbol=ticker, indicator='市净率', period='近一年'))
            pb = pb_df['value'].iloc[-1] if pb_df is not None and not pb_df.empty else cached.get("pb", "未知")
        except Exception as e:
            errors.append(f"PB估值接口失败: {e}")
            pb = cached.get("pb", "未知")
            data_quality = "cached" if cached else "limited"

        if any(str(v) != "未知" for v in [industry, market_cap, pe, pb]):
            cache[str(ticker)] = {
                "industry": industry,
                "market_cap": market_cap,
                "pe": pe,
                "pb": pb,
                "updated_cutoff": str(cutoff_date)[:10] if cutoff_date else "latest",
            }
            _save_financial_cache(cache)

        if data_quality == "limited":
            return (
                "[基本面降级] 实时财务接口暂时不可用，且本地无该标的缓存；"
                "本轮基本面证据不足，应降低基本面权重并倾向 neutral。"
            )

        source_note = "实时接口" if data_quality == "live" else "本地缓存"
        error_note = f"，部分接口异常: {'; '.join(errors[:2])}" if errors else ""
        return (
            f"数据来源: {source_note}{error_note}。"
            f"所属行业: {industry}, 总市值: {market_cap}, 动态市盈率(PE): {pe}, 市净率(PB): {pb}"
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
        try:
            market = "sh" if str(ticker).startswith("6") else "sz"
            fund_df = ak.stock_individual_fund_flow(stock=ticker, market=market)
            if cutoff_date:
                try:
                    cutoff_dt = cutoff_date[:10]
                    if 'date' in fund_df.columns:
                        fund_df['date_str'] = fund_df['date'].astype(str)
                        fund_df = fund_df[fund_df['date_str'] <= cutoff_dt]
                except Exception:
                    pass

            recent_fund = fund_df.tail(2)[['收盘价', '主力净流入-净额', '涨跌幅']].to_dict('records')
            return f"近2个交易日的大单主力资金净流入(元)与价格变动特征: {recent_fund}"
        except Exception as e:
            return f"主力资金API抓取异常: {e}"

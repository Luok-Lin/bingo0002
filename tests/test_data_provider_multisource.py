import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dataflows.providers import akshare_provider
from dataflows.providers.akshare_provider import AkShareProvider


class MultiSourceAkShareProviderTests(unittest.TestCase):
    def test_fundamental_data_uses_multiple_sources_with_partial_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "financial_cache.json")
            provider = AkShareProvider()
            with (
                patch.object(akshare_provider, "FINANCIAL_CACHE_PATH", cache_path),
                patch.object(
                    akshare_provider.ak,
                    "stock_individual_info_em",
                    return_value=pd.DataFrame(
                        [{"item": "行业", "value": "银行"}, {"item": "总市值", "value": "1000亿"}]
                    ),
                ),
                patch.object(
                    akshare_provider.ak,
                    "stock_zh_valuation_baidu",
                    side_effect=[
                        pd.DataFrame([{"date": "2026-05-20", "value": 8.5}]),
                        RuntimeError("pb timeout"),
                    ],
                ),
                patch.object(
                    akshare_provider.ak,
                    "stock_financial_abstract",
                    return_value=pd.DataFrame([{"报告期": "2025年报", "营业收入": "100亿", "归母净利润": "12亿"}]),
                ),
                patch.object(
                    akshare_provider.ak,
                    "stock_financial_analysis_indicator",
                    return_value=pd.DataFrame([{"日期": "2025-12-31", "净资产收益率": "12.5", "资产负债率": "45"}]),
                ),
                patch.object(akshare_provider.ak, "stock_profit_sheet_by_report_em", side_effect=RuntimeError("profit down")),
                patch.object(
                    akshare_provider.ak,
                    "stock_balance_sheet_by_report_em",
                    return_value=pd.DataFrame([{"公告日期": "2026-04-30", "资产负债率": "45"}]),
                ),
                patch.object(
                    akshare_provider.ak,
                    "stock_cash_flow_sheet_by_report_em",
                    return_value=pd.DataFrame([{"公告日期": "2026-04-30", "经营现金流量净额": "20亿"}]),
                ),
            ):
                text = provider.fetch_fundamental_data("000001", cutoff_date="2026-05-21")

        self.assertIn("实时多源基本面", text)
        self.assertIn("个股资料", text)
        self.assertIn("估值", text)
        self.assertIn("财务摘要", text)
        self.assertIn("现金流量表", text)
        self.assertIn("部分来源异常", text)

    def test_sentiment_data_uses_reports_and_announcements_when_news_fails(self):
        provider = AkShareProvider()
        with (
            patch.object(akshare_provider.ak, "stock_news_em", side_effect=RuntimeError("news down")),
            patch.object(
                akshare_provider.ak,
                "stock_research_report_em",
                return_value=pd.DataFrame(
                    [{"日期": "2026-05-20", "机构": "测试证券", "东财评级": "买入", "报告名称": "盈利修复"}]
                ),
            ),
            patch.object(
                akshare_provider.ak,
                "stock_zh_a_disclosure_report_cninfo",
                return_value=pd.DataFrame([{"公告时间": "2026-05-19", "公告标题": "年度权益分派公告"}]),
            ),
            patch.object(
                akshare_provider.ak,
                "stock_news_main_cx",
                return_value=pd.DataFrame([{"时间": "2026-05-18", "标题": "A股市场活跃"}]),
            ),
        ):
            text = provider.fetch_sentiment_data("000001", cutoff_date="2026-05-21")

        self.assertIn("多源舆情", text)
        self.assertIn("券商研报", text)
        self.assertIn("交易所公告", text)
        self.assertIn("市场要闻", text)
        self.assertIn("部分来源异常", text)

    def test_smart_money_uses_cache_when_live_flow_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "smart_money_cache.json")
            provider = AkShareProvider()
            with (
                patch.object(akshare_provider, "SMART_MONEY_CACHE_PATH", cache_path),
                patch.object(
                    akshare_provider.ak,
                    "stock_individual_fund_flow",
                    return_value=pd.DataFrame(
                        [
                            {"日期": "2026-05-19", "收盘价": 10.1, "主力净流入-净额": 1200000, "涨跌幅": 1.2},
                            {"日期": "2026-05-20", "收盘价": 10.3, "主力净流入-净额": 800000, "涨跌幅": 1.9},
                        ]
                    ),
                ),
            ):
                first = provider.fetch_smart_money_data("000001", cutoff_date="2026-05-21")

            self.assertIn("stock_individual_fund_flow", first)
            self.assertIn("净流入", first)

            with (
                patch.object(akshare_provider, "SMART_MONEY_CACHE_PATH", cache_path),
                patch.object(akshare_provider.ak, "stock_individual_fund_flow", side_effect=RuntimeError("flow down")),
            ):
                cached = provider.fetch_smart_money_data("000001", cutoff_date="2026-05-21")

            self.assertIn("本地主力资金缓存", cached)
            self.assertIn("实时接口异常", cached)

    def test_smart_money_uses_rank_backup_when_detail_flow_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = AkShareProvider()
            rank_df = pd.DataFrame(
                [
                    {
                        "代码": "000001",
                        "名称": "平安银行",
                        "最新价": 12.3,
                        "涨跌幅": 1.5,
                        "主力净流入": 23450000,
                        "主力净占比": 8.2,
                        "排名": 12,
                    }
                ]
            )
            with (
                patch.object(akshare_provider, "SMART_MONEY_CACHE_PATH", os.path.join(tmpdir, "smart_money_cache.json")),
                patch.object(akshare_provider.ak, "stock_individual_fund_flow", side_effect=RuntimeError("detail down")),
                patch.object(akshare_provider.ak, "stock_main_fund_flow", return_value=rank_df),
            ):
                text = provider.fetch_smart_money_data("000001", cutoff_date="2026-05-21")

        self.assertIn("stock_main_fund_flow", text)
        self.assertIn("平安银行", text)
        self.assertIn("净流入", text)

    def test_smart_money_falls_back_to_volume_price_proxy(self):
        provider = AkShareProvider()
        hist = pd.DataFrame(
            [
                {"日期": f"2026-05-{day:02d}", "收盘": 10 + day / 100, "成交量": 1000 + day, "成交额": 100000 + day, "换手率": 1.0, "涨跌幅": 0.1}
                for day in range(1, 21)
            ]
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(akshare_provider, "SMART_MONEY_CACHE_PATH", os.path.join(tmpdir, "smart_money_cache.json")),
            patch.object(akshare_provider.ak, "stock_individual_fund_flow", side_effect=RuntimeError("flow down")),
            patch.object(akshare_provider.ak, "stock_main_fund_flow", side_effect=RuntimeError("main down")),
            patch.object(akshare_provider.ak, "stock_individual_fund_flow_rank", side_effect=RuntimeError("rank down")),
            patch.object(akshare_provider.ak, "stock_zh_a_hist", return_value=hist),
        ):
            text = provider.fetch_smart_money_data("000001", cutoff_date="2026-05-21")

        self.assertIn("主力资金代理", text)
        self.assertIn("5/20日量比", text)


if __name__ == "__main__":
    unittest.main()

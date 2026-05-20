import unittest

from rl.market_rules import infer_a_share_limit_percent, simulate_a_share_execution


class MarketRulesTests(unittest.TestCase):
    def test_infer_a_share_limit_percent(self):
        self.assertEqual(infer_a_share_limit_percent("600519"), 10.0)
        self.assertEqual(infer_a_share_limit_percent("300750"), 20.0)
        self.assertEqual(infer_a_share_limit_percent("688981"), 20.0)
        self.assertEqual(infer_a_share_limit_percent("600000", name="ST 测试"), 5.0)

    def test_long_only_sell_is_not_executed(self):
        result = simulate_a_share_execution(
            ticker="600519",
            action="SELL",
            position=1.0,
            market_move_percent=-2.5,
            allow_short=False,
        )
        self.assertEqual(result.requested_action, "SELL")
        self.assertEqual(result.executed_action, "HOLD")
        self.assertEqual(result.net_pnl_percent, 0.0)
        self.assertIn("长仓", result.blocked_reason)

    def test_buy_net_pnl_deducts_round_trip_cost(self):
        result = simulate_a_share_execution(
            ticker="600519",
            action="BUY",
            position=0.5,
            market_move_percent=2.0,
            current_change_percent=0.0,
            commission_rate=0.0003,
            stamp_duty_rate=0.0005,
            slippage_bps=5.0,
        )
        self.assertEqual(result.executed_action, "BUY")
        self.assertEqual(result.gross_pnl_percent, 1.0)
        self.assertEqual(result.cost_percent, 0.105)
        self.assertEqual(result.net_pnl_percent, 0.895)

    def test_limit_up_blocks_new_buy(self):
        result = simulate_a_share_execution(
            ticker="600519",
            action="BUY",
            position=1.0,
            market_move_percent=1.0,
            current_change_percent=9.98,
        )
        self.assertEqual(result.executed_action, "HOLD")
        self.assertEqual(result.net_pnl_percent, 0.0)
        self.assertIn("涨停", result.blocked_reason)


if __name__ == "__main__":
    unittest.main()

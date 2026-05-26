import unittest

from backend.services import _apply_user_profile_to_advice


class UserProfilePersonalizationTests(unittest.TestCase):
    def _base_advice(self) -> dict:
        return {
            "recommendation": {
                "action": "BUY",
                "execution_action": "BUY 35.0%",
                "position_percent": 35.0,
                "confidence": 0.72,
                "reason": "趋势偏强。",
            },
            "risk": {"decision": "BUY", "position_percent": 35.0, "reason": "风控通过。"},
            "referee": {"trend_strength": 0.7},
            "data_quality": {"score": 0.8, "level": "high"},
            "risk_plan": {"stop_loss": "跌破20日线减仓。"},
            "multi_period_advice": {"risk_plan": {"stop_loss": "跌破20日线减仓。"}},
        }

    def test_empty_position_caps_first_entry(self):
        advice = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
                "prefer_stop_loss": True,
            },
        )

        self.assertEqual(advice["recommendation"]["action"], "BUY")
        self.assertEqual(advice["recommendation"]["position_percent"], 10.0)
        self.assertIn("空仓画像", advice["recommendation"]["reason"])
        self.assertTrue(advice["personalization"]["applied"])

    def test_conservative_low_quality_forces_hold(self):
        advice = self._base_advice()
        advice["data_quality"] = {"score": 0.3, "level": "low"}

        personalized = _apply_user_profile_to_advice(
            advice,
            {
                "risk_profile": "conservative",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "HOLD")
        self.assertEqual(personalized["recommendation"]["position_percent"], 0.0)
        self.assertEqual(personalized["risk"]["decision"], "HOLD")
        self.assertIn("数据质量偏低", personalized["recommendation"]["reason"])

    def test_buy_with_zero_position_becomes_hold(self):
        advice = self._base_advice()
        advice["recommendation"]["position_percent"] = 0.0
        advice["risk"]["position_percent"] = 0.0

        personalized = _apply_user_profile_to_advice(
            advice,
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "HOLD")
        self.assertEqual(personalized["recommendation"]["execution_action"], "HOLD")
        self.assertEqual(personalized["recommendation"]["position_percent"], 0.0)
        self.assertEqual(personalized["risk"]["decision"], "HOLD")
        self.assertIn("仓位约束压至 0%", personalized["recommendation"]["reason"])

    def test_holding_user_gets_stop_loss_from_cost(self):
        advice = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "conservative",
                "holding_period": "swing",
                "max_position_per_stock": 40,
                "already_holding": True,
                "current_position_percent": 30,
                "cost_price": 18.5,
                "prefer_stop_loss": True,
            },
        )

        self.assertEqual(advice["recommendation"]["position_percent"], 35.0)
        self.assertIn("17.57", advice["risk"]["stop_loss"])
        self.assertIn("17.57", advice["risk_plan"]["stop_loss"])
        self.assertIn("17.57", advice["multi_period_advice"]["risk_plan"]["stop_loss"])

    def test_custom_constraint_blocks_buy(self):
        personalized = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
                "personalization_preferences": {
                    "custom_constraints": [{"type": "block_buy", "text": "财报前不加仓"}],
                },
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "HOLD")
        self.assertEqual(personalized["recommendation"]["position_percent"], 0.0)
        self.assertIn("禁止买入", personalized["recommendation"]["reason"])
        self.assertEqual(personalized["personalization"]["custom_constraints"][0]["type"], "block_buy")

    def test_custom_user_strategy_blocks_buy(self):
        personalized = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
                "personalization_preferences": {
                    "custom_rules": [{"policy": "block_buy", "text": "只在放量回踩后买入"}],
                },
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "HOLD")
        self.assertEqual(personalized["recommendation"]["position_percent"], 0.0)
        self.assertIn("用户自定义策略禁止买入", personalized["recommendation"]["reason"])
        self.assertEqual(personalized["personalization"]["custom_rules"][0]["policy"], "block_buy")

    def test_custom_user_strategy_can_be_plain_language(self):
        personalized = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 20,
                "already_holding": False,
                "personalization_preferences": {
                    "custom_rules": [{"text": "只在放量回踩后买入"}],
                },
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "HOLD")
        self.assertEqual(personalized["personalization"]["custom_rules"][0]["policy"], "block_buy")

    def test_custom_user_strategy_can_infer_position_value_from_text(self):
        personalized = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 50,
                "already_holding": True,
                "current_position_percent": 10,
                "personalization_preferences": {
                    "custom_rules": [{"text": "这只票仓位不超过12%"}],
                },
            },
        )

        self.assertEqual(personalized["recommendation"]["position_percent"], 12.0)
        self.assertEqual(personalized["personalization"]["custom_rules"][0]["policy"], "position_cap")

    def test_custom_position_cap_limits_target_position(self):
        personalized = _apply_user_profile_to_advice(
            self._base_advice(),
            {
                "risk_profile": "balanced",
                "holding_period": "swing",
                "max_position_per_stock": 50,
                "already_holding": True,
                "current_position_percent": 10,
                "personalization_preferences": {
                    "custom_constraints": [{"type": "position_cap", "value": 12, "text": "只允许小仓试错"}],
                },
            },
        )

        self.assertEqual(personalized["recommendation"]["action"], "BUY")
        self.assertEqual(personalized["recommendation"]["position_percent"], 12.0)
        self.assertIn("自定义约束仓位上限", personalized["recommendation"]["reason"])


if __name__ == "__main__":
    unittest.main()

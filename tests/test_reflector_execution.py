import unittest

from agents.roles import QuantitativeRiskReflector


class _Memory:
    def __init__(self):
        self.memory = []
        self.updated = False

    def append(self, record):
        self.memory.append(record)

    def update_experience_score_by_action(self, **_kwargs):
        self.updated = True

    def crystallize_knowledge(self, _ticker):
        return []


class ReflectorExecutionTests(unittest.TestCase):
    def test_reflector_uses_execution_net_pnl_without_recomputing(self):
        memory = _Memory()
        reflector = QuantitativeRiskReflector(name="test-reflector", memory_bank=memory)

        reflector.step(
            ticker="600519",
            decision="BUY 50%",
            reports=[],
            pnl_percent=2.0,
            execution_result={
                "requested_action": "BUY",
                "executed_action": "BUY",
                "executed_position": 0.5,
                "market_move_percent": 2.0,
                "gross_pnl_percent": 1.0,
                "cost_percent": 0.105,
                "net_pnl_percent": 0.895,
                "blocked_reason": "",
            },
        )

        self.assertEqual(len(memory.memory), 1)
        record = memory.memory[0]
        self.assertEqual(record["decision"], "BUY")
        self.assertEqual(record["position"], 0.5)
        self.assertEqual(record["market_move_percent"], 2.0)
        self.assertEqual(record["pnl_percent"], 0.9)
        self.assertEqual(record["cost_percent"], 0.105)
        self.assertTrue(memory.updated)


if __name__ == "__main__":
    unittest.main()

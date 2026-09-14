import unittest
from core import Config, Prediction, run_pipeline


class RoutingTests(unittest.TestCase):
    def test_routing_preserves_all_rows_and_order(self):
        rows = [{"name": name} for name in ["exclude", "same", "same", "last"]]
        seen = {}
        def transformer(batch):
            seen["transformer"] = [r["pipeline_row_id"] for r in batch]
            return {"1": Prediction(2, .99), "2": Prediction(3, .99), "3": Prediction(5, .4)}
        def llm(batch):
            seen["llm"] = [r["pipeline_row_id"] for r in batch]
            return {"3": Prediction(9), "2": Prediction(1)}
        result = run_pipeline(rows, Config(
            [{"column": "name", "keyword": "exclude"}], [0, 2, 5, 9], {"5": .9}), transformer, llm)
        self.assertEqual(seen, {"transformer": ["1", "2", "3"], "llm": ["2", "3"]})
        self.assertEqual([r["pipeline_label"] for r in result], [0, 2, 1, 9])
        self.assertEqual([r["name"] for r in result], [r["name"] for r in rows])
        self.assertNotIn("pipeline_label", rows[0])

    def test_unconnected_is_pending_not_zero(self):
        result = run_pipeline([{"name": "a"}], Config())
        self.assertEqual(result[0]["pipeline_status"], "pending")
        self.assertEqual(result[0]["pipeline_label"], "")

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            run_pipeline([{}], Config(), lambda rows: {"wrong": Prediction(0)})

    def test_missing_confidence_routes_to_llm(self):
        result = run_pipeline([{}], Config(trusted_categories=[2], min_confidence={"2": .9}),
                              lambda rows: {"0": Prediction(2)}, lambda rows: {"0": Prediction(5)})
        self.assertEqual(result[0]["pipeline_source"], "llm")


if __name__ == "__main__":
    unittest.main()

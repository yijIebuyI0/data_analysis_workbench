from __future__ import annotations

import unittest

import pandas as pd

from src.analytics import aggregate, build_analysis_context, numeric_statistics


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "segment": ["新客", "新客", "老客"],
                "amount": [100.0, 200.0, 500.0],
                "city": ["上海", "北京", "北京"],
            }
        )

    def test_numeric_statistics_and_grouped_aggregation(self) -> None:
        summary = numeric_statistics(self.frame)
        self.assertEqual(summary.loc[0, "求和"], 800.0)
        result = aggregate(self.frame, "amount", "平均值", "segment")
        new_customer_mean = result.table.loc[result.table["segment"] == "新客", "amount_mean"].iloc[0]
        self.assertEqual(new_customer_mean, 150.0)

    def test_context_contains_structured_aggregation_without_raw_rows(self) -> None:
        result = aggregate(self.frame, "amount", "求和", "segment")
        context = build_analysis_context(self.frame, result, [{"step": 1, "summary": {"label": "空值处理"}}])
        self.assertIn("dataset_overview", context)
        self.assertEqual(context["user_aggregation"]["config"]["group_by"], "segment")
        self.assertNotIn("raw_data", context)
        self.assertIn("data_scope_note", context)

    def test_numeric_operation_rejects_text_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "数值字段"):
            aggregate(self.frame, "city", "求和")


if __name__ == "__main__":
    unittest.main()

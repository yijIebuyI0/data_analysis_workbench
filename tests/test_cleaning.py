from __future__ import annotations

import unittest

import pandas as pd

from src.cleaning import CleaningError, prepare_cleaning_preview


class CleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "id": [1, 1, 2, 3],
                "city": [" 上海市 ", "Shanghai", "北京", "北京"],
                "amount": [10.0, None, 30.0, None],
                "code": ["10", "bad", "30", "40"],
            }
        )

    def test_each_step_returns_a_copy_and_does_not_mutate_source(self) -> None:
        original = self.frame.copy(deep=True)
        preview = prepare_cleaning_preview(
            self.frame,
            {"action": "missing", "columns": ["amount"], "method": "median"},
            base_revision=0,
        )
        self.assertFalse(preview.result["amount"].isna().any())
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(preview.base_revision, 0)

    def test_duplicate_type_and_normalization_steps(self) -> None:
        deduplicated = prepare_cleaning_preview(
            self.frame,
            {"action": "duplicates", "columns": ["id"], "keep": "last"},
            0,
        ).result
        self.assertEqual(len(deduplicated), 3)

        converted = prepare_cleaning_preview(
            self.frame,
            {"action": "type", "columns": ["code"], "target": "integer", "errors": "coerce"},
            0,
        ).result
        self.assertEqual(str(converted["code"].dtype), "Int64")
        self.assertTrue(pd.isna(converted.loc[1, "code"]))

        normalized = prepare_cleaning_preview(
            self.frame,
            {
                "action": "normalize",
                "columns": ["city"],
                "strip": True,
                "case": None,
                "mapping": {"上海市": "上海", "Shanghai": "上海"},
            },
            0,
        ).result
        self.assertEqual(normalized["city"].tolist()[:2], ["上海", "上海"])

    def test_constant_fill_keeps_numeric_columns_numeric(self) -> None:
        preview = prepare_cleaning_preview(
            self.frame,
            {"action": "missing", "columns": ["amount"], "method": "constant", "value": "0"},
            0,
        )
        self.assertEqual(preview.result["amount"].tolist(), [10.0, 0.0, 30.0, 0.0])
        self.assertTrue(pd.api.types.is_numeric_dtype(preview.result["amount"]))

        with self.assertRaisesRegex(CleaningError, "类型不匹配"):
            prepare_cleaning_preview(
                self.frame,
                {"action": "missing", "columns": ["amount"], "method": "constant", "value": "未知"},
                0,
            )

    def test_invalid_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(CleaningError, "至少选择"):
            prepare_cleaning_preview(
                self.frame,
                {"action": "missing", "columns": [], "method": "median"},
                0,
            )


if __name__ == "__main__":
    unittest.main()

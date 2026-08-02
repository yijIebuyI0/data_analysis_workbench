from __future__ import annotations

import json
import unittest
from datetime import datetime

import pandas as pd

from src.exporting import audit_download, dataframe_download


class ExportingTests(unittest.TestCase):
    def test_exports_new_file_name_in_original_format(self) -> None:
        frame = pd.DataFrame({"id": [1, 2]})
        fixed = datetime(2026, 8, 2, 9, 30, 0)
        csv_bytes, csv_name, csv_mime = dataframe_download(frame, "source.csv", "csv", fixed)
        self.assertEqual(csv_name, "source_cleaned_20260802_093000.csv")
        self.assertEqual(csv_mime, "text/csv")
        self.assertIn(b"id", csv_bytes)
        excel_bytes, excel_name, _ = dataframe_download(frame, "source.xlsx", "excel", fixed)
        self.assertEqual(excel_name, "source_cleaned_20260802_093000.xlsx")
        self.assertTrue(excel_bytes.startswith(b"PK"))

    def test_audit_log_is_json(self) -> None:
        payload = audit_download({"file_name": "x.csv"}, [{"step": 1}], {"rows": 2})
        decoded = json.loads(payload)
        self.assertEqual(decoded["cleaning_steps"][0]["step"], 1)


if __name__ == "__main__":
    unittest.main()

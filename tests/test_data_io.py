from __future__ import annotations

import json
import unittest
from io import BytesIO

import pandas as pd

from src.data_io import DataLoadError, excel_sheet_names, load_uploaded_bytes


class DataIoTests(unittest.TestCase):
    def test_loads_semicolon_csv_and_preserves_metadata(self) -> None:
        data = "id;city\n1;上海\n2;北京\n".encode("utf-8-sig")
        loaded = load_uploaded_bytes(data, "sample.csv")
        self.assertEqual(loaded.dataframe.shape, (2, 2))
        self.assertEqual(loaded.metadata["delimiter"], ";")
        self.assertEqual(loaded.file_format, "csv")

    def test_loads_json_records_and_data_wrapper(self) -> None:
        records = json.dumps([{"id": 1, "profile": {"city": "上海"}}], ensure_ascii=False).encode()
        loaded = load_uploaded_bytes(records, "records.json")
        self.assertEqual(loaded.dataframe.columns.tolist(), ["id", "profile.city"])
        wrapped = json.dumps({"data": [{"id": 2}, {"id": 3}]}).encode()
        loaded_wrapped = load_uploaded_bytes(wrapped, "wrapped.json")
        self.assertEqual(loaded_wrapped.dataframe["id"].tolist(), [2, 3])

    def test_excel_requires_explicit_sheet_for_multiple_sheets(self) -> None:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"id": [1]}).to_excel(writer, sheet_name="一月", index=False)
            pd.DataFrame({"id": [2]}).to_excel(writer, sheet_name="二月", index=False)
        data = buffer.getvalue()
        self.assertEqual(excel_sheet_names(data), ["一月", "二月"])
        with self.assertRaisesRegex(DataLoadError, "多个工作表"):
            load_uploaded_bytes(data, "sample.xlsx")
        loaded = load_uploaded_bytes(data, "sample.xlsx", "二月")
        self.assertEqual(loaded.dataframe["id"].tolist(), [2])

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaisesRegex(DataLoadError, "仅支持"):
            load_uploaded_bytes(b"hello", "notes.txt")


if __name__ == "__main__":
    unittest.main()

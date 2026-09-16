from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.analytics import aggregate, build_analysis_context
from src.cleaning import prepare_cleaning_preview
from src.data_io import load_uploaded_bytes
from src.exporting import dataframe_download
from src.profiling import assess_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = PROJECT_ROOT / "sample_data" / "customer_orders.csv"


class SampleDataFlowTests(unittest.TestCase):
    """用仓库自带样例数据跑通上传到导出的完整链路。"""

    def test_sample_pipeline_runs_end_to_end(self) -> None:
        loaded = load_uploaded_bytes(SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name)
        self.assertEqual(loaded.dataframe.shape, (8, 7))

        profile = assess_dataframe(loaded.dataframe)
        self.assertEqual(profile.overview["rows"], 8)
        self.assertEqual(profile.overview["columns"], 7)
        self.assertEqual(profile.overview["missing_cells"], 2)
        # 两行 O-1001 的城市写法不同，属于口径问题而不是完全重复行
        self.assertEqual(profile.overview["duplicate_rows"], 0)
        self.assertFalse(profile.issues.empty)
        self.assertEqual(len(profile.columns), 7)

        normalized = prepare_cleaning_preview(
            loaded.dataframe,
            {
                "action": "normalize",
                "columns": ["city"],
                "strip": True,
                "case": None,
                "mapping": {"上海市": "上海", "Shanghai": "上海", "Guangzhou": "广州"},
            },
            base_revision=0,
        )
        self.assertEqual(normalized.result["city"].tolist()[0], "上海")
        self.assertNotIn("Shanghai", normalized.result["city"].tolist())

        deduplicated = prepare_cleaning_preview(
            normalized.result,
            {"action": "duplicates", "columns": ["order_id"], "keep": "first"},
            base_revision=1,
        )
        self.assertEqual(len(deduplicated.result), 7)

        aggregation = aggregate(deduplicated.result, "order_amount", "求和", "customer_segment")
        self.assertEqual(len(aggregation.table), 3)
        context = build_analysis_context(deduplicated.result, aggregation, [{"step": 1}])
        self.assertEqual(context["dataset_overview"]["rows"], 7)
        self.assertIsNotNone(context["user_aggregation"])
        self.assertNotIn("raw_data", context)

        payload, file_name, mime = dataframe_download(
            deduplicated.result,
            loaded.file_name,
            loaded.file_format,
            source_metadata=loaded.metadata,
        )
        self.assertTrue(file_name.startswith("customer_orders_cleaned_"))
        self.assertEqual(mime, "text/csv")
        self.assertIn("order_id".encode(), payload)


class SampleUploadSmokeTests(unittest.TestCase):
    """验证页面在真实上传样例文件后仍能正常渲染。"""

    def test_upload_sample_file_renders_profile_metrics(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "datalab.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)

        app.file_uploader[0].upload(
            SAMPLE_FILE.name,
            SAMPLE_FILE.read_bytes(),
            "text/csv",
        ).run()

        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("行数"), "8")
        self.assertEqual(metrics.get("字段数"), "7")
        self.assertEqual(metrics.get("空值单元格"), "2")
        self.assertEqual(metrics.get("完全重复行"), "0")


if __name__ == "__main__":
    unittest.main()

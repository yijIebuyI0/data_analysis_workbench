from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_initial_page_renders_without_api_key(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "datalab.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.file_uploader), 1)
        self.assertIn("DataLab", app.markdown[2].value)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


def cleaned_file_name(original_name: str, extension: str, timestamp: datetime | None = None) -> str:
    moment = timestamp or datetime.now()
    stem = Path(original_name).stem
    return f"{stem}_cleaned_{moment:%Y%m%d_%H%M%S}.{extension}"


def dataframe_download(
    frame: pd.DataFrame,
    original_name: str,
    original_format: str,
    timestamp: datetime | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[bytes, str, str]:
    if original_format == "excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="cleaned_data")
        return (
            buffer.getvalue(),
            cleaned_file_name(original_name, "xlsx", timestamp),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if original_format == "json":
        data = frame.to_json(orient="records", force_ascii=False, indent=2, date_format="iso").encode("utf-8")
        return data, cleaned_file_name(original_name, "json", timestamp), "application/json"
    metadata = source_metadata or {}
    delimiter = metadata.get("delimiter", "," if original_format == "csv" else "\t")
    data = frame.to_csv(index=False, sep=delimiter).encode("utf-8-sig")
    if original_format == "txt":
        return data, cleaned_file_name(original_name, "txt", timestamp), "text/plain"
    return data, cleaned_file_name(original_name, "csv", timestamp), "text/csv"


def audit_download(
    source: dict[str, Any],
    cleaning_steps: list[dict[str, Any]],
    analysis_context: dict[str, Any] | None,
) -> bytes:
    payload = {
        "source": source,
        "cleaning_steps": cleaning_steps,
        "analysis_context": analysis_context,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")

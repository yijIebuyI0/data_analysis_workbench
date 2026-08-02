from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd


MAX_ROWS = 100_000
SUPPORTED_SUFFIXES = {".csv": "csv", ".xlsx": "excel", ".json": "json"}


class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be interpreted as one table."""


@dataclass(frozen=True)
class LoadedData:
    dataframe: pd.DataFrame
    file_name: str
    file_format: str
    file_hash: str
    metadata: dict[str, Any]


def file_signature(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def excel_sheet_names(data: bytes) -> list[str]:
    try:
        with pd.ExcelFile(BytesIO(data)) as workbook:
            return list(workbook.sheet_names)
    except Exception as exc:
        raise DataLoadError(f"无法读取 Excel 工作表：{exc}") from exc


def _detect_csv(data: bytes) -> tuple[str, str]:
    sample = data[:65_536]
    encoding = None
    for candidate in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            sample.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if encoding is None:
        raise DataLoadError("无法识别 CSV 编码，请转换为 UTF-8 或 GB18030 后重试。")
    text = sample.decode(encoding)
    try:
        delimiter = csv.Sniffer().sniff(text, delimiters=",\t;|").delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def _load_json(data: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    text = data.decode("utf-8-sig")
    try:
        payload = json.loads(text)
        mode = "json"
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            payload = rows
            mode = "jsonl"
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"JSON 格式错误：{exc}") from exc

    if isinstance(payload, list):
        frame = pd.json_normalize(payload)
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            frame = pd.json_normalize(payload["data"])
        else:
            try:
                frame = pd.DataFrame(payload)
            except ValueError:
                frame = pd.json_normalize(payload)
    else:
        raise DataLoadError("JSON 顶层必须是对象或数组。")
    return frame, {"json_mode": mode}


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        raise DataLoadError("文件中没有可识别的表格数据。")
    if len(frame) > MAX_ROWS:
        raise DataLoadError(f"当前版本最多处理 {MAX_ROWS:,} 行数据。")
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if any(not column for column in result.columns):
        raise DataLoadError("存在空字段名，请先补充字段名。")
    if len(result.columns) != len(set(result.columns)):
        raise DataLoadError("存在重复字段名，请先修改字段名。")
    return result.reset_index(drop=True)


def load_uploaded_bytes(data: bytes, file_name: str, sheet_name: str | None = None) -> LoadedData:
    suffix = Path(file_name).suffix.lower()
    file_format = SUPPORTED_SUFFIXES.get(suffix)
    if file_format is None:
        raise DataLoadError("仅支持 CSV、XLSX 和 JSON 文件。")
    metadata: dict[str, Any] = {}
    try:
        if file_format == "csv":
            encoding, delimiter = _detect_csv(data)
            frame = pd.read_csv(BytesIO(data), encoding=encoding, sep=delimiter)
            metadata.update(encoding=encoding, delimiter=delimiter)
        elif file_format == "excel":
            sheets = excel_sheet_names(data)
            selected = sheet_name or (sheets[0] if len(sheets) == 1 else None)
            if selected is None:
                raise DataLoadError("该 Excel 包含多个工作表，请先选择一个工作表。")
            if selected not in sheets:
                raise DataLoadError(f"找不到工作表：{selected}")
            frame = pd.read_excel(BytesIO(data), sheet_name=selected)
            metadata.update(sheet_name=selected, available_sheets=sheets)
        else:
            frame, json_metadata = _load_json(data)
            metadata.update(json_metadata)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"读取文件失败：{exc}") from exc

    frame = _validate_frame(frame)
    metadata.update(rows=len(frame), columns=len(frame.columns))
    return LoadedData(
        dataframe=frame,
        file_name=file_name,
        file_format=file_format,
        file_hash=file_signature(data),
        metadata=metadata,
    )

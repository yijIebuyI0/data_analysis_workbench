from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(frozen=True)
class DataProfile:
    overview: dict[str, int]
    columns: pd.DataFrame
    issues: pd.DataFrame
    inconsistent_terms: dict[str, list[list[str]]]


def _display_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _safe_string_series(series: pd.Series) -> pd.Series:
    return series.map(_display_value).astype("string")


def _term_groups(series: pd.Series) -> list[list[str]]:
    values = _safe_string_series(series.dropna())
    if values.empty or values.nunique() > 100:
        return []
    groups: dict[str, set[str]] = {}
    for value in values.unique().tolist():
        normalized = str(value).strip().casefold()
        groups.setdefault(normalized, set()).add(str(value))
    return [sorted(group) for group in groups.values() if len(group) > 1]


def assess_dataframe(frame: pd.DataFrame) -> DataProfile:
    overview = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    column_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    inconsistent: dict[str, list[list[str]]] = {}

    for column in frame.columns:
        series = frame[column]
        display_series = _safe_string_series(series.dropna())
        missing = int(series.isna().sum())
        distinct = int(display_series.nunique())
        examples = "、".join(display_series.head(3).tolist())
        suggestion = "保持当前类型"

        if not is_numeric_dtype(series) and not is_datetime64_any_dtype(series) and not display_series.empty:
            numeric_ratio = float(pd.to_numeric(display_series, errors="coerce").notna().mean())
            if numeric_ratio >= 0.85:
                suggestion = "可能是数值字段，建议检查类型"
                issues.append({"问题": "类型可能不一致", "字段": column, "数量": int(len(display_series)), "建议": suggestion})

        groups = _term_groups(series) if not is_numeric_dtype(series) else []
        if groups:
            inconsistent[column] = groups
            issues.append(
                {
                    "问题": "存在疑似口径不一致",
                    "字段": column,
                    "数量": sum(len(group) for group in groups),
                    "建议": "检查空格、大小写或同义词映射",
                }
            )
        if missing:
            issues.append({"问题": "存在空值", "字段": column, "数量": missing, "建议": "根据业务含义选择填充或删除"})

        column_rows.append(
            {
                "字段": column,
                "当前类型": str(series.dtype),
                "空值数": missing,
                "空值率": missing / len(frame) if len(frame) else 0.0,
                "唯一值数": distinct,
                "示例值": examples,
                "类型建议": suggestion,
            }
        )

    if overview["duplicate_rows"]:
        issues.insert(
            0,
            {
                "问题": "存在完全重复行",
                "字段": "整表",
                "数量": overview["duplicate_rows"],
                "建议": "确认业务主键后再去重",
            },
        )
    issue_columns = ["问题", "字段", "数量", "建议"]
    return DataProfile(
        overview=overview,
        columns=pd.DataFrame(column_rows),
        issues=pd.DataFrame(issues, columns=issue_columns),
        inconsistent_terms=inconsistent,
    )

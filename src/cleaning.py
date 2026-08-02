from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype


class CleaningError(ValueError):
    """Raised when a cleaning step is incompatible with the data."""


ACTION_LABELS = {
    "missing": "空值处理",
    "duplicates": "重复值处理",
    "type": "类型处理",
    "normalize": "口径统一",
}


@dataclass(frozen=True)
class CleaningPreview:
    result: pd.DataFrame
    summary: dict[str, Any]
    before_sample: pd.DataFrame
    after_sample: pd.DataFrame
    spec: dict[str, Any]
    base_revision: int


def _overview(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    if not columns:
        raise CleaningError("至少选择一个字段。")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CleaningError("字段不存在：" + "、".join(missing))


def _boolean_series(series: pd.Series, errors: str) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "是"}
    falsy = {"false", "0", "no", "n", "否"}

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        normalized = str(value).strip().casefold()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        if errors == "coerce":
            return pd.NA
        raise CleaningError(f"无法转换为布尔值：{value}")

    return series.map(convert).astype("boolean")


def _coerce_constant(series: pd.Series, raw_value: Any) -> Any:
    if raw_value is None or (isinstance(raw_value, str) and raw_value == ""):
        raise CleaningError("请填写固定填充值。")
    try:
        if is_bool_dtype(series.dtype):
            return _boolean_series(pd.Series([raw_value]), errors="raise").iloc[0]
        if is_integer_dtype(series.dtype):
            numeric = pd.to_numeric(raw_value, errors="raise")
            if float(numeric).is_integer():
                return int(numeric)
            raise CleaningError(f"固定值 {raw_value} 不是整数。")
        if is_numeric_dtype(series.dtype):
            return float(pd.to_numeric(raw_value, errors="raise"))
        if is_datetime64_any_dtype(series.dtype):
            return pd.to_datetime(raw_value, errors="raise")
    except CleaningError:
        raise
    except (TypeError, ValueError) as exc:
        raise CleaningError(f"固定值 {raw_value} 与字段 {series.name} 的类型不匹配。") from exc
    return raw_value


def _changed_indices(before: pd.DataFrame, after: pd.DataFrame) -> list[int]:
    changed: set[int] = set(before.index.difference(after.index).tolist())
    common = before.index.intersection(after.index)
    for column in before.columns.intersection(after.columns):
        left = before.loc[common, column]
        right = after.loc[common, column]
        equal = left.eq(right).fillna(False) | (left.isna() & right.isna())
        changed.update(common[~equal.to_numpy()].tolist())
    return [index for index in before.index if index in changed]


def _apply_missing(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    columns = spec.get("columns", [])
    _ensure_columns(frame, columns)
    method = spec.get("method")
    result = frame.copy(deep=True)
    if method == "drop_rows":
        return result.dropna(subset=columns, how="any")
    for column in columns:
        if method == "constant":
            value = _coerce_constant(result[column], spec.get("value"))
            result[column] = result[column].fillna(value)
        elif method in {"mean", "median"}:
            numeric = pd.to_numeric(result[column], errors="coerce")
            value = numeric.mean() if method == "mean" else numeric.median()
            if pd.isna(value):
                raise CleaningError(f"字段 {column} 无法计算{method}。")
            result[column] = result[column].fillna(value)
        elif method == "mode":
            modes = result[column].mode(dropna=True)
            if modes.empty:
                raise CleaningError(f"字段 {column} 没有可用于填充的众数。")
            result[column] = result[column].fillna(modes.iloc[0])
        elif method == "ffill":
            result[column] = result[column].ffill()
        elif method == "bfill":
            result[column] = result[column].bfill()
        else:
            raise CleaningError("不支持的空值处理方式。")
    return result


def _apply_duplicates(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    columns = spec.get("columns", [])
    if columns:
        _ensure_columns(frame, columns)
    keep = spec.get("keep", "first")
    if keep not in {"first", "last"}:
        raise CleaningError("去重保留规则必须是 first 或 last。")
    return frame.drop_duplicates(subset=columns or None, keep=keep)


def _apply_type(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    columns = spec.get("columns", [])
    _ensure_columns(frame, columns)
    target = spec.get("target")
    errors = spec.get("errors", "coerce")
    if errors not in {"coerce", "raise"}:
        raise CleaningError("类型转换错误策略不受支持。")
    result = frame.copy(deep=True)
    for column in columns:
        try:
            if target == "string":
                converted = result[column].astype("string")
            elif target == "integer":
                converted = pd.to_numeric(result[column], errors=errors).astype("Int64")
            elif target == "float":
                converted = pd.to_numeric(result[column], errors=errors).astype("Float64")
            elif target == "boolean":
                converted = _boolean_series(result[column], errors)
            elif target == "datetime":
                converted = pd.to_datetime(result[column], errors=errors)
            else:
                raise CleaningError("不支持的目标类型。")
        except (TypeError, ValueError) as exc:
            raise CleaningError(f"字段 {column} 转换失败：{exc}") from exc
        result[column] = converted
    return result


def _apply_normalize(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    columns = spec.get("columns", [])
    _ensure_columns(frame, columns)
    mapping = spec.get("mapping", {})
    strip = bool(spec.get("strip", True))
    case_mode = spec.get("case")
    if case_mode not in {None, "lower", "upper"}:
        raise CleaningError("大小写规则不受支持。")
    if not mapping and not strip and case_mode is None:
        raise CleaningError("请至少选择去空格、大小写转换或填写映射。")
    result = frame.copy(deep=True)
    for column in columns:
        def normalize(value: Any) -> Any:
            if pd.isna(value) or not isinstance(value, str):
                return value
            normalized = value.strip() if strip else value
            if case_mode == "lower":
                normalized = normalized.lower()
            elif case_mode == "upper":
                normalized = normalized.upper()
            return mapping.get(normalized, normalized)

        result[column] = result[column].map(normalize)
    return result


def prepare_cleaning_preview(frame: pd.DataFrame, spec: dict[str, Any], base_revision: int) -> CleaningPreview:
    action = spec.get("action")
    before = frame.copy(deep=True)
    if action == "missing":
        after = _apply_missing(before, spec)
    elif action == "duplicates":
        after = _apply_duplicates(before, spec)
    elif action == "type":
        after = _apply_type(before, spec)
    elif action == "normalize":
        after = _apply_normalize(before, spec)
    else:
        raise CleaningError("不支持的清洗动作。")

    changed = _changed_indices(before, after)
    before_view = before.loc[changed[:10]].copy() if changed else before.head(0).copy()
    after_indices = [index for index in changed[:10] if index in after.index]
    after_view = after.loc[after_indices].copy() if after_indices else after.head(0).copy()
    before_overview = _overview(before)
    after_overview = _overview(after)
    summary = {
        "action": action,
        "label": ACTION_LABELS[action],
        "affected_rows": len(changed),
        "rows_before": before_overview["rows"],
        "rows_after": after_overview["rows"],
        "missing_before": before_overview["missing_cells"],
        "missing_after": after_overview["missing_cells"],
        "duplicates_before": before_overview["duplicate_rows"],
        "duplicates_after": after_overview["duplicate_rows"],
    }
    return CleaningPreview(
        result=after.reset_index(drop=True),
        summary=summary,
        before_sample=before_view.reset_index(names="原行索引"),
        after_sample=after_view.reset_index(names="原行索引"),
        spec=spec,
        base_revision=base_revision,
    )

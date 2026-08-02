from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


ALLOWED_AGGREGATIONS = {
    "求和": "sum",
    "平均值": "mean",
    "最大值": "max",
    "最小值": "min",
    "计数": "count",
    "去重计数": "nunique",
}


@dataclass(frozen=True)
class AggregationResult:
    table: pd.DataFrame
    description: str
    config: dict[str, Any]


def numeric_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    if not numeric_columns:
        return pd.DataFrame(columns=["字段", "有效值数", "求和", "平均值", "最小值", "最大值"])
    rows = []
    for column in numeric_columns:
        series = frame[column]
        rows.append(
            {
                "字段": column,
                "有效值数": int(series.count()),
                "求和": series.sum(skipna=True),
                "平均值": series.mean(skipna=True),
                "最小值": series.min(skipna=True),
                "最大值": series.max(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def categorical_statistics(frame: pd.DataFrame, max_columns: int = 10) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        if is_numeric_dtype(frame[column]):
            continue
        series = frame[column].dropna().astype("string")
        if series.empty:
            continue
        top = series.value_counts().head(3)
        rows.append(
            {
                "字段": column,
                "有效值数": int(series.count()),
                "唯一值数": int(series.nunique()),
                "Top 3": "；".join(f"{key} ({value})" for key, value in top.items()),
            }
        )
        if len(rows) >= max_columns:
            break
    return pd.DataFrame(rows, columns=["字段", "有效值数", "唯一值数", "Top 3"])


def aggregate(frame: pd.DataFrame, metric: str, operation_label: str, group_by: str | None = None) -> AggregationResult:
    if metric not in frame.columns:
        raise ValueError(f"指标字段不存在：{metric}")
    operation = ALLOWED_AGGREGATIONS.get(operation_label)
    if operation is None:
        raise ValueError(f"不支持的聚合操作：{operation_label}")
    if operation in {"sum", "mean", "max", "min"} and not is_numeric_dtype(frame[metric]):
        raise ValueError("当前操作需要数值字段。")
    if group_by and group_by not in frame.columns:
        raise ValueError(f"分组字段不存在：{group_by}")

    if group_by:
        result = (
            frame.groupby(group_by, dropna=False)[metric]
            .agg(operation)
            .reset_index(name=f"{metric}_{operation}")
            .sort_values(f"{metric}_{operation}", ascending=False)
            .reset_index(drop=True)
        )
        description = f"按「{group_by}」分组，对「{metric}」执行{operation_label}"
    else:
        value = getattr(frame[metric], operation)()
        result = pd.DataFrame({"指标": [metric], "操作": [operation_label], "结果": [value]})
        description = f"对「{metric}」执行{operation_label}"
    return AggregationResult(
        table=result,
        description=description,
        config={"metric": metric, "operation": operation_label, "group_by": group_by},
    )


def _records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    safe = frame.head(limit).copy()
    safe = safe.astype(object).where(pd.notna(safe), None)
    return json.loads(safe.to_json(orient="records", force_ascii=False, date_format="iso"))


def build_analysis_context(
    frame: pd.DataFrame,
    aggregation_result: AggregationResult | None,
    cleaning_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric = numeric_statistics(frame)
    categorical = categorical_statistics(frame)
    context: dict[str, Any] = {
        "dataset_overview": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
        },
        "cleaning_log": cleaning_steps,
        "numeric_statistics": _records(numeric, 20),
        "categorical_statistics": _records(categorical, 10),
        "user_aggregation": None,
        "data_scope_note": "上下文仅包含汇总统计，不包含完整原始明细。",
    }
    if aggregation_result is not None:
        context["user_aggregation"] = {
            "description": aggregation_result.description,
            "config": aggregation_result.config,
            "rows": _records(aggregation_result.table, 50),
            "truncated": len(aggregation_result.table) > 50,
        }
    return context

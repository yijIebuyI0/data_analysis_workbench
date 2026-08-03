from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.analytics import (
    ALLOWED_AGGREGATIONS,
    AggregationResult,
    aggregate,
    build_analysis_context,
    categorical_statistics,
    numeric_statistics,
)
from src.cleaning import CleaningError, CleaningPreview, prepare_cleaning_preview
from src.data_io import DataLoadError, LoadedData, excel_sheet_names, file_signature, load_uploaded_bytes
from src.exporting import audit_download, dataframe_download
from src.llm import DEFAULT_MODEL, SYSTEM_PROMPT, StrategyGenerationError, generate_strategy, resolve_api_key
from src.profiling import assess_dataframe


st.set_page_config(
    page_title="DataLab · AI 数据分析工作台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
  --lab-ink: #17252f;
  --lab-teal: #0f766e;
  --lab-teal-soft: #dcefeb;
  --lab-amber: #b35c12;
  --lab-paper: #f3f7f8;
  --lab-line: #cad7db;
}
.stApp { background: var(--lab-paper); color: var(--lab-ink); }
[data-testid="stSidebar"] { background: #e8f0f2; border-right: 1px solid var(--lab-line); }
.block-container { max-width: 1180px; padding-top: 2.25rem; }
.lab-kicker { color: var(--lab-teal); font-size: .78rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.lab-title { color: var(--lab-ink); font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 650; letter-spacing: -.035em; line-height: 1.05; margin: .25rem 0 .75rem; }
.lab-subtitle { color: #51636c; font-size: 1rem; max-width: 780px; }
.lab-rule { border-top: 1px solid var(--lab-line); margin: 1rem 0 1.5rem; }
.stage-rail { display: grid; gap: .55rem; margin: 1rem 0; }
.stage-item { align-items: center; border-left: 3px solid var(--lab-line); display: grid; gap: .15rem; grid-template-columns: 2rem 1fr; padding: .35rem 0 .35rem .65rem; }
.stage-item strong { color: var(--lab-ink); font-size: .88rem; }
.stage-item span { color: #65767d; font-size: .74rem; }
.stage-item.done { border-left-color: var(--lab-teal); }
.stage-item.active { background: var(--lab-teal-soft); border-left-color: var(--lab-teal); }
.stage-number { color: var(--lab-teal); font-family: Consolas, monospace; font-weight: 700; }
.checkpoint { border-left: 4px solid var(--lab-teal); padding: .4rem 0 .4rem .85rem; }
.checkpoint b { color: var(--lab-ink); }
.checkpoint small { color: #63757d; display: block; margin-top: .15rem; }
.stButton > button[kind="primary"] { background: var(--lab-teal); border-color: var(--lab-teal); }
.stButton > button, .stDownloadButton > button { border-radius: .3rem; }
[data-testid="stMetric"] { background: rgba(255,255,255,.56); border: 1px solid var(--lab-line); padding: .8rem 1rem; }
[data-testid="stMetricLabel"] { color: #5b6d75; }
h1, h2, h3 { color: var(--lab-ink); letter-spacing: -.02em; }
@media (max-width: 700px) { .block-container { padding-top: 1.2rem; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


STATE_DEFAULTS = {
    "source_key": None,
    "loaded": None,
    "working_df": None,
    "cleaning_steps": [],
    "pending_preview": None,
    "cleaning_finalized": False,
    "aggregation_result": None,
    "analysis_finalized": False,
    "analysis_context": None,
    "strategy_result": None,
}


def initialize_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_for_source(loaded: LoadedData, source_key: str) -> None:
    st.session_state.source_key = source_key
    st.session_state.loaded = loaded
    st.session_state.working_df = loaded.dataframe.copy(deep=True)
    st.session_state.cleaning_steps = []
    st.session_state.pending_preview = None
    st.session_state.cleaning_finalized = False
    st.session_state.aggregation_result = None
    st.session_state.analysis_finalized = False
    st.session_state.analysis_context = None
    st.session_state.strategy_result = None


def invalidate_downstream() -> None:
    st.session_state.cleaning_finalized = False
    st.session_state.aggregation_result = None
    st.session_state.analysis_finalized = False
    st.session_state.analysis_context = None
    st.session_state.strategy_result = None


def parse_mapping(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        separator = "=>" if "=>" in line else "=" if "=" in line else None
        if separator is None:
            raise CleaningError(f"映射第 {line_number} 行缺少 = 或 =>。")
        source, target = [part.strip() for part in line.split(separator, 1)]
        if not source:
            raise CleaningError(f"映射第 {line_number} 行缺少原值。")
        mapping[source] = target
    return mapping


def render_header() -> None:
    st.markdown('<div class="lab-kicker">Portfolio MVP · Data workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="lab-title">DataLab 数据分析工作台</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lab-subtitle">把数据清洗、描述性分析和 AI 策略建议串成一条可确认、可回溯的分析流程。</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="lab-rule"></div>', unsafe_allow_html=True)


def render_stage_rail(active: str) -> None:
    has_data = st.session_state.loaded is not None
    states = [
        ("01", "数据评估与清洗", "已确认" if st.session_state.cleaning_finalized else "进行中" if has_data else "待上传"),
        ("02", "描述统计与聚合", "已确认" if st.session_state.analysis_finalized else "可开始" if st.session_state.cleaning_finalized else "等待清洗"),
        ("03", "AI 业务策略", "已生成" if st.session_state.strategy_result else "可开始" if st.session_state.analysis_finalized else "等待分析"),
    ]
    html = ['<div class="stage-rail">']
    active_number = active[:2]
    for number, label, status in states:
        css = "active" if number == active_number else "done" if status in {"已确认", "已生成"} else ""
        html.append(
            f'<div class="stage-item {css}"><div class="stage-number">{number}</div>'
            f'<div><strong>{label}</strong><span>{status}</span></div></div>'
        )
    html.append("</div>")
    st.sidebar.markdown("".join(html), unsafe_allow_html=True)


def load_uploader() -> None:
    st.sidebar.markdown("### 数据入口")
    uploaded = st.sidebar.file_uploader(
        "上传 CSV、TXT、Excel 或 JSON",
        type=["csv", "txt", "xlsx", "json"],
        help="单文件不超过 200MB；当前版本最多处理 50 万行。",
    )
    if uploaded is None:
        return
    data = uploaded.getvalue()
    sheet_name = None
    if Path(uploaded.name).suffix.lower() == ".xlsx":
        try:
            sheets = excel_sheet_names(data)
        except DataLoadError as exc:
            st.sidebar.error(str(exc))
            return
        sheet_name = st.sidebar.selectbox("选择工作表", sheets)
    source_key = f"{file_signature(data)}:{sheet_name or ''}"
    if source_key == st.session_state.source_key:
        return
    try:
        loaded = load_uploaded_bytes(data, uploaded.name, sheet_name)
    except DataLoadError as exc:
        st.sidebar.error(str(exc))
        return
    reset_for_source(loaded, source_key)


def show_empty_state() -> None:
    st.info("从左侧上传一个 CSV、TXT、XLSX 或 JSON 文件开始。")
    columns = st.columns(3)
    copy = [
        ("01", "先看数据质量", "自动识别空值、重复行、字段类型和疑似口径问题。"),
        ("02", "逐步预览再执行", "每个清洗动作都先展示影响范围，确认后才写入工作副本。"),
        ("03", "让结论有数据依据", "聚合结果通过工具消息交给模型，再结合业务问题输出策略。"),
    ]
    for container, (number, title, description) in zip(columns, copy):
        with container:
            st.markdown(f"**{number} · {title}**")
            st.caption(description)


def render_profile(frame: pd.DataFrame) -> None:
    profile = assess_dataframe(frame)
    cols = st.columns(4)
    cols[0].metric("行数", f"{profile.overview['rows']:,}")
    cols[1].metric("字段数", f"{profile.overview['columns']:,}")
    cols[2].metric("空值单元格", f"{profile.overview['missing_cells']:,}")
    cols[3].metric("完全重复行", f"{profile.overview['duplicate_rows']:,}")
    if profile.issues.empty:
        st.success("基础质量检查未发现空值、重复行或明显的字段口径问题。")
    else:
        st.markdown("#### 自动评估发现")
        st.dataframe(profile.issues, hide_index=True, width="stretch")
    with st.expander("查看字段画像", expanded=False):
        column_profile = profile.columns.copy()
        column_profile["空值率"] = column_profile["空值率"].map(lambda value: f"{value:.1%}")
        st.dataframe(column_profile, hide_index=True, width="stretch")
    with st.expander("查看数据样例", expanded=False):
        st.dataframe(frame.head(20), hide_index=True, width="stretch")


def cleaning_spec_form(frame: pd.DataFrame) -> dict | None:
    st.markdown("### 新增一个清洗步骤")
    action_label = st.selectbox("选择清洗类型", ["空值处理", "重复值处理", "类型处理", "口径不一致词处理"])
    action_map = {
        "空值处理": "missing",
        "重复值处理": "duplicates",
        "类型处理": "type",
        "口径不一致词处理": "normalize",
    }
    action = action_map[action_label]
    spec: dict = {"action": action}
    with st.form(f"cleaning_form_{action}", clear_on_submit=False):
        if action == "missing":
            default_columns = [column for column in frame.columns if frame[column].isna().any()]
            spec["columns"] = st.multiselect("处理字段", list(frame.columns), default=default_columns[:3])
            method_label = st.selectbox(
                "处理方式",
                ["删除含空值的行", "固定值填充", "平均值填充", "中位数填充", "众数填充", "向前填充", "向后填充"],
                index=1,
                help="默认使用较保守的固定值填充；删除行仍需主动选择并预览。",
            )
            method_map = {
                "删除含空值的行": "drop_rows",
                "固定值填充": "constant",
                "平均值填充": "mean",
                "中位数填充": "median",
                "众数填充": "mode",
                "向前填充": "ffill",
                "向后填充": "bfill",
            }
            spec["method"] = method_map[method_label]
            spec["value"] = st.text_input("固定填充值", help="仅在选择“固定值填充”时使用。")
        elif action == "duplicates":
            spec["columns"] = st.multiselect("去重键", list(frame.columns), help="不选择字段时按整行去重。")
            keep_label = st.selectbox("重复时保留", ["第一条", "最后一条"])
            spec["keep"] = "first" if keep_label == "第一条" else "last"
        elif action == "type":
            spec["columns"] = st.multiselect("转换字段", list(frame.columns))
            target_label = st.selectbox("目标类型", ["文本", "整数", "小数", "布尔值", "日期时间"])
            spec["target"] = {"文本": "string", "整数": "integer", "小数": "float", "布尔值": "boolean", "日期时间": "datetime"}[target_label]
            error_label = st.selectbox("无法转换时", ["设为空值并继续", "停止本步骤"])
            spec["errors"] = "coerce" if error_label == "设为空值并继续" else "raise"
        else:
            spec["columns"] = st.multiselect("统一字段", list(frame.columns))
            spec["strip"] = st.checkbox("移除文本首尾空格", value=True)
            case_label = st.selectbox("大小写处理", ["保持不变", "统一小写", "统一大写"])
            spec["case"] = {"保持不变": None, "统一小写": "lower", "统一大写": "upper"}[case_label]
            mapping_text = st.text_area(
                "值映射（每行一条）",
                placeholder="上海市 => 上海\nShanghai => 上海",
                help="使用“原值 => 新值”格式。可以只做去空格或大小写处理，不填写映射。",
            )
            spec["mapping_text"] = mapping_text
        submitted = st.form_submit_button("生成步骤预览", type="primary", width="stretch")
    if not submitted:
        return None
    if action == "normalize":
        spec["mapping"] = parse_mapping(spec.pop("mapping_text"))
    if action == "missing" and spec["method"] != "constant":
        spec.pop("value", None)
    return spec


def render_pending_preview(preview: CleaningPreview) -> None:
    st.markdown("### 待确认步骤")
    summary = preview.summary
    columns = st.columns(4)
    columns[0].metric("影响行数", summary["affected_rows"])
    columns[1].metric("结果行数", summary["rows_after"], summary["rows_after"] - summary["rows_before"])
    columns[2].metric("结果空值", summary["missing_after"], summary["missing_after"] - summary["missing_before"])
    duplicate_label = "结果键重复行" if summary["action"] == "duplicates" else "结果重复行"
    columns[3].metric(duplicate_label, summary["duplicates_after"], summary["duplicates_after"] - summary["duplicates_before"])
    if summary["affected_rows"] == 0:
        st.warning("该步骤不会改变当前数据，请检查字段和参数。")
    elif not preview.before_sample.empty:
        if summary["action"] == "duplicates":
            st.info("同一“重复组”内，所选去重键的值完全一致；预览同时展示将保留和将删除的记录。")
        left, right = st.columns(2)
        with left:
            st.caption("去重前：重复组明细" if summary["action"] == "duplicates" else "执行前样例")
            st.dataframe(preview.before_sample, hide_index=True, width="stretch")
        with right:
            st.caption("去重后：每组保留记录" if summary["action"] == "duplicates" else "执行后样例")
            st.dataframe(preview.after_sample, hide_index=True, width="stretch")
    confirm_col, cancel_col = st.columns([1, 1])
    if confirm_col.button("确认并执行该步骤", type="primary", width="stretch"):
        if preview.base_revision != len(st.session_state.cleaning_steps):
            st.error("数据版本已变化，请重新生成预览。")
        else:
            st.session_state.working_df = preview.result.copy(deep=True)
            st.session_state.cleaning_steps.append(
                {
                    "step": len(st.session_state.cleaning_steps) + 1,
                    "confirmed_at": datetime.now().isoformat(timespec="seconds"),
                    "spec": preview.spec,
                    "summary": preview.summary,
                }
            )
            st.session_state.pending_preview = None
            invalidate_downstream()
            st.rerun()
    if cancel_col.button("取消该步骤", width="stretch"):
        st.session_state.pending_preview = None
        st.rerun()


def render_cleaning_log() -> None:
    steps = st.session_state.cleaning_steps
    if not steps:
        return
    st.markdown("### 已确认步骤")
    for step in reversed(steps):
        summary = step["summary"]
        st.markdown(
            f'<div class="checkpoint"><b>{step["step"]:02d} · {summary["label"]}</b>'
            f'<small>影响 {summary["affected_rows"]} 行 · {summary["rows_before"]} → {summary["rows_after"]} 行 · {step["confirmed_at"]}</small></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"查看第 {step['step']} 步参数"):
            st.json(step["spec"])


def render_cleaning_stage(frame: pd.DataFrame) -> None:
    st.markdown("## 第一板块 · 数据评估与分步清洗")
    st.caption("所有动作只作用于会话内工作副本；先预览，确认后才进入下一版本。")
    render_profile(frame)
    st.markdown('<div class="lab-rule"></div>', unsafe_allow_html=True)
    try:
        spec = cleaning_spec_form(frame)
        if spec is not None:
            st.session_state.pending_preview = prepare_cleaning_preview(
                frame, spec, base_revision=len(st.session_state.cleaning_steps)
            )
    except CleaningError as exc:
        st.error(str(exc))
    pending = st.session_state.pending_preview
    if pending is not None:
        render_pending_preview(pending)
    render_cleaning_log()
    st.markdown('<div class="lab-rule"></div>', unsafe_allow_html=True)
    if st.button("确认清洗完成", type="primary", width="stretch"):
        st.session_state.cleaning_finalized = True
        st.session_state.pending_preview = None
        st.success("清洗结果已锁定，可以切换到第二板块。")


def render_aggregation_chart(result: AggregationResult) -> None:
    group_by = result.config.get("group_by")
    if not group_by or result.table.empty:
        return
    value_column = [column for column in result.table.columns if column != group_by][0]
    chart_data = result.table.head(30)
    chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#0f766e", cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{value_column}:Q", title=result.description),
            y=alt.Y(f"{group_by}:N", sort="-x", title=group_by),
            tooltip=[alt.Tooltip(f"{group_by}:N"), alt.Tooltip(f"{value_column}:Q")],
        )
        .properties(height=max(220, min(620, len(chart_data) * 28)))
    )
    st.altair_chart(chart, width="stretch")


def render_analysis_stage(frame: pd.DataFrame) -> None:
    st.markdown("## 第二板块 · 描述统计与聚合分析")
    if not st.session_state.cleaning_finalized:
        st.warning("请先在第一板块确认清洗结果。")
        return
    st.caption("先展示标准描述统计，再通过下拉框完成一次自定义字段聚合。")
    numeric = numeric_statistics(frame)
    categorical = categorical_statistics(frame)
    st.markdown("### 数值字段概览")
    if numeric.empty:
        st.info("当前数据没有可直接汇总的数值字段。")
    else:
        st.dataframe(numeric, hide_index=True, width="stretch")
    with st.expander("分类字段概览", expanded=False):
        if categorical.empty:
            st.info("当前数据没有分类字段。")
        else:
            st.dataframe(categorical, hide_index=True, width="stretch")

    st.markdown("### 自定义聚合")
    metric = st.selectbox("指标字段", list(frame.columns), key="metric_column")
    operation = st.selectbox("计算方式", list(ALLOWED_AGGREGATIONS), key="aggregation_operation")
    group_option = st.selectbox("分组字段", ["不分组"] + list(frame.columns), key="group_column")
    group_by = None if group_option == "不分组" else group_option
    if st.button("执行聚合分析", type="primary"):
        try:
            st.session_state.aggregation_result = aggregate(frame, metric, operation, group_by)
            st.session_state.analysis_finalized = False
            st.session_state.analysis_context = None
            st.session_state.strategy_result = None
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.aggregation_result
    if result is not None:
        st.markdown(f"**{result.description}**")
        st.dataframe(result.table.head(100), hide_index=True, width="stretch")
        render_aggregation_chart(result)

    st.markdown('<div class="lab-rule"></div>', unsafe_allow_html=True)
    if st.button("确认分析结果", type="primary", width="stretch"):
        st.session_state.analysis_context = build_analysis_context(
            frame, st.session_state.aggregation_result, st.session_state.cleaning_steps
        )
        st.session_state.analysis_finalized = True
        st.success("分析上下文已锁定，可以切换到第三板块。")


def render_strategy_stage(frame: pd.DataFrame, loaded: LoadedData) -> None:
    st.markdown("## 第三板块 · AI 业务策略")
    if not st.session_state.analysis_finalized:
        st.warning("请先在第二板块确认分析结果。")
        return
    context = st.session_state.analysis_context
    st.caption("模型先调用分析摘要工具，再结合你的业务问题输出建议。完整明细不会发送给模型。")
    with st.expander("查看发送给模型的工具消息", expanded=False):
        st.json(context)
    with st.expander("查看系统提示词", expanded=False):
        st.code(SYSTEM_PROMPT, language="text")

    prompt = st.text_area(
        "业务问题",
        placeholder="例如：请从客户分层和收入增长角度，给出三条下阶段运营策略，并说明验证指标。",
        height=120,
    )
    key_input = st.text_input(
        "DeepSeek API Key",
        type="password",
        help="优先读取当前会话输入，其次读取 Streamlit secrets 或环境变量；不会写入导出文件。",
    )
    try:
        secret_key = str(st.secrets.get("DEEPSEEK_API_KEY", ""))
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        secret_key = ""
    available_models = ["deepseek-v4-flash", "deepseek-v4-pro"]
    configured_model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    if configured_model not in available_models:
        available_models.insert(0, configured_model)
    model = st.selectbox("模型", available_models, index=available_models.index(configured_model))
    if st.button("生成业务策略", type="primary", width="stretch"):
        api_key = resolve_api_key(key_input or secret_key)
        if not api_key:
            st.error("请配置 DEEPSEEK_API_KEY，或在当前会话中输入 API Key。")
        else:
            with st.spinner("正在读取分析工具结果并生成策略…"):
                try:
                    st.session_state.strategy_result = generate_strategy(prompt, context, api_key, model=model)
                except StrategyGenerationError as exc:
                    st.error(str(exc))
    if st.session_state.strategy_result:
        st.markdown("### 策略输出")
        st.markdown(st.session_state.strategy_result)

    st.markdown('<div class="lab-rule"></div>', unsafe_allow_html=True)
    st.markdown("### 另存结果")
    data_bytes, file_name, mime = dataframe_download(
        frame,
        loaded.file_name,
        loaded.file_format,
        source_metadata=loaded.metadata,
    )
    source_info = {
        "file_name": loaded.file_name,
        "file_format": loaded.file_format,
        "file_hash": loaded.file_hash,
        "original_rows": len(loaded.dataframe),
        "cleaned_rows": len(frame),
    }
    audit_bytes = audit_download(source_info, st.session_state.cleaning_steps, context)
    export_columns = st.columns(3)
    export_columns[0].download_button(
        "下载清洗后数据",
        data=data_bytes,
        file_name=file_name,
        mime=mime,
        on_click="ignore",
        type="primary",
        width="stretch",
    )
    export_columns[1].download_button(
        "下载流程记录",
        data=audit_bytes,
        file_name=f"{Path(loaded.file_name).stem}_audit.json",
        mime="application/json",
        on_click="ignore",
        width="stretch",
    )
    strategy_text = st.session_state.strategy_result or "尚未生成 AI 策略。"
    export_columns[2].download_button(
        "下载策略 Markdown",
        data=strategy_text.encode("utf-8"),
        file_name=f"{Path(loaded.file_name).stem}_strategy.md",
        mime="text/markdown",
        on_click="ignore",
        width="stretch",
    )
    st.caption("下载文件使用新文件名，原上传文件不会被覆盖。")


initialize_state()
load_uploader()
render_header()

stage_options = ["01 · 数据评估与清洗", "02 · 描述统计与聚合", "03 · AI 业务策略"]
active_stage = st.segmented_control("工作阶段", stage_options, default=stage_options[0], key="active_stage")
active_stage = active_stage or stage_options[0]
render_stage_rail(active_stage)

loaded = st.session_state.loaded
if loaded is None:
    show_empty_state()
    st.stop()

frame = st.session_state.working_df
st.sidebar.markdown("### 当前数据")
st.sidebar.caption(loaded.file_name)
st.sidebar.write(f"{len(frame):,} 行 × {len(frame.columns):,} 列")
st.sidebar.write(f"已确认清洗步骤：{len(st.session_state.cleaning_steps)}")

if active_stage.startswith("01"):
    render_cleaning_stage(frame)
elif active_stage.startswith("02"):
    render_analysis_stage(frame)
else:
    render_strategy_stage(frame, loaded)

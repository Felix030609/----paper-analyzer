from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_components import (  # noqa: E402
    CATEGORY_LABELS,
    inject_css,
    normalize_label_name,
    render_academic_space_map,
    render_hero,
    render_evidence_chain,
    render_label_bar_chart,
    render_label_matrix,
    render_core_label_pills,
    render_radar_chart,
    render_report,
    render_section_header,
    render_summary_judgement_card,
)
from scripts.analyze_uploaded_paper import (  # noqa: E402
    MAX_ANALYSIS_CHARS,
    MAX_TOP_K,
    QUICK_TEST_LABELS,
    analyze_text,
    load_label_definitions,
)
from scripts.deepseek_client import (  # noqa: E402
    MissingAPIKeyError,
    get_deepseek_api_key,
    get_deepseek_key_status,
    get_deepseek_model,
)
from scripts.report_export import markdown_to_pdf_bytes  # noqa: E402
from scripts.text_preprocessing import (  # noqa: E402
    build_cleaned_document,
    extract_raw_text_from_file,
)
from scripts.usage_logger import log_event, read_usage_events, usage_summary  # noqa: E402


EXCEL_PATH = PROJECT_ROOT / "data" / "training" / "人文社科论文思想谱系训练数据模板_已补证据.xlsx"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_OPTIONS = {
    "V4 Flash｜速度优先": "deepseek-v4-flash",
    "V4 Pro｜质量优先": "deepseek-v4-pro",
}
MODEL_LABELS = {value: label for label, value in MODEL_OPTIONS.items()}


st.set_page_config(
    page_title="人文社科论文思想谱系分析工具",
    layout="wide",
)


def safe_value(row: pd.Series, column: str, default: str = "暂无") -> Any:
    if column not in row.index:
        return default
    value = row.get(column)
    if pd.isna(value) or str(value).strip() == "":
        return default
    return value


def display_value(value: Any) -> str:
    if pd.isna(value):
        return "暂无"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_demo_excel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not EXCEL_PATH.exists():
        st.warning("未找到示例论文数据，示例分析暂不可用。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        papers = pd.read_excel(EXCEL_PATH, sheet_name="01_papers")
        annotations = pd.read_excel(EXCEL_PATH, sheet_name="02_annotations")
        evidence = pd.read_excel(EXCEL_PATH, sheet_name="03_evidence")
        return papers, annotations, evidence
    except Exception as exc:
        st.warning(f"示例论文数据读取失败：{exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def label_category_map(label_definitions: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {
        normalize_label_name(item.get("label_name", "")): item.get("category", "")
        for item in label_definitions
    }
    for category, labels in CATEGORY_LABELS.items():
        for label in labels:
            mapping.setdefault(normalize_label_name(label), category)
    return mapping


def score_from_annotation(annotation_row: pd.Series, label_name: str) -> float:
    candidates = [label_name, label_name.replace("底层", "/底层")]
    for candidate in candidates:
        if candidate in annotation_row.index:
            value = annotation_row.get(candidate)
            if not pd.isna(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def p001_label_results(
    annotations: pd.DataFrame,
    evidence: pd.DataFrame,
    label_definitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if annotations.empty or "paper_id" not in annotations.columns:
        return []

    rows = annotations[annotations["paper_id"].astype(str).str.strip().eq("P001")]
    if rows.empty:
        return []

    annotation_row = rows.iloc[0]
    category_map = label_category_map(label_definitions)
    evidence_by_label: dict[str, list[dict[str, Any]]] = {}

    if not evidence.empty and {"paper_id", "label"}.issubset(evidence.columns):
        p001_evidence = evidence[evidence["paper_id"].astype(str).str.strip().eq("P001")]
        for _, row in p001_evidence.iterrows():
            label = normalize_label_name(row.get("label", ""))
            evidence_by_label.setdefault(label, []).append(
                {
                    "evidence_text": str(row.get("evidence_text", "") or "").strip(),
                    "reason": str(row.get("reason", "") or "").strip(),
                    "evidence_type": str(row.get("evidence_type", "证据") or "证据").strip(),
                    "confidence": row.get("confidence", ""),
                }
            )

    results = []
    for definition in label_definitions:
        raw_label = str(definition.get("label_name", "")).strip()
        label = normalize_label_name(raw_label)
        score = score_from_annotation(annotation_row, raw_label)
        evidence_items = evidence_by_label.get(label, [])
        confidence_values = pd.to_numeric(
            [item.get("confidence") for item in evidence_items],
            errors="coerce",
        )
        confidence_values = confidence_values[~pd.isna(confidence_values)]
        confidence = (
            float(confidence_values.max())
            if len(confidence_values)
            else safe_value(annotation_row, "confidence_1_5", 0)
        )
        reason = (
            evidence_items[0]["reason"]
            if evidence_items
            else str(safe_value(annotation_row, "overall_judgment", ""))
        )

        results.append(
            {
                "label_name": raw_label,
                "category": category_map.get(label, definition.get("category", "")),
                "score": score,
                "confidence": confidence,
                "evidence": [item["evidence_text"] for item in evidence_items[:3] if item["evidence_text"]],
                "evidence_items": evidence_items,
                "reason": reason,
                "uncertainty": "样本标注仍需结合全文语境复核。",
                "retrieved_paragraphs": [
                    {"rank": index + 1, "similarity": "示例证据", "text": item["evidence_text"]}
                    for index, item in enumerate(evidence_items[:3])
                    if item["evidence_text"]
                ],
            }
        )
    return results


def p001_metadata(papers: pd.DataFrame, annotations: pd.DataFrame) -> dict[str, Any]:
    default = {
        "title": "当代中国的思想状况与现代性问题",
        "author": "汪晖",
        "year": "暂无",
        "core_question": "暂无",
        "overall_judgment": "暂无",
    }
    if not papers.empty and "paper_id" in papers.columns:
        rows = papers[papers["paper_id"].astype(str).str.strip().eq("P001")]
        if not rows.empty:
            row = rows.iloc[0]
            default.update(
                {
                    "title": safe_value(row, "title", default["title"]),
                    "author": safe_value(row, "author", default["author"]),
                    "year": display_value(safe_value(row, "year", default["year"])),
                    "core_question": safe_value(row, "core_question", default["core_question"]),
                }
            )
    if not annotations.empty and "paper_id" in annotations.columns:
        rows = annotations[annotations["paper_id"].astype(str).str.strip().eq("P001")]
        if not rows.empty:
            default["overall_judgment"] = safe_value(
                rows.iloc[0],
                "overall_judgment",
                default["overall_judgment"],
            )
    return default


def demo_data(label_definitions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    papers, annotations, evidence = read_demo_excel()
    return p001_metadata(papers, annotations), p001_label_results(annotations, evidence, label_definitions)


def parse_uploaded_file(uploaded_file) -> str:
    return extract_raw_text_from_file(uploaded_file)


def uploaded_file_size(uploaded_file) -> int:
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int):
        return size
    try:
        return len(uploaded_file.getbuffer())
    except Exception:
        return len(uploaded_file.getvalue())


def demo_paper_display_title(meta: dict[str, Any]) -> str:
    author = str(meta.get("author") or "作者未知").strip()
    title = str(meta.get("title") or "示例论文").strip()
    year = str(meta.get("year") or "").strip()
    year_part = f"（{year}）" if year and year != "暂无" else ""
    return f"{author}《{title}》{year_part}"


def result_timestamp(result: dict[str, Any]) -> str:
    value = str(result.get("created_at") or "").strip()
    if value:
        try:
            return datetime.fromisoformat(value).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_analysis_state() -> None:
    defaults = {
        "analysis_status": "idle",
        "uploaded_file_name": "",
        "analysis_result": None,
        "report_markdown": "",
        "analysis_error": "",
        "current_step": "等待上传论文",
        "progress_value": 0.0,
        "completed_labels": [],
        "current_label": "",
        "total_labels": 0,
        "pending_analysis": None,
        "selected_model_name": "deepseek-v4-flash",
        "estimated_time": "",
        "analysis_mode": "快速测试",
        "use_enhanced_cleaning": False,
        "app_open_logged": False,
        "last_uploaded_signature": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if not st.session_state.app_open_logged:
        log_event("app_open")
        st.session_state.app_open_logged = True


def reset_analysis_state() -> None:
    st.session_state.analysis_status = "idle"
    st.session_state.uploaded_file_name = ""
    st.session_state.analysis_result = None
    st.session_state.report_markdown = ""
    st.session_state.analysis_error = ""
    st.session_state.current_step = "等待上传论文"
    st.session_state.progress_value = 0.0
    st.session_state.completed_labels = []
    st.session_state.current_label = ""
    st.session_state.total_labels = 0
    st.session_state.pending_analysis = None
    st.session_state.selected_model_name = "deepseek-v4-flash"
    st.session_state.estimated_time = ""
    st.session_state.analysis_mode = "快速测试"
    st.rerun()


def estimate_analysis_time(model_name: str, quick_test_mode: bool, text_length: int = 0) -> str:
    if quick_test_mode and model_name == "deepseek-v4-flash":
        base = "约 1—2 分钟"
    elif quick_test_mode:
        base = "约 2—3 分钟"
    elif model_name == "deepseek-v4-flash":
        base = "约 3—5 分钟"
    else:
        base = "约 5—10 分钟"
    if text_length > 30_000:
        return f"{base}。长文本可能额外增加 1—3 分钟。"
    return base


def model_display_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def set_running_state(
    *,
    file_name: str,
    raw_text: str,
    top_k: int,
    quick_test_mode: bool,
    core_only: bool,
    use_enhanced_cleaning: bool,
    original_text_length: int,
    upload_file_size: int,
    model_name: str,
    estimated_time: str,
    file_type: str,
) -> None:
    selected_labels = QUICK_TEST_LABELS if quick_test_mode else None
    st.session_state.analysis_status = "running"
    st.session_state.uploaded_file_name = file_name
    st.session_state.analysis_result = None
    st.session_state.report_markdown = ""
    st.session_state.analysis_error = ""
    st.session_state.current_step = "正在准备分析"
    st.session_state.progress_value = 0.01
    st.session_state.completed_labels = []
    st.session_state.current_label = ""
    st.session_state.total_labels = len(selected_labels) if selected_labels else 19
    st.session_state.selected_model_name = model_name
    st.session_state.estimated_time = estimated_time
    st.session_state.analysis_mode = "快速测试" if quick_test_mode else "完整分析"
    st.session_state.use_enhanced_cleaning = use_enhanced_cleaning
    st.session_state.pending_analysis = {
        "raw_text": raw_text,
        "top_k": top_k,
        "title": file_name,
        "selected_label_names": selected_labels,
        "core_only": core_only,
        "use_enhanced_cleaning": use_enhanced_cleaning,
        "quick_test_mode": quick_test_mode,
        "upload_file_size": upload_file_size,
        "original_text_length": original_text_length,
        "model_name": model_name,
        "estimated_time": estimated_time,
        "file_type": file_type,
        "started_at": time.time(),
    }
    st.rerun()


def update_progress_state(event: dict[str, Any], slots: dict[str, Any] | None = None) -> None:
    progress = float(event.get("progress", 0.0))
    status = str(event.get("status", "") or "正在分析")
    stage = str(event.get("stage", "") or "")
    label_name = str(event.get("label_name", "") or "")
    current = int(event.get("current", 0) or 0)
    total = int(event.get("total", 0) or 0)

    st.session_state.progress_value = max(0.0, min(1.0, progress))
    st.session_state.current_step = status
    if label_name:
        st.session_state.current_label = label_name
    if total:
        st.session_state.total_labels = total
    if stage == "label_done" and label_name:
        completed = list(st.session_state.get("completed_labels", []))
        if label_name not in completed:
            completed.append(label_name)
        st.session_state.completed_labels = completed

    if slots:
        slots["progress"].progress(st.session_state.progress_value)
        slots["status"].markdown(
            f"""
            <div class="progress-card">
              <div class="summary-label">当前步骤</div>
              <div class="subtle-text">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        current_label = st.session_state.get("current_label") or "等待标签分析"
        completed_count = len(st.session_state.get("completed_labels", []))
        total_count = st.session_state.get("total_labels") or total or 0
        slots["labels"].markdown(
            f"""
            <span class="status-pill">当前标签：{current_label}</span>
            <span class="status-pill status-pill-done">已完成：{completed_count}/{total_count}</span>
            """,
            unsafe_allow_html=True,
        )
        if current and total:
            slots["counter"].caption(f"标签进度：第 {current}/{total} 个")
        completed_labels = st.session_state.get("completed_labels", [])
        if completed_labels:
            slots["completed"].markdown(
                f"""
                <div class="progress-card">
                  <div class="summary-label">已完成标签</div>
                  <div class="subtle-text">{'、'.join(completed_labels[-8:])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_running_panel() -> dict[str, Any]:
    progress_value = float(st.session_state.get("progress_value", 0.0) or 0.0)
    current_step = st.session_state.get("current_step", "正在分析")
    current_label = st.session_state.get("current_label") or "等待标签分析"
    completed_count = len(st.session_state.get("completed_labels", []))
    total_labels = st.session_state.get("total_labels") or 0
    file_name = st.session_state.get("uploaded_file_name") or "已上传论文"
    model = st.session_state.get("selected_model_name") or get_deepseek_model()
    mode = st.session_state.get("analysis_mode") or "快速测试"
    estimated_time = st.session_state.get("estimated_time") or "正在估算"
    cleaning_mode = "增强文本清洗" if st.session_state.get("use_enhanced_cleaning") else "标准文本处理"

    st.markdown(
        f"""
        <div class="running-panel">
          <div class="section-title">正在分析你的论文</div>
          <div class="subtle-text">系统正在提取正文、召回证据并生成思想谱系报告。</div>
          <div style="margin-top: 12px;">
            <span class="status-pill">当前文件：{file_name}</span>
            <span class="status-pill">当前模型：{model_display_name(model)}</span>
            <span class="status-pill">当前模式：{mode}</span>
            <span class="status-pill">文本处理：{cleaning_mode}</span>
            <span class="status-pill">预计耗时：{estimated_time}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress = st.progress(progress_value)
    status = st.empty()
    labels = st.empty()
    counter = st.empty()
    completed = st.empty()
    status.markdown(
        f"""
        <div class="progress-card">
          <div class="summary-label">当前步骤</div>
          <div class="subtle-text">{current_step}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    labels.markdown(
        f"""
        <span class="status-pill">当前标签：{current_label}</span>
        <span class="status-pill status-pill-done">已完成：{completed_count}/{total_labels}</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-card">
          <div class="summary-label">分析流程</div>
          <div class="step-list">
            <div class="step-item">正文提取</div>
            <div class="step-item">文本清洗</div>
            <div class="step-item">段落切分</div>
            <div class="step-item">证据召回</div>
            <div class="step-item">标签评分</div>
            <div class="step-item">报告生成</div>
            <div class="step-item">PDF 导出准备</div>
          </div>
        </div>
        <div class="skeleton-card">
          <div class="summary-label">结果预览</div>
          <div class="subtle-text">分析完成后将在这里展示核心判断、标签矩阵、证据链与学术谱系图。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return {"progress": progress, "status": status, "labels": labels, "counter": counter, "completed": completed}


def render_error_panel() -> None:
    error = st.session_state.get("analysis_error") or "分析过程中发生未知错误。"
    st.markdown(
        f"""
        <div class="error-card">
          <div class="section-title">分析失败</div>
          <div class="subtle-text">{error}</div>
          <div class="subtle-text" style="margin-top: 10px;">请检查 API key、文件格式或网络连接后重新分析。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("重新分析", type="primary"):
        reset_analysis_state()


def render_upload_card(label_definitions: list[dict[str, Any]]) -> None:
    analysis_status = st.session_state.get("analysis_status", "idle")
    st.markdown('<div class="sticky-note">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="upload-title">上传论文开始分析</div>
            <div class="upload-copy">
              支持 PDF / TXT，系统将自动提取正文并生成分析报告。
            </div>
            """,
            unsafe_allow_html=True,
        )
        if analysis_status == "running":
            st.info("分析进行中，请勿重复提交。")
            st.caption(f"当前文件：{st.session_state.get('uploaded_file_name') or '已上传论文'}")
            st.caption(f"当前模型：{model_display_name(st.session_state.get('selected_model_name') or get_deepseek_model())}")
            st.caption(f"预计耗时：{st.session_state.get('estimated_time') or '正在估算'}")
            st.button("开始分析", disabled=True, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if analysis_status == "done" and st.session_state.get("analysis_result"):
            st.success("分析已完成。可继续上传新的论文重新分析。")
            st.caption(f"当前文件：{st.session_state.get('uploaded_file_name') or '已上传论文'}")
            st.caption(f"当前模型：{model_display_name(st.session_state.get('analysis_result', {}).get('model') or get_deepseek_model())}")
            if st.button("重新上传论文", width="stretch"):
                reset_analysis_state()

        if analysis_status == "error":
            st.error("上一次分析失败。请检查文件或 API 设置后重新分析。")
            if st.button("重新上传", width="stretch"):
                reset_analysis_state()

        uploaded_file = st.file_uploader(
            "上传论文文件",
            type=["pdf", "txt"],
            accept_multiple_files=False,
        )
        model_label = st.selectbox(
            "模型选择",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
        )
        selected_model_name = MODEL_OPTIONS[model_label]
        st.caption("V4 Flash：响应更快，适合快速预览。V4 Pro：分析更细，耗时更长，适合正式报告。")
        quick_test_mode = st.checkbox(
            "快速测试模式",
            value=True,
            help="快速测试模式用于验证 API、召回和报告链路，耗时更短；正式分析可关闭该模式。",
        )
        use_enhanced_cleaning = st.checkbox(
            "启用增强文本清洗（实验）",
            value=False,
            help=(
                "启用后，系统会在 RAG 分析前对 PDF 提取文本进行格式清洗与结构识别，"
                "尽量去除页眉页脚、脚注、参考文献和乱码干扰。该功能不会改写论文内容，"
                "但可能增加分析耗时。"
            ),
        )
        if quick_test_mode:
            top_k = st.slider(
                "每个标签召回证据段落数量",
                min_value=1,
                max_value=5,
                value=3,
                key="top_k_quick",
            )
        else:
            top_k = st.slider(
                "每个标签召回证据段落数量",
                min_value=3,
                max_value=MAX_TOP_K,
                value=5,
                key="top_k_full",
            )
        if quick_test_mode:
            st.caption("快速测试模式仅分析：现代性批判、思想史研究、社会历史批评、历史唯物主义、启蒙理性。")
        if use_enhanced_cleaning:
            st.caption("增强文本清洗会先进行规则清洗，并可调用大模型逐块做结构识别；清洗失败时会自动回退。")
        core_only = st.toggle("证据链只展示分数 >= 2 的核心标签", value=True)
        st.caption(f"当前模型：{selected_model_name}")
        key_status, key_message = get_deepseek_key_status()
        if key_status == "configured":
            st.success("DeepSeek API 已配置，用户无需输入 key。")
        elif key_status == "invalid":
            st.error(key_message)
        else:
            st.warning("未配置 DeepSeek API Key，无法生成报告。")
        base_estimated_time = estimate_analysis_time(selected_model_name, quick_test_mode, 0)
        mode_name = "快速测试" if quick_test_mode else "完整分析"
        st.info(
            f"预计耗时：{model_label} {mode_name}约 {base_estimated_time}。"
            "请保持页面打开，不要重复点击提交。"
        )
        st.caption("边界：只分析论文文本呈现出的思想倾向，不判断作者本人真实政治立场。")
        st.caption(
            "隐私提示：当前版本用于原型测试。上传论文仅用于本次分析；如启用增强文本清洗或大模型分析，"
            "系统会将相关文本片段发送至大模型 API。请勿上传涉密、未授权或不希望被第三方模型处理的文本。"
        )

        if not uploaded_file:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        file_type = Path(uploaded_file.name).suffix.lower().lstrip(".")
        file_size_bytes = uploaded_file_size(uploaded_file)
        upload_signature = f"{file_type}:{file_size_bytes}"
        if st.session_state.get("last_uploaded_signature") != upload_signature:
            log_event(
                "file_uploaded",
                {
                    "file_type": file_type,
                    "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
                },
            )
            st.session_state.last_uploaded_signature = upload_signature

        if file_size_bytes > MAX_UPLOAD_BYTES:
            st.error("文件超过 10MB。当前 V0 为控制成本，仅支持上传 10MB 以内的 PDF/TXT。")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        try:
            raw_text = parse_uploaded_file(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            st.markdown("</div>", unsafe_allow_html=True)
            return

        original_text_length = len(raw_text)
        preview_text = raw_text[:MAX_ANALYSIS_CHARS]
        estimated_time = estimate_analysis_time(selected_model_name, quick_test_mode, original_text_length)
        st.caption(f"已提取约 {original_text_length} 个字符。")
        if original_text_length > MAX_ANALYSIS_CHARS:
            st.warning(f"当前 V0 为控制成本，仅分析前 {MAX_ANALYSIS_CHARS:,} 字。")
        if estimated_time != base_estimated_time:
            st.info(
                f"当前文本预计耗时：{model_label} {mode_name}约 {estimated_time}。"
                "请保持页面打开，不要重复点击提交。"
            )
        with st.expander("查看前 1000 字预览", expanded=False):
            st.text_area("正文预览", preview_text[:1000], height=180)

        if st.button("开始分析", type="primary", width="stretch"):
            if not get_deepseek_api_key():
                st.error("未配置 DEEPSEEK_API_KEY。请在本地环境变量或 Streamlit Cloud Secrets 中配置后再分析。")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            log_event(
                "analysis_started",
                {
                    "file_type": file_type,
                    "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
                    "text_length": original_text_length,
                    "model_name": selected_model_name,
                    "analysis_mode": mode_name,
                    "top_k": top_k,
                    "success": True,
                },
            )
            set_running_state(
                file_name=uploaded_file.name,
                raw_text=raw_text,
                top_k=top_k,
                quick_test_mode=quick_test_mode,
                core_only=core_only,
                use_enhanced_cleaning=use_enhanced_cleaning,
                original_text_length=original_text_length,
                upload_file_size=file_size_bytes,
                model_name=selected_model_name,
                estimated_time=estimated_time,
                file_type=file_type,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_header_section(
    label_definitions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta, results = demo_data(label_definitions)
    status = st.session_state.get("analysis_status", "idle")
    left, right = st.columns([0.62, 0.38], gap="large")
    with right:
        render_upload_card(label_definitions)
    with left:
        render_hero()
        if status == "idle" and results:
            st.markdown(
                f"""
                <div class="sample-heading">
                  示例论文核心判断
                </div>
                <div class="subtle-text">以下展示系统如何将论文定位到具体的学术谱系与证据链中。</div>
                """,
                unsafe_allow_html=True,
            )
            render_summary_judgement_card(results, demo_paper_display_title(meta))
            render_core_label_pills(results)
        elif status == "running":
            st.markdown(
                """
                <div class="progress-card">
                  <div class="section-title">分析进行中</div>
                  <div class="subtle-text">下方工作台正在实时更新分析进度。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "done" and st.session_state.get("analysis_result"):
            st.markdown(
                """
                <div class="progress-card">
                  <div class="section-title">分析已完成</div>
                  <div class="subtle-text">下方工作台已生成完整分析结果、证据链与导出文件。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "error":
            st.markdown(
                """
                <div class="error-card">
                  <div class="section-title">分析未完成</div>
                  <div class="subtle-text">请在下方查看错误信息，或在右侧重新上传论文。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "idle":
            st.info("暂无可渲染的示例论文数据。")
    return meta, results


def render_main_content(
    label_definitions: list[dict[str, Any]],
    meta: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    status = st.session_state.get("analysis_status", "idle")
    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
    running_slots = None
    if status == "idle":
        render_sample_analysis_full_width(meta, results, label_definitions)
    elif status == "running":
        running_slots = render_running_panel()
    elif status == "done":
        render_upload_result(label_definitions)
    elif status == "error":
        render_error_panel()
    st.markdown("</div>", unsafe_allow_html=True)
    return running_slots


def demo_report_markdown() -> str:
    demo_report_path = PROJECT_ROOT / "outputs" / "P001_report.md"
    if demo_report_path.exists():
        return demo_report_path.read_text(encoding="utf-8")
    return "暂无完整报告。"


def render_sample_analysis_full_width(
    meta: dict[str, Any],
    results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
) -> None:
    render_section_header(
        "示例分析：汪晖《当代中国的思想状况与现代性问题》",
        "以下展示系统如何将论文定位到具体的学术谱系与证据链中。",
    )
    if not results:
        st.info("暂无可渲染的示例论文标签数据。")
        return
    render_analysis_tabs(
        results,
        label_definitions,
        demo_paper_display_title(meta),
        report_markdown=demo_report_markdown(),
        key_prefix="sample",
    )


def render_distribution_tab(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    render_label_bar_chart(label_results, label_definitions, key=f"{key_prefix}_bar")
    render_radar_chart(label_results, key=f"{key_prefix}_radar")


def render_export_tab(result: dict[str, Any]) -> None:
    report_markdown = str(result.get("report_markdown", ""))
    timestamp = result_timestamp(result)
    report_bytes = report_markdown.encode("utf-8")
    json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = markdown_to_pdf_bytes(
            report_markdown,
            title=str(result.get("title") or "用户上传论文"),
            created_at=str(result.get("created_at") or ""),
        )
    except Exception as exc:
        print(f"PDF 生成失败：{exc}", flush=True)

    st.markdown(
        """
        <div class="export-card">
          <div class="export-title">导出分析结果</div>
          <div class="subtle-text">可下载完整报告与结构化 JSON 数据。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if pdf_bytes:
            st.download_button(
                "下载 PDF 报告",
                data=pdf_bytes,
                file_name=f"paper_analysis_report_{timestamp}.pdf",
                mime="application/pdf",
                width="stretch",
                on_click=log_event,
                args=("download_pdf", {"model_name": result.get("model"), "success": True}),
            )
        else:
            st.warning("PDF 生成失败，请先下载 Markdown 报告。")
    with col2:
        st.download_button(
            "下载 Markdown 报告",
            data=report_bytes,
            file_name=f"paper_analysis_report_{timestamp}.md",
            mime="text/markdown",
            width="stretch",
            on_click=log_event,
            args=("download_markdown", {"model_name": result.get("model"), "success": True}),
        )
    with col3:
        st.download_button(
            "下载 JSON 数据",
            data=json_bytes,
            file_name=f"paper_analysis_result_{timestamp}.json",
            mime="application/json",
            width="stretch",
            on_click=log_event,
            args=("download_json", {"model_name": result.get("model"), "success": True}),
        )


def render_analysis_tabs(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    paper_title: str,
    *,
    report_markdown: str | None = None,
    export_result: dict[str, Any] | None = None,
    key_prefix: str,
) -> None:
    tab_names = ["证据链", "学术谱系图", "标签分布", "完整报告"]
    if export_result is not None:
        tab_names.append("导出报告")
    st.markdown('<div class="analysis-tabs">', unsafe_allow_html=True)
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_evidence_chain(label_results)
    with tabs[1]:
        render_academic_space_map(
            label_results,
            label_definitions,
            paper_title,
            key=f"{key_prefix}_academic_space",
        )
    with tabs[2]:
        render_distribution_tab(label_results, label_definitions, key_prefix=key_prefix)
    with tabs[3]:
        render_report(report_markdown or "暂无完整报告。")
    if export_result is not None:
        with tabs[4]:
            render_export_tab(export_result)
    st.markdown("</div>", unsafe_allow_html=True)


def render_cleaning_summary(result: dict[str, Any]) -> None:
    if not result.get("use_enhanced_cleaning"):
        return
    summary = result.get("cleaning_summary") or {}
    if not summary:
        st.info("已启用增强文本清洗，但暂无可展示的清洗摘要。")
        return

    method_map = {
        "rule_based": "规则清洗",
        "rule_based_plus_llm": "规则 + 大模型清洗",
        "raw_fallback": "原文回退",
    }
    st.markdown("#### 文本清洗摘要")
    cols = st.columns(3)
    metrics = [
        ("清洗方式", method_map.get(str(summary.get("cleaning_method")), str(summary.get("cleaning_method") or "未知"))),
        ("原始文本字数", summary.get("raw_text_length", 0)),
        ("清洗后字数", summary.get("cleaned_text_length", 0)),
        ("删除噪声", summary.get("removed_noise_count", 0)),
        ("疑似脚注", summary.get("possible_footnotes_count", 0)),
        ("疑似参考文献", summary.get("possible_references_count", 0)),
        ("清洗警告", summary.get("warnings_count", 0)),
    ]
    for index, (label, value) in enumerate(metrics):
        with cols[index % 3]:
            st.metric(label, value)
    warnings = summary.get("warnings") or []
    if warnings:
        with st.expander("查看清洗警告", expanded=False):
            for warning in warnings:
                st.write(f"- {warning}")


def run_pending_analysis(running_slots: dict[str, Any] | None = None) -> None:
    if st.session_state.get("analysis_status") != "running":
        return
    pending = st.session_state.get("pending_analysis")
    if not pending:
        st.session_state.analysis_status = "error"
        st.session_state.analysis_error = "未找到待分析任务，请重新上传论文。"
        st.rerun()

    def progress_callback(event: dict[str, Any]) -> None:
        update_progress_state(event, running_slots)

    try:
        update_progress_state(
            {
                "progress": 0.02,
                "status": "正在读取文件",
                "stage": "read_file",
                "total": st.session_state.get("total_labels", 0),
            },
            running_slots,
        )
        selected_labels = pending.get("selected_label_names")
        raw_text = str(pending.get("raw_text") or "")
        use_enhanced_cleaning = bool(pending.get("use_enhanced_cleaning", False))
        raw_for_analysis = raw_text[:MAX_ANALYSIS_CHARS]
        analysis_text = raw_for_analysis
        cleaning_summary: dict[str, Any] | None = None

        if use_enhanced_cleaning:
            update_progress_state(
                {
                    "progress": 0.04,
                    "status": "正在进行增强文本清洗与结构识别",
                    "stage": "text_preprocessing",
                    "total": st.session_state.get("total_labels", 0),
                },
                running_slots,
            )
            try:
                cleaning_result = build_cleaned_document(
                    raw_for_analysis,
                    use_llm_cleaning=True,
                    model_name=str(pending.get("model_name") or get_deepseek_model()),
                )
                cleaning_summary = cleaning_result.to_summary()
                cleaning_summary["raw_text_length"] = len(raw_text)
                if len(raw_text) > MAX_ANALYSIS_CHARS:
                    warnings = list(cleaning_summary.get("warnings") or [])
                    warnings.append(f"当前 V0 为控制成本，仅对前 {MAX_ANALYSIS_CHARS:,} 字进行清洗和分析。")
                    cleaning_summary["warnings"] = warnings[:8]
                    cleaning_summary["warnings_count"] = int(cleaning_summary.get("warnings_count", 0) or 0) + 1
                analysis_text = (cleaning_result.cleaned_text or "").strip()
                if not analysis_text:
                    raise ValueError("文本清洗后正文为空。")
                if len(analysis_text) > MAX_ANALYSIS_CHARS:
                    analysis_text = analysis_text[:MAX_ANALYSIS_CHARS]
            except Exception as exc:
                print(f"增强文本清洗失败，已回退到原始文本：{exc}", flush=True)
                cleaning_summary = {
                    "cleaning_method": "raw_fallback",
                    "raw_text_length": len(raw_text),
                    "cleaned_text_length": len(raw_for_analysis),
                    "block_count": 0,
                    "removed_noise_count": 0,
                    "possible_footnotes_count": 0,
                    "possible_references_count": 0,
                    "warnings_count": 1,
                    "warnings": [f"增强文本清洗失败，已回退到原始文本：{exc}"],
                }
                analysis_text = raw_for_analysis
        else:
            update_progress_state(
                {
                    "progress": 0.04,
                    "status": "正在准备正文",
                    "stage": "prepare_text",
                    "total": st.session_state.get("total_labels", 0),
                },
                running_slots,
            )

        result = analyze_text(
            analysis_text,
            top_k=int(pending.get("top_k", 3)),
            title=str(pending.get("title") or "用户上传论文"),
            save_outputs=True,
            selected_label_names=selected_labels,
            progress_callback=progress_callback,
            model_name=str(pending.get("model_name") or get_deepseek_model()),
        )
        result["core_only"] = bool(pending.get("core_only", True))
        result["quick_test_mode"] = bool(pending.get("quick_test_mode", False))
        result["upload_file_size"] = pending.get("upload_file_size", 0)
        result["original_text_length"] = pending.get("original_text_length", len(raw_text))
        result["use_enhanced_cleaning"] = use_enhanced_cleaning
        if cleaning_summary:
            result["cleaning_summary"] = cleaning_summary
        duration_seconds = round(time.time() - float(pending.get("started_at") or time.time()), 2)
        result["duration_seconds"] = duration_seconds

        st.session_state.analysis_result = result
        st.session_state.report_markdown = result.get("report_markdown", "")
        st.session_state.analysis_status = "done"
        st.session_state.analysis_error = ""
        st.session_state.progress_value = 1.0
        st.session_state.current_step = "分析完成"
        st.session_state.pending_analysis = None
        log_event(
            "analysis_completed",
            {
                "file_type": pending.get("file_type"),
                "file_size_mb": round(float(pending.get("upload_file_size", 0)) / (1024 * 1024), 3),
                "text_length": pending.get("original_text_length"),
                "model_name": result.get("model"),
                "analysis_mode": "快速测试" if pending.get("quick_test_mode") else "完整分析",
                "top_k": pending.get("top_k"),
                "duration_seconds": duration_seconds,
                "success": True,
            },
        )
        st.rerun()
    except MissingAPIKeyError as exc:
        duration_seconds = round(time.time() - float(pending.get("started_at") or time.time()), 2)
        log_event(
            "analysis_failed",
            {
                "file_type": pending.get("file_type"),
                "file_size_mb": round(float(pending.get("upload_file_size", 0)) / (1024 * 1024), 3),
                "text_length": pending.get("original_text_length"),
                "model_name": pending.get("model_name"),
                "analysis_mode": "快速测试" if pending.get("quick_test_mode") else "完整分析",
                "top_k": pending.get("top_k"),
                "duration_seconds": duration_seconds,
                "success": False,
                "error_type": type(exc).__name__,
                "error_message_short": str(exc),
            },
        )
        st.session_state.analysis_status = "error"
        st.session_state.analysis_error = str(exc)
        st.session_state.pending_analysis = None
        st.rerun()
    except Exception as exc:
        duration_seconds = round(time.time() - float(pending.get("started_at") or time.time()), 2)
        log_event(
            "analysis_failed",
            {
                "file_type": pending.get("file_type"),
                "file_size_mb": round(float(pending.get("upload_file_size", 0)) / (1024 * 1024), 3),
                "text_length": pending.get("original_text_length"),
                "model_name": pending.get("model_name"),
                "analysis_mode": "快速测试" if pending.get("quick_test_mode") else "完整分析",
                "top_k": pending.get("top_k"),
                "duration_seconds": duration_seconds,
                "success": False,
                "error_type": type(exc).__name__,
                "error_message_short": str(exc),
            },
        )
        st.session_state.analysis_status = "error"
        st.session_state.analysis_error = f"分析失败：{exc}"
        st.session_state.pending_analysis = None
        st.rerun()


def render_upload_result(label_definitions: list[dict[str, Any]]) -> None:
    result = st.session_state.get("analysis_result")
    if not result:
        return

    render_section_header("分析结果")
    label_results = result.get("label_results", [])
    paper_title = result.get("title", "用户上传论文")
    render_summary_judgement_card(label_results, paper_title)
    render_label_matrix(label_results, label_definitions)
    render_cleaning_summary(result)
    render_analysis_tabs(
        label_results,
        label_definitions,
        paper_title,
        report_markdown=result.get("report_markdown"),
        export_result=result,
        key_prefix="upload_result",
    )

    if result.get("save_warning"):
        st.warning(result["save_warning"])


def is_admin_view() -> bool:
    try:
        return st.query_params.get("admin") == "1"
    except Exception:
        return False


def render_usage_admin_panel() -> None:
    if not is_admin_view():
        return
    events = read_usage_events()
    summary = usage_summary(events)
    counts = summary["counts"]
    st.markdown("---")
    render_section_header("使用统计")
    columns = st.columns(4)
    metrics = [
        ("打开页面", counts.get("app_open", 0)),
        ("上传论文", counts.get("file_uploaded", 0)),
        ("开始分析", counts.get("analysis_started", 0)),
        ("成功报告", counts.get("analysis_completed", 0)),
        ("分析失败", counts.get("analysis_failed", 0)),
        ("PDF 下载", counts.get("download_pdf", 0)),
        ("Markdown 下载", counts.get("download_markdown", 0)),
        ("JSON 下载", counts.get("download_json", 0)),
    ]
    for index, (label, value) in enumerate(metrics):
        with columns[index % 4]:
            st.metric(label, value)

    col1, col2, col3 = st.columns(3)
    col1.metric("Flash 使用次数", summary["flash_count"])
    col2.metric("Pro 使用次数", summary["pro_count"])
    col3.metric("平均分析耗时", f"{summary['avg_duration']} 秒")

    recent = list(reversed(events[-20:]))
    if recent:
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.caption("暂无统计事件。")


def main() -> None:
    init_analysis_state()
    inject_css()
    try:
        label_definitions = load_label_definitions()
    except Exception as exc:
        st.error(f"标签配置读取失败：{exc}")
        return

    meta, results = render_header_section(label_definitions)
    running_slots = render_main_content(label_definitions, meta, results)
    render_usage_admin_panel()
    run_pending_analysis(running_slots)


if __name__ == "__main__":
    main()

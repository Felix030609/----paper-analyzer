from __future__ import annotations

import math
import html as html_lib
from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CATEGORY_LABELS = {
    "方法论标签": ["社会历史批评", "文本细读", "文学史研究", "思想史研究", "文化研究"],
    "哲学资源标签": ["历史唯物主义", "现代性批判", "启蒙理性", "后结构主义", "精神分析"],
    "文学观标签": ["审美自治", "文学社会功能论", "现实主义传统", "现代主义形式实验"],
    "政治—美学倾向": ["民族国家叙事", "革命叙事", "共同体叙事", "个体主体性", "阶级底层视角"],
}

CATEGORY_SHORT = {
    "方法论标签": "方法论",
    "哲学资源标签": "哲学资源",
    "文学观标签": "文学观",
    "政治—美学倾向": "政治—美学",
}

DIMENSION_EXPLANATIONS = {
    "方法论标签": "观察论文主要依靠怎样的研究路径展开论证。",
    "哲学资源标签": "观察论文调用了哪些思想资源、理论框架和问题意识。",
    "文学观标签": "观察论文如何理解文学的形式、审美与社会功能。",
    "政治—美学倾向": "观察论文在政治想象、主体结构和美学判断上的倾向。",
}

HIGHLIGHT_EXPLANATIONS = {
    "现代性批判": "该文以中国现代性问题为核心，对启蒙主义、社会主义与市场化进程进行反思。",
    "思想史研究": "该文追踪概念、知识分子论述和历史语境之间的关系。",
    "社会历史批评": "该文把思想问题放入社会结构、制度变化和历史条件中理解。",
    "历史唯物主义": "该文关注资本、市场、社会关系和历史结构对思想问题的塑造。",
    "启蒙理性": "该文围绕启蒙、理性、主体和现代进步观展开辨析。",
}


def normalize_label_name(label: str) -> str:
    return str(label).replace("/", "").strip()


LABEL_COORDS = {
    "文本细读": (-0.85, -0.15),
    "文学史研究": (0.20, 0.30),
    "社会历史批评": (0.80, 0.45),
    "思想史研究": (0.45, 0.65),
    "文化研究": (0.35, 0.15),
    "审美自治": (-0.90, -0.35),
    "现代主义形式实验": (-0.75, -0.65),
    "精神分析": (-0.35, -0.75),
    "个体主体性": (-0.15, -0.55),
    "启蒙理性": (0.10, 0.20),
    "后结构主义": (-0.20, 0.05),
    "现代性批判": (0.55, 0.35),
    "历史唯物主义": (0.75, 0.70),
    "文学社会功能论": (0.65, 0.20),
    "现实主义传统": (0.50, 0.05),
    "民族国家叙事": (0.85, 0.75),
    "革命叙事": (0.70, 0.55),
    "共同体叙事": (0.55, 0.80),
    "阶级底层视角": (0.80, 0.25),
}

CATEGORY_COLORS = {
    "方法论标签": "#2f6fcb",
    "哲学资源标签": "#8557c8",
    "文学观标签": "#2f9b6a",
    "政治—美学倾向": "#d9673f",
}


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, "#6b7280")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f7fb;
            color: #172033;
        }
        div[data-testid="stHeader"] {
            background: rgba(246, 247, 251, 0.88);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }
        .hero-card {
            border: 1px solid #e3e7ef;
            border-radius: 20px;
            padding: 26px 28px;
            background:
                radial-gradient(circle at 90% 10%, rgba(86, 111, 194, 0.14), transparent 30%),
                linear-gradient(135deg, #ffffff 0%, #f4f7ff 100%);
            box-shadow: 0 18px 48px rgba(24, 39, 75, 0.10);
            margin-bottom: 16px;
        }
        .hero-title {
            font-size: 36px;
            line-height: 1.12;
            font-weight: 800;
            margin: 0 0 14px 0;
            color: #121a2d;
            letter-spacing: 0;
        }
        .hero-subtitle {
            font-size: 16px;
            line-height: 1.75;
            max-width: 880px;
            color: #435069;
            margin-bottom: 20px;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 12px;
            border-radius: 999px;
            background: #eef3ff;
            color: #294a8d;
            border: 1px solid #dbe6ff;
            font-size: 13px;
            font-weight: 650;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .warning-box {
            margin-top: 18px;
            padding: 12px 14px;
            border-radius: 12px;
            background: #fff8e8;
            border: 1px solid #f0dfb0;
            color: #735414;
            font-size: 14px;
        }
        .section-card {
            border: 1px solid #e4e8f0;
            border-radius: 18px;
            padding: 24px 26px;
            background: #ffffff;
            box-shadow: 0 10px 32px rgba(24, 39, 75, 0.07);
            margin: 20px 0;
        }
        .map-card {
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 18px 18px 10px 18px;
            background:
                radial-gradient(circle at 16% 20%, rgba(47, 111, 203, 0.08), transparent 30%),
                radial-gradient(circle at 84% 18%, rgba(217, 103, 63, 0.08), transparent 30%),
                #ffffff;
            box-shadow: 0 14px 38px rgba(24, 39, 75, 0.08);
            margin: 14px 0 18px 0;
        }
        .upload-card {
            border: 1px solid #dce4f0;
            border-radius: 20px;
            padding: 22px 22px 18px 22px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            box-shadow: 0 18px 44px rgba(24, 39, 75, 0.12);
            margin-top: 2px;
        }
        .upload-title {
            font-size: 22px;
            font-weight: 780;
            color: #142033;
            margin-bottom: 6px;
        }
        .upload-copy {
            color: #637089;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 14px;
        }
        .sticky-note {
            position: sticky;
            top: 76px;
        }
        .section-title {
            font-size: 26px;
            font-weight: 760;
            color: #151d30;
            margin: 0 0 8px 0;
        }
        .subtle-text {
            color: #60708c;
            line-height: 1.7;
            font-size: 15px;
        }
        .sample-heading {
            font-size: 22px;
            line-height: 1.35;
            font-weight: 820;
            color: #111827;
            margin: 18px 0 6px 0;
        }
        .sample-section,
        .result-section {
            border: 1px solid #e4e8f0;
            border-radius: 20px;
            padding: 22px;
            background: #ffffff;
            box-shadow: 0 12px 34px rgba(24, 39, 75, 0.07);
            margin: 18px 0;
        }
        .hero-grid {
            margin-bottom: 18px;
        }
        .full-width-section {
            width: 100%;
            margin-top: 22px;
        }
        .analysis-tabs {
            margin-top: 14px;
        }
        div[data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #e4e8f0;
        }
        button[data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 10px 14px;
            font-weight: 720;
        }
        .chart-card {
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 14px 16px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(24, 39, 75, 0.06);
            margin: 12px 0 18px 0;
        }
        .metric-card {
            border-radius: 16px;
            border: 1px solid #e4e8f0;
            background: #ffffff;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 8px 26px rgba(24, 39, 75, 0.08);
            min-height: 172px;
        }
        .metric-kicker {
            color: #65738d;
            font-size: 13px;
            font-weight: 650;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 34px;
            line-height: 1;
            font-weight: 800;
            color: #172033;
            margin-bottom: 10px;
        }
        .metric-label {
            color: #294a8d;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .metric-explain {
            color: #5b687e;
            font-size: 13px;
            line-height: 1.55;
        }
        .evidence-meta {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            background: #eef2f7;
            color: #4b5b73;
            font-size: 12px;
            font-weight: 650;
            margin-right: 6px;
        }
        .evidence-card {
            border: 1px solid #e1e7f0;
            border-radius: 16px;
            background: #ffffff;
            padding: 14px 16px;
            margin: 12px 0;
        }
        .evidence-block {
            border-left: 4px solid #315fbd;
            background: #f7f9fd;
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0 12px 0;
            color: #263347;
            line-height: 1.75;
            font-size: 14px;
        }
        .evidence-reason {
            background: #fffaf0;
            border: 1px solid #f3dfac;
            border-radius: 12px;
            padding: 10px 12px;
            color: #5f4515;
            line-height: 1.65;
            margin: 10px 0;
            font-size: 14px;
        }
        .export-card {
            border: 1px solid #e3e8f0;
            border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(24, 39, 75, 0.07);
            margin: 8px 0 14px 0;
        }
        .export-title {
            font-size: 20px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 6px;
        }
        .running-panel,
        .result-panel {
            border: 1px solid #e0e7f2;
            border-radius: 22px;
            padding: 24px;
            background:
                radial-gradient(circle at 12% 12%, rgba(49, 95, 189, 0.09), transparent 30%),
                #ffffff;
            box-shadow: 0 16px 42px rgba(24, 39, 75, 0.10);
            margin: 18px 0;
        }
        .progress-card,
        .step-card,
        .skeleton-card {
            border: 1px solid #e4e8f0;
            border-radius: 16px;
            background: #ffffff;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(24, 39, 75, 0.06);
            margin: 12px 0;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 6px 11px;
            background: #eef3ff;
            color: #294a8d;
            border: 1px solid #d9e5ff;
            font-size: 13px;
            font-weight: 750;
            margin: 4px 6px 4px 0;
        }
        .status-pill-done {
            background: #eef9f3;
            border-color: #cdebd9;
            color: #23724f;
        }
        .status-pill-muted {
            background: #f4f6f9;
            border-color: #e2e7ef;
            color: #64748b;
        }
        .step-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 8px;
        }
        .step-item {
            border-radius: 12px;
            background: #f7f9fd;
            border: 1px solid #e5ebf4;
            padding: 9px 10px;
            color: #52617a;
            font-size: 13px;
            font-weight: 680;
        }
        .error-card {
            border: 1px solid #f1c7c7;
            border-radius: 20px;
            padding: 22px;
            background: #fffafa;
            box-shadow: 0 12px 32px rgba(127, 29, 29, 0.08);
            margin: 18px 0;
        }
        .upload-panel {
            border-top: 1px solid #dbe2ed;
            margin-top: 32px;
            padding-top: 24px;
        }
        .core-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 7px 12px;
            margin: 0 8px 8px 0;
            background: #f1f5ff;
            border: 1px solid #d8e4ff;
            color: #294a8d;
            font-size: 13px;
            font-weight: 700;
        }
        .system-panel {
            border: 1px solid #e4e8f0;
            border-radius: 16px;
            padding: 18px 18px 8px 18px;
            background: #ffffff;
            box-shadow: 0 8px 26px rgba(24, 39, 75, 0.06);
            margin: 14px 0 18px 0;
        }
        .system-row {
            padding: 10px 0;
            border-bottom: 1px solid #eef2f7;
        }
        .system-row:last-child {
            border-bottom: none;
        }
        .system-name {
            font-weight: 760;
            color: #172033;
        }
        .system-desc {
            color: #68758d;
            font-size: 13px;
        }
        .map-caption {
            padding: 12px 14px;
            border-radius: 14px;
            background: #f7f9fd;
            border: 1px solid #e4e8f0;
            color: #52617a;
            font-size: 14px;
            line-height: 1.65;
            margin-top: -2px;
            margin-bottom: 16px;
        }
        .summary-card {
            border: 1px solid #dfe6f1;
            border-radius: 20px;
            padding: 22px 24px;
            background:
                linear-gradient(135deg, #ffffff 0%, #f7faff 100%);
            box-shadow: 0 14px 36px rgba(24, 39, 75, 0.09);
            margin: 14px 0 16px 0;
        }
        .summary-kicker {
            color: #64748b;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .summary-title {
            font-size: 22px;
            line-height: 1.35;
            font-weight: 820;
            color: #101827;
            margin-bottom: 14px;
        }
        .summary-row {
            margin: 9px 0;
            color: #334155;
            font-size: 14px;
            line-height: 1.65;
        }
        .summary-label {
            color: #0f172a;
            font-weight: 760;
        }
        .summary-focus {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 3px 4px 3px 0;
            background: #172033;
            color: #ffffff;
            font-size: 13px;
            font-weight: 760;
        }
        .summary-related {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 3px 4px 3px 0;
            background: #eef3ff;
            border: 1px solid #d9e5ff;
            color: #294a8d;
            font-size: 13px;
            font-weight: 700;
        }
        .summary-weak {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 3px 4px 3px 0;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            color: #64748b;
            font-size: 13px;
            font-weight: 650;
        }
        .label-matrix {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin: 14px 0 18px 0;
        }
        .label-group {
            border: 1px solid #e3e8f0;
            border-radius: 18px;
            background: #ffffff;
            padding: 16px;
            box-shadow: 0 10px 28px rgba(24, 39, 75, 0.07);
        }
        .label-group-title {
            font-size: 15px;
            color: #172033;
            font-weight: 800;
            margin-bottom: 11px;
        }
        .label-pill {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            min-height: 38px;
            border-radius: 13px;
            padding: 8px 10px;
            margin-bottom: 8px;
            font-size: 13px;
            line-height: 1.35;
        }
        .label-pill-name {
            font-weight: 740;
            color: inherit;
        }
        .label-pill-score {
            font-weight: 820;
            white-space: nowrap;
            color: inherit;
        }
        .label-pill-3 {
            background: #172033;
            color: #ffffff;
            box-shadow: 0 10px 22px rgba(23, 32, 51, 0.18);
        }
        .label-pill-2 {
            background: #eef3ff;
            border: 1px solid #d6e3ff;
            color: #244681;
        }
        .label-pill-1 {
            background: #f7f9fc;
            border: 1px solid #e4e8f0;
            color: #64748b;
        }
        .label-pill-0 {
            background: #f4f4f5;
            border: 1px solid #e5e7eb;
            color: #9aa3b2;
        }
        @media (max-width: 900px) {
            .label-matrix {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">人文社科论文思想谱系分析工具</div>
          <div class="hero-subtitle">
            上传中文人文社科论文，系统将从方法论、哲学资源、文学观与政治—美学倾向四个层面生成结构化分析。
          </div>
          <div>
            <span class="badge">19 个思想标签</span>
            <span class="badge">RAG 证据召回</span>
            <span class="badge">DeepSeek 自动报告生成</span>
          </div>
          <div class="warning-box">
            本工具只分析论文文本中呈现出的思想倾向，不判断作者本人真实政治立场。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None) -> None:
    description_html = f'<div class="subtle-text">{description}</div>' if description else ""
    st.markdown(
        f"""
        <div class="section-card">
          <div class="section-title">{title}</div>
          {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def label_chip_html(name: str, class_name: str) -> str:
    return f'<span class="{class_name}">{html_lib.escape(name)}</span>'


def sorted_by_score(label_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        label_results,
        key=lambda item: (get_score(item), str(item.get("label_name", ""))),
        reverse=True,
    )


def render_summary_judgement_card(label_results: list[dict[str, Any]], paper_title: str) -> None:
    paper_x, paper_y, _ = paper_position(label_results)
    quadrant = quadrant_name(paper_x, paper_y)
    core = [item for item in sorted_by_score(label_results) if get_score(item) >= 3]
    related = [item for item in sorted_by_score(label_results) if get_score(item) == 2]
    weak_by_label = result_map([item for item in label_results if get_score(item) <= 1])
    weak_priority = ["文本细读", "审美自治", "精神分析", "现代主义形式实验"]
    weak_names = [
        name
        for name in weak_priority
        if name in weak_by_label
    ]
    if len(weak_names) < 4:
        weak_names.extend(
            str(item.get("label_name", "")).strip()
            for item in sorted_by_score(label_results)
            if get_score(item) <= 1 and str(item.get("label_name", "")).strip() not in weak_names
        )
    weak_names = [name for name in weak_names if name][:4]

    core_html = "".join(label_chip_html(str(item.get("label_name", "")), "summary-focus") for item in core)
    related_html = "".join(label_chip_html(str(item.get("label_name", "")), "summary-related") for item in related)
    weak_html = "".join(label_chip_html(name, "summary-weak") for name in weak_names)
    core_text = core_html or '<span class="summary-weak">暂无 3 分核心标签</span>'
    related_text = related_html or '<span class="summary-weak">暂无 2 分明显相关标签</span>'
    weak_text = weak_html or '<span class="summary-weak">暂无弱相关标签</span>'

    if "当代中国的思想状况与现代性问题" in paper_title or "汪晖" in paper_title:
        explanation = (
            "该文主要不是以文学形式细读或审美自治为中心，而是将中国知识界思想变化放入"
            "现代性、市场化、国家、资本与历史结构中分析。"
        )
    else:
        explanation = (
            f"系统根据高分标签将论文定位在“{quadrant}”区域，核心判断仍需结合证据链人工复核。"
        )

    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-kicker">核心判断</div>
          <div class="summary-title">{html_lib.escape(paper_title)}</div>
          <div class="summary-row"><span class="summary-label">主要定位：</span>{html_lib.escape(quadrant)}</div>
          <div class="summary-row"><span class="summary-label">核心谱系：</span>{core_text}</div>
          <div class="summary-row"><span class="summary-label">明显相关：</span>{related_text}</div>
          <div class="summary-row"><span class="summary-label">弱相关方向：</span>{weak_text}</div>
          <div class="subtle-text">{html_lib.escape(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_label_matrix(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
) -> None:
    by_label = result_map(label_results)
    definitions = label_definition_map(label_definitions)
    groups_html = []
    for category, labels in CATEGORY_LABELS.items():
        pills = []
        for label in labels:
            normalized = normalize_label_name(label)
            item = by_label.get(normalized, {})
            score = int(get_score(item))
            definition = definitions.get(normalized, {})
            reason = str(item.get("reason") or definition.get("definition") or "").strip()
            tooltip = html_lib.escape(reason[:80])
            pills.append(
                f'<div class="label-pill label-pill-{score}" title="{tooltip}">'
                f'<span class="label-pill-name">{html_lib.escape(label)}</span>'
                f'<span class="label-pill-score">{score}/3</span>'
                "</div>"
            )
        groups_html.append(
            '<div class="label-group">'
            f'<div class="label-group-title">{html_lib.escape(CATEGORY_SHORT.get(category, category))}</div>'
            f'{"".join(pills)}'
            "</div>"
        )

    st.markdown(
        f'<div class="label-matrix">{"".join(groups_html)}</div>',
        unsafe_allow_html=True,
    )


def label_definition_map(label_definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_label_name(item.get("label_name", "")): item for item in label_definitions}


def result_map(label_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_label_name(item.get("label_name", "")): item for item in label_results}


def get_score(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    try:
        return float(result.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def dimension_stats(label_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = result_map(label_results)
    rows = []
    for category, labels in CATEGORY_LABELS.items():
        values = []
        category_results = []
        for label in labels:
            item = by_label.get(normalize_label_name(label))
            score = get_score(item)
            values.append(score)
            if item:
                category_results.append((label, score, item))

        top_label, top_score, top_item = max(
            category_results or [("暂无", 0.0, {})],
            key=lambda value: value[1],
        )
        avg_score = sum(values) / len(values) if values else 0.0
        rows.append(
            {
                "category": category,
                "short": CATEGORY_SHORT.get(category, category),
                "average": avg_score,
                "max": top_score,
                "top_label": top_label,
                "top_item": top_item,
                "explanation": HIGHLIGHT_EXPLANATIONS.get(
                    normalize_label_name(top_label),
                    DIMENSION_EXPLANATIONS.get(category, ""),
                ),
            }
        )
    return rows


def paper_position(label_results: list[dict[str, Any]]) -> tuple[float, float, list[dict[str, Any]]]:
    positioned = []
    for item in label_results:
        label = normalize_label_name(item.get("label_name", ""))
        if label not in LABEL_COORDS:
            continue
        score = get_score(item)
        if score >= 2:
            x, y = LABEL_COORDS[label]
            positioned.append({"item": item, "x": x, "y": y, "weight": score})

    if not positioned:
        return 0.0, 0.0, []

    total_weight = sum(point["weight"] for point in positioned)
    x = sum(point["x"] * point["weight"] for point in positioned) / total_weight
    y = sum(point["y"] * point["weight"] for point in positioned) / total_weight
    return x, y, positioned


def quadrant_name(x: float, y: float) -> str:
    if x > 0.3 and y > 0.3:
        return "社会历史—结构秩序"
    if x > 0.3 and y <= 0.3:
        return "社会现实—文化政治"
    if x <= 0.3 and y > 0.3:
        return "观念谱系—审美传统"
    return "主体经验—形式分析"


def core_label_names(label_results: list[dict[str, Any]], limit: int = 4) -> str:
    core = sorted(
        [item for item in label_results if get_score(item) >= 2],
        key=lambda item: get_score(item),
        reverse=True,
    )
    names = [str(item.get("label_name", "")).strip() for item in core if item.get("label_name")]
    return "、".join(names[:limit]) if names else "暂无高分标签"


def academic_space_explanation(x: float, y: float, label_results: list[dict[str, Any]]) -> str:
    quadrant = quadrant_name(x, y)
    labels = core_label_names(label_results, limit=3)
    return (
        f"根据高分标签加权计算，该论文位于【{quadrant}】区域，"
        f"说明其更接近 {labels} 路径。"
    )


def render_academic_space_map(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    paper_title: str,
    *,
    key: str = "academic_space_map",
) -> tuple[float, float, str]:
    definitions = label_definition_map(label_definitions)
    paper_x, paper_y, positioned = paper_position(label_results)
    explanation = academic_space_explanation(paper_x, paper_y, label_results)

    figure = go.Figure()
    figure.add_shape(
        type="rect",
        x0=-1,
        x1=0,
        y0=0,
        y1=1,
        fillcolor="rgba(133,87,200,0.045)",
        line_width=0,
        layer="below",
    )
    figure.add_shape(
        type="rect",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        fillcolor="rgba(217,103,63,0.045)",
        line_width=0,
        layer="below",
    )
    figure.add_shape(
        type="rect",
        x0=-1,
        x1=0,
        y0=-1,
        y1=0,
        fillcolor="rgba(47,155,106,0.045)",
        line_width=0,
        layer="below",
    )
    figure.add_shape(
        type="rect",
        x0=0,
        x1=1,
        y0=-1,
        y1=0,
        fillcolor="rgba(47,111,203,0.04)",
        line_width=0,
        layer="below",
    )

    for point in positioned:
        score = get_score(point["item"])
        alpha = 0.25 + score * 0.18
        figure.add_trace(
            go.Scatter(
                x=[paper_x, point["x"]],
                y=[paper_y, point["y"]],
                mode="lines",
                line=dict(
                    color=f"rgba(49, 95, 189, {min(alpha, 0.82):.2f})",
                    width=1.5 + score,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in positioned:
        category = point["item"].get("category", "")
        by_category[category].append(point)

    for category in ["方法论标签", "哲学资源标签", "文学观标签", "政治—美学倾向"]:
        xs = []
        ys = []
        sizes = []
        hover = []
        text_labels = []
        for point in by_category.get(category, []):
            item = point["item"]
            label = normalize_label_name(item.get("label_name", ""))
            score = get_score(item)
            definition = definitions.get(label, {}).get("definition", "")
            xs.append(point["x"])
            ys.append(point["y"])
            sizes.append(20 + score * 9)
            text_labels.append(f"{item.get('label_name')} {score:.0f}")
            hover.append(
                f"标签：{label}<br>"
                f"类别：{category}<br>"
                f"分数：{score:.0f}<br>"
                f"定义：{definition}<br>"
                f"{(item or {}).get('reason', '')}"
            )

        if not xs:
            continue
        base_color = category_color(category)
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=CATEGORY_SHORT.get(category, category),
                text=text_labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=base_color,
                    opacity=0.92,
                    line=dict(color="#ffffff", width=1.5),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[paper_x],
            y=[paper_y],
            mode="markers+text",
            text=["当前论文"],
            textposition="bottom center",
            marker=dict(
                size=34,
                color="#111827",
                symbol="diamond",
                line=dict(color="#ffffff", width=2.5),
            ),
            hovertext=[f"{paper_title}<br>x={paper_x:.2f}, y={paper_y:.2f}<br>{explanation}"],
            hoverinfo="text",
            name="当前论文",
        )
    )

    annotations = [
        dict(x=-0.63, y=0.92, text="审美传统 / 观念谱系", showarrow=False, font=dict(size=12, color="#6b5aa6")),
        dict(x=0.62, y=0.92, text="国家—资本—共同体结构", showarrow=False, font=dict(size=12, color="#9b4d31")),
        dict(x=-0.62, y=-0.92, text="主体经验 / 形式实验", showarrow=False, font=dict(size=12, color="#2f7a58")),
        dict(x=0.62, y=-0.92, text="社会现实 / 文化政治", showarrow=False, font=dict(size=12, color="#315fbd")),
    ]

    figure.update_layout(
        title="学术谱系空间定位图",
        height=530,
        margin=dict(l=24, r=24, t=62, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=annotations,
        xaxis=dict(
            title="文本形式 ←→ 社会历史",
            range=[-1.05, 1.05],
            zeroline=True,
            zerolinecolor="rgba(51,65,85,0.28)",
            gridcolor="rgba(148,163,184,0.20)",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        yaxis=dict(
            title="个体经验 ←→ 结构秩序",
            range=[-1.05, 1.05],
            zeroline=True,
            zerolinecolor="rgba(51,65,85,0.28)",
            gridcolor="rgba(148,163,184,0.20)",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    )

    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="map-caption">{explanation}</div>', unsafe_allow_html=True)
    return paper_x, paper_y, explanation


def render_core_label_pills(label_results: list[dict[str, Any]]) -> None:
    core = sorted(
        [item for item in label_results if get_score(item) >= 2],
        key=lambda item: get_score(item),
        reverse=True,
    )
    if not core:
        st.caption("暂无高分核心标签。")
        return
    pills = "".join(
        f'<span class="core-pill">{item.get("label_name")} {get_score(item):.0f}</span>'
        for item in core[:10]
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_label_system_panel(label_results: list[dict[str, Any]]) -> None:
    by_label = result_map(label_results)
    rows = []
    for category, labels in CATEGORY_LABELS.items():
        scored = []
        for label in labels:
            item = by_label.get(normalize_label_name(label))
            scored.append((label, get_score(item)))
        scored.sort(key=lambda item: item[1], reverse=True)
        top = [f"{label} {score:.0f}" for label, score in scored if score > 0][:3]
        if not top:
            top = ["暂无明显高分标签"]
        rows.append((CATEGORY_SHORT.get(category, category), DIMENSION_EXPLANATIONS.get(category, ""), "、".join(top)))

    parts = ['<div class="system-panel">']
    parts.append('<div class="system-name">四层标签体系：不是给论文贴标签，而是在结构化空间里定位论文。</div>')
    for name, description, top in rows:
        parts.append(
            '<div class="system-row">'
            f'<div class="system-name">{html_lib.escape(name)}</div>'
            f'<div class="system-desc">{html_lib.escape(description)}</div>'
            f'<div class="subtle-text">当前高分：{html_lib.escape(top)}</div>'
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_dimension_cards(label_results: list[dict[str, Any]]) -> None:
    stats = dimension_stats(label_results)
    columns = st.columns(4)
    titles = {
        "方法论标签": "方法论强度",
        "哲学资源标签": "哲学资源强度",
        "文学观标签": "文学观强度",
        "政治—美学倾向": "政治—美学强度",
    }
    for column, item in zip(columns, stats):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-kicker">{titles.get(item["category"], item["category"])}</div>
                  <div class="metric-value">{item["average"]:.1f}</div>
                  <div class="metric-label">核心标签：{item["top_label"]} {item["max"]:.0f}</div>
                  <div class="metric-explain">{item["explanation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_radar_chart(label_results: list[dict[str, Any]], *, key: str = "radar_chart") -> None:
    stats = dimension_stats(label_results)
    categories = [item["short"] for item in stats]
    values = [round(item["average"], 2) for item in stats]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name="平均强度",
            line=dict(color="#315fbd", width=3),
            fillcolor="rgba(49, 95, 189, 0.22)",
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="四层标签平均强度",
        polar=dict(radialaxis=dict(visible=True, range=[0, 3], tickvals=[0, 1, 2, 3])),
        showlegend=False,
        margin=dict(l=40, r=40, t=70, b=40),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_label_bar_chart(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    *,
    key: str = "label_bar_chart",
) -> None:
    definitions = label_definition_map(label_definitions)
    rows = []
    for item in label_results:
        label = normalize_label_name(item.get("label_name", ""))
        definition = definitions.get(label, {})
        score = get_score(item)
        rows.append(
            {
                "label": item.get("label_name", label),
                "score": score,
                "category": item.get("category") or definition.get("category", ""),
                "definition": definition.get("definition", ""),
                "color": "#315fbd" if score >= 3 else "#6e94d4" if score >= 2 else "#c9d2e3",
            }
        )
    rows.sort(key=lambda row: row["score"])

    figure = go.Figure(
        go.Bar(
            x=[row["score"] for row in rows],
            y=[row["label"] for row in rows],
            orientation="h",
            marker=dict(color=[row["color"] for row in rows]),
            customdata=[[row["category"], row["definition"]] for row in rows],
            hovertemplate=(
                "<b>%{y}</b><br>分数：%{x}<br>类别：%{customdata[0]}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="标签强度分布",
        xaxis=dict(range=[0, 3.1], tickvals=[0, 1, 2, 3], title="强度分数"),
        yaxis=dict(title=""),
        height=620,
        margin=dict(l=120, r=30, t=70, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_spectrum_network(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    paper_title: str,
    *,
    key: str = "spectrum_network",
) -> None:
    definitions = label_definition_map(label_definitions)
    core_results = [item for item in label_results if get_score(item) >= 2]
    category_positions = {
        "方法论标签": (-1.4, 0.95),
        "哲学资源标签": (1.4, 0.95),
        "文学观标签": (-1.35, -0.95),
        "政治—美学倾向": (1.35, -0.95),
    }

    nodes = [
        {
            "id": "paper",
            "label": paper_title,
            "x": 0,
            "y": 0,
            "size": 34,
            "color": "#172033",
            "hover": paper_title,
        }
    ]
    edges = []
    for category, position in category_positions.items():
        category_id = f"cat:{category}"
        nodes.append(
            {
                "id": category_id,
                "label": CATEGORY_SHORT.get(category, category),
                "x": position[0],
                "y": position[1],
                "size": 24,
                "color": "#6e94d4",
                "hover": category,
            }
        )
        edges.append(("paper", category_id))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in core_results:
        by_category[item.get("category", "")].append(item)

    for category, items in by_category.items():
        base_x, base_y = category_positions.get(category, (0, 0))
        radius = 0.62
        for index, item in enumerate(items):
            angle = (math.pi * 2 * index / max(len(items), 1)) + 0.5
            x = base_x + math.cos(angle) * radius
            y = base_y + math.sin(angle) * radius
            score = get_score(item)
            label = normalize_label_name(item.get("label_name", ""))
            definition = definitions.get(label, {}).get("definition", "")
            node_id = f"label:{label}"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{item.get('label_name')} {score:.0f}",
                    "x": x,
                    "y": y,
                    "size": 18 + score * 6,
                    "color": "#315fbd" if score >= 3 else "#789bd7",
                    "hover": (
                        f"标签：{item.get('label_name')}<br>"
                        f"分数：{score:.0f}<br>"
                        f"类别：{item.get('category')}<br>"
                        f"{item.get('reason') or definition}"
                    ),
                }
            )
            edges.append((f"cat:{category}", node_id))

    node_by_id = {node["id"]: node for node in nodes}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for start, end in edges:
        if start not in node_by_id or end not in node_by_id:
            continue
        edge_x.extend([node_by_id[start]["x"], node_by_id[end]["x"], None])
        edge_y.extend([node_by_id[start]["y"], node_by_id[end]["y"], None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.4, color="rgba(84, 105, 140, 0.32)"),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers+text",
            text=[node["label"] for node in nodes],
            textposition="bottom center",
            marker=dict(
                size=[node["size"] for node in nodes],
                color=[node["color"] for node in nodes],
                line=dict(color="#ffffff", width=2),
            ),
            hovertext=[node["hover"] for node in nodes],
            hoverinfo="text",
        )
    )
    figure.update_layout(
        title="论文—维度—标签思想谱系网络",
        height=560,
        showlegend=False,
        xaxis=dict(visible=False, range=[-2.35, 2.35]),
        yaxis=dict(visible=False, range=[-1.85, 1.85]),
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    st.markdown("</div>", unsafe_allow_html=True)


def truncate_text(text: str, limit: int = 800) -> tuple[str, bool]:
    text = str(text).strip()
    if len(text) <= limit:
        return text, False
    return text[:limit] + "……", True


def render_evidence_chain(label_results: list[dict[str, Any]]) -> None:
    st.markdown("### 核心标签证据链")
    st.caption("证据链基于语义召回生成，每个标签展示与其定义最相关的原文段落。分数越高，说明该标签在当前文本中的支撑越强。")
    core_results = [item for item in label_results if get_score(item) >= 2]
    if not core_results:
        st.info("暂无分数 >= 2 的核心标签证据链。")
        return

    for item in sorted(core_results, key=lambda value: get_score(value), reverse=True):
        score = int(get_score(item))
        confidence = item.get("confidence", "待复核")
        title = f"{item.get('label_name')}｜{score}/3｜置信度 {confidence}"
        with st.expander(title, expanded=score >= 3):
            evidence_items = item.get("evidence_items") or []
            retrieved_items = item.get("retrieved_paragraphs") or []
            display_items = []
            if evidence_items:
                display_items = [
                    {
                        "text": evidence.get("evidence_text", ""),
                        "reason": evidence.get("reason") or item.get("reason"),
                        "evidence_type": evidence.get("evidence_type", "证据"),
                        "confidence": evidence.get("confidence", confidence),
                        "chunk_index": evidence.get("chunk_index", ""),
                        "similarity": evidence.get("similarity_score", evidence.get("similarity", "")),
                    }
                    for evidence in evidence_items
                ]
            elif retrieved_items:
                display_items = [
                    {
                        "text": evidence.get("evidence_full_text") or evidence.get("text", ""),
                        "reason": item.get("reason"),
                        "evidence_type": "语义召回段落",
                        "confidence": confidence,
                        "chunk_index": evidence.get("chunk_index", ""),
                        "section_title": evidence.get("section_title", ""),
                        "similarity": evidence.get("similarity_score", evidence.get("similarity", "")),
                    }
                    for evidence in retrieved_items
                ]
            else:
                display_items = [
                    {
                        "text": str(text),
                        "reason": item.get("reason"),
                        "evidence_type": "模型摘录",
                        "confidence": confidence,
                        "chunk_index": "",
                        "similarity": "",
                    }
                    for text in (item.get("evidence") or [])
                ]

            if display_items:
                if len(display_items) == 1 and len(retrieved_items) == 0:
                    st.caption("当前标签只召回到 1 条可展示证据。可提高 top_k 或切换 V4 Pro 重新分析。")
                for index, evidence in enumerate(display_items, start=1):
                    text = evidence.get("text", "")
                    preview, has_more = truncate_text(text)
                    chunk_meta = f"｜chunk {evidence.get('chunk_index')}" if evidence.get("chunk_index") not in ("", None) else ""
                    section_meta = f"｜{evidence.get('section_title')}" if evidence.get("section_title") else ""
                    similarity_meta = f"｜相似度 {evidence.get('similarity')}" if evidence.get("similarity") not in ("", None) else ""
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                          <div class="summary-label">原文证据 {index}{html_lib.escape(section_meta)}{html_lib.escape(chunk_meta)}{html_lib.escape(similarity_meta)}</div>
                          <div class="evidence-block">{html_lib.escape(preview)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if has_more:
                        with st.expander("查看完整证据", expanded=False):
                            st.write(text)
                    reason = evidence.get("reason") or item.get("reason")
                    if reason:
                        st.markdown(
                            f'<div class="evidence-reason"><strong>判断理由：</strong>{html_lib.escape(str(reason))}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"""
                        <span class="evidence-meta">{evidence.get("evidence_type", "证据")}</span>
                        <span class="evidence-meta">置信度 {evidence.get("confidence", confidence)}</span>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("当前标签未召回到足够证据，建议提高 top_k 或切换 V4 Pro 重新分析。")
            uncertainty = str(item.get("uncertainty") or "").strip()
            if uncertainty:
                st.caption(f"不确定性：{uncertainty}")


def render_report(report_markdown: str | None) -> None:
    if not report_markdown:
        return
    st.markdown(report_markdown)


def render_more_analysis_charts(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    paper_title: str,
    *,
    key_prefix: str = "more_charts",
) -> None:
    with st.expander("更多分析图表", expanded=False):
        left, right = st.columns([0.9, 1.1])
        with left:
            render_radar_chart(label_results, key=f"{key_prefix}_radar")
        with right:
            render_label_bar_chart(label_results, label_definitions, key=f"{key_prefix}_bar")
        render_spectrum_network(
            label_results,
            label_definitions,
            paper_title,
            key=f"{key_prefix}_network",
        )


def render_visual_suite(
    label_results: list[dict[str, Any]],
    label_definitions: list[dict[str, Any]],
    paper_title: str,
    *,
    report_markdown: str | None = None,
    key_prefix: str = "visual_suite",
    include_more_charts: bool = True,
) -> None:
    render_summary_judgement_card(label_results, paper_title)
    render_label_matrix(label_results, label_definitions)
    render_academic_space_map(
        label_results,
        label_definitions,
        paper_title,
        key=f"{key_prefix}_academic_space",
    )
    st.subheader("核心标签证据链")
    render_evidence_chain(label_results)
    if report_markdown:
        st.subheader("Markdown 分析报告")
        render_report(report_markdown)
    if include_more_charts:
        render_more_analysis_charts(
            label_results,
            label_definitions,
            paper_title,
            key_prefix=key_prefix,
        )

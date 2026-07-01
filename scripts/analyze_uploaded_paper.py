from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.deepseek_client import (
    DeepSeekTimeoutError,
    MissingAPIKeyError,
    call_deepseek_chat,
    get_deepseek_api_key,
    get_deepseek_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "label_definitions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MAX_ANALYSIS_CHARS = 60_000
MAX_LABELS_PER_ANALYSIS = 19
MAX_TOP_K = 8
MAX_LLM_EVIDENCE_PER_LABEL = 8
MAX_EVIDENCE_CHARS = 800
MAX_VISIBLE_EVIDENCE_CHARS = 1200
MAX_REPORT_EVIDENCE_CHARS = 500
MAX_REPORT_REASON_CHARS = 500
QUICK_TEST_LABELS = ["现代性批判", "思想史研究", "社会历史批评", "历史唯物主义", "启蒙理性"]
ProgressCallback = Callable[[dict[str, Any]], None]


def load_label_definitions() -> list[dict[str, Any]]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到标签配置文件：{CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def shorten_text(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "……"


def trim_text_for_analysis(text: str) -> tuple[str, bool, int]:
    original_length = len(text or "")
    if original_length <= MAX_ANALYSIS_CHARS:
        return text, False, original_length
    return text[:MAX_ANALYSIS_CHARS], True, original_length


def normalize_label_result(
    raw: dict[str, Any],
    label_definition: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = raw.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []

    return {
        "label_name": label_definition["label_name"],
        "category": label_definition["category"],
        "score": clamp_int(raw.get("score"), 0, 3, 0),
        "confidence": clamp_int(raw.get("confidence"), 1, 5, 1),
        "evidence": [str(item).strip() for item in evidence[:3] if str(item).strip()],
        "reason": str(raw.get("reason", "")).strip() or "DeepSeek 未返回明确理由。",
        "uncertainty": str(raw.get("uncertainty", "")).strip() or "需要人工复核。",
        "retrieved_paragraphs": retrieved,
    }


def fallback_label_result(
    label_definition: dict[str, Any],
    retrieved: list[dict[str, Any]],
    message: str,
) -> dict[str, Any]:
    return normalize_label_result(
        {
            "score": 0,
            "confidence": 1,
            "evidence": [],
            "reason": f"该标签自动评分失败：{message}",
            "uncertainty": "需要人工复核该标签。DeepSeek 调用或 JSON 解析未成功。",
        },
        label_definition,
        retrieved,
    )


def timeout_label_result(
    label_definition: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    return normalize_label_result(
        {
            "score": 0,
            "confidence": 1,
            "evidence": [],
            "reason": "DeepSeek 请求超时，未完成该标签分析",
            "uncertainty": "该标签需要重新分析",
        },
        label_definition,
        retrieved,
    )


def emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    progress: float,
    status: str,
    stage: str,
    current: int = 0,
    total: int = 0,
    label_name: str = "",
) -> None:
    print(status, flush=True)
    if progress_callback:
        progress_callback(
            {
                "progress": max(0.0, min(1.0, progress)),
                "status": status,
                "stage": stage,
                "current": current,
                "total": total,
                "label_name": label_name,
            }
        )


def select_label_definitions(
    labels: list[dict[str, Any]],
    selected_label_names: list[str] | None,
) -> list[dict[str, Any]]:
    if not selected_label_names:
        return labels[:MAX_LABELS_PER_ANALYSIS]
    wanted = {str(name).replace("/", "").strip() for name in selected_label_names}
    return [
        item
        for item in labels
        if str(item.get("label_name", "")).replace("/", "").strip() in wanted
    ][:MAX_LABELS_PER_ANALYSIS]


def split_long_paragraph(paragraph: str, max_chars: int = 650) -> list[str]:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if len(paragraph) <= 1200:
        return [paragraph] if paragraph else []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?])", paragraph)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > 900 and len(current) >= 500:
            chunks.append(current.strip())
            overlap = current[-100:] if len(current) > 100 else current
            current = overlap + sentence
        else:
            current += sentence
    if current:
        chunks.append(current.strip())
    return chunks or [paragraph]


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n+", normalized)
        if part.strip()
    ]

    if len(raw_paragraphs) < 3:
        raw_paragraphs = [part.strip() for part in normalized.splitlines() if part.strip()]

    if len(raw_paragraphs) < 3:
        raw_paragraphs = [normalized]

    merged: list[str] = []
    current = ""
    for paragraph in raw_paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) < 80:
            current = f"{current}\n{paragraph}".strip()
            if len(current) < 300:
                continue
            merged.append(current)
            current = ""
        else:
            if current:
                merged.append(current)
                current = ""
            merged.append(paragraph)
    if current:
        merged.append(current)

    chunks: list[str] = []
    for paragraph in merged:
        chunks.extend(split_long_paragraph(paragraph))
    return [paragraph for paragraph in chunks if len(paragraph) >= 80]


def detect_section_title(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    first_line = lines[0]
    if len(first_line) <= 40 and re.match(
        r"^(\[.*?\]\s*)?(摘要|引言|导论|结语|结论|参考文献|注释|第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|\d+[.、])",
        first_line,
    ):
        return first_line
    return ""


def build_text_chunks(text: str) -> list[dict[str, Any]]:
    paragraphs = split_paragraphs(text)
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for index, paragraph in enumerate(paragraphs, start=1):
        probe = paragraph[:80]
        start_char = text.find(probe, cursor) if probe else -1
        if start_char < 0:
            start_char = cursor
        end_char = min(start_char + len(paragraph), len(text))
        cursor = max(end_char, cursor)
        section_title = detect_section_title(paragraph)
        is_body = not paragraph.startswith(("[可能参考文献]", "[参考文献/注释区域]"))
        warnings: list[str] = []
        if not is_body:
            warnings.append("该 chunk 可能属于参考文献或注释区域。")
        chunks.append(
            {
                "chunk_id": f"C{index:03d}",
                "chunk_index": index,
                "cleaned_text": paragraph,
                "raw_text_reference": paragraph,
                "section_title": section_title,
                "start_char": start_char,
                "end_char": end_char,
                "is_body": is_body,
                "warnings": warnings,
            }
        )
    return chunks


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def retrieve_label_evidence(
    label_definition: dict[str, Any],
    paragraphs: list[str],
    chunks: list[dict[str, Any]],
    paragraph_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    clues = "；".join(label_definition.get("positive_clues", []))
    query = f"{label_definition['definition']}\n正向线索：{clues}"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.asarray(paragraph_embeddings) @ np.asarray(query_embedding)
    candidate_k = min(max(top_k * 2, 10), len(paragraphs))
    top_indices = np.argsort(similarities)[::-1][:candidate_k]
    final_indices = top_indices[: min(top_k, len(top_indices))]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(final_indices, start=1):
        chunk = chunks[int(index)] if int(index) < len(chunks) else {}
        full_text = str(chunk.get("cleaned_text") or paragraphs[index])
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id") or f"C{int(index) + 1:03d}",
                "chunk_index": int(chunk.get("chunk_index") or int(index) + 1),
                "section_title": chunk.get("section_title", ""),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "is_body": bool(chunk.get("is_body", True)),
                "warnings": chunk.get("warnings", []),
                "raw_text_reference": shorten_text(str(chunk.get("raw_text_reference") or full_text), MAX_VISIBLE_EVIDENCE_CHARS),
                "similarity": round(float(similarities[index]), 4),
                "similarity_score": round(float(similarities[index]), 4),
                "text": shorten_text(full_text, MAX_EVIDENCE_CHARS),
                "evidence_excerpt": shorten_text(full_text, MAX_EVIDENCE_CHARS),
                "evidence_full_text": shorten_text(full_text, MAX_VISIBLE_EVIDENCE_CHARS),
            }
        )
    return results


def parse_json_response(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("DeepSeek 未返回可解析的 JSON。")
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def score_label_with_deepseek(
    label_definition: dict[str, Any],
    retrieved: list[dict[str, Any]],
    model_name: str | None = None,
) -> dict[str, Any]:
    compact_definition = {
        "label_name": label_definition.get("label_name"),
        "category": label_definition.get("category"),
        "definition": label_definition.get("definition"),
        "score_0": label_definition.get("score_0"),
        "score_1": label_definition.get("score_1"),
        "score_2": label_definition.get("score_2"),
        "score_3": label_definition.get("score_3"),
        "positive_clues": label_definition.get("positive_clues", [])[:6],
        "negative_clues": label_definition.get("negative_clues", [])[:4],
    }
    prompt = {
        "task": "请根据标签定义、评分标准和候选证据，为论文在该标签上打分。",
        "label_definition": compact_definition,
        "retrieved_paragraphs": [
            {
                "rank": item.get("rank"),
                "chunk_index": item.get("chunk_index"),
                "similarity": item.get("similarity"),
                "text": item.get("evidence_excerpt") or item.get("text", ""),
            }
            for item in retrieved
        ],
        "output_schema": {
            "label_name": "string",
            "score": "integer, one of 0,1,2,3",
            "confidence": "integer, one of 1,2,3,4,5",
            "evidence": ["1到3条原文证据，必须来自召回段落"],
            "reason": "为什么这样打分",
            "uncertainty": "不确定性说明",
        },
    }

    try:
        content = call_deepseek_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是人文社科论文思想谱系分析助手。"
                        "只根据提供的论文段落和标签标准评分，不判断作者本人真实政治立场。"
                        "必须输出严格 JSON，不要输出 Markdown。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            model_name=model_name,
            json_mode=True,
            max_tokens=1400,
            timeout=120,
        )
        raw = parse_json_response(content)
        return normalize_label_result(raw, label_definition, retrieved)
    except DeepSeekTimeoutError:
        return timeout_label_result(label_definition, retrieved)
    except Exception as exc:
        return fallback_label_result(label_definition, retrieved, str(exc))


def generate_markdown_report(
    label_results: list[dict[str, Any]],
    title: str = "用户上传论文",
    model_name: str | None = None,
) -> str:
    compact_results = [
        {
            "label_name": item["label_name"],
            "category": item["category"],
            "score": item["score"],
            "confidence": item["confidence"],
            "evidence": [
                shorten_text(evidence, MAX_REPORT_EVIDENCE_CHARS)
                for evidence in item.get("evidence", [])[:3]
            ],
            "reason": shorten_text(item.get("reason", ""), MAX_REPORT_REASON_CHARS),
            "uncertainty": shorten_text(item.get("uncertainty", ""), 300),
        }
        for item in label_results
    ]
    prompt = f"""
请基于以下标签评分和证据，为论文《{title}》生成一份中文 Markdown 结构化分析报告。

必须包含以下章节：
1. 论文核心问题
2. 方法论倾向
3. 哲学资源
4. 文学观
5. 政治—美学倾向
6. 高分标签解释
7. 原文证据
8. 不确定性说明
9. 产品边界声明

产品边界声明必须明确写出：
本工具只分析论文文本中呈现出的思想倾向，不判断作者本人真实政治立场。

标签分析结果：
{json.dumps(compact_results, ensure_ascii=False, indent=2)}
""".strip()

    try:
        return call_deepseek_chat(
            [
                {
                    "role": "system",
                    "content": "你是严谨的人文社科论文分析助手，输出中文 Markdown 报告。",
                },
                {"role": "user", "content": prompt},
            ],
            model_name=model_name,
            temperature=0.3,
            max_tokens=3600,
            timeout=120,
        )
    except Exception as exc:
        return build_fallback_report(title, label_results, str(exc))


def build_fallback_report(
    title: str,
    label_results: list[dict[str, Any]],
    message: str,
) -> str:
    high_scores = [item for item in label_results if item["score"] >= 2]
    lines = [
        f"# {title} 思想谱系分析报告",
        "",
        "## 生成状态",
        f"DeepSeek 报告生成失败：{message}",
        "",
        "## 高分标签",
    ]
    if high_scores:
        for item in high_scores:
            lines.append(f"- {item['label_name']}：{item['score']} 分，置信度 {item['confidence']}")
    else:
        lines.append("- 暂无分数 >= 2 的标签，或自动评分未成功。")
    lines.extend(
        [
            "",
            "## 不确定性说明",
            "该报告为降级结果，仅供检查流程使用。请配置并确认 DeepSeek API 可用后重新生成。",
            "",
            "## 产品边界声明",
            "本工具只分析论文文本中呈现出的思想倾向，不判断作者本人真实政治立场。",
        ]
    )
    return "\n".join(lines)


def analyze_text(
    text: str,
    *,
    top_k: int = 3,
    title: str = "用户上传论文",
    save_outputs: bool = True,
    selected_label_names: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    if not text or len(text.strip()) < 100:
        raise ValueError("正文过短，无法进行可靠分析。请上传包含完整正文的 PDF 或 TXT。")

    if not get_deepseek_api_key():
        raise MissingAPIKeyError("未配置 DeepSeek API Key，无法生成报告。")

    original_char_count = len(text)
    emit_progress(
        progress_callback,
        progress=0.04,
        status="正在清洗文本",
        stage="clean_text",
    )
    text, truncated, source_char_count = trim_text_for_analysis(text)
    effective_model_name = get_deepseek_model(model_name)
    top_k = clamp_int(top_k, 1, MAX_TOP_K, 5)
    llm_top_k = min(top_k, MAX_LLM_EVIDENCE_PER_LABEL)
    labels = select_label_definitions(load_label_definitions(), selected_label_names)
    emit_progress(
        progress_callback,
        progress=0.08,
        status="正在切分段落",
        stage="split_paragraphs",
    )
    chunks = build_text_chunks(text)
    if not chunks:
        raise ValueError("未能从正文中切分出有效段落。")
    paragraphs = [chunk["cleaned_text"] for chunk in chunks]
    print(f"总段落数：{len(paragraphs)}", flush=True)

    emit_progress(
        progress_callback,
        progress=0.14,
        status="正在加载 embedding 模型",
        stage="load_embedding_model",
    )
    model = load_embedding_model()
    emit_progress(
        progress_callback,
        progress=0.20,
        status="正在生成段落 embedding",
        stage="build_embeddings",
    )
    paragraph_embeddings = model.encode(
        paragraphs,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    label_results: list[dict[str, Any]] = []
    total_labels = len(labels)
    for index, label_definition in enumerate(labels, start=1):
        label_name = str(label_definition.get("label_name", "")).strip()
        loop_base = 0.26 + ((index - 1) / max(total_labels, 1)) * 0.58
        emit_progress(
            progress_callback,
            progress=loop_base,
            status=f"正在召回标签证据：第 {index}/{total_labels} 个标签，标签名 {label_name}",
            stage="retrieve_evidence",
            current=index,
            total=total_labels,
            label_name=label_name,
        )
        print(f"当前分析标签名：{label_name}", flush=True)
        retrieved = retrieve_label_evidence(
            label_definition,
            paragraphs,
            chunks,
            paragraph_embeddings,
            model,
            top_k=llm_top_k,
        )
        print(f"当前标签召回到的证据数量：{len(retrieved)}", flush=True)
        emit_progress(
            progress_callback,
            progress=loop_base + 0.02,
            status=f"正在调用 DeepSeek 分析：第 {index}/{total_labels} 个标签，标签名 {label_name}",
            stage="deepseek_label",
            current=index,
            total=total_labels,
            label_name=label_name,
        )
        print(f"当前标签 DeepSeek 请求开始：{label_name}", flush=True)
        label_result = score_label_with_deepseek(label_definition, retrieved, model_name=effective_model_name)
        reason = str(label_result.get("reason", ""))
        if "超时" in reason or "失败" in reason:
            print(f"当前标签是否超时或失败：是，{label_name}，{reason}", flush=True)
        else:
            print(f"当前标签 DeepSeek 请求结束：{label_name}", flush=True)
        label_results.append(label_result)
        emit_progress(
            progress_callback,
            progress=0.26 + (index / max(total_labels, 1)) * 0.58,
            status=f"已完成标签：{label_name}",
            stage="label_done",
            current=index,
            total=total_labels,
            label_name=label_name,
        )

    emit_progress(
        progress_callback,
        progress=0.88,
        status="正在生成最终报告",
        stage="generate_report",
    )
    report = generate_markdown_report(label_results, title=title, model_name=effective_model_name)
    result = {
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": top_k,
        "llm_evidence_per_label": llm_top_k,
        "model": effective_model_name,
        "original_char_count": original_char_count,
        "source_char_count": source_char_count,
        "analyzed_char_count": len(text),
        "text_truncated": truncated,
        "paragraph_count": len(paragraphs),
        "chunk_count": len(chunks),
        "label_count": total_labels,
        "selected_label_names": selected_label_names,
        "label_results": label_results,
        "report_markdown": report,
    }

    if save_outputs:
        try:
            result.update(save_analysis_outputs(result))
        except Exception as exc:
            result["save_warning"] = f"分析结果未能保存到 outputs：{exc}"

    emit_progress(
        progress_callback,
        progress=1.0,
        status="分析完成",
        stage="done",
    )
    return result


def save_analysis_outputs(result: dict[str, Any]) -> dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"analysis_{timestamp}.json"
    report_path = OUTPUT_DIR / f"report_{timestamp}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(result["report_markdown"], encoding="utf-8")
    return {
        "analysis_json_path": str(json_path),
        "report_markdown_path": str(report_path),
    }

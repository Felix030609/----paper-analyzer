from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from scripts.deepseek_client import DeepSeekTimeoutError, call_deepseek_chat


PRIVATE_UNICODE_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U0010FFFF]")
SUSPICIOUS_SYMBOL_RE = re.compile(r"[�□■◆◇●○★☆▲△▼▽�]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
REFERENCE_TITLE_RE = re.compile(r"^\s*(【?参考文献】?|【?注释】?|References|Bibliography|Notes)\s*$", re.I)
PAGE_NUMBER_RE = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")
NOISE_ONLY_RE = re.compile(r"^[\W_]{6,}$")
CNKI_NOISE_RE = re.compile(
    r"(CNKI|中国知网|China Academic Journal Electronic Publishing House|www\.cnki\.net|"
    r"DOI\s*[:：]?\s*10\.\d+|https?://|Southern Cultural Forum|南方文坛\s*\d{4}\.?\d*|"
    r"最新文本|网络出版|下载时间|版权所有|电子杂志社)",
    re.I,
)
FOOTNOTE_MARK_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿ]")
COMMON_TEXT_CHARS_RE = re.compile(
    r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？；：、“”‘’（）《》〈〉—…·,.!?;:'\"()\[\]{}<>/\-+%=*&@#￥$]"
)


@dataclass
class CleaningBlockResult:
    block_id: str
    raw_text: str
    cleaned_text: str
    section_title: str = ""
    start_char: int = 0
    end_char: int = 0
    removed_noise: list[str] = field(default_factory=list)
    possible_footnotes: list[str] = field(default_factory=list)
    possible_references: list[str] = field(default_factory=list)
    uncertain_segments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentCleaningResult:
    raw_text: str
    cleaned_text: str
    cleaning_method: str
    blocks: list[CleaningBlockResult] = field(default_factory=list)
    removed_noise: list[str] = field(default_factory=list)
    possible_footnotes: list[str] = field(default_factory=list)
    possible_references: list[str] = field(default_factory=list)
    references_text: str = ""
    notes_text: str = ""
    quality_before: dict[str, Any] = field(default_factory=dict)
    quality_after: dict[str, Any] = field(default_factory=dict)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    global_warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        backend_scores = self.extraction_metadata.get("backend_scores", {})
        return {
            "cleaning_method": self.cleaning_method,
            "raw_text_length": len(self.raw_text or ""),
            "cleaned_text_length": len(self.cleaned_text or ""),
            "block_count": len(self.blocks),
            "backend": self.extraction_metadata.get("backend", ""),
            "backend_scores": backend_scores,
            "raw_quality_score": self.quality_before.get("quality_score"),
            "cleaned_quality_score": self.quality_after.get("quality_score"),
            "garbled_char_count_before": self.quality_before.get("garbled_char_count", 0),
            "garbled_char_count_after": self.quality_after.get("garbled_char_count", 0),
            "removed_noise_count": len(self.removed_noise),
            "removed_header_footer_count": sum(1 for item in self.removed_noise if CNKI_NOISE_RE.search(item)),
            "possible_footnotes_count": sum(len(block.possible_footnotes) for block in self.blocks)
            + len(self.possible_footnotes),
            "possible_references_count": sum(len(block.possible_references) for block in self.blocks)
            + len(self.possible_references),
            "references_text_length": len(self.references_text or ""),
            "notes_text_length": len(self.notes_text or ""),
            "warnings_count": sum(len(block.warnings) for block in self.blocks) + len(self.global_warnings),
            "warnings": self.global_warnings[:10],
        }


def count_chinese(text: str) -> int:
    return len(CHINESE_RE.findall(text or ""))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def evaluate_text_quality(text: str) -> dict[str, Any]:
    text = text or ""
    total_chars = len(text)
    non_space_chars = len(re.sub(r"\s+", "", text))
    chinese_count = count_chinese(text)
    punctuation_count = len(re.findall(r"[，。！？；：、“”‘’（）《》,.!?;:]", text))
    whitespace_count = len(re.findall(r"\s", text))
    private_count = len(PRIVATE_UNICODE_RE.findall(text))
    suspicious_count = len(SUSPICIOUS_SYMBOL_RE.findall(text))
    cnki_count = len(CNKI_NOISE_RE.findall(text))
    isolated_breaks = len(re.findall(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_line_count = sum(1 for line in lines if 0 < len(line) <= 8)
    chinese_spacing_count = len(re.findall(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", text))

    chinese_char_ratio = safe_ratio(chinese_count, max(non_space_chars, 1))
    punctuation_ratio = safe_ratio(punctuation_count, max(non_space_chars, 1))
    whitespace_ratio = safe_ratio(whitespace_count, max(total_chars, 1))
    private_unicode_ratio = safe_ratio(private_count, max(total_chars, 1))
    line_break_density = safe_ratio(len(lines), max(total_chars / 500, 1))

    score = 100.0
    score -= min(private_count * 1.4, 25)
    score -= min(suspicious_count * 1.2, 20)
    score -= min(cnki_count * 4.0, 16)
    score -= min(isolated_breaks * 0.7, 15)
    score -= min(chinese_spacing_count * 0.4, 12)
    if chinese_char_ratio < 0.25:
        score -= 18
    if whitespace_ratio > 0.35:
        score -= 10
    if short_line_count > max(len(lines) * 0.35, 10):
        score -= 8
    quality_score = int(max(0, min(100, round(score))))

    warnings: list[str] = []
    if private_count:
        warnings.append("检测到 Unicode 私有区乱码字符。")
    if suspicious_count:
        warnings.append("检测到异常符号或无法识别字符。")
    if cnki_count:
        warnings.append("检测到 CNKI/DOI/期刊页脚等出版信息。")
    if isolated_breaks > 20 or chinese_spacing_count > 20:
        warnings.append("检测到较多中文断行或中文字符间空格。")
    if quality_score < 60:
        warnings.append("PDF 提取质量较低，建议上传 TXT / Word / 可复制文本版，或启用增强清洗。")

    return {
        "total_chars": total_chars,
        "chinese_char_ratio": chinese_char_ratio,
        "punctuation_ratio": punctuation_ratio,
        "whitespace_ratio": whitespace_ratio,
        "private_unicode_ratio": private_unicode_ratio,
        "garbled_char_count": private_count + suspicious_count,
        "suspicious_symbol_count": suspicious_count,
        "cnki_noise_count": cnki_count,
        "line_break_density": line_break_density,
        "isolated_line_break_count": isolated_breaks,
        "chinese_spacing_count": chinese_spacing_count,
        "quality_score": quality_score,
        "warnings": warnings,
    }


def _uploaded_file_bytes(file_obj: Any) -> bytes:
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    data = file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return data


def _extract_pdf_pymupdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF

    parts: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
            page_lines = []
            for block in blocks:
                text = str(block[4] or "").strip()
                if text:
                    page_lines.append(text)
            if page_lines:
                parts.append(f"[PDF第{page_index}页]\n" + "\n".join(page_lines))
    return "\n\n".join(parts).strip()


def _extract_pdf_pdfplumber(file_bytes: bytes) -> str:
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            if text.strip():
                parts.append(f"[PDF第{page_index}页]\n{text.strip()}")
    return "\n\n".join(parts).strip()


def extract_text_with_multiple_backends(file_obj: Any) -> dict[str, Any]:
    suffix = Path(file_obj.name).suffix.lower()
    warnings: list[str] = []

    if suffix == ".txt":
        raw = _uploaded_file_bytes(file_obj)
        if not raw:
            raise ValueError("TXT 文件为空。")
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                raw_text = raw.decode(encoding).strip()
                quality = evaluate_text_quality(raw_text)
                return {
                    "raw_text": raw_text,
                    "backend": f"txt:{encoding}",
                    "backend_scores": {f"txt:{encoding}": quality["quality_score"]},
                    "backend_quality": {f"txt:{encoding}": quality},
                    "warnings": quality.get("warnings", []),
                }
            except UnicodeDecodeError:
                continue
        raise ValueError("TXT 编码无法识别。请保存为 UTF-8 后重新上传。")

    if suffix != ".pdf":
        raise ValueError("仅支持 PDF 或 TXT 文件。")

    file_bytes = _uploaded_file_bytes(file_obj)
    candidates: dict[str, str] = {}
    backend_quality: dict[str, dict[str, Any]] = {}

    try:
        candidates["pymupdf"] = _extract_pdf_pymupdf(file_bytes)
    except Exception as exc:
        warnings.append(f"PyMuPDF 提取失败：{exc}")

    try:
        candidates["pdfplumber"] = _extract_pdf_pdfplumber(file_bytes)
    except Exception as exc:
        warnings.append(f"pdfplumber 提取失败：{exc}")

    if not any(text.strip() for text in candidates.values()):
        raise ValueError("PDF 解析失败。请确认文件不是扫描图片版 PDF，或改用可复制文本的 PDF/TXT。")

    backend_scores: dict[str, int] = {}
    for backend, text in candidates.items():
        quality = evaluate_text_quality(text)
        backend_quality[backend] = quality
        backend_scores[backend] = int(quality["quality_score"])

    backend = max(backend_scores, key=backend_scores.get)
    raw_text = candidates[backend].strip()
    selected_quality = backend_quality[backend]
    warnings.extend(selected_quality.get("warnings", []))

    return {
        "raw_text": raw_text,
        "backend": backend,
        "backend_scores": backend_scores,
        "backend_quality": backend_quality,
        "warnings": warnings,
    }


def extract_raw_text_from_file(uploaded_file) -> str:
    result = extract_text_with_multiple_backends(uploaded_file)
    raw_text = str(result.get("raw_text") or "").strip()
    if not raw_text:
        raise ValueError("未能从文件中提取正文。请检查文件内容后重新上传。")
    return raw_text


def remove_private_and_suspicious_symbols(text: str) -> tuple[str, list[str]]:
    removed = PRIVATE_UNICODE_RE.findall(text)
    removed.extend(SUSPICIOUS_SYMBOL_RE.findall(text))
    text = PRIVATE_UNICODE_RE.sub("", text)
    text = SUSPICIOUS_SYMBOL_RE.sub("", text)
    text = COMMON_TEXT_CHARS_RE.sub("", text)
    return text, removed[:300]


def normalize_chinese_spacing(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def normalize_punctuation_for_chinese_line(line: str) -> str:
    if re.search(r"https?://|www\.|DOI\s*[:：]?\s*10\.|[A-Za-z]\.[A-Za-z]", line, re.I):
        return line
    if count_chinese(line) < max(2, len(line) * 0.25):
        return line
    return (
        line.replace(",", "，")
        .replace(";", "；")
        .replace(":", "：")
        .replace("?", "？")
        .replace("!", "！")
    )


def is_cnki_or_publication_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if PAGE_NUMBER_RE.match(stripped):
        return True
    if CNKI_NOISE_RE.search(stripped):
        return True
    if re.match(r"^\s*(第\s*)?\d+\s*页\s*$", stripped):
        return True
    if re.match(r"^\s*\d{4}\s*年第?\s*\d+\s*期\s*$", stripped):
        return True
    if NOISE_ONLY_RE.match(stripped):
        return True
    return False


def should_merge_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith(("。", "？", "！", "：", "；", "”", "」", "》", ".", "?", "!", ":")):
        return False
    if re.match(r"^(第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|\d+[.、])", current):
        return False
    if CHINESE_RE.search(previous[-1:]) and CHINESE_RE.search(current[:1]):
        return True
    return False


def fix_pdf_line_breaks(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    fixed: list[str] = []
    buffer = ""
    for raw_line in lines:
        line = normalize_chinese_spacing(raw_line.strip())
        line = normalize_punctuation_for_chinese_line(line)
        if not line:
            if buffer:
                fixed.append(buffer)
                buffer = ""
            if fixed and fixed[-1] != "":
                fixed.append("")
            continue
        if buffer and should_merge_lines(buffer, line):
            buffer += line
        else:
            if buffer:
                fixed.append(buffer)
            buffer = line
    if buffer:
        fixed.append(buffer)
    return "\n".join(fixed)


def clean_pdf_extracted_text(raw_text: str) -> dict[str, Any]:
    quality_before = evaluate_text_quality(raw_text)
    warnings = list(quality_before.get("warnings", []))
    removed_noise: list[str] = []
    notes_lines: list[str] = []
    reference_lines: list[str] = []

    text, removed_symbols = remove_private_and_suspicious_symbols(raw_text or "")
    removed_noise.extend([f"乱码符号：{symbol}" for symbol in removed_symbols])
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    line_counts: dict[str, int] = {}
    for line in raw_lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    cleaned_lines: list[str] = []
    current_region = "body"
    seen_short_titles: set[str] = set()
    repeated_removed: set[str] = set()
    body_candidate_lines: list[str] = []

    for line in raw_lines:
        stripped = normalize_chinese_spacing(re.sub(r"\s+", " ", line).strip())
        stripped = normalize_punctuation_for_chinese_line(stripped)
        if not stripped:
            if current_region == "body" and cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if is_cnki_or_publication_noise(stripped):
            removed_noise.append(stripped)
            continue

        if REFERENCE_TITLE_RE.match(stripped):
            if "注" in stripped or re.search(r"Notes", stripped, re.I):
                current_region = "notes"
                notes_lines.append(stripped)
            else:
                current_region = "references"
                reference_lines.append(stripped)
            continue

        if current_region == "references":
            reference_lines.append(stripped)
            continue
        if current_region == "notes":
            notes_lines.append(stripped)
            continue

        if len(stripped) < 40 and line_counts.get(stripped, 0) >= 3:
            if stripped not in repeated_removed:
                removed_noise.append(stripped)
                repeated_removed.add(stripped)
            continue

        if len(stripped) < 80 and stripped in seen_short_titles:
            removed_noise.append(stripped)
            continue
        if len(stripped) < 80:
            seen_short_titles.add(stripped)

        body_candidate_lines.append(stripped)

    fixed_body = fix_pdf_line_breaks("\n".join(body_candidate_lines))
    for line in fixed_body.splitlines():
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(stripped)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    quality_after = evaluate_text_quality(cleaned_text)

    if not cleaned_text:
        cleaned_text = raw_text
        quality_after = evaluate_text_quality(cleaned_text)
        warnings.append("规则清洗后文本为空，已回退到 raw_text。")

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "removed_noise": removed_noise[:500],
        "references_text": "\n".join(reference_lines).strip(),
        "notes_text": "\n".join(notes_lines).strip(),
        "quality_before": quality_before,
        "quality_after": quality_after,
        "warnings": list(dict.fromkeys(warnings + quality_after.get("warnings", [])))[:20],
    }


def rule_based_clean_text(raw_text: str) -> dict[str, Any]:
    try:
        return clean_pdf_extracted_text(raw_text)
    except Exception as exc:
        return {
            "raw_text": raw_text,
            "cleaned_text": raw_text,
            "removed_noise": [],
            "references_text": "",
            "notes_text": "",
            "quality_before": evaluate_text_quality(raw_text),
            "quality_after": evaluate_text_quality(raw_text),
            "warnings": [f"规则清洗失败，已回退到 raw_text：{exc}"],
        }


def split_into_cleaning_blocks(cleaned_text: str, max_chars: int = 2500) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start = 0
    block_index = 1
    text = cleaned_text or ""
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            break_at = max(text.rfind("\n\n", start, end), text.rfind("。", start, end))
            if break_at > start + max_chars * 0.55:
                end = break_at + 1
        block_text = text[start:end].strip()
        if block_text:
            blocks.append(
                {
                    "block_id": f"B{block_index:03d}",
                    "raw_text": block_text,
                    "block_text": block_text,
                    "start_char": start,
                    "end_char": end,
                }
            )
            block_index += 1
        start = end
    return blocks


def parse_llm_cleaning_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("LLM 清洗结果不是 JSON。")
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def llm_clean_block(block_text: str, model_name: str | None = None) -> dict[str, Any]:
    prompt = {
        "task": "对中文学术论文片段做格式清洗和结构识别。",
        "strict_rules": [
            "只修复格式和断行，不要总结。",
            "不要改写、补充或改变作者原意。",
            "不要删除不确定内容。",
            "删除明显页眉、页脚、页码。",
            "标记疑似脚注、注释、参考文献和乱码。",
            "保留正文原句。",
            "不确定时放入 uncertain_segments 或 warnings。",
        ],
        "output_schema": {
            "cleaned_text": "清洗后的文本，尽量保留原文表达",
            "section_title": "识别到的章节标题，没有则为空",
            "removed_noise": ["删除的明显噪声"],
            "possible_footnotes": ["疑似脚注或注释"],
            "possible_references": ["疑似参考文献"],
            "uncertain_segments": ["不确定片段"],
            "warnings": ["清洗警告"],
        },
        "block_text": block_text,
    }
    try:
        content = call_deepseek_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是中文学术论文文本清洗助手。必须输出严格 JSON。"
                        "只做格式清洗和结构识别，不得总结、改写、补充或改变作者原意。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            model_name=model_name,
            json_mode=True,
            thinking=False,
            temperature=0.0,
            max_tokens=2600,
            timeout=90,
        )
        parsed = parse_llm_cleaning_json(content)
        cleaned_text = str(parsed.get("cleaned_text") or block_text).strip() or block_text
        return {
            "cleaned_text": cleaned_text,
            "section_title": str(parsed.get("section_title") or "").strip(),
            "removed_noise": parsed.get("removed_noise") if isinstance(parsed.get("removed_noise"), list) else [],
            "possible_footnotes": parsed.get("possible_footnotes")
            if isinstance(parsed.get("possible_footnotes"), list)
            else [],
            "possible_references": parsed.get("possible_references")
            if isinstance(parsed.get("possible_references"), list)
            else [],
            "uncertain_segments": parsed.get("uncertain_segments")
            if isinstance(parsed.get("uncertain_segments"), list)
            else [],
            "warnings": parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else [],
        }
    except DeepSeekTimeoutError as exc:
        return {
            "cleaned_text": block_text,
            "section_title": "",
            "removed_noise": [],
            "possible_footnotes": [],
            "possible_references": [],
            "uncertain_segments": [],
            "warnings": [f"LLM 清洗超时，已保留原 block：{exc}"],
        }
    except Exception as exc:
        return {
            "cleaned_text": block_text,
            "section_title": "",
            "removed_noise": [],
            "possible_footnotes": [],
            "possible_references": [],
            "uncertain_segments": [],
            "warnings": [f"LLM 清洗失败，已保留原 block：{exc}"],
        }


def build_cleaned_document(
    raw_text: str,
    use_llm_cleaning: bool = False,
    model_name: str | None = None,
    extraction_metadata: dict[str, Any] | None = None,
) -> DocumentCleaningResult:
    rule_result = rule_based_clean_text(raw_text)
    base_cleaned_text = str(rule_result.get("cleaned_text") or raw_text or "").strip()
    global_warnings = list(rule_result.get("warnings") or [])
    removed_noise = list(rule_result.get("removed_noise") or [])

    if extraction_metadata:
        global_warnings.extend(str(item) for item in extraction_metadata.get("warnings", []))

    if not base_cleaned_text:
        return DocumentCleaningResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            cleaning_method="raw_fallback",
            removed_noise=removed_noise,
            references_text=str(rule_result.get("references_text") or ""),
            notes_text=str(rule_result.get("notes_text") or ""),
            quality_before=rule_result.get("quality_before", evaluate_text_quality(raw_text)),
            quality_after=evaluate_text_quality(raw_text),
            extraction_metadata=extraction_metadata or {},
            global_warnings=global_warnings + ["cleaned_text 为空，已回退到 raw_text。"],
        )

    if not use_llm_cleaning:
        return DocumentCleaningResult(
            raw_text=raw_text,
            cleaned_text=base_cleaned_text,
            cleaning_method="rule_based",
            removed_noise=removed_noise,
            references_text=str(rule_result.get("references_text") or ""),
            notes_text=str(rule_result.get("notes_text") or ""),
            quality_before=rule_result.get("quality_before", evaluate_text_quality(raw_text)),
            quality_after=rule_result.get("quality_after", evaluate_text_quality(base_cleaned_text)),
            extraction_metadata=extraction_metadata or {},
            global_warnings=global_warnings,
        )

    blocks = split_into_cleaning_blocks(base_cleaned_text)
    block_results: list[CleaningBlockResult] = []
    for block in blocks:
        print(f"LLM 清洗 block：{block['block_id']}", flush=True)
        llm_result = llm_clean_block(block["block_text"], model_name=model_name)
        block_results.append(
            CleaningBlockResult(
                block_id=block["block_id"],
                raw_text=block["block_text"],
                cleaned_text=llm_result["cleaned_text"],
                section_title=llm_result.get("section_title", ""),
                start_char=int(block["start_char"]),
                end_char=int(block["end_char"]),
                removed_noise=[str(item) for item in llm_result.get("removed_noise", [])],
                possible_footnotes=[str(item) for item in llm_result.get("possible_footnotes", [])],
                possible_references=[str(item) for item in llm_result.get("possible_references", [])],
                uncertain_segments=[str(item) for item in llm_result.get("uncertain_segments", [])],
                warnings=[str(item) for item in llm_result.get("warnings", [])],
            )
        )

    cleaned_text = "\n\n".join(block.cleaned_text for block in block_results if block.cleaned_text).strip()
    if not cleaned_text:
        cleaned_text = base_cleaned_text
        global_warnings.append("LLM 清洗合并后为空，已回退到规则清洗结果。")

    return DocumentCleaningResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        cleaning_method="rule_based_plus_llm",
        blocks=block_results,
        removed_noise=removed_noise,
        references_text=str(rule_result.get("references_text") or ""),
        notes_text=str(rule_result.get("notes_text") or ""),
        quality_before=rule_result.get("quality_before", evaluate_text_quality(raw_text)),
        quality_after=evaluate_text_quality(cleaned_text),
        extraction_metadata=extraction_metadata or {},
        global_warnings=global_warnings,
    )

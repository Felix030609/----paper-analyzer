from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from scripts.deepseek_client import DeepSeekTimeoutError, call_deepseek_chat


REFERENCE_PATTERNS = [
    r"^参考文献\s*$",
    r"^注释\s*$",
    r"^References\s*$",
    r"^Bibliography\s*$",
]
PAGE_NUMBER_RE = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")
NOISE_RE = re.compile(r"^[\W_]{6,}$")


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
    global_warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "cleaning_method": self.cleaning_method,
            "raw_text_length": len(self.raw_text or ""),
            "cleaned_text_length": len(self.cleaned_text or ""),
            "block_count": len(self.blocks),
            "removed_noise_count": len(self.removed_noise),
            "possible_footnotes_count": sum(len(block.possible_footnotes) for block in self.blocks)
            + len(self.possible_footnotes),
            "possible_references_count": sum(len(block.possible_references) for block in self.blocks)
            + len(self.possible_references),
            "warnings_count": sum(len(block.warnings) for block in self.blocks) + len(self.global_warnings),
            "warnings": self.global_warnings[:8],
        }


def extract_raw_text_from_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        try:
            parts: list[str] = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        parts.append(f"[PDF第{page_index}页]\n{text.strip()}")
            raw_text = "\n\n".join(parts).strip()
        except Exception as exc:
            raise ValueError("PDF 解析失败。请确认文件不是扫描图片版 PDF，或改用可复制文本的 PDF/TXT。") from exc
    elif suffix == ".txt":
        raw = uploaded_file.getvalue()
        if not raw:
            raise ValueError("TXT 文件为空。")
        raw_text = ""
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                raw_text = raw.decode(encoding).strip()
                break
            except UnicodeDecodeError:
                continue
        if not raw_text:
            raise ValueError("TXT 编码无法识别。请保存为 UTF-8 后重新上传。")
    else:
        raise ValueError("仅支持 PDF 或 TXT 文件。")

    if not raw_text.strip():
        raise ValueError("未能从文件中提取正文。请检查文件内容后重新上传。")
    return raw_text


def fix_pdf_line_breaks(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    fixed: list[str] = []
    buffer = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                fixed.append(buffer)
                buffer = ""
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


def should_merge_lines(previous: str, current: str) -> bool:
    if previous.endswith(("。", "？", "！", "：", "；", ".", "?", "!", "”", "」", "》")):
        return False
    if re.match(r"^(第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|\d+[.、])", current):
        return False
    if len(previous) < 8 or len(current) < 4:
        return False
    return True


def rule_based_clean_text(raw_text: str) -> dict[str, Any]:
    try:
        text = fix_pdf_line_breaks(raw_text or "")
        lines = text.splitlines()
        line_counts = {line.strip(): 0 for line in lines if line.strip()}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        cleaned_lines: list[str] = []
        removed_noise: list[str] = []
        warnings: list[str] = []
        in_reference_area = False
        repeated_removed = set()
        title_seen = set()

        for line in lines:
            stripped = re.sub(r"\s+", " ", line).strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            if PAGE_NUMBER_RE.match(stripped):
                removed_noise.append(stripped)
                continue

            if NOISE_RE.match(stripped):
                removed_noise.append(stripped)
                continue

            if any(re.match(pattern, stripped, flags=re.I) for pattern in REFERENCE_PATTERNS):
                in_reference_area = True
                cleaned_lines.append(f"[参考文献/注释区域] {stripped}")
                continue

            if len(stripped) < 40 and line_counts.get(stripped, 0) >= 3:
                if stripped not in repeated_removed:
                    removed_noise.append(stripped)
                    repeated_removed.add(stripped)
                continue

            if len(stripped) < 80 and stripped in title_seen:
                removed_noise.append(stripped)
                continue
            if len(stripped) < 80:
                title_seen.add(stripped)

            if in_reference_area:
                cleaned_lines.append(f"[可能参考文献] {stripped}")
            else:
                cleaned_lines.append(stripped)

        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        if not cleaned_text:
            warnings.append("规则清洗后文本为空，已回退到 raw_text。")
            cleaned_text = raw_text
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "removed_noise": removed_noise[:200],
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "raw_text": raw_text,
            "cleaned_text": raw_text,
            "removed_noise": [],
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
            "只做格式清洗和结构识别，不要总结。",
            "不要改写、补充或改变作者原意。",
            "修复明显 PDF 断行。",
            "删除明显页眉、页脚、页码。",
            "标记疑似脚注、注释、参考文献。",
            "保留正文原句。",
            "不确定时放入 uncertain_segments 或 warnings，不要直接删除。",
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
                        "不得总结、改写、补充或改变作者原意。"
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
) -> DocumentCleaningResult:
    rule_result = rule_based_clean_text(raw_text)
    base_cleaned_text = str(rule_result.get("cleaned_text") or raw_text or "").strip()
    global_warnings = list(rule_result.get("warnings") or [])
    removed_noise = list(rule_result.get("removed_noise") or [])

    if not base_cleaned_text:
        return DocumentCleaningResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            cleaning_method="raw_fallback",
            removed_noise=removed_noise,
            global_warnings=global_warnings + ["cleaned_text 为空，已回退到 raw_text。"],
        )

    if not use_llm_cleaning:
        return DocumentCleaningResult(
            raw_text=raw_text,
            cleaned_text=base_cleaned_text,
            cleaning_method="rule_based",
            removed_noise=removed_noise,
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
        global_warnings=global_warnings,
    )

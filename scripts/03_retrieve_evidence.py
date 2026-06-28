from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "data" / "training" / "人文社科论文思想谱系训练数据模板_已补证据.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

LABEL_DEFINITIONS = {
    "现代性批判": "关注现代化、资本主义、市场社会、启蒙理性、全球化与中国社会转型中的结构性矛盾，对现代性本身进行历史反思和批判。",
    "社会历史批评": "从社会结构、历史条件、制度变迁、阶级关系和现实语境解释文本、思想或文化现象。",
    "思想史研究": "追踪概念、观念、知识分子论述和思想传统在具体历史语境中的形成、转化与争论。",
}


def is_non_empty(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def resolve_text_path(text_path: object) -> Path | None:
    if not is_non_empty(text_path):
        return None

    raw_path = Path(str(text_path).strip())
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        PROJECT_ROOT / raw_path,
        PROCESSED_DIR / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_p001_text() -> str:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"找不到训练数据文件：{EXCEL_PATH}")

    papers = pd.read_excel(EXCEL_PATH, sheet_name="01_papers")
    required = {"paper_id", "full_text", "text_path"}
    missing = required - set(papers.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"01_papers 缺少必要字段：{missing_text}")

    rows = papers[papers["paper_id"].astype(str).str.strip().eq("P001")]
    if rows.empty:
        raise ValueError("01_papers 中找不到 paper_id=P001")

    row = rows.iloc[0]
    if is_non_empty(row.get("full_text")):
        return str(row["full_text"]).strip()

    text_path = resolve_text_path(row.get("text_path"))
    if text_path and text_path.exists():
        return text_path.read_text(encoding="utf-8").strip()

    raise ValueError("P001 的 full_text 为空，且 text_path 指向的 txt 不存在")


def split_long_paragraph(paragraph: str, max_chars: int = 550) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?])", paragraph)
        if sentence.strip()
    ]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)

    return chunks or [paragraph]


def split_paragraphs(text: str) -> list[str]:
    raw_paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n+", text)
        if part.strip()
    ]
    if len(raw_paragraphs) >= 3:
        paragraphs: list[str] = []
        for paragraph in raw_paragraphs:
            paragraphs.extend(split_long_paragraph(paragraph))
        return paragraphs

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    if len(lines) >= 3:
        paragraphs = []
        for line in lines:
            paragraphs.extend(split_long_paragraph(line))
        return paragraphs

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?])", text)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > 550 and current:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)

    return chunks or [text]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索 P001 中最相关的证据段落")
    parser.add_argument(
        "label",
        nargs="?",
        help=f"标签名，可选：{', '.join(LABEL_DEFINITIONS)}",
    )
    parser.add_argument("--top-k", type=int, default=3, help="返回段落数量，默认 3")
    return parser.parse_args()


def choose_label(label: str | None) -> str:
    if label:
        label = label.strip()
    else:
        print("可选标签：")
        for name in LABEL_DEFINITIONS:
            print(f"- {name}")
        label = input("请输入标签名：").strip()

    if label not in LABEL_DEFINITIONS:
        allowed = ", ".join(LABEL_DEFINITIONS)
        raise ValueError(f"未知标签：{label}。可选标签：{allowed}")

    return label


def main() -> None:
    args = parse_args()
    label = choose_label(args.label)
    text = load_p001_text()
    paragraphs = split_paragraphs(text)

    print(f"加载模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    label_embedding = model.encode(
        [LABEL_DEFINITIONS[label]],
        normalize_embeddings=True,
    )[0]
    paragraph_embeddings = model.encode(
        paragraphs,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    similarities = np.asarray(paragraph_embeddings) @ np.asarray(label_embedding)
    top_k = min(args.top_k, len(paragraphs))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n标签：{label}")
    print(f"标签定义：{LABEL_DEFINITIONS[label]}")
    print(f"段落总数：{len(paragraphs)}")
    print(f"\n最相关的 {top_k} 个段落：")

    for rank, index in enumerate(top_indices, start=1):
        score = float(similarities[index])
        paragraph = paragraphs[index].replace("\n", " ")
        print(f"\n{rank}. 相似度：{score:.4f}")
        print(paragraph)


if __name__ == "__main__":
    main()

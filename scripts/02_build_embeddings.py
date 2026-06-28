from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "data" / "training" / "人文社科论文思想谱系训练数据模板_已补证据.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EMBEDDING_PATH = OUTPUT_DIR / "paper_embeddings.npy"
PAPER_IDS_PATH = OUTPUT_DIR / "paper_ids.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"01_papers 缺少必要字段：{missing_text}")


def is_non_empty(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def resolve_text_path(text_path: object) -> Path | None:
    if not is_non_empty(text_path):
        return None

    raw_path = Path(str(text_path).strip())
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / raw_path,
                PROCESSED_DIR / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1] if candidates else None


def load_text(row: pd.Series) -> str | None:
    if is_non_empty(row.get("full_text")):
        return str(row["full_text"]).strip()

    candidate = resolve_text_path(row.get("text_path"))
    if candidate and candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()

    return None


def main() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"找不到训练数据文件：{EXCEL_PATH}")

    papers = pd.read_excel(EXCEL_PATH, sheet_name="01_papers")
    require_columns(papers, {"paper_id", "full_text", "text_path"})

    paper_ids: list[str] = []
    texts: list[str] = []
    skipped: list[str] = []

    for _, row in papers.iterrows():
        if not is_non_empty(row.get("paper_id")):
            continue

        paper_id = str(row["paper_id"]).strip()
        text = load_text(row)
        if text:
            paper_ids.append(paper_id)
            texts.append(text)
        else:
            skipped.append(paper_id)

    if not texts:
        raise ValueError("没有找到可用于生成 embedding 的论文正文")

    if skipped:
        print(f"跳过无正文论文：{', '.join(skipped)}")

    print(f"加载模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_PATH, np.asarray(embeddings, dtype=np.float32))
    PAPER_IDS_PATH.write_text(
        json.dumps(paper_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已保存 embedding：{EMBEDDING_PATH}")
    print(f"已保存 paper_id 顺序：{PAPER_IDS_PATH}")
    print(f"论文数量：{len(paper_ids)}")
    print(f"向量维度：{embeddings.shape[1]}")


if __name__ == "__main__":
    main()

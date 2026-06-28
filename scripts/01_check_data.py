from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "data" / "training" / "人文社科论文思想谱系训练数据模板_已补证据.xlsx"

REQUIRED_SHEETS = {
    "01_papers": {"paper_id", "full_text", "text_path"},
    "02_annotations": {"paper_id"},
    "03_evidence": {"paper_id", "label", "score", "evidence_text"},
}

ANNOTATION_META_COLUMNS = {
    "paper_id",
    "core_tags",
    "overall_judgment",
    "annotator",
    "confidence_1_5",
    "status",
    "notes",
}


def read_sheet(sheet_name: str) -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"找不到训练数据文件：{EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
    missing = REQUIRED_SHEETS[sheet_name] - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{sheet_name} 缺少必要字段：{missing_text}")

    return df


def clean_ids(series: pd.Series) -> set[str]:
    return {
        str(value).strip()
        for value in series.dropna()
        if str(value).strip()
    }


def print_sheet_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}]")
    print(f"行数：{len(df)}")
    print(f"列名：{list(df.columns)}")


def print_p001_scores(annotations: pd.DataFrame) -> None:
    p001_rows = annotations[
        annotations["paper_id"].astype(str).str.strip().eq("P001")
    ]
    if p001_rows.empty:
        raise ValueError("02_annotations 中找不到 paper_id=P001 的标注行")

    p001 = p001_rows.iloc[0]
    label_columns = [
        column
        for column in annotations.columns
        if column not in ANNOTATION_META_COLUMNS
    ]

    print("\n[P001 标签分数]")
    for column in label_columns:
        print(f"{column}: {p001[column]}")


def print_p001_evidence_count(evidence: pd.DataFrame) -> None:
    p001_evidence = evidence[
        evidence["paper_id"].astype(str).str.strip().eq("P001")
    ]
    print(f"\nP001 证据数量：{len(p001_evidence)}")


def check_paper_id_links(papers: pd.DataFrame, annotations: pd.DataFrame, evidence: pd.DataFrame) -> None:
    paper_ids = clean_ids(papers["paper_id"])
    annotation_ids = clean_ids(annotations["paper_id"])
    evidence_ids = clean_ids(evidence["paper_id"])

    missing_in_papers_from_annotations = annotation_ids - paper_ids
    missing_in_papers_from_evidence = evidence_ids - paper_ids

    if missing_in_papers_from_annotations:
        ids = ", ".join(sorted(missing_in_papers_from_annotations))
        raise ValueError(f"02_annotations 中存在 01_papers 找不到的 paper_id：{ids}")

    if missing_in_papers_from_evidence:
        ids = ", ".join(sorted(missing_in_papers_from_evidence))
        raise ValueError(f"03_evidence 中存在 01_papers 找不到的 paper_id：{ids}")

    print("\n[paper_id 对应检查]")
    print("02_annotations.paper_id 均可在 01_papers 中找到")
    print("03_evidence.paper_id 均可在 01_papers 中找到")

    papers_without_annotations = paper_ids - annotation_ids
    papers_without_evidence = paper_ids - evidence_ids
    if papers_without_annotations:
        ids = ", ".join(sorted(papers_without_annotations))
        print(f"提示：以下论文暂未标注：{ids}")
    if papers_without_evidence:
        ids = ", ".join(sorted(papers_without_evidence))
        print(f"提示：以下论文暂未配置证据：{ids}")


def main() -> None:
    data = {sheet: read_sheet(sheet) for sheet in REQUIRED_SHEETS}

    for sheet_name, df in data.items():
        print_sheet_summary(sheet_name, df)

    print_p001_scores(data["02_annotations"])
    print_p001_evidence_count(data["03_evidence"])
    check_paper_id_links(
        data["01_papers"],
        data["02_annotations"],
        data["03_evidence"],
    )


if __name__ == "__main__":
    main()

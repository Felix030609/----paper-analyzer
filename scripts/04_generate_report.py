from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_uploaded_paper import (  # noqa: E402
    MissingAPIKeyError,
    analyze_text,
)


EXCEL_PATH = PROJECT_ROOT / "data" / "training" / "人文社科论文思想谱系训练数据模板_已补证据.xlsx"


def read_p001_text() -> tuple[str, str]:
    import pandas as pd

    papers = pd.read_excel(EXCEL_PATH, sheet_name="01_papers")
    rows = papers[papers["paper_id"].astype(str).str.strip().eq("P001")]
    if rows.empty:
        raise ValueError("01_papers 中找不到 paper_id=P001")

    row = rows.iloc[0]
    title = str(row.get("title", "P001")).strip() or "P001"
    full_text = row.get("full_text")
    if not pd.isna(full_text) and str(full_text).strip():
        return str(full_text).strip(), title

    text_path = PROJECT_ROOT / "data" / "processed" / str(row.get("text_path", "")).strip()
    if text_path.exists():
        return text_path.read_text(encoding="utf-8"), title

    raise ValueError("P001 的 full_text 为空，且 data/processed 下找不到对应 txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 P001 生成 V0 自动分析报告")
    parser.add_argument("--top-k", type=int, default=3, help="每个标签召回段落数，默认 3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        text, title = read_p001_text()
        result = analyze_text(text, top_k=args.top_k, title=title, save_outputs=False)

        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "P001_report.md"
        json_path = output_dir / "P001_analysis.json"
        report_path.write_text(result["report_markdown"], encoding="utf-8")
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"报告已保存：{report_path}")
        print(f"JSON 已保存：{json_path}")
        return 0
    except MissingAPIKeyError as exc:
        print(f"配置错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

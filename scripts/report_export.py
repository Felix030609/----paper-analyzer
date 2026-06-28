from __future__ import annotations

import html
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any


BOUNDARY_STATEMENT = "本工具只分析论文文本中呈现出的思想倾向，不判断作者本人真实政治立场。"
TOOL_NOTE = "本报告由人文社科论文思想谱系分析工具自动生成，仅供学术阅读辅助。"


class PDFExportError(RuntimeError):
    pass


def register_reportlab_chinese_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        ("MicrosoftYaHei", Path("C:/Windows/Fonts/msyh.ttc")),
        ("SimSun", Path("C:/Windows/Fonts/simsun.ttc")),
        ("NotoSansCJK", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")),
        ("NotoSansCJK", Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")),
    ]
    for font_name, font_path in font_candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                return font_name
            except Exception:
                continue

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def markdown_to_html(markdown_text: str) -> str:
    try:
        import markdown
    except Exception as exc:
        raise PDFExportError("缺少 markdown 依赖，无法执行 HTML 转换。") from exc

    return markdown.markdown(
        markdown_text or "",
        extensions=["extra", "tables", "fenced_code", "nl2br"],
        output_format="html5",
    )


def build_report_html(
    markdown_text: str,
    *,
    title: str | None = None,
    author: str | None = None,
    created_at: str | None = None,
) -> str:
    report_html = markdown_to_html(markdown_text)
    title_text = title or "论文思想谱系分析报告"
    author_text = author or "未提供"
    created_text = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <style>
        @page {{
          size: A4;
          margin: 20mm 18mm;
        }}
        body {{
          font-family: "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS", sans-serif;
          font-size: 11pt;
          color: #172033;
          line-height: 1.6;
        }}
        .cover {{
          border-bottom: 1px solid #d9e0eb;
          padding-bottom: 14px;
          margin-bottom: 20px;
        }}
        .report-title {{
          font-size: 22pt;
          font-weight: 800;
          margin-bottom: 12px;
        }}
        .meta {{
          color: #52617a;
          margin: 3px 0;
        }}
        .tool-note {{
          margin-top: 12px;
          padding: 10px 12px;
          border-radius: 8px;
          background: #f4f7fb;
          border-left: 4px solid #315fbd;
        }}
        h1, h2, h3 {{
          color: #101827;
          line-height: 1.35;
          margin-top: 20px;
          margin-bottom: 10px;
        }}
        h1 {{ font-size: 19pt; }}
        h2 {{ font-size: 15pt; border-bottom: 1px solid #e5eaf2; padding-bottom: 4px; }}
        h3 {{ font-size: 13pt; }}
        p {{ margin: 7px 0; }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin: 12px 0;
        }}
        th, td {{
          border: 1px solid #cfd8e6;
          padding: 6px 8px;
          vertical-align: top;
        }}
        th {{ background: #f2f5fa; }}
        pre, code {{
          font-family: "Consolas", "Courier New", monospace;
          background: #f5f6f8;
        }}
        pre {{
          padding: 10px;
          border-radius: 6px;
          overflow-wrap: break-word;
          white-space: pre-wrap;
        }}
        blockquote {{
          margin: 10px 0;
          padding: 9px 12px;
          background: #f7f9fc;
          border-left: 4px solid #8aa4d6;
          color: #34445d;
        }}
        .boundary {{
          margin-top: 18px;
          padding: 10px 12px;
          background: #fff8e6;
          border-left: 4px solid #d8a642;
        }}
      </style>
    </head>
    <body>
      <section class="cover">
        <div class="report-title">人文社科论文思想谱系分析报告</div>
        <div class="meta"><strong>论文标题：</strong>{html.escape(title_text)}</div>
        <div class="meta"><strong>作者：</strong>{html.escape(author_text)}</div>
        <div class="meta"><strong>生成时间：</strong>{html.escape(str(created_text))}</div>
        <div class="tool-note">{html.escape(TOOL_NOTE)}</div>
      </section>
      <main>{report_html}</main>
      <section class="boundary"><strong>边界声明：</strong>{html.escape(BOUNDARY_STATEMENT)}</section>
    </body>
    </html>
    """


def html_to_pdf_with_weasyprint(html_text: str) -> bytes:
    try:
        from weasyprint import HTML
    except Exception as exc:
        raise PDFExportError("WeasyPrint 不可用。") from exc

    try:
        return HTML(string=html_text).write_pdf()
    except Exception as exc:
        raise PDFExportError(f"WeasyPrint 生成 PDF 失败：{exc}") from exc


def strip_markdown(text: str) -> str:
    text = re.sub(r"`{1,3}", "", text or "")
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.M)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text.strip()


def markdown_to_pdf_with_reportlab(
    markdown_text: str,
    *,
    title: str | None = None,
    author: str | None = None,
    created_at: str | None = None,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as exc:
        raise PDFExportError("ReportLab 不可用。") from exc

    try:
        font_name = register_reportlab_chinese_font()
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title or "论文思想谱系分析报告",
        )
        styles = getSampleStyleSheet()
        base = ParagraphStyle(
            "ChineseBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=18,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
        h1 = ParagraphStyle("ChineseH1", parent=base, fontSize=18, leading=24, spaceBefore=12, spaceAfter=10)
        h2 = ParagraphStyle("ChineseH2", parent=base, fontSize=14, leading=20, spaceBefore=10, spaceAfter=8)
        meta = ParagraphStyle("ChineseMeta", parent=base, textColor=colors.HexColor("#52617a"))
        note = ParagraphStyle(
            "ChineseNote",
            parent=base,
            backColor=colors.HexColor("#f4f7fb"),
            borderColor=colors.HexColor("#d9e0eb"),
            borderWidth=0.6,
            borderPadding=8,
        )
        story: list[Any] = [
            Paragraph("人文社科论文思想谱系分析报告", h1),
            Paragraph(f"<b>论文标题：</b>{html.escape(title or '论文思想谱系分析报告')}", meta),
            Paragraph(f"<b>作者：</b>{html.escape(author or '未提供')}", meta),
            Paragraph(f"<b>生成时间：</b>{html.escape(created_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", meta),
            Spacer(1, 8),
            Paragraph(html.escape(TOOL_NOTE), note),
            Spacer(1, 10),
        ]

        for raw_line in (markdown_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                story.append(Spacer(1, 5))
                continue
            clean = html.escape(strip_markdown(line))
            if raw_line.startswith("# "):
                story.append(Paragraph(clean, h1))
            elif raw_line.startswith("## "):
                story.append(Paragraph(clean, h2))
            elif raw_line.startswith(("### ", "#### ")):
                story.append(Paragraph(f"<b>{clean}</b>", base))
            elif raw_line.lstrip().startswith(">"):
                story.append(Paragraph(clean.lstrip("&gt;").strip(), note))
            else:
                story.append(Paragraph(clean, base))

        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>边界声明：</b>{html.escape(BOUNDARY_STATEMENT)}", note))
        doc.build(story)
        return buffer.getvalue()
    except Exception as exc:
        raise PDFExportError(f"ReportLab 生成 PDF 失败：{exc}") from exc


def markdown_to_pdf_bytes(
    markdown_text: str,
    title: str | None = None,
    author: str | None = None,
    created_at: str | None = None,
) -> bytes:
    try:
        html_text = build_report_html(
            markdown_text,
            title=title,
            author=author,
            created_at=created_at,
        )
        return html_to_pdf_with_weasyprint(html_text)
    except Exception as first_error:
        print(f"PDF HTML/WeasyPrint 生成失败，尝试 ReportLab fallback：{first_error}", flush=True)

    return markdown_to_pdf_with_reportlab(
        markdown_text,
        title=title,
        author=author,
        created_at=created_at,
    )

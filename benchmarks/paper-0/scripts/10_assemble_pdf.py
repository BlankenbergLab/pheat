from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Flowable
    from reportlab.platypus import Image as ReportLabImage
    from reportlab.platypus import PageBreak
    from reportlab.platypus import Paragraph
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.platypus import Spacer
    from reportlab.platypus import Table
    from reportlab.platypus import TableStyle
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise SystemExit(
        "reportlab is required to assemble the paper-0 PDF. Install it into the "
        "repo-local environment with `mamba install -p .conda/free-energy-calc -c conda-forge reportlab`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
BODY_WIDTH = LETTER[0] - 1.5 * inch
CODE_RE = re.compile(r"`([^`]+)`")


class DemoFigure(Flowable):
    def __init__(self, summary: dict[str, object], width: float = BODY_WIDTH, height: float = 118) -> None:
        super().__init__()
        self.width = width
        self.height = height
        groups = summary.get("groups", {})
        if isinstance(groups, dict):
            self.row_count = sum(int(group.get("row_count", 0)) for group in groups.values() if isinstance(group, dict))
            self.optional_skip_count = sum(
                int(group.get("optional_skip_count", 0)) for group in groups.values() if isinstance(group, dict)
            )
            self.failure_count = sum(
                int(group.get("failure_count", 0)) for group in groups.values() if isinstance(group, dict)
            )
        else:
            self.row_count = 0
            self.optional_skip_count = 0
            self.failure_count = 0

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        return min(self.width, available_width), self.height

    def draw(self) -> None:
        canvas = self.canv
        width = self.width
        canvas.setStrokeColor(colors.HexColor("#4c78a8"))
        canvas.setFillColor(colors.white)
        canvas.roundRect(0, 0, width, self.height, 4, stroke=1, fill=1)
        canvas.setFillColor(colors.HexColor("#222222"))
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(14, self.height - 24, "PHEAT paper-0 demo workflow")
        canvas.setFont("Helvetica", 9)
        canvas.drawString(14, self.height - 44, f"Demo rows: {self.row_count}")
        canvas.drawString(14, self.height - 60, f"Optional/configuration status rows: {self.optional_skip_count}")
        canvas.drawString(14, self.height - 76, f"Real demo failures: {self.failure_count}")
        canvas.drawString(14, self.height - 92, "Workflow plumbing only; not a scientific result.")

        max_bar_width = width - 28
        bar_width = min(max_bar_width, 20 + self.row_count * 12)
        canvas.setFillColor(colors.HexColor("#4c78a8"))
        canvas.rect(14, 18, bar_width, 14, stroke=0, fill=1)


def _inline_text(text: str) -> str:
    parts: list[str] = []
    start = 0
    for match in CODE_RE.finditer(text):
        parts.append(escape(text[start : match.start()]))
        parts.append(f'<font name="Courier">{escape(match.group(1))}</font>')
        start = match.end()
    parts.append(escape(text[start:]))
    return "".join(parts)


def _parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _make_table(rows: list[list[str]], style) -> Table:
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    data = [[Paragraph(_inline_text(cell), style) for cell in row] for row in normalized]
    col_widths = [BODY_WIDTH / column_count] * column_count
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_flowables(path: Path, styles) -> list[object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    flowables: list[object] = []
    body_style = styles["BodyText"]
    bullet_style = styles["Bullet"]
    table_style = styles["TableCell"]
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith("### "):
            flowables.append(Paragraph(_inline_text(stripped[4:]), styles["Heading3"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flowables.append(Paragraph(_inline_text(stripped[3:]), styles["Heading2"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flowables.append(Paragraph(_inline_text(stripped[2:]), styles["Heading1"]))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            rows: list[list[str]] = []
            while i < len(lines):
                table_line = lines[i].strip()
                if not (table_line.startswith("|") and table_line.endswith("|")):
                    break
                cells = _parse_table_row(table_line)
                if not _is_separator_row(cells):
                    rows.append(cells)
                i += 1
            if rows:
                flowables.append(_make_table(rows, table_style))
                flowables.append(Spacer(1, 10))
            continue

        if stripped.startswith("- "):
            while i < len(lines):
                line = lines[i]
                line_stripped = line.strip()
                if line_stripped.startswith("- "):
                    item = line_stripped[2:]
                    i += 1
                    while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                        item += " " + lines[i].strip()
                        i += 1
                    flowables.append(Paragraph("- " + _inline_text(item), bullet_style))
                    continue
                break
            flowables.append(Spacer(1, 6))
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            next_stripped = next_line.strip()
            if (
                not next_stripped
                or next_stripped.startswith("#")
                or next_stripped.startswith("- ")
                or (next_stripped.startswith("|") and next_stripped.endswith("|"))
            ):
                break
            paragraph_lines.append(next_stripped)
            i += 1
        flowables.append(Paragraph(_inline_text(" ".join(paragraph_lines)), body_style))
        flowables.append(Spacer(1, 6))

    return flowables


def _csv_table(path: Path, styles) -> Table:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return _make_table(rows, styles["TableCell"])


def _image_flowable(path: Path, summary: dict[str, object]):
    if not path.exists():
        return DemoFigure(summary)
    image = ReportLabImage(str(path))
    scale = min(1.0, BODY_WIDTH / image.imageWidth)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _source_note(path: Path, styles) -> Paragraph:
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    return Paragraph(f"Source: {_inline_text(str(display_path))}", styles["Caption"])


def _page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, 0.42 * inch, "paper-0 draft")
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _build_styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontSize = 24
    styles["Title"].leading = 30
    styles["Heading1"].spaceBefore = 12
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].spaceBefore = 10
    styles["Heading2"].spaceAfter = 6
    styles["BodyText"].leading = 12
    styles["Bullet"].leftIndent = 14
    styles["Bullet"].firstLineIndent = -8
    styles.add(styles["BodyText"].clone("TableCell", fontSize=7, leading=8))
    styles.add(styles["BodyText"].clone("Caption", fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    styles.add(styles["BodyText"].clone("Disclaimer", fontSize=10, leading=13, textColor=colors.HexColor("#333333")))
    return styles


def assemble_pdf(
    *,
    output: Path,
    summary_path: Path,
    table_path: Path,
    figure_path: Path,
    doc_paths: list[Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    story: list[object] = [
        Paragraph("paper-0 draft", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "PHEAT: an open toolkit for reproducible protein heavy-atom reference corpora, "
            "geometry reconstruction, and molecular-modelling benchmarks",
            styles["Heading2"],
        ),
        Spacer(1, 16),
        Paragraph(f"Generated: {generated_at}", styles["BodyText"]),
        Spacer(1, 16),
        Paragraph(
            "Draft assembly for internal review. Demo outputs in this PDF are workflow checks only "
            "and are not scientific benchmark evidence.",
            styles["Disclaimer"],
        ),
        PageBreak(),
    ]

    for doc_path in doc_paths:
        story.extend(_markdown_flowables(doc_path, styles))
        story.append(PageBreak())

    story.extend(
        [
            Paragraph("Demo Workflow Artifacts", styles["Heading1"]),
            Paragraph(_inline_text(str(summary.get("note", "Demo summary only."))), styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Demo Summary Table", styles["Heading2"]),
            _source_note(table_path, styles),
            Spacer(1, 6),
            _csv_table(table_path, styles),
            Spacer(1, 14),
        ]
    )
    comparison_table_path = table_path.parent / "reconstruction-comparison.csv"
    if comparison_table_path.exists():
        story.extend(
            [
                Paragraph("Reconstruction Comparison Table", styles["Heading2"]),
                _source_note(comparison_table_path, styles),
                Spacer(1, 6),
                _csv_table(comparison_table_path, styles),
                Spacer(1, 14),
            ]
        )
    story.extend(
        [
            Paragraph("Demo Workflow Figure", styles["Heading2"]),
            _source_note(figure_path, styles),
            Spacer(1, 6),
            _image_flowable(figure_path, summary),
        ]
    )

    doc = SimpleDocTemplate(
        str(output),
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="paper-0 draft",
        author="PHEAT contributors",
    )
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the paper-0 draft PDF from local docs and demo artifacts.")
    parser.add_argument("--summary", default="results/demo/summary.json")
    parser.add_argument("--table", default="tables/demo/demo-summary.csv")
    parser.add_argument("--figure", default="figures/demo/demo-workflow-summary.png")
    parser.add_argument("--output", default="paper/paper-0-draft.pdf")
    args = parser.parse_args(argv)

    doc_paths = [
        REPO_ROOT / "docs" / "paper" / "paper-0-outline.md",
        REPO_ROOT / "docs" / "paper" / "paper-0-figures-and-tables.md",
        REPO_ROOT / "docs" / "paper" / "paper-0-claims-and-evidence.md",
        REPO_ROOT / "docs" / "paper" / "paper-0-related-work-matrix.md",
        REPO_ROOT / "docs" / "paper-0-submission-readiness.md",
    ]
    assemble_pdf(
        output=Path(args.output),
        summary_path=Path(args.summary),
        table_path=Path(args.table),
        figure_path=Path(args.figure),
        doc_paths=doc_paths,
    )
    print(json.dumps({"ok": True, "output": str(Path(args.output))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

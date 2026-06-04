from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create tiny demo SVG and PNG figures from summary JSON.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    row_count = sum(group.get("row_count", 0) for group in summary.get("groups", {}).values())
    optional_skip_count = sum(group.get("optional_skip_count", 0) for group in summary.get("groups", {}).values())
    failure_count = sum(group.get("failure_count", 0) for group in summary.get("groups", {}).values())
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="520" height="200" viewBox="0 0 520 200">
  <rect width="520" height="200" fill="#ffffff"/>
  <text x="24" y="36" font-family="Arial" font-size="18">PHEAT paper-0 demo workflow</text>
  <text x="24" y="70" font-family="Arial" font-size="13">Demo rows: {row_count}</text>
  <text x="24" y="94" font-family="Arial" font-size="13">Optional/configuration status rows: {optional_skip_count}</text>
  <text x="24" y="118" font-family="Arial" font-size="13">Real demo failures: {failure_count}</text>
  <text x="24" y="146" font-family="Arial" font-size="13">This is workflow plumbing, not a scientific result.</text>
  <rect x="24" y="164" width="{min(460, 20 + row_count * 12)}" height="18" fill="#4c78a8"/>
</svg>
"""
    (output_root / "demo-workflow-summary.svg").write_text(svg, encoding="utf-8")
    _write_png(output_root / "demo-workflow-summary.png", row_count, optional_skip_count, failure_count)
    return 0


def _write_png(path: Path, row_count: int, optional_skip_count: int, failure_count: int) -> None:
    image = Image.new("RGB", (1040, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1039, 359), outline="#4c78a8", width=2)
    draw.text((48, 42), "PHEAT paper-0 demo workflow", fill="#222222")
    draw.text((48, 112), f"Demo rows: {row_count}", fill="#222222")
    draw.text((48, 158), f"Optional/configuration status rows: {optional_skip_count}", fill="#222222")
    draw.text((48, 204), f"Real demo failures: {failure_count}", fill="#222222")
    draw.text((48, 268), "Workflow plumbing only; not a scientific result.", fill="#222222")
    draw.rectangle((48, 314, min(968, 48 + 40 + row_count * 24), 338), fill="#4c78a8")
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())

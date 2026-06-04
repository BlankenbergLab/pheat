from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _common import read_jsonl


RECONSTRUCTION_COMPARISON_COLUMNS = [
    "task",
    "record",
    "pheat",
    "pulchra",
    "pheat_ca_orig",
    "pulchra_ca_orig",
    "pheat_pulchra_ca",
    "pheat_bb_orig",
    "pulchra_bb_orig",
    "pheat_pulchra_bb",
    "pheat_heavy_orig",
    "pulchra_heavy_orig",
    "pheat_pulchra_heavy",
    "pheat_side_orig",
    "pulchra_side_orig",
    "pheat_pulchra_side",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create tiny demo CSV tables from summary JSON.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "demo-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "row_count",
                "ok_count",
                "optional_skip_count",
                "failure_count",
                "failure_rate",
                "runtime_seconds_mean",
            ],
        )
        writer.writeheader()
        for group, payload in sorted(summary.get("groups", {}).items()):
            writer.writerow(
                {
                    "group": group,
                    "row_count": payload.get("row_count"),
                    "ok_count": payload.get("ok_count"),
                    "optional_skip_count": payload.get("optional_skip_count"),
                    "failure_count": payload.get("failure_count"),
                    "failure_rate": payload.get("failure_rate"),
                    "runtime_seconds_mean": payload.get("runtime_seconds_mean"),
                }
            )
    _write_reconstruction_comparison_table(summary, output_root / "reconstruction-comparison.csv")
    return 0


def _write_reconstruction_comparison_table(summary: dict, output: Path) -> None:
    rows: list[dict] = []
    for input_path in summary.get("inputs", []):
        if Path(str(input_path)).name != "reconstruction-results.jsonl":
            continue
        for row in read_jsonl(input_path):
            if row.get("benchmark") != "reconstruction_comparison":
                continue
            rows.append(
                {
                    "task": row.get("comparison_task"),
                    "record": row.get("record_id"),
                    "pheat": row.get("pheat_status"),
                    "pulchra": row.get("pulchra_status"),
                    "pheat_ca_orig": _rounded(row.get("pheat_vs_original_ca_rmsd")),
                    "pulchra_ca_orig": _rounded(row.get("pulchra_vs_original_ca_rmsd")),
                    "pheat_pulchra_ca": _rounded(row.get("pheat_vs_pulchra_ca_rmsd")),
                    "pheat_bb_orig": _rounded(row.get("pheat_vs_original_backbone_rmsd")),
                    "pulchra_bb_orig": _rounded(row.get("pulchra_vs_original_backbone_rmsd")),
                    "pheat_pulchra_bb": _rounded(row.get("pheat_vs_pulchra_backbone_rmsd")),
                    "pheat_heavy_orig": _rounded(row.get("pheat_vs_original_all_heavy_rmsd")),
                    "pulchra_heavy_orig": _rounded(row.get("pulchra_vs_original_all_heavy_rmsd")),
                    "pheat_pulchra_heavy": _rounded(row.get("pheat_vs_pulchra_all_heavy_rmsd")),
                    "pheat_side_orig": _rounded(row.get("pheat_vs_original_sidechain_heavy_rmsd")),
                    "pulchra_side_orig": _rounded(row.get("pulchra_vs_original_sidechain_heavy_rmsd")),
                    "pheat_pulchra_side": _rounded(row.get("pheat_vs_pulchra_sidechain_heavy_rmsd")),
                }
            )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECONSTRUCTION_COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rounded(value: object) -> object:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value


if __name__ == "__main__":
    raise SystemExit(main())

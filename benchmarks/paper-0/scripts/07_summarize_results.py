from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from _common import read_jsonl


OPTIONAL_STATUSES = {"not_installed", "installed_not_run", "unsupported_input"}
ERROR_STATUSES = {"error", "unknown_backend", "unknown_comparator", "unsupported_format"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize demo benchmark JSONL outputs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    groups = {}
    for input_path in args.inputs:
        rows = read_jsonl(input_path)
        groups[Path(input_path).stem] = _summarize_rows(rows)
    payload = {
        "format": "pheat.paper-0-demo-summary",
        "version": 1,
        "note": "Demo summary only; not a scientific benchmark result.",
        "inputs": args.inputs,
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _summarize_rows(rows: list[dict]) -> dict:
    runtimes = [float(row.get("runtime_seconds") or 0.0) for row in rows]
    failures = [row for row in rows if _is_failure(row)]
    optional_skips = [row for row in rows if _row_status(row) in OPTIONAL_STATUSES]
    statuses = {}
    for row in rows:
        status = _row_status(row)
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "row_count": len(rows),
        "ok_count": len(rows) - len(failures) - len(optional_skips),
        "optional_skip_count": len(optional_skips),
        "failure_count": len(failures),
        "failure_rate": (len(failures) / len(rows)) if rows else 0.0,
        "runtime_seconds_mean": mean(runtimes) if runtimes else 0.0,
        "statuses": statuses,
    }


def _row_status(row: dict) -> str:
    return str(row.get("comparator_status") or row.get("backend_status") or row.get("score_origin") or "unknown")


def _is_failure(row: dict) -> bool:
    status = _row_status(row)
    if status in OPTIONAL_STATUSES:
        return False
    if status in ERROR_STATUSES:
        return True
    return bool(row.get("failure_reason"))


if __name__ == "__main__":
    raise SystemExit(main())

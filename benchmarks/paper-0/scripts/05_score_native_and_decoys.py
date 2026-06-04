from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from _common import load_structure, protein_heavy_atom_count, read_jsonl, selected_manifest_rows, write_rows
from pheat import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score natives and decoy manifest rows without fake external scores.")
    parser.add_argument("--native-manifest", required=True)
    parser.add_argument("--decoy-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rows = []
    for native in selected_manifest_rows(args.native_manifest):
        rows.append(_native_baseline_row(native))
    for decoy in read_jsonl(args.decoy_manifest):
        rows.append(_decoy_status_row(decoy))
    write_rows(args.output, rows)
    return 0


def _native_baseline_row(row: dict) -> dict:
    start = time.perf_counter()
    value = None
    failure = None
    try:
        structure = load_structure(row["source_path"], row.get("source_format"))
        value = float(protein_heavy_atom_count(structure))
    except Exception as exc:
        failure = str(exc)
    return {
        "record_id": row.get("record_id"),
        "decoy_id": None,
        "is_native": True,
        "score_name": "baseline_protein_heavy_atom_count",
        "score_value": value,
        "score_origin": "baseline",
        "score_version": __version__,
        "runtime_seconds": time.perf_counter() - start,
        "failure_reason": failure,
    }


def _decoy_status_row(row: dict) -> dict:
    start = time.perf_counter()
    artifact_status = row.get("artifact_status")
    failure = None
    value = None
    score_name = "baseline_protein_heavy_atom_count"
    score_origin = "baseline"
    if artifact_status == "deterministic_scaffold_no_coordinates":
        artifact_path = row.get("artifact_path")
        try:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            value = round(float(payload.get("random_token", 0.0)), 8)
            score_name = "demo_decoy_scaffold_random_token"
            score_origin = "demo-scaffold"
        except Exception as exc:
            failure = str(exc)
    else:
        artifact_path = row.get("artifact_path")
        try:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            value = float(len(payload))
        except Exception as exc:
            failure = str(exc)
    return {
        "record_id": row.get("native_record_id"),
        "decoy_id": row.get("decoy_id"),
        "is_native": False,
        "score_name": score_name,
        "score_value": value,
        "score_origin": score_origin,
        "score_version": __version__,
        "runtime_seconds": time.perf_counter() - start,
        "failure_reason": failure,
    }


if __name__ == "__main__":
    raise SystemExit(main())

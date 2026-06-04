from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from _common import selected_manifest_rows, write_rows
from pheat import __version__
from pheat.schemas import validate_json_object


SUPPORTED_RECIPES = {"pheat-torsion-v1", "pheat-coordinate-noise-v1"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic decoy manifest scaffolds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recipe", choices=sorted(SUPPORTED_RECIPES), required=True)
    parser.add_argument("--n-per-native", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(args.seed)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for row in selected_manifest_rows(args.manifest):
        for index in range(args.n_per_native):
            decoy_id = f"{row['record_id']}-{args.recipe}-{index + 1:03d}"
            artifact = output_root / "artifacts" / f"{decoy_id}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": "pheat.decoy-scaffold",
                "version": 1,
                "decoy_id": decoy_id,
                "native_record_id": row["record_id"],
                "recipe": args.recipe,
                "seed": args.seed,
                "random_token": rng.random(),
                "note": "Deterministic scaffold only; not a scientific structural decoy.",
            }
            artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_row = {
                "record_id": row["record_id"],
                "decoy_id": decoy_id,
                "native_record_id": row["record_id"],
                "recipe": args.recipe,
                "seed": args.seed,
                "artifact_path": str(artifact),
                "artifact_sha256": _file_sha256(artifact),
                "artifact_status": "deterministic_scaffold_no_coordinates",
                "failure_reason": None,
                "pheat_version": __version__,
                "created_at": created_at,
            }
            validate_json_object(manifest_row, "decoy-set-manifest")
            rows.append(manifest_row)
    output = output_root / "decoy-set-manifest.jsonl"
    write_rows(output, rows)
    return 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

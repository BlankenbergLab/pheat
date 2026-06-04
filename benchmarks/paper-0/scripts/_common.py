from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from pheat import __version__
from pheat.bcif import load_bcif
from pheat.domains import filter_structure_for_domain
from pheat.mmcif import load_mmcif
from pheat.models import HeavyAtomStructure
from pheat.pdbio import load_pdb
from pheat.residues import SUPPORTED_RESIDUES


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def selected_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if row.get("selected", True)]


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
        return
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def load_structure(path: str | Path, source_format: str | None = None) -> HeavyAtomStructure:
    source = Path(path)
    fmt = source_format or source_format_from_path(source)
    if fmt == "pdb":
        return load_pdb(source, hydrogens="drop")
    if fmt == "mmcif":
        return load_mmcif(source, hydrogens="drop")
    if fmt == "bcif":
        return load_bcif(source, hydrogens="drop")
    raise ValueError(f"Unsupported source format: {fmt}")


def source_format_from_path(path: str | Path) -> str:
    text = str(path).lower()
    if text.endswith(".pdb"):
        return "pdb"
    if text.endswith(".cif") or text.endswith(".mmcif"):
        return "mmcif"
    if text.endswith(".bcif") or text.endswith(".bcif.gz"):
        return "bcif"
    raise ValueError(f"Cannot infer source format from {path}")


def structure_counts(structure: HeavyAtomStructure) -> dict[str, Any]:
    filtered, _coverage = filter_structure_for_domain(structure, "protein-heavy")
    return {
        "atom_count": len(structure.atoms),
        "protein_heavy_atom_count": len(filtered.atoms),
        "residue_count": len(structure.residue_keys()),
        "chain_count": len({atom.chain_id or "" for atom in structure.atoms}),
    }


def protein_heavy_atom_count(structure: HeavyAtomStructure) -> int:
    return sum(
        1
        for atom in structure.atoms
        if atom.element.strip().upper() not in {"H", "D", "T"}
        and atom.resname.strip().upper() in SUPPORTED_RESIDUES
    )


def package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def runtime_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.exists():
        return str(candidate)
    return None


def base_result(row: Mapping[str, Any], *, comparator: str, benchmark: str) -> dict[str, Any]:
    return {
        "record_id": str(row.get("record_id") or ""),
        "pdb_id": row.get("pdb_id"),
        "chain_id": row.get("chain_id"),
        "benchmark": benchmark,
        "comparator": comparator,
        "comparator_status": "not_run",
        "runtime_seconds": 0.0,
        "peak_memory_mb": None,
        "source_path": row.get("source_path"),
        "pheat_version": __version__,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value

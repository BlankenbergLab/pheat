"""Schema-validated corpus specs and small local reference-corpus builds."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

from pheat import __version__
from pheat.bcif import load_bcif
from pheat.ccd import annotate_structure_components, load_component_table
from pheat.domains import filter_structure_for_domain
from pheat.mmcif import load_mmcif
from pheat.models import HeavyAtomStructure
from pheat.pdbio import load_pdb, write_structure_json
from pheat.residues import SUPPORTED_RESIDUES
from pheat.schemas import validate_json_object


SUPPORTED_CORPUS_SPEC_TEMPLATES = (
    "user-defined-ids",
    "xray-aqueous-30id",
    "xray-membrane-30id",
    "ligand-bound-ccd-demo",
)


def load_corpus_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a corpus specification from JSON or YAML."""

    spec = _read_structured_file(Path(path))
    validate_json_object(spec, "corpus-spec")
    return spec


def validate_corpus_spec(path: str | Path) -> dict[str, Any]:
    """Return a machine-readable validation result for a corpus spec."""

    spec_path = Path(path)
    spec = load_corpus_spec(spec_path)
    return {
        "ok": True,
        "schema": "corpus-spec",
        "path": str(spec_path),
        "corpus_id": spec["corpus_id"],
        "version": spec["version"],
        "source_type": spec.get("source", {}).get("type"),
        "source_format": spec.get("source", {}).get("format"),
        "atom_domain": spec.get("selection", {}).get("atom_domain"),
    }


def summarize_corpus_spec(path: str | Path) -> dict[str, Any]:
    """Return a compact, user-facing summary of a corpus spec."""

    spec_path = Path(path)
    spec = load_corpus_spec(spec_path)
    selection = dict(spec.get("selection") or {})
    source = dict(spec.get("source") or {})
    decoys = dict(spec.get("decoys") or {})
    return {
        "corpus": f"{spec['corpus_id']} {spec['version']}",
        "description": spec.get("description", ""),
        "source": {
            "type": source.get("type"),
            "format": source.get("format"),
            "source_root": source.get("source_root"),
            "ids_file": source.get("ids_file"),
            "snapshot_label": source.get("snapshot_label"),
        },
        "selection": {
            "atom_domain": selection.get("atom_domain"),
            "experimental_methods": selection.get("experimental_methods", []),
            "max_resolution_angstrom": selection.get("max_resolution_angstrom"),
            "length_range": [selection.get("min_length"), selection.get("max_length")],
            "sequence_identity": selection.get("sequence_identity"),
        },
        "decoys": {
            "enabled": bool(decoys.get("enabled")),
            "recipe": decoys.get("recipe"),
            "n_per_native": decoys.get("n_per_native"),
            "seed": decoys.get("seed"),
        },
        "path": str(spec_path),
    }


def init_corpus_spec(template: str, output: str | Path) -> dict[str, Any]:
    """Write a starter corpus spec template."""

    payload = corpus_spec_template(template)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_structured_file(output_path, payload)
    return {"ok": True, "template": template, "output": str(output_path)}


def corpus_spec_template(template: str) -> dict[str, Any]:
    """Return a supported corpus spec template payload."""

    normalized = template.strip().lower()
    if normalized == "user-defined-ids":
        return _base_template(
            corpus_id="pheat-user-defined-ids-demo",
            version="v1",
            description="Offline demo corpus built from local PHEAT fixture IDs; not a scientific benchmark.",
            source={
                "type": "id_list",
                "format": "pdb",
                "source_root": "../../tests/fixtures",
                "ids_file": "user_defined_ids.txt",
                "snapshot_label": "local-fixture-demo",
                "license_note": "Local repository fixtures for offline testing and demonstration.",
            },
        )
    if normalized == "xray-aqueous-30id":
        return _base_template(
            corpus_id="pheat-xray-aqueous-30id",
            version="v1",
            description="Archived-snapshot aqueous X-ray corpus template; dry-run locally unless a snapshot is supplied.",
            source={
                "type": "rcsb_snapshot",
                "format": "bcif",
                "snapshot_label": "TODO-rcsb-current-bcif",
                "license_note": "RCSB/wwPDB archive data with explicit snapshot provenance.",
            },
        )
    if normalized == "xray-membrane-30id":
        payload = corpus_spec_template("xray-aqueous-30id")
        payload["corpus_id"] = "pheat-xray-membrane-30id"
        payload["description"] = "Archived-snapshot membrane X-ray corpus template; dry-run locally unless a snapshot is supplied."
        payload["selection"]["membrane_status"] = "membrane"
        return payload
    if normalized == "ligand-bound-ccd-demo":
        payload = _base_template(
            corpus_id="pheat-ligand-bound-ccd-demo",
            version="v1",
            description="CCD-aware local fixture demo; not a ligand-affinity benchmark.",
            source={
                "type": "id_list",
                "format": "pdb",
                "source_root": "../../tests/fixtures",
                "ids_file": "user_defined_ids.txt",
                "snapshot_label": "local-fixture-demo",
                "license_note": "Local repository fixtures with optional user CCD table annotation.",
            },
        )
        payload["selection"]["include_ligands"] = ["HEM", "ATP"]
        payload["provenance"]["notes"] = [
            "Use --ccd with pheat reference build to add local CCD-like annotations.",
            "Demo metadata is intentionally small and not a full CCD mirror.",
        ]
        return payload
    raise ValueError(
        f"Unknown corpus spec template '{template}'. Known templates: "
        + ", ".join(SUPPORTED_CORPUS_SPEC_TEMPLATES)
    )


def build_reference_corpus(
    *,
    corpus_spec: str | Path,
    output_root: str | Path,
    ccd: Optional[str | Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a tiny local/user-defined reference corpus from a validated spec."""

    spec_path = Path(corpus_spec)
    spec = load_corpus_spec(spec_path)
    root = Path(output_root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"output root already contains files; pass --overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)

    now = _utc_now()
    source = dict(spec.get("source") or {})
    source_type = str(source.get("type") or "")
    spec_sha256 = _file_sha256(spec_path)
    metadata = {
        "format": "pheat.reference-build-metadata",
        "version": 1,
        "corpus_id": spec["corpus_id"],
        "corpus_version": spec["version"],
        "created_at": now,
        "pheat_version": __version__,
        "python_version": sys.version.split()[0],
        "corpus_spec": str(spec_path),
        "corpus_spec_sha256": spec_sha256,
        "command": list(argv),
        "dry_run": bool(dry_run),
        "ccd": str(ccd) if ccd else None,
    }

    if dry_run or source_type == "rcsb_snapshot":
        plan = {
            "ok": True,
            "dry_run": True,
            "message": "Validated corpus spec and planned outputs without downloading structures.",
            "corpus_id": spec["corpus_id"],
            "corpus_version": spec["version"],
            "source": source,
            "outputs": _output_paths(root),
        }
        _write_json(root / "plan.json", plan)
        _write_jsonl(root / "manifest.jsonl", [])
        _write_jsonl(root / "excluded-records.jsonl", [])
        _write_structure_summary(root / "structure-summary.csv", [])
        _write_json(root / "build-metadata.json", {**metadata, "selected_count": 0, "excluded_count": 0})
        checksums = _write_checksums(root)
        return {
            "ok": True,
            "dry_run": True,
            "output_root": str(root),
            "plan": str(root / "plan.json"),
            "manifest": str(root / "manifest.jsonl"),
            "checksums": str(root / "checksums.sha256"),
            "checksums_sha256": checksums,
        }

    component_table = load_component_table(ccd)
    manifest_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    candidates = _candidate_sources(spec, spec_path.parent)
    for candidate in candidates:
        try:
            row = _build_manifest_row(
                candidate,
                spec=spec,
                spec_path=spec_path,
                output_root=root,
                component_table=component_table,
                created_at=now,
            )
            validate_json_object(row, "reference-dataset-manifest")
            manifest_rows.append(row)
        except Exception as exc:
            excluded_rows.append(
                {
                    "corpus_id": spec["corpus_id"],
                    "corpus_version": spec["version"],
                    "record_id": candidate.get("record_id"),
                    "pdb_id": candidate.get("pdb_id"),
                    "source_path": str(candidate.get("source_path") or ""),
                    "selected": False,
                    "exclusion_reason": str(exc),
                    "created_at": now,
                    "pheat_version": __version__,
                }
            )

    _write_jsonl(root / "manifest.jsonl", manifest_rows)
    _write_jsonl(root / "excluded-records.jsonl", excluded_rows)
    _write_structure_summary(root / "structure-summary.csv", manifest_rows)
    _write_json(
        root / "build-metadata.json",
        {
            **metadata,
            "selected_count": len(manifest_rows),
            "excluded_count": len(excluded_rows),
            "candidate_count": len(candidates),
        },
    )
    checksums = _write_checksums(root)
    return {
        "ok": True,
        "dry_run": False,
        "output_root": str(root),
        "manifest": str(root / "manifest.jsonl"),
        "excluded_records": str(root / "excluded-records.jsonl"),
        "structure_summary": str(root / "structure-summary.csv"),
        "build_metadata": str(root / "build-metadata.json"),
        "checksums": str(root / "checksums.sha256"),
        "checksums_sha256": checksums,
        "selected_count": len(manifest_rows),
        "excluded_count": len(excluded_rows),
    }


def _base_template(
    *,
    corpus_id: str,
    version: str,
    description: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "corpus_id": corpus_id,
        "version": version,
        "description": description,
        "source": dict(source),
        "selection": {
            "atom_domain": "protein-heavy",
            "experimental_methods": ["X-RAY DIFFRACTION"],
            "max_resolution_angstrom": 2.5,
            "min_length": 1,
            "max_length": 10000,
            "sequence_identity": {
                "method": "recorded-clusters-or-user-provided",
                "threshold": 0.30,
            },
            "require_complete_backbone": False,
            "allow_modified_residues": True,
            "allow_altlocs": True,
            "allow_insertion_codes": True,
        },
        "geometry": {
            "store_angles": ["omega", "tau", "theta"],
            "store_lengths": "protein-heavy",
            "store_chi": "all",
            "reconstruction_mode": "pheat-default",
        },
        "scoring": {
            "profiles": ["generic"],
            "burial_methods": ["contacts"],
        },
        "decoys": {
            "enabled": True,
            "recipe": "pheat-coordinate-noise-v1",
            "n_per_native": 2,
            "seed": 20260529,
        },
        "outputs": {
            "include_manifest": True,
            "include_features": False,
            "include_score_tables": False,
            "include_decoys": False,
            "compression": "none",
        },
        "provenance": {
            "pheat_version": __version__,
            "created_by": "PHEAT corpus spec template",
            "notes": "Template/demo only; larger corpora must be rebuilt with archived inputs.",
        },
    }


def _candidate_sources(spec: Mapping[str, Any], spec_dir: Path) -> list[dict[str, Any]]:
    source = dict(spec.get("source") or {})
    source_type = str(source.get("type") or "")
    source_root = _resolve_spec_path(source.get("source_root"), spec_dir)
    source_format = str(source.get("format") or "mixed")
    if source_root is None:
        raise ValueError("source.source_root is required for local corpus builds")
    if not source_root.exists():
        raise FileNotFoundError(f"source.source_root does not exist: {source_root}")

    ids_path = _resolve_spec_path(source.get("ids_file"), spec_dir)
    if source_type == "id_list":
        if ids_path is None:
            raise ValueError("source.ids_file is required when source.type is id_list")
        ids = _read_ids_file(ids_path)
        return [
            _candidate_from_id(record_id, source_root=source_root, source_format=source_format)
            for record_id in ids
        ]

    if source_type == "local_archive":
        return [
            _candidate_from_path(path, source_format=_source_format_from_path(path))
            for path in sorted(_scan_source_files(source_root, source_format))
        ]
    raise ValueError(f"source.type {source_type!r} is not supported for non-dry-run local builds")


def _candidate_from_id(record_id: str, *, source_root: Path, source_format: str) -> dict[str, Any]:
    source_path = _find_source_file(source_root, record_id, source_format)
    if source_path is None:
        raise FileNotFoundError(f"no source file for record {record_id!r} under {source_root}")
    candidate = _candidate_from_path(source_path, source_format=_source_format_from_path(source_path))
    candidate["record_id"] = record_id
    candidate["pdb_id"] = _pdb_id_from_record_id(record_id)
    return candidate


def _candidate_from_path(path: Path, *, source_format: str) -> dict[str, Any]:
    record_id = _record_id_from_path(path)
    return {
        "record_id": record_id,
        "pdb_id": _pdb_id_from_record_id(record_id),
        "source_path": path,
        "source_format": source_format,
    }


def _build_manifest_row(
    candidate: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    output_root: Path,
    component_table: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    source_path = Path(candidate["source_path"])
    structure = _load_structure_for_source(source_path, str(candidate["source_format"]))
    structure = _filter_structure_chains(structure, dict(spec.get("selection") or {}))
    domain = str((spec.get("selection") or {}).get("atom_domain") or "protein-heavy")
    normalized_domain = "protein-heavy" if domain in {"backbone", "ca-only"} else domain
    filtered, coverage = filter_structure_for_domain(structure, normalized_domain)
    if domain == "backbone":
        filtered = _filter_atom_names(filtered, {"N", "CA", "C", "O"})
    elif domain == "ca-only":
        filtered = _filter_atom_names(filtered, {"CA"})

    record_id = str(candidate["record_id"])
    artifact_path = output_root / "structures" / f"{record_id}.atom-structure.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_structure_json(filtered, artifact_path)
    source_sha256 = _file_sha256(source_path)
    artifact_sha256 = _file_sha256(artifact_path)
    component_summary = annotate_structure_components(
        structure,
        component_table=component_table,
    )
    chain_ids = sorted({atom.chain_id or "" for atom in filtered.atoms})
    model_ids = sorted({atom.model for atom in filtered.atoms if atom.model is not None})
    all_heavy_atom_count = sum(1 for atom in structure.atoms if atom.element.strip().upper() not in {"H", "D", "T"})
    protein_heavy_atom_count = sum(
        1
        for atom in structure.atoms
        if atom.element.strip().upper() not in {"H", "D", "T"}
        and atom.resname.strip().upper() in SUPPORTED_RESIDUES
    )
    relative_artifact = _relative_or_absolute(artifact_path, output_root)
    row = {
        "corpus_id": spec["corpus_id"],
        "corpus_version": spec["version"],
        "record_id": record_id,
        "pdb_id": str(candidate["pdb_id"]),
        "model_id": model_ids[0] if len(model_ids) == 1 else None,
        "chain_id": chain_ids[0] if len(chain_ids) == 1 else "all",
        "entity_id": None,
        "source_format": candidate["source_format"],
        "source_path": str(source_path),
        "source_url": None,
        "source_checksum_sha256": source_sha256,
        "corpus_spec": str(spec_path),
        "corpus_spec_sha256": _file_sha256(spec_path),
        "pheat_version": __version__,
        "parser_name": f"pheat-{candidate['source_format']}",
        "parser_version": __version__,
        "atom_domain": domain,
        "residue_count": len(filtered.residue_keys()),
        "atom_count": len(filtered.atoms),
        "protein_heavy_atom_count": protein_heavy_atom_count,
        "all_heavy_atom_count": all_heavy_atom_count,
        "experimental_method": _single_or_none((spec.get("selection") or {}).get("experimental_methods")),
        "resolution_angstrom": (spec.get("selection") or {}).get("max_resolution_angstrom"),
        "selected": True,
        "exclusion_reason": None,
        "domain_coverage": coverage,
        "generated_artifacts": {"atom_structure_json": relative_artifact},
        "generated_checksums": {"atom_structure_json": artifact_sha256},
        "created_at": created_at,
    }
    row.update(component_summary)
    return row


def _load_structure_for_source(path: Path, source_format: str) -> HeavyAtomStructure:
    if source_format == "pdb":
        return load_pdb(path, hydrogens="drop")
    if source_format == "mmcif":
        return load_mmcif(path, hydrogens="drop")
    if source_format == "bcif":
        return load_bcif(path, hydrogens="drop")
    raise ValueError(f"unsupported source format: {source_format}")


def _filter_structure_chains(
    structure: HeavyAtomStructure,
    selection: Mapping[str, Any],
) -> HeavyAtomStructure:
    include = {str(value) for value in selection.get("include_chains") or []}
    exclude = {str(value) for value in selection.get("exclude_chains") or []}
    if not include and not exclude:
        return structure
    atoms = [
        atom
        for atom in structure.atoms
        if (not include or atom.chain_id in include) and atom.chain_id not in exclude
    ]
    return HeavyAtomStructure(
        atoms=atoms,
        name=structure.name,
        metadata=dict(structure.metadata),
        disulfide_bonds=list(structure.disulfide_bonds),
        bonds=list(structure.bonds),
        atom_scope=structure.atom_scope,
    )


def _filter_atom_names(structure: HeavyAtomStructure, names: set[str]) -> HeavyAtomStructure:
    atoms = [atom for atom in structure.atoms if atom.name.strip().upper() in names]
    return HeavyAtomStructure(
        atoms=atoms,
        name=structure.name,
        metadata=dict(structure.metadata),
        disulfide_bonds=list(structure.disulfide_bonds),
        bonds=list(structure.bonds),
        atom_scope=structure.atom_scope,
    )


def _read_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - minimal install only
            raise RuntimeError(
                "YAML corpus specs require PyYAML. Use JSON or install pheat[all]."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported spec extension for {path}; use .json, .yml, or .yaml")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return dict(payload)


def _write_structured_file(path: Path, payload: Mapping[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        _write_json(path, payload)
        return
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - minimal install only
            raise RuntimeError("Writing YAML corpus specs requires PyYAML.") from exc
        path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
        return
    raise ValueError(f"Unsupported output extension for {path}; use .json, .yml, or .yaml")


def _resolve_spec_path(value: Any, spec_dir: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (spec_dir / path).resolve()


def _read_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(token)
    return ids


def _find_source_file(root: Path, record_id: str, source_format: str) -> Optional[Path]:
    for extension in _extensions_for_format(source_format):
        for stem in (record_id, record_id.lower(), record_id.upper()):
            candidate = root / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def _scan_source_files(root: Path, source_format: str) -> list[Path]:
    extensions = tuple(_extensions_for_format(source_format))
    return [
        path
        for path in root.iterdir()
        if path.is_file() and any(str(path).lower().endswith(ext) for ext in extensions)
    ]


def _extensions_for_format(source_format: str) -> list[str]:
    if source_format == "pdb":
        return [".pdb"]
    if source_format == "mmcif":
        return [".cif", ".mmcif"]
    if source_format == "bcif":
        return [".bcif", ".bcif.gz"]
    if source_format == "mixed":
        return [".pdb", ".cif", ".mmcif", ".bcif", ".bcif.gz"]
    raise ValueError(f"unsupported source format: {source_format}")


def _source_format_from_path(path: Path) -> str:
    lower = str(path).lower()
    if lower.endswith(".pdb"):
        return "pdb"
    if lower.endswith(".cif") or lower.endswith(".mmcif"):
        return "mmcif"
    if lower.endswith(".bcif") or lower.endswith(".bcif.gz"):
        return "bcif"
    raise ValueError(f"cannot infer source format from {path}")


def _record_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".bcif.gz", ".bcif", ".mmcif", ".cif", ".pdb"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _pdb_id_from_record_id(record_id: str) -> str:
    return record_id.upper() if len(record_id) == 4 and record_id.isalnum() else record_id


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_structure_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id",
        "pdb_id",
        "chain_id",
        "source_format",
        "selected",
        "residue_count",
        "atom_count",
        "protein_heavy_atom_count",
        "all_heavy_atom_count",
        "water_count",
        "ion_count",
        "unknown_component_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ",".join(row.get(field, []))
                    if isinstance(row.get(field), list)
                    else row.get(field)
                    for field in fields
                }
            )


def _write_checksums(root: Path) -> str:
    checksum_path = root / "checksums.sha256"
    generated = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in generated:
            handle.write(f"{_file_sha256(path)}  {_relative_or_absolute(path, root)}\n")
    return _file_sha256(checksum_path)


def _output_paths(root: Path) -> dict[str, str]:
    return {
        "manifest": str(root / "manifest.jsonl"),
        "excluded_records": str(root / "excluded-records.jsonl"),
        "structure_summary": str(root / "structure-summary.csv"),
        "build_metadata": str(root / "build-metadata.json"),
        "checksums": str(root / "checksums.sha256"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _single_or_none(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[0] if len(value) == 1 else None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

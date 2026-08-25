"""Training-corpus and score-table utilities for PHEAT.

This module implements the reproducible framework for building PHEAT-owned
statistical tables.  It intentionally does not bundle full-corpus outputs; those
can be generated later from a local archive snapshot and reviewed before being
committed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

from pheat import __version__
from pheat.archive import resolve_manifest_row_path
from pheat.domains import filter_structure_for_domain, normalize_domain
from pheat.geometry import centroid, distance
from pheat.mmcif import structure_from_mmcif_string
from pheat.models import Atom, HeavyAtomStructure, ResidueKey
from pheat.pdbio import load_heavy_json, structure_from_pdb_string
from pheat.residue_geometry import structure_to_residue_geometry
from pheat.residues import CANONICAL_RESIDUES, three_to_one
from pheat.sasa import SASA_BACKENDS, residue_burial
from pheat.score_contracts import score_input_contract
from pheat.score_tables import (
    SCORE_TABLE_SET_FORMAT,
    SCORE_TABLE_SET_VERSION,
    write_score_table_set,
)
from pheat.scoring import KYTE_DOOLITTLE_HYDROPATHY

Point3D = tuple[float, float, float]


DEFAULT_TRAINING_ROOT = Path(".pheat-cache/training")
DEFAULT_TRAINING_SNAPSHOT_ID = "rcsb-current-bcif"
DEFAULT_TRAINING_DOMAIN = "protein-heavy"
DEFAULT_ARTIFACT_VERSION = "v1"
DEFAULT_TRAINING_MODELS = (
    "pheat-dfire",
    "pheat-goap",
    "pheat-mj",
    "pheat-hydropathy",
    "pheat-backbone",
    "pheat-rotamer",
    "pheat-hbond",
    "pheat-rg",
)
SUCCESSFUL_ARCHIVE_STATUSES = ("downloaded", "present", "reused", "verified")
TRAINING_TABLE_FILENAME = "score-tables.json"
TABLE_BUILD_METHODS = ("contacts", "sasa", "both")


DECOY_DATASETS = (
    {
        "id": "3drobot",
        "name": "3DRobot and Zhang Lab decoy collections",
        "description": (
            "Reference entry for Zhang Lab decoy datasets. The public landing page "
            "contains multiple decoy downloads; PHEAT records provenance and leaves "
            "large payload selection to the user."
        ),
        "urls": ["https://seq2fun.dcmb.med.umich.edu/decoys/"],
        "license_note": "External dataset; do not vendor decoy payloads in PHEAT.",
    },
    {
        "id": "casp-prediction-center",
        "name": "CASP Prediction Center download area",
        "description": "Reference entry for CASP prediction/target download areas used for validation datasets.",
        "urls": ["https://predictioncenter.org/download_area/"],
        "license_note": "External dataset; users should follow CASP data-use terms for fetched files.",
    },
)


@dataclass(frozen=True)
class InventoryOptions:
    snapshot_root: Path
    output: Optional[Path] = None
    domain: str = DEFAULT_TRAINING_DOMAIN
    metadata_jsonl: Optional[Path] = None
    max_entries: Optional[int] = None
    workers: Union[str, int] = 1
    progress: Optional[Callable[[str], None]] = None


@dataclass(frozen=True)
class SelectionOptions:
    inventory: Path
    output_root: Path
    method: Optional[str] = None
    max_resolution: Optional[float] = None
    sequence_identity: Union[str, float] = 0.30
    canonical_only: bool = False
    min_length: int = 50
    max_length: int = 800
    sequence_clusters: Optional[Path] = None
    corpus_id: Optional[str] = None
    corpus_version: str = DEFAULT_ARTIFACT_VERSION
    include_file: Optional[Path] = None
    exclude_file: Optional[Path] = None
    holdout_file: Optional[Path] = None
    command_args: Optional[Sequence[str]] = None


def list_decoy_datasets() -> list[dict[str, Any]]:
    return [dict(dataset) for dataset in DECOY_DATASETS]


def get_decoy_dataset(dataset_id: str) -> dict[str, Any]:
    normalized = dataset_id.strip().lower()
    for dataset in DECOY_DATASETS:
        if dataset["id"] == normalized:
            return dict(dataset)
    known = ", ".join(str(dataset["id"]) for dataset in DECOY_DATASETS)
    raise ValueError(f"Unknown decoy dataset '{dataset_id}'. Known datasets: {known}")


def fetch_decoy_dataset(
    dataset_id: str,
    *,
    output_root: Path = DEFAULT_TRAINING_ROOT / "decoys",
    yes: bool = False,
) -> dict[str, Any]:
    """Record decoy dataset provenance without vendoring large external files."""

    dataset = get_decoy_dataset(dataset_id)
    destination = Path(output_root) / dataset["id"]
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "pheat.decoy-dataset",
        "version": 1,
        "dataset": dataset,
        "downloaded_files": [],
        "status": "metadata-only",
        "fetched_at": _utc_now(),
        "yes": bool(yes),
        "notes": [
            "PHEAT does not vendor external decoy payloads.",
            "Use the recorded source URLs to select and download the desired benchmark payloads.",
        ],
    }
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset_id": dataset["id"], "output_root": str(destination), "manifest": str(manifest), **payload}


def verify_decoy_root(input_root: Path = DEFAULT_TRAINING_ROOT / "decoys") -> dict[str, Any]:
    root = Path(input_root)
    manifests = sorted(root.glob("*/manifest.json")) if root.exists() else []
    datasets = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        downloaded_files = payload.get("downloaded_files", [])
        missing = []
        if isinstance(downloaded_files, list):
            for item in downloaded_files:
                if isinstance(item, Mapping) and item.get("path"):
                    path = Path(str(item["path"]))
                    if not path.exists():
                        missing.append(str(path))
        datasets.append(
            {
                "dataset_id": payload.get("dataset", {}).get("id"),
                "manifest": str(manifest),
                "status": "missing-files" if missing else payload.get("status", "unknown"),
                "missing_files": missing,
            }
        )
    return {
        "input_root": str(root),
        "dataset_count": len(datasets),
        "ok": all(not item["missing_files"] for item in datasets),
        "datasets": datasets,
    }


def snapshot_ids_from_manifest(
    output_root: Path,
    *,
    manifest_dir: Optional[Path] = None,
    statuses: Sequence[str] = SUCCESSFUL_ARCHIVE_STATUSES,
) -> list[str]:
    rows = _read_archive_manifest_rows(output_root, manifest_dir=manifest_dir)
    allowed = {status.strip().lower() for status in statuses}
    ids = [
        str(row.get("pdb_id")).strip().upper()
        for row in rows
        if str(row.get("status", "")).strip().lower() in allowed and row.get("pdb_id")
    ]
    return sorted(set(ids))


def write_snapshot_ids(
    output_root: Path,
    output: Optional[Path],
    *,
    manifest_dir: Optional[Path] = None,
    statuses: Sequence[str] = SUCCESSFUL_ARCHIVE_STATUSES,
) -> list[str]:
    ids = snapshot_ids_from_manifest(output_root, manifest_dir=manifest_dir, statuses=statuses)
    text = "".join(f"{pdb_id}\n" for pdb_id in ids)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
    return ids


def inventory_snapshot(options: InventoryOptions) -> list[dict[str, Any]]:
    domain = normalize_domain(options.domain)
    metadata = _metadata_by_pdb_id(options.metadata_jsonl)
    snapshot_manifest = Path(options.snapshot_root) / "manifests" / "files.jsonl"
    snapshot_manifest_sha256 = _file_sha256(snapshot_manifest) if snapshot_manifest.exists() else None
    rows = _read_archive_manifest_rows(options.snapshot_root)
    candidate_rows = [
        dict(row)
        for row in rows
        if str(row.get("status", "")).lower() in SUCCESSFUL_ARCHIVE_STATUSES
    ]
    if options.max_entries is not None:
        candidate_rows = candidate_rows[: options.max_entries]
    tasks = [
        (
            row,
            domain,
            metadata.get(str(row.get("pdb_id") or "").upper(), {}),
            str(options.snapshot_root),
            snapshot_manifest_sha256,
        )
        for row in candidate_rows
    ]
    inventory = [
        item
        for chunk in _map_ordered(
            _inventory_snapshot_worker,
            tasks,
            workers=options.workers,
            progress=options.progress,
            progress_label="inventory",
        )
        for item in chunk
    ]
    if options.output is not None:
        _write_jsonl(options.output, inventory)
    return inventory


def select_corpus(options: SelectionOptions) -> dict[str, Any]:
    sequence_identity_threshold = normalize_sequence_identity_threshold(options.sequence_identity)
    sequence_identity_label = sequence_identity_threshold_label(sequence_identity_threshold)
    include_members = _read_member_set(options.include_file)
    exclude_members = _read_member_set(options.exclude_file)
    holdout_members = _read_member_set(options.holdout_file)
    rows = _read_jsonl(options.inventory)
    candidates = [_normalize_inventory_candidate(row) for row in rows]
    filtered_before_member_lists = [
        row
        for row in candidates
        if _candidate_passes_filters(
            row,
            method=options.method,
            max_resolution=options.max_resolution,
            canonical_only=options.canonical_only,
            min_length=options.min_length,
            max_length=options.max_length,
        )
    ]
    filtered_after_include = [
        row for row in filtered_before_member_lists if not include_members or _candidate_matches_members(row, include_members)
    ]
    filtered_after_exclude = [
        row for row in filtered_after_include if not _candidate_matches_members(row, exclude_members)
    ]
    holdout = [row for row in filtered_after_exclude if _candidate_matches_members(row, holdout_members)]
    filtered = [row for row in filtered_after_exclude if not _candidate_matches_members(row, holdout_members)]
    ranked = sorted(filtered, key=_selection_rank_key)
    cluster_map = _read_sequence_cluster_map(options.sequence_clusters)
    if cluster_map:
        clusters_by_id: dict[str, list[dict[str, Any]]] = {}
        for candidate in ranked:
            cluster_id = _candidate_cluster_id(candidate, cluster_map)
            clusters_by_id.setdefault(cluster_id, []).append(candidate)
        clusters = [clusters_by_id[key] for key in sorted(clusters_by_id)]
        selected = [cluster[0] for cluster in clusters]
        cluster_source = str(options.sequence_clusters)
    elif any(_candidate_metadata_cluster_id(candidate, sequence_identity_threshold) for candidate in ranked):
        clusters_by_id = {}
        for candidate in ranked:
            metadata_cluster_id = _candidate_metadata_cluster_id(candidate, sequence_identity_threshold)
            if metadata_cluster_id is None:
                metadata_cluster_id = f"singleton:{candidate.get('pdb_id')}:{candidate.get('chain_id')}"
            clusters_by_id.setdefault(metadata_cluster_id, []).append(candidate)
        clusters = [clusters_by_id[key] for key in sorted(clusters_by_id)]
        selected = [cluster[0] for cluster in clusters]
        cluster_source = f"metadata-sequence-identity-{sequence_identity_label}"
    else:
        clusters = []
        selected = []
        for candidate in ranked:
            assigned = False
            for cluster in clusters:
                if sequence_identity(candidate["sequence"], cluster[0]["sequence"]) >= sequence_identity_threshold:
                    cluster.append(candidate)
                    assigned = True
                    break
            if not assigned:
                clusters.append([candidate])
                selected.append(candidate)
        cluster_source = "internal-sequence-identity"

    output_root = Path(options.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = _split_selected(selected)
    corpus_id = options.corpus_id or _default_corpus_id(candidates, sequence_identity_label)
    corpus_version = options.corpus_version or DEFAULT_ARTIFACT_VERSION
    artifact_label = artifact_label_for(corpus_id, corpus_version)
    selected_entry_checksum = _entries_sha256(selected)
    holdout_entry_checksum = _entries_sha256(holdout)
    member_files = {
        "include": _member_file_provenance(options.include_file, include_members),
        "exclude": _member_file_provenance(options.exclude_file, exclude_members),
        "holdout": _member_file_provenance(options.holdout_file, holdout_members),
    }
    source_snapshot = _source_snapshot_summary(candidates)
    selected_json = {
        "format": "pheat.training-corpus",
        "version": 1,
        "artifact_id": corpus_id,
        "artifact_version": corpus_version,
        "artifact_label": artifact_label,
        "created_at": _utc_now(),
        "filters": {
            "method": options.method,
            "max_resolution": options.max_resolution,
            "sequence_identity": sequence_identity_threshold,
            "sequence_identity_label": sequence_identity_label,
            "canonical_only": options.canonical_only,
            "min_length": options.min_length,
            "max_length": options.max_length,
            "sequence_clusters": str(options.sequence_clusters) if options.sequence_clusters else None,
            "cluster_source": cluster_source,
            "include_file": str(options.include_file) if options.include_file else None,
            "exclude_file": str(options.exclude_file) if options.exclude_file else None,
            "holdout_file": str(options.holdout_file) if options.holdout_file else None,
        },
        "provenance": {
            "pheat_version": __version__,
            "command_args": list(options.command_args) if options.command_args is not None else None,
            "input_inventory": str(options.inventory),
            "input_inventory_sha256": _file_sha256(options.inventory),
            "source_snapshot": source_snapshot,
            "member_files": member_files,
            "selected_entry_sha256": selected_entry_checksum,
            "holdout_entry_sha256": holdout_entry_checksum,
        },
        "input_inventory": str(options.inventory),
        "candidate_count": len(candidates),
        "filtered_count": len(filtered),
        "filtered_before_member_lists_count": len(filtered_before_member_lists),
        "included_count": len(filtered_after_include),
        "excluded_count": len(filtered_after_include) - len(filtered_after_exclude),
        "holdout_count": len(holdout),
        "cluster_count": len(clusters),
        "selected_count": len(selected),
        "selected_entry_sha256": selected_entry_checksum,
        "holdout_entry_sha256": holdout_entry_checksum,
        "entries": selected,
        "holdout_entries": holdout,
    }
    (output_root / "selected.json").write_text(
        json.dumps(selected_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_root / "selected.jsonl", selected)
    _write_jsonl(output_root / "holdout.jsonl", holdout)
    for split_name, split_rows in splits.items():
        _write_jsonl(output_root / f"{split_name}.jsonl", split_rows)
    return {**selected_json, "output_root": str(output_root), "splits": {k: len(v) for k, v in splits.items()}}


def build_score_tables(
    training_set: Path,
    *,
    output_root: Path,
    models: Sequence[str] = DEFAULT_TRAINING_MODELS,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    burial_method: str = "contacts",
    sasa_backend: str = "auto",
    max_entries: Optional[int] = None,
    table_set_id: Optional[str] = None,
    table_set_version: str = DEFAULT_ARTIFACT_VERSION,
    command_args: Optional[Sequence[str]] = None,
    workers: Union[str, int] = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    normalized_models = normalize_model_list(models)
    domain = normalize_domain(domain)
    burial_method = _normalize_choice(burial_method, TABLE_BUILD_METHODS, "burial method")
    sasa_backend = _normalize_choice(sasa_backend, SASA_BACKENDS, "SASA backend")
    corpus_manifest = _training_set_manifest(training_set)
    entries = _training_set_entries(training_set)
    if max_entries is not None:
        entries = entries[:max_entries]

    methods = ("contacts", "sasa") if burial_method == "both" else (burial_method,)
    source_corpus = _source_corpus_summary(training_set, corpus_manifest, entries)
    profile_prefix = str(source_corpus.get("artifact_id") or "").strip() or None
    table_id = table_set_id or _default_table_set_id(profile_prefix, domain, burial_method)
    table_version = table_set_version or DEFAULT_ARTIFACT_VERSION
    table_artifact_label = artifact_label_for(table_id, table_version)
    profiles = {}
    for method in methods:
        profile_id = score_table_profile_id(domain, method, profile_prefix=profile_prefix)
        profiles[profile_id] = _build_score_table_profile(
            entries,
            models=normalized_models,
            domain=domain,
            burial_method=method,
            sasa_backend=sasa_backend,
            source_corpus=source_corpus,
            workers=workers,
            progress=progress,
            progress_label=f"score tables {method}",
        )

    payload = {
        "format": SCORE_TABLE_SET_FORMAT,
        "version": SCORE_TABLE_SET_VERSION,
        "artifact_id": table_id,
        "artifact_version": table_version,
        "artifact_label": table_artifact_label,
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "default_profile": score_table_profile_id(domain, methods[0], profile_prefix=profile_prefix),
        "metadata": {
            "training_set": str(training_set),
            "entry_count": len(entries),
            "entry_sha256": _entries_sha256(entries),
            "source_corpus": source_corpus,
            "domain": domain,
            "burial_method": burial_method,
            "sasa_backend": sasa_backend if "sasa" in methods else None,
            "generated_full_corpus": False,
            "provenance": {
                "pheat_version": __version__,
                "command_args": list(command_args) if command_args is not None else None,
            },
        },
        "profiles": profiles,
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / TRAINING_TABLE_FILENAME
    write_score_table_set(output, payload)
    return {**payload, "output": str(output)}


def score_table_profile_id(domain: str, burial_method: str, *, profile_prefix: Optional[str] = None) -> str:
    prefix = _safe_artifact_label(profile_prefix) if profile_prefix else normalize_domain(domain)
    return f"{prefix}-{burial_method}"


def _build_score_table_profile(
    entries: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    domain: str,
    burial_method: str,
    sasa_backend: str,
    source_corpus: Optional[Mapping[str, Any]] = None,
    workers: Union[str, int] = 1,
    progress: Optional[Callable[[str], None]] = None,
    progress_label: str = "score tables",
) -> dict[str, Any]:
    accumulators = _new_table_accumulators(models)
    loaded_count = 0
    warnings = []
    tasks = [
        (dict(entry), tuple(models), domain, burial_method, sasa_backend)
        for entry in entries
    ]
    for result in _map_ordered(
        _score_table_entry_worker,
        tasks,
        workers=workers,
        progress=progress,
        progress_label=progress_label,
    ):
        warnings.extend(result["warnings"])
        if result["loaded"]:
            _merge_accumulators(accumulators, result["accumulators"])
            loaded_count += 1
    return {
        "metadata": {
            "domain": domain,
            "burial_method": burial_method,
            "sasa_backend": sasa_backend if burial_method == "sasa" else None,
            "entry_count": len(entries),
            "loaded_entry_count": loaded_count,
            "source_corpus": dict(source_corpus or {}),
            "status": "ok" if loaded_count else "unavailable",
            "warnings": warnings,
        },
        "models": _finalize_tables(accumulators, models, burial_method=burial_method),
    }


def validate_tables(
    table_set: Path,
    *,
    decoy_root: Optional[Path] = None,
) -> dict[str, Any]:
    from pheat.score_tables import load_score_table_set

    payload = load_score_table_set(table_set)
    assert payload is not None
    profiles = payload.get("profiles", {})
    profile_models = {
        str(profile_id): sorted(profile.get("models", {}))
        for profile_id, profile in profiles.items()
        if isinstance(profile, Mapping) and isinstance(profile.get("models"), Mapping)
    } if isinstance(profiles, Mapping) else {}
    decoy_summary = verify_decoy_root(decoy_root) if decoy_root is not None else None
    return {
        "table_set": str(table_set),
        "ok": bool(profile_models),
        "profile_count": len(profile_models),
        "profiles": profile_models,
        "decoys": decoy_summary,
    }


def describe_training_corpus(training_set: Path) -> dict[str, Any]:
    manifest = _training_set_manifest(training_set)
    entries = _training_set_entries(training_set)
    payload: Mapping[str, Any] = manifest or {}
    return {
        "format": "pheat.training-corpus-description",
        "version": 1,
        "training_set": str(training_set),
        "artifact_id": payload.get("artifact_id"),
        "artifact_version": payload.get("artifact_version"),
        "artifact_label": payload.get("artifact_label"),
        "entry_count": len(entries),
        "entry_sha256": _entries_sha256(entries),
        "selected_count": payload.get("selected_count", len(entries)),
        "holdout_count": payload.get("holdout_count"),
        "cluster_count": payload.get("cluster_count"),
        "filters": payload.get("filters", {}),
        "provenance": payload.get("provenance", {}),
    }


def describe_score_tables(table_set: Path) -> dict[str, Any]:
    from pheat.score_tables import load_score_table_set

    payload = load_score_table_set(table_set)
    assert payload is not None
    profiles = payload.get("profiles", {})
    profile_summaries = {}
    if isinstance(profiles, Mapping):
        for profile_id, profile in profiles.items():
            if not isinstance(profile, Mapping):
                continue
            models = profile.get("models", {})
            metadata = profile.get("metadata", {})
            profile_summaries[str(profile_id)] = {
                "models": sorted(str(model) for model in models) if isinstance(models, Mapping) else [],
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            }
    return {
        "format": "pheat.score-table-set-description",
        "version": 1,
        "table_set": str(table_set),
        "artifact_id": payload.get("artifact_id"),
        "artifact_version": payload.get("artifact_version"),
        "artifact_label": payload.get("artifact_label"),
        "default_profile": payload.get("default_profile"),
        "profile_count": len(profile_summaries),
        "profiles": profile_summaries,
        "metadata": payload.get("metadata", {}),
    }


def extract_features(
    training_set: Path,
    *,
    output: Path,
    models: Sequence[str],
    domain: str = DEFAULT_TRAINING_DOMAIN,
    workers: Union[str, int] = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    normalized_models = _normalize_feature_model_list(models)
    tasks = [
        (dict(entry), normalized_models, domain)
        for entry in _training_set_entries(training_set)
    ]
    rows = [
        row
        for row in _map_ordered(
            _feature_entry_worker,
            tasks,
            workers=workers,
            progress=progress,
            progress_label="features",
        )
        if row is not None
    ]
    _write_jsonl(output, rows)
    return {"output": str(output), "feature_count": len(rows)}


def train_linear_model(
    features: Path,
    *,
    output: Path,
    target: str = "native_vs_decoy",
) -> dict[str, Any]:
    rows = _read_jsonl(features)
    feature_rows = [_numeric_feature_payload(row) for row in rows]
    numeric_columns = sorted({key for row in feature_rows for key in row})
    coefficient = 1.0 / len(numeric_columns) if numeric_columns else 0.0
    feature_stats = _feature_column_stats(feature_rows, numeric_columns)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SCORE_TABLE_SET_FORMAT,
        "version": SCORE_TABLE_SET_VERSION,
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "default_profile": DEFAULT_TRAINING_DOMAIN + "-contacts",
        "metadata": {
            "training_features": str(features),
            "training_features_sha256": _file_sha256(features),
            "target": target,
            "feature_count": len(rows),
            "feature_columns": numeric_columns,
            "feature_stats": feature_stats,
            "score_direction": "lower-is-better",
            "note": "lightweight deterministic linear baseline; not a deep-learning potential",
        },
        "profiles": {
            DEFAULT_TRAINING_DOMAIN + "-contacts": {
                "metadata": {
                    "domain": DEFAULT_TRAINING_DOMAIN,
                    "burial_method": "contacts",
                    "status": "ok",
                },
                "models": {
                    "pheat-ml-linear": {
                        "type": "linear-feature-combination",
                        "target": target,
                        "intercept": 0.0,
                        "coefficients": {column: coefficient for column in numeric_columns},
                        "feature_means": {column: feature_stats[column]["mean"] for column in numeric_columns},
                        "feature_stds": {column: feature_stats[column]["std"] for column in numeric_columns},
                        "score_direction": "lower-is-better",
                    }
                },
            }
        },
    }
    write_score_table_set(output, payload)
    return {**payload, "output": str(output)}


def load_training_structure(path: Path) -> HeavyAtomStructure:
    source = Path(path)
    lower = source.name.lower()
    if lower.endswith(".bcif") or lower.endswith(".bcif.gz"):
        from pheat.bcif import load_bcif

        return load_bcif(source)
    if lower.endswith(".cif") or lower.endswith(".mmcif"):
        return structure_from_mmcif_string(source.read_text(encoding="utf-8"), name=source.name)
    if lower.endswith(".cif.gz") or lower.endswith(".mmcif.gz"):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return structure_from_mmcif_string(handle.read(), name=source.name)
    if lower.endswith(".pdb") or lower.endswith(".ent"):
        return structure_from_pdb_string(source.read_text(encoding="utf-8"), name=source.name)
    if lower.endswith(".pdb.gz") or lower.endswith(".ent.gz"):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return structure_from_pdb_string(handle.read(), name=source.name)
    return load_heavy_json(source)


def normalize_workers(value: Union[str, int, None] = 1) -> int:
    """Return a concrete worker count for deterministic local reference builds."""

    if value is None:
        return 1
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    workers = int(text)
    if workers < 1:
        raise ValueError("workers must be 'auto' or an integer greater than zero")
    return workers


def normalize_model_list(models: Union[Sequence[str], str]) -> tuple[str, ...]:
    if isinstance(models, str):
        tokens = [token.strip().lower() for token in models.split(",") if token.strip()]
    else:
        tokens = [str(model).strip().lower() for model in models if str(model).strip()]
    if not tokens:
        raise ValueError("at least one model is required")
    unknown = sorted(set(tokens) - set(DEFAULT_TRAINING_MODELS) - {"pheat-ml-linear"})
    if unknown:
        raise ValueError(f"unsupported training table model(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(tokens))


def _map_ordered(
    function: Any,
    items: Sequence[Any],
    *,
    workers: Union[str, int],
    progress: Optional[Callable[[str], None]] = None,
    progress_label: str = "progress",
) -> list[Any]:
    worker_count = normalize_workers(workers)
    if not items:
        return []
    reporter = _ProgressReporter(progress, progress_label, len(items))
    reporter.start()
    results = []
    if worker_count == 1:
        for item in items:
            results.append(function(item))
            reporter.record()
        reporter.finish()
        return results
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for result in pool.map(function, items):
            results.append(result)
            reporter.record()
    reporter.finish()
    return results


class _ProgressReporter:
    def __init__(
        self,
        callback: Optional[Callable[[str], None]],
        label: str,
        total: int,
        *,
        interval: int = 1000,
        seconds: float = 30.0,
    ) -> None:
        self.callback = callback
        self.label = label
        self.total = int(total)
        self.interval = max(1, int(interval))
        self.seconds = max(0.0, float(seconds))
        self.started_at = monotonic()
        self.last_at = self.started_at
        self.processed = 0

    def start(self) -> None:
        self._emit()

    def record(self) -> None:
        self.processed += 1
        now = monotonic()
        if self.processed == self.total or self.processed % self.interval == 0 or (
            self.seconds and now - self.last_at >= self.seconds
        ):
            self._emit(now=now)

    def finish(self) -> None:
        if self.processed != self.total:
            self.processed = self.total
            self._emit()

    def _emit(self, *, now: Optional[float] = None) -> None:
        if self.callback is None:
            return
        current = monotonic() if now is None else now
        elapsed = max(0.0, current - self.started_at)
        rate = self.processed / elapsed if elapsed else 0.0
        remaining = max(0, self.total - self.processed)
        eta = remaining / rate if rate else None
        percent = (self.processed / self.total * 100.0) if self.total else 100.0
        message = (
            f"{self.label} progress: {self.processed}/{self.total} "
            f"({percent:.1f}%), {rate:.2f}/s, elapsed {_format_duration(elapsed)}"
        )
        if eta is not None:
            message += f", ETA {_format_duration(eta)}"
        self.callback(message)
        self.last_at = current


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _inventory_snapshot_worker(task: tuple[dict[str, Any], str, Mapping[str, Any], str, Optional[str]]) -> list[dict[str, Any]]:
    row, domain, metadata, snapshot_root_text, snapshot_manifest_sha256 = task
    path = _manifest_row_path(row, snapshot_root=Path(snapshot_root_text))
    if path is None or not path.exists():
        return []
    pdb_id = str(row.get("pdb_id") or path.stem).upper()
    try:
        structure = load_training_structure(path)
    except Exception as exc:
        return [
            {
                "pdb_id": pdb_id,
                "path": str(path),
                "status": "failed",
                "error": str(exc),
            }
        ]
    filtered, coverage = filter_structure_for_domain(structure, domain=domain)
    return [
        _chain_inventory_row(
            pdb_id,
            chain_id,
            chain_atoms,
            path=path,
            domain=domain,
            coverage=coverage,
            metadata=metadata,
            snapshot_root=Path(snapshot_root_text),
            snapshot_manifest_sha256=snapshot_manifest_sha256,
        )
        for chain_id, chain_atoms in _atoms_by_chain(filtered).items()
    ]


def _score_table_entry_worker(task: tuple[dict[str, Any], tuple[str, ...], str, str, str]) -> dict[str, Any]:
    entry, models, domain, burial_method, sasa_backend = task
    accumulators = _new_table_accumulators(models)
    path = Path(str(entry.get("path") or ""))
    if not path.exists():
        return {"loaded": False, "warnings": [f"missing selected structure: {path}"], "accumulators": accumulators}
    try:
        filtered = _load_training_entry_structure(entry, domain=domain)
        _accumulate_tables(
            accumulators,
            filtered,
            models=models,
            burial_method=burial_method,
            sasa_backend=sasa_backend,
        )
        return {"loaded": True, "warnings": [], "accumulators": accumulators}
    except Exception as exc:
        return {"loaded": False, "warnings": [f"{path}: {exc}"], "accumulators": accumulators}


def _feature_entry_worker(task: tuple[dict[str, Any], tuple[str, ...], str]) -> Optional[dict[str, Any]]:
    from pheat.scoring import score_structure

    entry, models, domain = task
    path = Path(str(entry.get("path") or ""))
    if not path.exists():
        return None
    structure = _load_training_entry_structure(entry, domain=domain)
    row = {
        "pdb_id": entry.get("pdb_id"),
        "chain_id": entry.get("chain_id"),
        "path": str(path),
    }
    for model in models:
        try:
            result = score_structure(structure, model=model, domain=domain)
            row[f"{model}_total"] = result.total
        except Exception as exc:
            row[f"{model}_error"] = str(exc)
    return row


def load_training_entry_structure(entry: Mapping[str, Any], *, domain: str = DEFAULT_TRAINING_DOMAIN) -> HeavyAtomStructure:
    """Load one selected training entry as the selected chain/domain structure.

    Reference selection rows are chain-level records, while their ``path`` values
    point to whole deposited structure files.  Score-table, decoy, and feature
    builders must therefore materialize only the selected chain after applying
    the requested scoring domain.  Otherwise large multi-chain entries are
    counted repeatedly and downstream feature extraction becomes prohibitively
    slow.
    """

    return _load_training_entry_structure(entry, domain=domain)


def _load_training_entry_structure(entry: Mapping[str, Any], *, domain: str) -> HeavyAtomStructure:
    path = Path(str(entry.get("path") or entry.get("structure_path") or ""))
    structure = load_training_structure(path)
    filtered, _coverage = filter_structure_for_domain(structure, domain=domain)
    chain_id = str(entry.get("chain_id") or "")
    if not chain_id:
        return filtered
    chain_atoms = [atom for atom in filtered.atoms if (atom.chain_id or "") == chain_id]
    if not chain_atoms:
        raise ValueError(f"selected chain {chain_id!r} is not present in {path}")
    return HeavyAtomStructure(
        atoms=chain_atoms,
        name=filtered.name,
        metadata={
            **dict(filtered.metadata),
            "selected_chain_id": chain_id,
            "source_path": str(path),
        },
        disulfide_bonds=_disulfides_for_atoms(filtered.disulfide_bonds, chain_atoms),
        atom_scope=filtered.atom_scope,
    )


def _disulfides_for_atoms(disulfides: Sequence[Any], atoms: Sequence[Atom]) -> list[Any]:
    kept_refs = {(atom.chain_id or "", int(atom.resseq), atom.icode or "") for atom in atoms}
    return [
        bond
        for bond in disulfides
        if bond.residue_ref_1 in kept_refs and bond.residue_ref_2 in kept_refs
    ]


def _merge_accumulators(target: dict[str, dict[str, Any]], source: Mapping[str, Mapping[str, Any]]) -> None:
    for model, source_payload in source.items():
        target_payload = target.setdefault(model, {"counts": {}, "totals": {}})
        _merge_nested_counts(target_payload.setdefault("counts", {}), source_payload.get("counts", {}))
        totals = target_payload.setdefault("totals", {})
        for key, value in source_payload.get("totals", {}).items():
            totals[key] = float(totals.get(key, 0.0)) + float(value)


def _merge_nested_counts(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            nested = target.setdefault(str(key), {})
            _merge_nested_counts(nested, value)
        else:
            target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _numeric_feature_payload(row: Mapping[str, Any]) -> dict[str, float]:
    if isinstance(row.get("features"), Mapping):
        return {
            str(key): float(value)
            for key, value in row["features"].items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    return {
        str(key): float(value)
        for key, value in row.items()
        if key.endswith("_total") and isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _feature_column_stats(rows: Sequence[Mapping[str, float]], columns: Sequence[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for column in columns:
        values = [float(row[column]) for row in rows if column in row]
        if not values:
            stats[column] = {"mean": 0.0, "std": 0.0, "count": 0.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats[column] = {"mean": mean, "std": math.sqrt(variance), "count": float(len(values))}
    return stats


def _normalize_feature_model_list(models: Sequence[str]) -> tuple[str, ...]:
    tokens = [str(model).strip().lower() for model in models if str(model).strip()]
    if not tokens:
        raise ValueError("at least one feature model is required")
    return tuple(dict.fromkeys(tokens))


def sequence_identity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    length = min(len(first), len(second))
    matches = sum(1 for left, right in zip(first[:length], second[:length]) if left == right)
    return matches / max(len(first), len(second))


def normalize_sequence_identity_threshold(value: Union[str, float]) -> float:
    text = str(value).strip()
    percent_style = text.endswith("%")
    if percent_style:
        text = text[:-1].strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"sequence identity threshold must be numeric: {value!r}") from exc
    threshold = numeric / 100.0 if percent_style or numeric > 1.0 else numeric
    if not 0.0 < threshold <= 1.0:
        raise ValueError("sequence identity threshold must be greater than 0 and no more than 1.0 or 100%")
    return threshold


def sequence_identity_threshold_label(value: Union[str, float]) -> str:
    threshold = normalize_sequence_identity_threshold(value)
    percent = threshold * 100.0
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-9):
        label = str(int(round(percent)))
    else:
        label = f"{percent:g}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "id"


def artifact_label_for(artifact_id: str, artifact_version: str) -> str:
    return _safe_artifact_label(f"{artifact_id}-{artifact_version}")


def _read_archive_manifest_rows(
    output_root: Path,
    *,
    manifest_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    manifest = (Path(manifest_dir) if manifest_dir is not None else root / "manifests") / "files.jsonl"
    if not manifest.exists():
        raise ValueError(f"archive files manifest not found: {manifest}")
    return _read_jsonl(manifest)


def _manifest_row_path(row: Mapping[str, Any], *, snapshot_root: Optional[Path] = None) -> Optional[Path]:
    if snapshot_root is not None:
        return resolve_manifest_row_path(row, output_root=snapshot_root)
    path = Path(str(row["path"])) if row.get("path") else None
    if path is not None and path.exists():
        return path
    source_path = Path(str(row["source_path"])) if row.get("source_path") else None
    if source_path is not None and source_path.exists():
        return source_path
    return path


def _metadata_by_pdb_id(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path)
    metadata = {}
    for row in rows:
        pdb_id = str(row.get("pdb_id") or row.get("entry_id") or "").upper()
        if pdb_id:
            metadata[pdb_id] = row
    return metadata


def _atoms_by_chain(structure: HeavyAtomStructure) -> dict[str, list[Atom]]:
    chains: dict[str, list[Atom]] = {}
    for atom in structure.atoms:
        chains.setdefault(atom.chain_id or "A", []).append(atom)
    return chains


def _chain_inventory_row(
    pdb_id: str,
    chain_id: str,
    atoms: Sequence[Atom],
    *,
    path: Path,
    domain: str,
    coverage: Mapping[str, Any],
    metadata: Mapping[str, Any],
    snapshot_root: Path,
    snapshot_manifest_sha256: Optional[str],
) -> dict[str, Any]:
    residues: dict[ResidueKey, list[Atom]] = {}
    for atom in atoms:
        residues.setdefault(atom.residue_key, []).append(atom)
    ordered_keys = list(residues)
    letters = []
    canonical = 0
    missing_backbone = 0
    for key in ordered_keys:
        resname = key[3].strip().upper()
        if resname in CANONICAL_RESIDUES:
            canonical += 1
        try:
            letters.append(three_to_one(resname))
        except Exception:
            letters.append("X")
        names = {atom.name.strip().upper() for atom in residues[key]}
        missing_backbone += len({"N", "CA", "C"} - names)
    length = len(ordered_keys)
    resolution = _optional_float(metadata.get("resolution") or metadata.get("resolution_angstrom"))
    method = metadata.get("method") or metadata.get("experimental_method")
    metadata_provenance = {
        "source": metadata.get("metadata_source"),
        "retrieved_at": metadata.get("metadata_retrieved_at"),
        "source_snapshot_id": metadata.get("source_snapshot_id"),
    }
    return {
        "format": "pheat.training-inventory-row",
        "version": 1,
        "status": "ok",
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "path": str(path),
        "domain": domain,
        "sequence": "".join(letters),
        "length": length,
        "atom_count": len(atoms),
        "canonical_residue_count": canonical,
        "canonical_fraction": (canonical / length) if length else 0.0,
        "missing_backbone_atom_count": missing_backbone,
        "method": str(method).lower() if method is not None else None,
        "resolution": resolution,
        "source_snapshot_id": Path(snapshot_root).name,
        "source_snapshot_root": str(snapshot_root),
        "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "coverage": dict(coverage),
        "environment": dict(metadata.get("environment") or {}),
        "quality": dict(metadata.get("quality") or {}),
        "composition": dict(metadata.get("composition") or {}),
        "sequence_clusters": dict(metadata.get("sequence_clusters") or {}),
        "metadata_provenance": {key: value for key, value in metadata_provenance.items() if value is not None},
    }


def _normalize_inventory_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["length"] = int(payload.get("length") or len(str(payload.get("sequence") or "")))
    payload["canonical_fraction"] = float(payload.get("canonical_fraction") or 0.0)
    payload["missing_backbone_atom_count"] = int(payload.get("missing_backbone_atom_count") or 0)
    payload["resolution"] = _optional_float(payload.get("resolution"))
    payload["method"] = str(payload.get("method") or "").lower() or None
    payload["sequence"] = str(payload.get("sequence") or "")
    return payload


def _candidate_passes_filters(
    row: Mapping[str, Any],
    *,
    method: Optional[str],
    max_resolution: Optional[float],
    canonical_only: bool,
    min_length: int,
    max_length: int,
) -> bool:
    if row.get("status") != "ok":
        return False
    length = int(row.get("length") or 0)
    if length < min_length or length > max_length:
        return False
    if canonical_only and float(row.get("canonical_fraction") or 0.0) < 1.0:
        return False
    if method and str(method).lower() not in str(row.get("method") or ""):
        return False
    resolution = row.get("resolution")
    if max_resolution is not None and (resolution is None or float(resolution) > max_resolution):
        return False
    return True


def _selection_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    resolution = row.get("resolution")
    return (
        float(resolution) if resolution is not None else float("inf"),
        int(row.get("missing_backbone_atom_count") or 0),
        -float(row.get("canonical_fraction") or 0.0),
        -int(row.get("length") or 0),
        str(row.get("pdb_id") or ""),
        str(row.get("chain_id") or ""),
    )


def _split_selected(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        bucket = index % 10
        if bucket == 8:
            splits["validation"].append(row)
        elif bucket == 9:
            splits["test"].append(row)
        else:
            splits["train"].append(row)
    return splits


def _read_sequence_cluster_map(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    cluster_map = {}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        tokens = [token for token in line.replace(",", " ").split() if token]
        if not tokens:
            continue
        cluster_id = f"cluster-{index}"
        for token in tokens:
            cluster_map[_normalize_cluster_member(token)] = cluster_id
    return cluster_map


def _candidate_cluster_id(candidate: Mapping[str, Any], cluster_map: Mapping[str, str]) -> str:
    pdb_id = str(candidate.get("pdb_id") or "").upper()
    chain_id = str(candidate.get("chain_id") or "")
    keys = [
        _normalize_cluster_member(f"{pdb_id}_{chain_id}"),
        _normalize_cluster_member(f"{pdb_id}.{chain_id}"),
        _normalize_cluster_member(f"{pdb_id}:{chain_id}"),
        _normalize_cluster_member(pdb_id + chain_id),
        _normalize_cluster_member(pdb_id),
    ]
    for key in keys:
        if key in cluster_map:
            return cluster_map[key]
    return f"singleton:{pdb_id}:{chain_id}"


def _candidate_metadata_cluster_id(candidate: Mapping[str, Any], threshold: float) -> Optional[str]:
    clusters = candidate.get("sequence_clusters")
    if not isinstance(clusters, Mapping):
        return None
    percent = threshold * 100.0
    keys = []
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-9):
        keys.extend([str(int(round(percent))), f"{int(round(percent))}.0"])
    keys.append(f"{percent:g}")
    for key in keys:
        values = clusters.get(key)
        if isinstance(values, list) and values:
            return f"metadata:{key}:{values[0]}"
        if isinstance(values, str) and values:
            return f"metadata:{key}:{values}"
    return None


def _normalize_cluster_member(value: str) -> str:
    return str(value).strip().upper().replace(".", "_").replace(":", "_")


def _read_member_set(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    members = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = line.split("#", 1)[0].replace(",", " ")
        for token in payload.split():
            if token.strip():
                members.add(_normalize_cluster_member(token))
    return members


def _member_file_provenance(path: Optional[Path], members: set[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "member_count": len(members),
    }


def _candidate_member_keys(candidate: Mapping[str, Any]) -> set[str]:
    pdb_id = str(candidate.get("pdb_id") or "").upper()
    chain_id = str(candidate.get("chain_id") or "")
    return {
        _normalize_cluster_member(f"{pdb_id}_{chain_id}"),
        _normalize_cluster_member(f"{pdb_id}.{chain_id}"),
        _normalize_cluster_member(f"{pdb_id}:{chain_id}"),
        _normalize_cluster_member(pdb_id + chain_id),
        _normalize_cluster_member(pdb_id),
    }


def _candidate_matches_members(candidate: Mapping[str, Any], members: set[str]) -> bool:
    return bool(members and _candidate_member_keys(candidate) & members)


def _default_corpus_id(candidates: Sequence[Mapping[str, Any]], sequence_label: str) -> str:
    domain = next((str(candidate.get("domain")) for candidate in candidates if candidate.get("domain")), DEFAULT_TRAINING_DOMAIN)
    return _safe_artifact_label(f"{domain}-{sequence_label}")


def _source_snapshot_summary(candidates: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    snapshot_ids = sorted({str(candidate.get("source_snapshot_id")) for candidate in candidates if candidate.get("source_snapshot_id")})
    snapshot_roots = sorted(
        {str(candidate.get("source_snapshot_root")) for candidate in candidates if candidate.get("source_snapshot_root")}
    )
    manifest_sha256 = sorted(
        {
            str(candidate.get("source_snapshot_manifest_sha256"))
            for candidate in candidates
            if candidate.get("source_snapshot_manifest_sha256")
        }
    )
    if not snapshot_ids and not snapshot_roots and not manifest_sha256:
        return None
    return {
        "snapshot_id": snapshot_ids[0] if len(snapshot_ids) == 1 else snapshot_ids,
        "snapshot_root": snapshot_roots[0] if len(snapshot_roots) == 1 else snapshot_roots,
        "manifest_sha256": manifest_sha256[0] if len(manifest_sha256) == 1 else manifest_sha256,
    }


def _training_set_manifest(training_set: Path) -> Optional[dict[str, Any]]:
    path = Path(training_set)
    candidate = path / "selected.json" if path.is_dir() else path
    if not candidate.exists() or candidate.suffix == ".jsonl":
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("format") == "pheat.training-corpus":
        return payload
    return None


def _source_corpus_summary(
    training_set: Path,
    manifest: Optional[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    filters = manifest.get("filters", {}) if isinstance(manifest, Mapping) else {}
    provenance = manifest.get("provenance", {}) if isinstance(manifest, Mapping) else {}
    return {
        "training_set": str(training_set),
        "artifact_id": manifest.get("artifact_id") if isinstance(manifest, Mapping) else None,
        "artifact_version": manifest.get("artifact_version") if isinstance(manifest, Mapping) else None,
        "artifact_label": manifest.get("artifact_label") if isinstance(manifest, Mapping) else None,
        "entry_count": len(entries),
        "entry_sha256": _entries_sha256(entries),
        "selected_entry_sha256": manifest.get("selected_entry_sha256") if isinstance(manifest, Mapping) else None,
        "sequence_identity": filters.get("sequence_identity") if isinstance(filters, Mapping) else None,
        "sequence_identity_label": filters.get("sequence_identity_label") if isinstance(filters, Mapping) else None,
        "source_snapshot": provenance.get("source_snapshot") if isinstance(provenance, Mapping) else None,
    }


def _default_table_set_id(profile_prefix: Optional[str], domain: str, burial_method: str) -> str:
    if profile_prefix:
        return _safe_artifact_label(profile_prefix if burial_method == "both" else f"{profile_prefix}-{burial_method}")
    return _safe_artifact_label(f"{normalize_domain(domain)}-{burial_method}")


def _entries_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    normalized = [dict(entry) for entry in entries]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact_label(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    output = []
    previous_dash = False
    for char in text:
        if char.isalnum():
            output.append(char)
            previous_dash = False
        elif not previous_dash:
            output.append("-")
            previous_dash = True
    label = "".join(output).strip("-")
    return label or "artifact"


def _training_set_entries(training_set: Path) -> list[dict[str, Any]]:
    path = Path(training_set)
    if path.is_dir():
        selected_jsonl = path / "selected.jsonl"
        selected_json = path / "selected.json"
        if selected_jsonl.exists():
            return _read_jsonl(selected_jsonl)
        if selected_json.exists():
            payload = json.loads(selected_json.read_text(encoding="utf-8"))
            entries = payload.get("entries", [])
            if isinstance(entries, list):
                return [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, Mapping) else []
    return [dict(entry) for entry in entries if isinstance(entry, Mapping)]


def _new_table_accumulators(models: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {model: {"counts": {}, "totals": {}} for model in models}


def _accumulate_tables(
    accumulators: dict[str, dict[str, Any]],
    structure: HeavyAtomStructure,
    *,
    models: Sequence[str],
    burial_method: str,
    sasa_backend: str,
) -> None:
    residues = structure.atoms_by_residue()
    residue_points = _residue_representative_points(residues)
    if "pheat-dfire" in models or "pheat-goap" in models:
        _accumulate_distance_bins(accumulators, residue_points)
    if "pheat-mj" in models:
        _accumulate_mj(accumulators["pheat-mj"], residue_points)
    if "pheat-hydropathy" in models:
        burial, _metadata = residue_burial(structure, method=burial_method, backend=sasa_backend)
        _accumulate_hydropathy(accumulators["pheat-hydropathy"], residue_points, burial)
    if "pheat-backbone" in models:
        _accumulate_backbone(accumulators["pheat-backbone"], structure)
    if "pheat-rotamer" in models:
        _accumulate_rotamer(accumulators["pheat-rotamer"], structure)
    if "pheat-hbond" in models:
        _accumulate_hbond(accumulators["pheat-hbond"], structure)
    if "pheat-rg" in models:
        _accumulate_rg(accumulators["pheat-rg"], structure)


def _residue_representative_points(
    residues: Mapping[ResidueKey, Sequence[Atom]]
) -> dict[ResidueKey, Point3D]:
    points = {}
    for key, atoms in residues.items():
        sidechain = [
            atom.coord
            for atom in atoms
            if atom.name.strip().upper() not in {"N", "CA", "C", "O", "OXT"}
        ]
        if sidechain:
            points[key] = centroid(sidechain)
            continue
        ca = next((atom.coord for atom in atoms if atom.name.strip().upper() == "CA"), None)
        if ca is not None:
            points[key] = ca
    return points


def _accumulate_distance_bins(
    accumulators: Mapping[str, dict[str, Any]],
    residue_points: Mapping[ResidueKey, Point3D],
) -> None:
    for key_a, key_b, dist in _nearby_residue_point_pairs(residue_points, cutoff=20.0):
        residue_pair = _residue_pair_key(key_a[3], key_b[3])
        bin_key = f"{int(dist // 1.0):02d}"
        for model in ("pheat-dfire", "pheat-goap"):
            if model in accumulators:
                _increment_nested(accumulators[model]["counts"], residue_pair, bin_key)


def _accumulate_mj(
    accumulator: dict[str, Any],
    residue_points: Mapping[ResidueKey, Point3D],
    *,
    cutoff: float = 6.5,
) -> None:
    for key_a, key_b, _dist in _nearby_residue_point_pairs(residue_points, cutoff=cutoff):
        _increment(accumulator["counts"], _residue_pair_key(key_a[3], key_b[3]))


def _accumulate_hydropathy(
    accumulator: dict[str, Any],
    residue_points: Mapping[ResidueKey, Point3D],
    burial: Mapping[ResidueKey, float],
) -> None:
    for key in residue_points:
        hydropathy = KYTE_DOOLITTLE_HYDROPATHY.get(key[3], 0.0)
        buried = float(burial.get(key, 0.0))
        accumulator["totals"]["weighted_hydropathy_burial"] = (
            accumulator["totals"].get("weighted_hydropathy_burial", 0.0) + hydropathy * buried
        )
        _increment(accumulator["counts"], key[3])


def _accumulate_backbone(accumulator: dict[str, Any], structure: HeavyAtomStructure) -> None:
    geometry = structure_to_residue_geometry(structure, angle_units="degrees", stored_angles="all")
    for residue in geometry.residues:
        if residue.phi is not None and residue.psi is not None:
            _increment_nested(accumulator["counts"], residue.name, _angle_bin(residue.phi, residue.psi))
        if residue.omega is not None:
            _increment_nested(accumulator["counts"], residue.name, "omega:" + _one_angle_bin(residue.omega))


def _accumulate_rotamer(accumulator: dict[str, Any], structure: HeavyAtomStructure) -> None:
    geometry = structure_to_residue_geometry(structure, angle_units="degrees")
    for residue in geometry.residues:
        for index, chi in enumerate(residue.chi, start=1):
            _increment_nested(accumulator["counts"], residue.name, f"chi{index}:{_one_angle_bin(chi)}")


def _accumulate_hbond(accumulator: dict[str, Any], structure: HeavyAtomStructure) -> None:
    polar_atoms = [
        atom for atom in structure.atoms if atom.element.strip().upper() in {"N", "O", "S", "SE"}
    ]
    for atom_a, atom_b, dist in _nearby_atom_pairs(polar_atoms, cutoff=3.6):
        if atom_a.chain_id == atom_b.chain_id and abs(atom_a.resseq - atom_b.resseq) <= 1:
            continue
        if dist >= 2.4:
            _increment(accumulator["counts"], f"{atom_a.element}-{atom_b.element}")


def _accumulate_rg(accumulator: dict[str, Any], structure: HeavyAtomStructure) -> None:
    from pheat.metrics import structure_radius_of_gyration

    residue_count = len(structure.residue_keys())
    if residue_count <= 0:
        return
    payload = structure_radius_of_gyration(structure, mode="unweighted", atom_set="ca")
    observed = float(payload["values"]["unweighted"])
    accumulator["counts"]["observations"] = int(accumulator["counts"].get("observations", 0)) + 1
    accumulator["totals"]["sum_log_residue_count"] = accumulator["totals"].get("sum_log_residue_count", 0.0) + math.log(
        float(residue_count)
    )
    accumulator["totals"]["sum_log_rg"] = accumulator["totals"].get("sum_log_rg", 0.0) + math.log(max(observed, 1e-12))
    accumulator["totals"]["sum_log_residue_count_squared"] = accumulator["totals"].get(
        "sum_log_residue_count_squared",
        0.0,
    ) + math.log(float(residue_count)) ** 2
    accumulator["totals"]["sum_log_residue_count_log_rg"] = accumulator["totals"].get(
        "sum_log_residue_count_log_rg",
        0.0,
    ) + math.log(float(residue_count)) * math.log(max(observed, 1e-12))


def _finalize_tables(
    accumulators: Mapping[str, dict[str, Any]],
    models: Sequence[str],
    *,
    burial_method: str,
) -> dict[str, Any]:
    finalized = {}
    for model in models:
        data = accumulators[model]
        finalized[model] = {
            "type": model,
            "counts": data.get("counts", {}),
            "totals": data.get("totals", {}),
            "pseudocount": 1.0,
            "units": "arbitrary",
            "generated_by": "pheat training tables build",
            "input_contract": score_input_contract(model),
        }
    if "pheat-hydropathy" in finalized:
        finalized["pheat-hydropathy"]["hydropathy_scale"] = dict(KYTE_DOOLITTLE_HYDROPATHY)
        finalized["pheat-hydropathy"]["burial_method"] = burial_method
    if "pheat-rg" in finalized:
        finalized["pheat-rg"].update(_fit_rg_table(accumulators["pheat-rg"]))
    return finalized


def _fit_rg_table(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    counts = accumulator.get("counts", {})
    totals = accumulator.get("totals", {})
    n = int(counts.get("observations", 0)) if isinstance(counts, Mapping) else 0
    if n < 2:
        return {
            "atom_set": "ca",
            "mode": "unweighted",
            "a": 2.2,
            "b": 1.0 / 3.0,
            "sigma_fraction": 0.25,
            "fitted": False,
            "fit_observation_count": n,
            "score_direction": "lower-is-better",
        }
    sum_x = float(totals.get("sum_log_residue_count", 0.0))
    sum_y = float(totals.get("sum_log_rg", 0.0))
    sum_xx = float(totals.get("sum_log_residue_count_squared", 0.0))
    sum_xy = float(totals.get("sum_log_residue_count_log_rg", 0.0))
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-12:
        b = 1.0 / 3.0
        log_a = (sum_y / n) - b * (sum_x / n)
    else:
        b = (n * sum_xy - sum_x * sum_y) / denominator
        log_a = (sum_y - b * sum_x) / n
    return {
        "atom_set": "ca",
        "mode": "unweighted",
        "a": math.exp(log_a),
        "b": b,
        "sigma_fraction": 0.25,
        "fitted": True,
        "fit_observation_count": n,
        "score_direction": "lower-is-better",
    }


def _nearby_residue_point_pairs(
    residue_points: Mapping[ResidueKey, Point3D],
    *,
    cutoff: float,
) -> Iterable[tuple[ResidueKey, ResidueKey, float]]:
    items = list(residue_points.items())
    for index_a, index_b, dist in _nearby_point_index_pairs(
        [point for _key, point in items],
        cutoff=cutoff,
    ):
        key_a = items[index_a][0]
        key_b = items[index_b][0]
        if key_a[0] == key_b[0] and abs(key_a[1] - key_b[1]) <= 2:
            continue
        yield key_a, key_b, dist


def _nearby_atom_pairs(
    atoms: Sequence[Atom],
    *,
    cutoff: float,
) -> Iterable[tuple[Atom, Atom, float]]:
    for index_a, index_b, dist in _nearby_point_index_pairs(
        [atom.coord for atom in atoms],
        cutoff=cutoff,
    ):
        yield atoms[index_a], atoms[index_b], dist


def _nearby_point_index_pairs(
    points: Sequence[Point3D],
    *,
    cutoff: float,
) -> Iterable[tuple[int, int, float]]:
    """Yield point pairs within ``cutoff`` using a simple fixed-width spatial grid."""

    if cutoff <= 0.0 or len(points) < 2:
        return
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(points):
        grid.setdefault(_point_cell(point, cutoff), []).append(index)
    for index_a, point_a in enumerate(points):
        cell = _point_cell(point_a, cutoff)
        for neighbor in _neighbor_cells(cell):
            for index_b in grid.get(neighbor, []):
                if index_b <= index_a:
                    continue
                dist = distance(point_a, points[index_b])
                if dist <= cutoff:
                    yield index_a, index_b, dist


def _point_cell(point: Point3D, cell_size: float) -> tuple[int, int, int]:
    return (
        int(math.floor(point[0] / cell_size)),
        int(math.floor(point[1] / cell_size)),
        int(math.floor(point[2] / cell_size)),
    )


def _neighbor_cells(cell: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def _residue_pair_key(left: str, right: str) -> str:
    return "-".join(sorted((left.strip().upper(), right.strip().upper())))


def _angle_bin(phi: float, psi: float, *, size: int = 30) -> str:
    return f"phi:{_one_angle_bin(phi, size=size)}|psi:{_one_angle_bin(psi, size=size)}"


def _one_angle_bin(angle: float, *, size: int = 30) -> str:
    shifted = (float(angle) + 180.0) % 360.0
    start = int(math.floor(shifted / size) * size - 180)
    return f"{start}:{start + size}"


def _increment(counts: dict[str, Any], key: str, amount: int = 1) -> None:
    counts[key] = int(counts.get(key, 0)) + amount


def _increment_nested(counts: dict[str, Any], first: str, second: str, amount: int = 1) -> None:
    nested = counts.setdefault(first, {})
    nested[second] = int(nested.get(second, 0)) + amount


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row is not an object in {path}")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _normalize_choice(value: str, choices: Sequence[str], label: str) -> str:
    normalized = str(value or choices[0]).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)}")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

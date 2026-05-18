"""Reference-corpus, decoy, and ML artifact workflow helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

from pheat import __version__
from pheat.archive import default_archive_paths, get_snapshot, write_snapshot_metadata
from pheat.metrics import (
    align_structure_to_reference,
    radius_of_gyration_delta,
    structure_radius_of_gyration,
    structure_rmsd,
)
from pheat.models import HeavyAtomStructure
from pheat.pdbio import write_structure_json
from pheat.residue_geometry import structure_from_residue_geometry, structure_to_residue_geometry
from pheat.score_tables import (
    SCORE_TABLE_XZ_PRESET,
    load_score_table_set,
    score_table_set_json_bytes,
    write_score_table_set,
)
from pheat.scoring import score_structure
from pheat.training import (
    DEFAULT_TRAINING_DOMAIN,
    DEFAULT_TRAINING_MODELS,
    InventoryOptions,
    SelectionOptions,
    build_score_tables,
    load_training_entry_structure,
    load_training_structure,
    normalize_sequence_identity_threshold,
    normalize_workers,
    select_corpus,
    train_linear_model,
    inventory_snapshot,
)


DEFAULT_REFERENCE_ROOT = Path(".pheat-cache/reference-builds")
DEFAULT_REFERENCE_ARTIFACT_VERSION = "v0"
DEFAULT_REFERENCE_SNAPSHOT_ID = "rcsb-current-bcif"
DEFAULT_REFERENCE_SEQUENCE_IDENTITY = "0.30"
DEFAULT_REFERENCE_METHOD = "x-ray"
DEFAULT_REFERENCE_MAX_RESOLUTION = 2.5
DEFAULT_REFERENCE_SEED = 20260522
REFERENCE_SUBSETS = ("aqueous", "membrane")
PHEAT_DECOY_RECIPES = (
    "backbone-noise-small",
    "backbone-noise-medium",
    "chi-rotamer-random",
    "compactness-perturb",
    "mixed",
    "sidechain-repack",
    "torsion-near",
    "torsion-medium",
    "torsion-far",
    "segment-near",
    "segment-medium",
    "compactness-contract",
    "compactness-expand",
)
PHEAT_DECOY_PROFILES = {
    "pheat-torsion-v1": (
        "sidechain-repack",
        "torsion-near",
        "torsion-medium",
        "torsion-far",
        "segment-near",
        "segment-medium",
        "compactness-contract",
        "compactness-expand",
    ),
    "smoke": ("backbone-noise-small",),
}
DEFAULT_REFERENCE_DECOY_PROFILE = "pheat-torsion-v1"
DEFAULT_REFERENCE_DECOY_ATTEMPTS = 8
REFERENCE_UNATTENDED_STAGE_ORDER = (
    "fetch",
    "metadata",
    "inventory",
    "select-aqueous",
    "select-membrane",
    "canary-aqueous",
    "canary-membrane",
    "decoys-aqueous",
    "decoys-membrane",
    "scores-aqueous",
    "scores-membrane",
    "features-aqueous",
    "features-membrane",
    "ml-aqueous",
    "ml-membrane",
    "validate-aqueous",
    "validate-membrane",
)
DEFAULT_REFERENCE_FEATURE_MODELS = (
    "generic",
    "pheat-dfire",
    "pheat-goap",
    "pheat-mj",
    "pheat-hydropathy",
    "pheat-backbone",
    "pheat-rotamer",
    "pheat-hbond",
    "pheat-rg",
    "heavy-mm",
    "pheat-geometry-integrity",
)

REFERENCE_VERSION_TOKEN_RE = re.compile(r"(?:(?<=/)|(?<=-)|^)v[0-9]+(?=$|/|-|\.)")
REFERENCE_ARTIFACT_VERSION_KEYS = {
    "artifact_version",
    "packaged_artifact_version",
    "packaged_as_artifact_version",
    "source_artifact_version",
    "source_run_artifact_version",
}
REFERENCE_ARTIFACT_LABEL_KEYS = {
    "artifact_label",
    "default_ml_model",
    "default_score_table",
    "packaged_asset_id",
}
REFERENCE_ARTIFACT_PATH_KEYS = {
    "decoys",
    "destination",
    "features",
    "log_root",
    "manifest",
    "metadata_jsonl",
    "model_aqueous",
    "model_membrane",
    "output",
    "output_root",
    "parents",
    "path",
    "reference_root",
    "rejections",
    "run_root",
    "source",
    "source_path",
    "source_run_summary",
    "summary",
    "tables_aqueous",
    "tables_membrane",
    "training_features",
    "training_set",
    "validation_aqueous",
    "validation_membrane",
}


REFERENCE_DECOY_DATASETS = (
    {
        "id": "3drobot",
        "name": "3DRobot decoy generator and decoy set",
        "urls": ["https://aideepmed.com/3DRobot/"],
        "expected_payloads": ["3DRobot_set.tar.bz", "3DRobot1.0.tar.bz"],
        "license_status": "unclear",
        "local_use_only": True,
        "packageable": False,
        "notes": [
            "PHEAT records provenance for local use but does not redistribute 3DRobot payloads.",
            "The published method can be used to guide independent PHEAT-owned decoy recipes.",
        ],
    },
    {
        "id": "casp",
        "name": "CASP Prediction Center download area",
        "urls": ["https://predictioncenter.org/download_area/"],
        "expected_payloads": [],
        "license_status": "dataset-specific",
        "local_use_only": True,
        "packageable": False,
        "notes": ["Use CASP data under the terms declared for each CASP release or paper."],
    },
    {
        "id": "itasser",
        "name": "I-TASSER/Zhang Lab decoy resources",
        "urls": ["https://zhanggroup.org/"],
        "expected_payloads": [],
        "license_status": "dataset-specific",
        "local_use_only": True,
        "packageable": False,
        "notes": ["Register local files explicitly unless a stable public payload URL is supplied."],
    },
    {
        "id": "rosetta",
        "name": "Rosetta decoy resources",
        "urls": ["https://www.rosettacommons.org/"],
        "expected_payloads": [],
        "license_status": "dataset-specific",
        "local_use_only": True,
        "packageable": False,
        "notes": ["Register local files explicitly unless a stable public payload URL is supplied."],
    },
)


def list_reference_decoy_datasets() -> list[dict[str, Any]]:
    return [dict(dataset) for dataset in REFERENCE_DECOY_DATASETS]


def get_reference_decoy_dataset(dataset_id: str) -> dict[str, Any]:
    normalized = str(dataset_id).strip().lower()
    for dataset in REFERENCE_DECOY_DATASETS:
        if dataset["id"] == normalized:
            return dict(dataset)
    known = ", ".join(str(dataset["id"]) for dataset in REFERENCE_DECOY_DATASETS)
    raise ValueError(f"Unknown reference decoy dataset '{dataset_id}'. Known datasets: {known}")


def fetch_reference_inputs(
    *,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    snapshot_id: str = DEFAULT_REFERENCE_SNAPSHOT_ID,
    snapshot_root: Optional[Path] = None,
    datasets: Sequence[str] = (),
    include_payloads: bool = False,
    payload_urls: Mapping[str, Sequence[str]] | None = None,
    local_files: Mapping[str, Sequence[Path]] | None = None,
    local_dirs: Mapping[str, Sequence[Path]] | None = None,
    overwrite: bool = False,
    workers: Union[str, int] = "auto",
    seed: int = DEFAULT_REFERENCE_SEED,
) -> dict[str, Any]:
    """Create a reference-input manifest and optional decoy dataset manifests."""

    root = Path(reference_root)
    root.mkdir(parents=True, exist_ok=True)
    fetch_root = root / "fetches" / artifact_version
    _ensure_output_path(fetch_root / "manifest.json", overwrite=overwrite)
    snapshot = get_snapshot(snapshot_id)
    resolved_snapshot_root = snapshot_root or snapshot.default_output_root
    dataset_ids = tuple(dict.fromkeys(str(dataset).strip().lower() for dataset in datasets if str(dataset).strip()))
    dataset_payloads = [
        _write_reference_dataset_manifest(
            root,
            dataset_id,
            artifact_version=artifact_version,
            include_payloads=include_payloads,
            payload_urls=tuple((payload_urls or {}).get(dataset_id, ())),
            local_files=tuple((local_files or {}).get(dataset_id, ())),
            local_dirs=tuple((local_dirs or {}).get(dataset_id, ())),
            overwrite=overwrite,
        )
        for dataset_id in dataset_ids
    ]
    payload = {
        "format": "pheat.reference-fetch",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "workers": normalize_workers(workers),
        "seed": int(seed),
        "reference_root": str(root),
        "snapshot": {
            "snapshot_id": snapshot.id,
            "snapshot_root": str(resolved_snapshot_root),
            "files_manifest": str(Path(resolved_snapshot_root) / "manifests" / "files.jsonl"),
            "files_manifest_sha256": _file_sha256(Path(resolved_snapshot_root) / "manifests" / "files.jsonl")
            if (Path(resolved_snapshot_root) / "manifests" / "files.jsonl").exists()
            else None,
            "exists": Path(resolved_snapshot_root).exists(),
        },
        "datasets": dataset_payloads,
        "lineage": _lineage_payload(stage="fetch", parents=[]),
    }
    fetch_root.mkdir(parents=True, exist_ok=True)
    output = fetch_root / "manifest.json"
    _write_json(output, payload)
    return {**payload, "output": str(output)}


def inventory_reference(
    *,
    snapshot_root: Path,
    output: Path,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    metadata_jsonl: Optional[Path] = None,
    max_entries: Optional[int] = None,
    workers: Union[str, int] = "auto",
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    _ensure_output_path(output, overwrite=overwrite)
    resolved_metadata_jsonl = metadata_jsonl
    if resolved_metadata_jsonl is None:
        candidate = Path(snapshot_root) / "manifests" / "metadata.jsonl"
        if candidate.exists():
            resolved_metadata_jsonl = candidate
    rows = inventory_snapshot(
        InventoryOptions(
            snapshot_root=Path(snapshot_root),
            output=Path(output),
            domain=domain,
            metadata_jsonl=resolved_metadata_jsonl,
            max_entries=max_entries,
            workers=workers,
            progress=progress,
        )
    )
    manifest = {
        "format": "pheat.reference-inventory",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "workers": normalize_workers(workers),
        "domain": domain,
        "snapshot_root": str(snapshot_root),
        "metadata_jsonl": str(resolved_metadata_jsonl) if resolved_metadata_jsonl else None,
        "inventory": str(output),
        "inventory_sha256": _file_sha256(output),
        "row_count": len(rows),
        "audit": _inventory_audit(rows),
        "lineage": _lineage_payload(stage="inventory", parents=[str(snapshot_root)]),
    }
    manifest_path = Path(reference_root) / "inventories" / artifact_version / "manifest.json"
    _ensure_output_path(manifest_path, overwrite=overwrite)
    _write_json(manifest_path, manifest)
    return {**manifest, "output": str(output), "manifest": str(manifest_path)}


def select_reference_corpus(
    *,
    inventory: Path,
    output_root: Path,
    subset: str,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    method: str = DEFAULT_REFERENCE_METHOD,
    max_resolution: Optional[float] = DEFAULT_REFERENCE_MAX_RESOLUTION,
    sequence_identity: Union[str, float] = DEFAULT_REFERENCE_SEQUENCE_IDENTITY,
    sequence_clusters: Optional[Path] = None,
    include_file: Optional[Path] = None,
    exclude_file: Optional[Path] = None,
    holdout_file: Optional[Path] = None,
    min_length: int = 50,
    max_length: int = 800,
    canonical_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    subset = _normalize_subset(subset)
    output_root = Path(output_root)
    _ensure_output_path(output_root / "selected.json", overwrite=overwrite)
    inventory_rows = _read_jsonl(inventory)
    filtered_rows, environment_warnings = _filter_subset_rows(inventory_rows, subset)
    working_inventory = output_root / "_reference-filtered-inventory.jsonl"
    _write_jsonl(working_inventory, filtered_rows)
    sequence_label = _sequence_identity_label(sequence_identity)
    corpus_id = f"{domain}-{sequence_label}-xray-{subset}"
    result = select_corpus(
        SelectionOptions(
            inventory=working_inventory,
            output_root=output_root,
            method=method,
            max_resolution=max_resolution,
            sequence_identity=sequence_identity,
            canonical_only=canonical_only,
            min_length=min_length,
            max_length=max_length,
            sequence_clusters=sequence_clusters,
            corpus_id=corpus_id,
            corpus_version=artifact_version,
            include_file=include_file,
            exclude_file=exclude_file,
            holdout_file=holdout_file,
        )
    )
    selected_json = output_root / "selected.json"
    payload = json.loads(selected_json.read_text(encoding="utf-8"))
    payload["provisional"] = artifact_version == "v0"
    payload.setdefault("filters", {})
    payload["filters"].update(
        {
            "reference_subset": subset,
            "reference_method": method,
            "default_xray_only": method == DEFAULT_REFERENCE_METHOD,
            "environment_warnings": environment_warnings,
        }
    )
    payload["lineage"] = _lineage_payload(stage="select", parents=[str(inventory)])
    _write_json(selected_json, payload)
    audit = _corpus_audit(payload)
    _write_json(output_root / "audit.json", audit)
    return {**result, **payload, "output_root": str(output_root), "audit": audit}


def build_reference_decoys(
    *,
    training_set: Path,
    output_root: Path,
    recipes: Sequence[str] = (DEFAULT_REFERENCE_DECOY_PROFILE,),
    domain: str = DEFAULT_TRAINING_DOMAIN,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    seed: int = DEFAULT_REFERENCE_SEED,
    max_entries: Optional[int] = None,
    attempts_per_decoy: int = DEFAULT_REFERENCE_DECOY_ATTEMPTS,
    workers: Union[str, int] = "auto",
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    _ensure_output_path(output_root / "decoys.jsonl", overwrite=overwrite)
    entries = _training_set_entries(training_set)
    if max_entries is not None:
        entries = entries[:max_entries]
    normalized_recipes = _expand_decoy_recipes(recipes)
    domain = DEFAULT_TRAINING_DOMAIN if domain is None else str(domain)
    tasks = [
        (dict(entry), recipe, str(output_root), int(seed), task_index, domain, int(attempts_per_decoy))
        for task_index, (entry, recipe) in enumerate(
            (entry, recipe)
            for entry in entries
            for recipe in normalized_recipes
        )
    ]
    worker_rows = [
        row
        for row in _map_ordered(
            _decoy_worker,
            tasks,
            workers=workers,
            progress=progress,
            progress_label="decoys",
        )
        if row is not None
    ]
    rows = [row for row in worker_rows if row.get("status") == "accepted"]
    rejected_rows = [row for row in worker_rows if row.get("status") != "accepted"]
    decoys_jsonl = output_root / "decoys.jsonl"
    rejections_jsonl = output_root / "rejections.jsonl"
    _write_jsonl(decoys_jsonl, rows)
    _write_jsonl(rejections_jsonl, rejected_rows)
    manifest = {
        "format": "pheat.reference-decoys",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "training_set": str(training_set),
        "domain": domain,
        "seed": int(seed),
        "recipes": normalized_recipes,
        "recipe_profile": _recipe_profile_label(recipes, normalized_recipes),
        "attempts_per_decoy": int(attempts_per_decoy),
        "workers": normalize_workers(workers),
        "native_entry_count": len(entries),
        "requested_decoy_count": len(tasks),
        "decoy_count": len(rows),
        "rejection_count": len(rejected_rows),
        "decoys": str(decoys_jsonl),
        "decoys_sha256": _file_sha256(decoys_jsonl),
        "rejections": str(rejections_jsonl),
        "rejections_sha256": _file_sha256(rejections_jsonl),
        "acceptance": _decoy_acceptance_summary(rows, rejected_rows),
        "lineage": _lineage_payload(stage="build-decoys", parents=[str(training_set)]),
    }
    _write_json(output_root / "manifest.json", manifest)
    return {**manifest, "output_root": str(output_root)}


def run_reference_unattended(
    *,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    snapshot_id: str = DEFAULT_REFERENCE_SNAPSHOT_ID,
    snapshot_root: Optional[Path] = None,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    method: str = DEFAULT_REFERENCE_METHOD,
    max_resolution: Optional[float] = DEFAULT_REFERENCE_MAX_RESOLUTION,
    sequence_identity: Union[str, float] = DEFAULT_REFERENCE_SEQUENCE_IDENTITY,
    min_length: int = 50,
    max_length: int = 800,
    decoy_recipes: Sequence[str] = (DEFAULT_REFERENCE_DECOY_PROFILE,),
    attempts_per_decoy: int = DEFAULT_REFERENCE_DECOY_ATTEMPTS,
    models: Sequence[str] = DEFAULT_TRAINING_MODELS,
    feature_models: Sequence[str] = DEFAULT_REFERENCE_FEATURE_MODELS,
    sasa_backend: str = "auto",
    metadata_source: str = "auto",
    workers: Union[str, int] = "auto",
    seed: int = DEFAULT_REFERENCE_SEED,
    overwrite: bool = False,
    backup_existing: bool = False,
    run_canary: bool = True,
    canary_entries: int = 25,
    dry_run: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run the complete non-interactive reference build workflow.

    The unattended workflow is deliberately just orchestration around the public
    reference-building stages. It writes enough logs and lineage metadata that a
    long run can be reviewed after completion without requiring terminal scrollback.
    """

    reference_root = Path(reference_root)
    snapshot = get_snapshot(snapshot_id)
    resolved_snapshot_root = Path(snapshot_root) if snapshot_root is not None else snapshot.default_output_root
    normalized_recipes = _expand_decoy_recipes(decoy_recipes)
    aqueous_set = f"{domain}-{_sequence_identity_label(sequence_identity)}-xray-aqueous-{artifact_version}"
    membrane_set = f"{domain}-{_sequence_identity_label(sequence_identity)}-xray-membrane-{artifact_version}"
    paths = _unattended_paths(
        reference_root,
        artifact_version=artifact_version,
        snapshot_root=resolved_snapshot_root,
        aqueous_set=aqueous_set,
        membrane_set=membrane_set,
    )
    plan = {
        "format": "pheat.reference-unattended-plan",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "pheat_version": __version__,
        "reference_root": str(reference_root),
        "snapshot_id": snapshot.id,
        "snapshot_root": str(resolved_snapshot_root),
        "domain": domain,
        "method": method,
        "max_resolution": max_resolution,
        "sequence_identity": normalize_sequence_identity_threshold(sequence_identity),
        "min_length": int(min_length),
        "max_length": int(max_length),
        "subsets": ["aqueous", "membrane"],
        "decoy_recipes": normalized_recipes,
        "attempts_per_decoy": int(attempts_per_decoy),
        "models": list(models),
        "feature_models": list(feature_models),
        "sasa_backend": sasa_backend,
        "metadata_source": metadata_source,
        "workers": normalize_workers(workers),
        "seed": int(seed),
        "canary_entries": int(canary_entries),
        "run_canary": bool(run_canary),
        "overwrite": bool(overwrite),
        "backup_existing": bool(backup_existing),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    if dry_run:
        return {**plan, "dry_run": True, "status": "planned"}

    backup = _backup_reference_outputs(reference_root, artifact_version) if backup_existing else None
    reference_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(paths["run_root"])
    log_root = Path(paths["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    logger = _UnattendedReferenceLogger(log_root, progress=progress)
    summary: dict[str, Any] = {
        **plan,
        "format": "pheat.reference-unattended-run",
        "dry_run": False,
        "status": "running",
        "started_at": _utc_now(),
        "backup": backup,
        "stages": {},
    }
    _write_json(run_root / "plan.json", plan)

    def stage(name: str, action: Any) -> dict[str, Any]:
        return _run_unattended_stage(name, action, logger=logger, summary=summary)

    stage(
        "fetch",
        lambda emit: fetch_reference_inputs(
            reference_root=reference_root,
            artifact_version=artifact_version,
            snapshot_id=snapshot.id,
            snapshot_root=resolved_snapshot_root,
            datasets=(),
            overwrite=overwrite,
            workers=workers,
            seed=seed,
        ),
    )
    archive_paths = default_archive_paths(resolved_snapshot_root)
    stage(
        "metadata",
        lambda emit: write_snapshot_metadata(
            snapshot.id,
            paths=archive_paths,
            source=metadata_source,
            output=Path(paths["metadata"]),
            workers=workers,
            overwrite=overwrite,
            status_stream=_LogStatusStream(emit),
        ),
    )
    stage(
        "inventory",
        lambda emit: inventory_reference(
            snapshot_root=resolved_snapshot_root,
            output=Path(paths["inventory"]),
            reference_root=reference_root,
            artifact_version=artifact_version,
            domain=domain,
            metadata_jsonl=Path(paths["metadata"]),
            workers=workers,
            overwrite=overwrite,
            progress=emit,
        ),
    )
    for subset in REFERENCE_SUBSETS:
        stage(
            f"select-{subset}",
            lambda emit, subset=subset: select_reference_corpus(
                inventory=Path(paths["inventory"]),
                output_root=Path(paths[f"set_{subset}"]),
                subset=subset,
                artifact_version=artifact_version,
                domain=domain,
                method=method,
                max_resolution=max_resolution,
                sequence_identity=sequence_identity,
                min_length=min_length,
                max_length=max_length,
                overwrite=overwrite,
            ),
        )

    if run_canary:
        for subset in REFERENCE_SUBSETS:
            canary_root = Path(paths[f"canary_{subset}"])
            stage(
                f"canary-{subset}",
                lambda emit, subset=subset, canary_root=canary_root: _run_reference_canary(
                    subset=subset,
                    training_set=Path(paths[f"set_{subset}"]),
                    canary_root=canary_root,
                    artifact_version=artifact_version,
                    domain=domain,
                    recipes=normalized_recipes,
                    attempts_per_decoy=attempts_per_decoy,
                    feature_models=feature_models,
                    max_entries=canary_entries,
                    workers=workers,
                    seed=seed,
                    overwrite=overwrite,
                    progress=emit,
                ),
            )

    for subset in REFERENCE_SUBSETS:
        stage(
            f"decoys-{subset}",
            lambda emit, subset=subset: build_reference_decoys(
                training_set=Path(paths[f"set_{subset}"]),
                output_root=Path(paths[f"decoys_{subset}"]),
                recipes=normalized_recipes,
                domain=domain,
                artifact_version=artifact_version,
                seed=seed,
                attempts_per_decoy=attempts_per_decoy,
                workers=workers,
                overwrite=overwrite,
                progress=emit,
            ),
        )
        stage(
            f"scores-{subset}",
            lambda emit, subset=subset: build_reference_scores(
                training_set=Path(paths[f"set_{subset}"]),
                output_root=Path(paths[f"tables_{subset}"]),
                artifact_version=artifact_version,
                table_set_id=f"{domain}-{_sequence_identity_label(sequence_identity)}-xray-{subset}",
                models=models,
                domain=domain,
                burial_method="both",
                sasa_backend=sasa_backend,
                workers=workers,
                overwrite=overwrite,
                progress=emit,
            ),
        )
        stage(
            f"features-{subset}",
            lambda emit, subset=subset: extract_reference_features(
                training_set=Path(paths[f"set_{subset}"]),
                output=Path(paths[f"features_{subset}"]),
                decoys=Path(paths[f"decoys_{subset}"]) / "decoys.jsonl",
                models=feature_models,
                domain=domain,
                artifact_version=artifact_version,
                workers=workers,
                overwrite=overwrite,
                progress=emit,
            ),
        )
        stage(
            f"ml-{subset}",
            lambda emit, subset=subset: train_reference_ml(
                features=Path(paths[f"features_{subset}"]),
                output=Path(paths[f"model_{subset}"]),
                artifact_version=artifact_version,
                overwrite=overwrite,
            ),
        )
        stage(
            f"validate-{subset}",
            lambda emit, subset=subset: validate_reference_features(
                features=Path(paths[f"features_{subset}"]),
                output=Path(paths[f"validation_{subset}"]),
                artifact_version=artifact_version,
                overwrite=overwrite,
            ),
        )

    summary["status"] = "completed"
    summary["completed_at"] = _utc_now()
    summary["outputs"] = _unattended_output_summary(paths)
    _write_json(run_root / "summary.json", summary)
    logger.close()
    return {**summary, "summary": str(run_root / "summary.json")}


def build_reference_scores(
    *,
    training_set: Path,
    output_root: Path,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    table_set_id: Optional[str] = None,
    models: Sequence[str] = DEFAULT_TRAINING_MODELS,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    burial_method: str = "both",
    sasa_backend: str = "auto",
    max_entries: Optional[int] = None,
    workers: Union[str, int] = "auto",
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    _ensure_output_path(Path(output_root) / "score-tables.json", overwrite=overwrite)
    result = build_score_tables(
        Path(training_set),
        output_root=Path(output_root),
        models=models,
        domain=domain,
        burial_method=burial_method,
        sasa_backend=sasa_backend,
        max_entries=max_entries,
        table_set_id=table_set_id,
        table_set_version=artifact_version,
        workers=workers,
        progress=progress,
    )
    result["provisional"] = artifact_version == "v0"
    result.setdefault("metadata", {})
    result["metadata"]["workers"] = normalize_workers(workers)
    result["metadata"]["lineage"] = _lineage_payload(stage="build-scores", parents=[str(training_set)])
    write_score_table_set(Path(output_root) / "score-tables.json", result)
    return result


def extract_reference_features(
    *,
    training_set: Path,
    output: Path,
    decoys: Optional[Path] = None,
    models: Sequence[str] = DEFAULT_REFERENCE_FEATURE_MODELS,
    domain: str = DEFAULT_TRAINING_DOMAIN,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    max_entries: Optional[int] = None,
    workers: Union[str, int] = "auto",
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    _ensure_output_path(output, overwrite=overwrite)
    native_samples = _native_feature_samples(training_set)
    if max_entries is not None:
        native_samples = native_samples[:max_entries]
    native_ids = {sample.get("sample_id") for sample in native_samples}
    decoy_samples = _decoy_feature_samples(decoys)
    if max_entries is not None:
        filtered_decoys = [
            sample
            for sample in decoy_samples
            if sample.get("native_reference_id") in native_ids
        ]
        decoy_samples = filtered_decoys if filtered_decoys else decoy_samples[:max_entries]
    samples = native_samples + decoy_samples
    tasks = [(sample, tuple(models), domain) for sample in samples]
    rows = _map_ordered(
        _reference_feature_worker,
        tasks,
        workers=workers,
        progress=progress,
        progress_label="features",
    )
    _write_jsonl(output, rows)
    manifest = {
        "format": "pheat.reference-features",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "training_set": str(training_set),
        "decoys": str(decoys) if decoys else None,
        "models": list(models),
        "domain": domain,
        "max_entries": max_entries,
        "native_sample_count": len(native_samples),
        "decoy_sample_count": len(decoy_samples),
        "workers": normalize_workers(workers),
        "feature_count": len(rows),
        "features": str(output),
        "features_sha256": _file_sha256(output),
        "lineage": _lineage_payload(stage="extract-features", parents=[str(training_set), str(decoys) if decoys else None]),
    }
    manifest_path = Path(output).with_suffix(Path(output).suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    return {**manifest, "output": str(output), "manifest": str(manifest_path)}


def train_reference_ml(
    *,
    features: Path,
    output: Path,
    target: str = "native_vs_decoy",
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    overwrite: bool = False,
) -> dict[str, Any]:
    _ensure_output_path(output, overwrite=overwrite)
    result = train_linear_model(Path(features), output=Path(output), target=target)
    result["artifact_version"] = artifact_version
    result["artifact_label"] = f"pheat-ml-linear-{artifact_version}"
    result["provisional"] = artifact_version == "v0"
    result.setdefault("metadata", {})
    result["metadata"]["lineage"] = _lineage_payload(stage="train-ml", parents=[str(features)])
    write_score_table_set(output, result)
    return result


def validate_reference_features(
    *,
    features: Path,
    output: Path,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    overwrite: bool = False,
) -> dict[str, Any]:
    _ensure_output_path(output, overwrite=overwrite)
    rows = _read_jsonl(features)
    report = {
        "format": "pheat.reference-validation",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "features": str(features),
        "features_sha256": _file_sha256(features),
        "sample_count": len(rows),
        "labels": _label_counts(rows),
        "metrics": _feature_validation_metrics(rows),
        "failure_counts": _failure_counts(rows),
        "lineage": _lineage_payload(stage="validate", parents=[str(features)]),
    }
    _write_json(output, report)
    return {**report, "output": str(output)}


def promote_reference_artifact(
    *,
    source: Path,
    destination: Path,
    note: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not note.strip():
        raise ValueError("promotion requires a non-empty promotion note")
    source = Path(source)
    destination = Path(destination)
    if _contains_unpackageable_payload(source):
        raise ValueError("source contains local-use-only or unpackageable payload metadata; promotion is blocked")
    _ensure_output_path(destination, overwrite=overwrite)
    if source.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        checksum = _tree_sha256(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksum = _file_sha256(destination)
    promotion = {
        "format": "pheat.reference-promotion",
        "version": 1,
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "source": str(source),
        "destination": str(destination),
        "destination_sha256": checksum,
        "promotion_note": note,
    }
    promotion_path = destination / "promotion.json" if destination.is_dir() else destination.with_suffix(destination.suffix + ".promotion.json")
    _write_json(promotion_path, promotion)
    return {**promotion, "promotion_manifest": str(promotion_path)}


def package_reference_scoring_assets(
    *,
    reference_root: Path,
    destination_root: Path,
    artifact_version: str = DEFAULT_REFERENCE_ARTIFACT_VERSION,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference_root = Path(reference_root)
    destination_root = Path(destination_root)
    manifest_path = destination_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {manifest_path}")

    assets: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for spec in _reference_scoring_asset_specs(artifact_version):
        source = reference_root / spec["source"]
        if not source.exists():
            skipped.append({"asset_id": spec["asset_id"], "source": _reference_archive_path(source, reference_root)})
            continue
        loaded_payload = load_score_table_set(source)
        if loaded_payload is None:
            raise ValueError(f"{source} did not contain a score table set")
        payload = copy.deepcopy(loaded_payload)
        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source} has non-object metadata")
        metadata["packaged_asset_id"] = spec["asset_id"]
        metadata["packaged_artifact_version"] = artifact_version
        metadata["source_run_artifact_version"] = artifact_version
        metadata["source_run_summary"] = _reference_archive_path(
            reference_root / "runs" / artifact_version / "summary.json",
            reference_root,
        )
        payload["artifact_version"] = artifact_version
        payload["provisional"] = artifact_version == "v0"
        raw = score_table_set_json_bytes(payload)
        if _contains_local_absolute_path(json.dumps(payload, sort_keys=True)):
            raise ValueError(f"{source} contains local absolute paths; refusing to package")

        destination = destination_root / spec["destination"]
        if destination.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {destination}")
        write_score_table_set(destination, payload)
        compressed = destination.read_bytes()
        source_raw = source.read_bytes()
        assets[spec["asset_id"]] = {
            "bytes": len(compressed),
            "compression": "xz",
            "compression_preset": SCORE_TABLE_XZ_PRESET,
            "kind": spec["kind"],
            "path": spec["destination"],
            "sha256": _bytes_sha256(compressed),
            "source_bytes": len(source_raw),
            "source_path": _reference_archive_path(source, reference_root),
            "source_sha256": _bytes_sha256(source_raw),
            "subset": spec["subset"],
            "uncompressed_bytes": len(raw),
            "uncompressed_sha256": _bytes_sha256(raw),
        }

    if not assets:
        raise ValueError("no reference scoring assets were found to package")

    manifest = _packaged_scoring_manifest_template(
        reference_root=reference_root,
        destination_root=destination_root,
        artifact_version=artifact_version,
    )
    manifest["assets"] = assets
    manifest["skipped_assets"] = skipped
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def audit_reference_artifact_version(
    *,
    reference_root: Path,
    artifact_version: str,
) -> dict[str, Any]:
    """Audit a reference archive for active-path and manifest version consistency.

    This intentionally validates metadata and manifests, not every large JSONL
    payload. It is meant for explicit local archive checks and for small fixture
    tests; standard package tests should not depend on a real external archive.
    """

    root = Path(reference_root)
    expected = str(artifact_version)
    issues: list[dict[str, Any]] = []
    checked_paths: list[str] = []
    checked_files: list[str] = []

    if not root.exists():
        issues.append(
            {
                "path": str(root),
                "message": "reference root does not exist",
            }
        )
        return _reference_audit_result(root, expected, checked_paths, checked_files, issues)
    if not root.is_dir():
        issues.append(
            {
                "path": str(root),
                "message": "reference root is not a directory",
            }
        )
        return _reference_audit_result(root, expected, checked_paths, checked_files, issues)

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _skip_reference_audit_path(rel):
            continue
        rel_text = rel.as_posix()
        checked_paths.append(rel_text)
        for token in _artifact_version_tokens(rel_text):
            if token != expected:
                issues.append(
                    {
                        "path": rel_text,
                        "message": "path contains a different artifact version",
                        "expected": expected,
                        "observed": token,
                    }
                )
        if path.is_file() and _should_audit_json_payload(path):
            checked_files.append(rel_text)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "path": rel_text,
                        "message": "JSON payload could not be decoded",
                        "error": str(exc),
                    }
                )
                continue
            _audit_reference_payload_versions(
                payload,
                expected=expected,
                issues=issues,
                path=rel_text,
                key_path=(),
            )

    return _reference_audit_result(root, expected, checked_paths, checked_files, issues)


def _reference_audit_result(
    root: Path,
    artifact_version: str,
    checked_paths: Sequence[str],
    checked_files: Sequence[str],
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "pheat.reference-version-audit",
        "version": 1,
        "created_at": _utc_now(),
        "pheat_version": __version__,
        "reference_root": str(root),
        "artifact_version": artifact_version,
        "ok": not issues,
        "checked_path_count": len(checked_paths),
        "checked_file_count": len(checked_files),
        "checked_paths": list(checked_paths),
        "checked_files": list(checked_files),
        "issue_count": len(issues),
        "issues": [dict(issue) for issue in issues],
    }


def _skip_reference_audit_path(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in {"backups", "logs"}:
        return True
    return "structures" in parts


def _should_audit_json_payload(path: Path) -> bool:
    name = path.name
    if not (name.endswith(".json") or name.endswith(".manifest.json")):
        return False
    # Large line-oriented artifacts are intentionally excluded from the audit.
    if name.endswith(".jsonl"):
        return False
    return True


def _artifact_version_tokens(value: str) -> list[str]:
    return REFERENCE_VERSION_TOKEN_RE.findall(value)


def _audit_reference_payload_versions(
    value: Any,
    *,
    expected: str,
    issues: list[dict[str, Any]],
    path: str,
    key_path: Sequence[str],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "backup":
                continue
            next_path = tuple(key_path) + (str(key),)
            if str(key) in REFERENCE_ARTIFACT_VERSION_KEYS and item != expected:
                issues.append(
                    {
                        "path": path,
                        "field": ".".join(next_path),
                        "message": "artifact version field does not match requested version",
                        "expected": expected,
                        "observed": item,
                    }
                )
            _audit_reference_payload_versions(
                item,
                expected=expected,
                issues=issues,
                path=path,
                key_path=next_path,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _audit_reference_payload_versions(
                item,
                expected=expected,
                issues=issues,
                path=path,
                key_path=tuple(key_path) + (str(index),),
            )
    elif isinstance(value, str) and key_path:
        key = key_path[-1]
        parent = key_path[-2] if len(key_path) > 1 else ""
        is_path_list_item = key.isdigit() and parent in {"parents"}
        if key in REFERENCE_ARTIFACT_LABEL_KEYS or key in REFERENCE_ARTIFACT_PATH_KEYS or is_path_list_item:
            for token in _artifact_version_tokens(value):
                if token != expected:
                    issues.append(
                        {
                            "path": path,
                            "field": ".".join(key_path),
                            "message": "artifact label or path field contains a different artifact version",
                            "expected": expected,
                            "observed": token,
                            "value": value,
                        }
                    )


def _reference_scoring_asset_specs(artifact_version: str) -> list[dict[str, str]]:
    aqueous = f"protein-heavy-30id-xray-aqueous-{artifact_version}"
    membrane = f"protein-heavy-30id-xray-membrane-{artifact_version}"
    return [
        {
            "asset_id": f"pheat-ml-linear-aqueous-{artifact_version}",
            "destination": "pheat-ml-linear-aqueous.json.xz",
            "kind": "ml-linear-model",
            "source": f"models/{artifact_version}/pheat-ml-linear-aqueous.json",
            "subset": "aqueous",
        },
        {
            "asset_id": f"pheat-ml-linear-membrane-{artifact_version}",
            "destination": "pheat-ml-linear-membrane.json.xz",
            "kind": "ml-linear-model",
            "source": f"models/{artifact_version}/pheat-ml-linear-membrane.json",
            "subset": "membrane",
        },
        {
            "asset_id": aqueous,
            "destination": "protein-heavy-30id-xray-aqueous/score-tables.json.xz",
            "kind": "score-table-set",
            "source": f"tables/{aqueous}/score-tables.json",
            "subset": "aqueous",
        },
        {
            "asset_id": membrane,
            "destination": "protein-heavy-30id-xray-membrane/score-tables.json.xz",
            "kind": "score-table-set",
            "source": f"tables/{membrane}/score-tables.json",
            "subset": "membrane",
        },
    ]


def _packaged_scoring_manifest_template(
    *,
    reference_root: Path,
    destination_root: Path,
    artifact_version: str,
) -> dict[str, Any]:
    existing = destination_root / "manifest.json"
    if existing.exists():
        try:
            payload = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload.get("format") == "pheat.packaged-score-assets":
            manifest = copy.deepcopy(payload)
        else:
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "artifact_version": artifact_version,
            "created_at": _utc_now(),
            "format": "pheat.packaged-score-assets",
            "pheat_version": __version__,
            "provisional": artifact_version == "v0",
            "status": "initial" if artifact_version == "v0" else "generated",
            "version": 1,
        }
    )
    manifest.setdefault("default_ml_model", f"pheat-ml-linear-aqueous-{artifact_version}")
    manifest.setdefault("default_score_table", f"protein-heavy-30id-xray-aqueous-{artifact_version}")
    manifest["notes"] = [
        "Packaged score/model JSON is committed as xz-compressed JSON using Python's standard-library lzma module.",
        "Large reference snapshot, inventory, decoy, feature, and log artifacts remain external archive outputs.",
        "These are initial v0 PHEAT-owned scoring assets; broader manual and scientific vetting is still expected before a later stable version.",
        "Backbone and rotamer validation AUCs are diagnostic signals in this build and should not be treated as standalone rankers.",
    ]
    summary_path = reference_root / "runs" / artifact_version / "summary.json"
    if summary_path.exists():
        manifest.setdefault(
            "source_run",
            {
                "packaged_as_artifact_version": artifact_version,
                "source_artifact_version": artifact_version,
                "summary": _reference_archive_path(summary_path, reference_root),
            },
        )
    return manifest


def _reference_archive_path(path: Path, reference_root: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(Path(reference_root).resolve())
    except ValueError:
        return str(path)
    return str(Path(reference_root).name / rel)


def _contains_local_absolute_path(value: str) -> bool:
    markers = (
        "/".join(("", "Users", "")),
        "/".join(("", "private", "")),
        "/".join(("", "mnt", "")),
    )
    return any(marker in value for marker in markers)


def _bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _run_reference_canary(
    *,
    subset: str,
    training_set: Path,
    canary_root: Path,
    artifact_version: str,
    domain: str,
    recipes: Sequence[str],
    attempts_per_decoy: int,
    feature_models: Sequence[str],
    max_entries: int,
    workers: Union[str, int],
    seed: int,
    overwrite: bool,
    progress: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    canary_root = Path(canary_root)
    selected_entries = _training_set_entries(training_set)
    if not selected_entries:
        return {
            "subset": subset,
            "max_entries": max_entries,
            "status": "skipped",
            "reason": "selected corpus is empty",
        }
    decoy_root = canary_root / "decoys"
    features_path = canary_root / "features.jsonl"
    validation_path = canary_root / "validation.json"
    decoys = build_reference_decoys(
        training_set=training_set,
        output_root=decoy_root,
        recipes=recipes,
        domain=domain,
        artifact_version=artifact_version,
        seed=seed,
        max_entries=max_entries,
        attempts_per_decoy=attempts_per_decoy,
        workers=workers,
        overwrite=overwrite,
        progress=progress,
    )
    features = extract_reference_features(
        training_set=training_set,
        output=features_path,
        decoys=decoy_root / "decoys.jsonl",
        models=feature_models,
        domain=domain,
        artifact_version=artifact_version,
        max_entries=max_entries,
        workers=workers,
        overwrite=overwrite,
        progress=progress,
    )
    validation = validate_reference_features(
        features=features_path,
        output=validation_path,
        artifact_version=artifact_version,
        overwrite=overwrite,
    )
    if int(decoys.get("decoy_count", 0)) <= 0:
        raise ValueError(f"{subset} canary produced no accepted decoys")
    if int(features.get("feature_count", 0)) <= 0:
        raise ValueError(f"{subset} canary produced no feature rows")
    return {
        "subset": subset,
        "max_entries": max_entries,
        "decoys": decoys,
        "features": features,
        "validation": validation,
    }


def _run_unattended_stage(
    name: str,
    action: Any,
    *,
    logger: "_UnattendedReferenceLogger",
    summary: dict[str, Any],
) -> dict[str, Any]:
    started = _utc_now()
    logger.stage(name, f"starting {name}")
    try:
        result = action(lambda message: logger.stage(name, message))
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_stage"] = name
        summary["failed_at"] = _utc_now()
        summary["stages"][name] = {
            "status": "failed",
            "started_at": started,
            "completed_at": _utc_now(),
            "error": str(exc),
        }
        logger.stage(name, f"failed {name}: {exc}")
        raise
    completed = _utc_now()
    summary["stages"][name] = {
        "status": "completed",
        "started_at": started,
        "completed_at": completed,
        "result": _stage_result_summary(result),
    }
    logger.stage(name, f"completed {name}")
    return result


def _stage_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "output",
        "output_root",
        "summary",
        "entry_count",
        "row_count",
        "selected_count",
        "holdout_count",
        "decoy_count",
        "rejection_count",
        "feature_count",
        "sample_count",
        "failure_count",
        "metadata_jsonl",
        "decoys",
        "features",
    }
    return {key: value for key, value in result.items() if key in keep}


class _UnattendedReferenceLogger:
    def __init__(self, log_root: Path, *, progress: Optional[Callable[[str], None]]) -> None:
        self.log_root = Path(log_root)
        self.progress = progress
        self.main_log = self.log_root / "run.log"
        self.main_log.parent.mkdir(parents=True, exist_ok=True)

    def stage(self, stage: str, message: str) -> None:
        line = f"{_utc_now()} [{stage}] {message}"
        with self.main_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        stage_path = self.log_root / f"{_safe_label(stage)}.log"
        with stage_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.progress is not None:
            self.progress(line)

    def close(self) -> None:
        return None


class _LogStatusStream:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.emit(line.strip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.emit(self._buffer.strip())
        self._buffer = ""


def _unattended_paths(
    reference_root: Path,
    *,
    artifact_version: str,
    snapshot_root: Path,
    aqueous_set: str,
    membrane_set: str,
) -> dict[str, Path]:
    run_root = Path(reference_root) / "runs" / artifact_version
    return {
        "run_root": run_root,
        "log_root": run_root / "logs",
        "metadata": Path(snapshot_root) / "manifests" / "metadata.jsonl",
        "inventory": Path(reference_root) / "inventories" / artifact_version / "inventory.jsonl",
        "set_aqueous": Path(reference_root) / "sets" / aqueous_set,
        "set_membrane": Path(reference_root) / "sets" / membrane_set,
        "canary_aqueous": Path(reference_root) / "canary" / artifact_version / "aqueous",
        "canary_membrane": Path(reference_root) / "canary" / artifact_version / "membrane",
        "decoys_aqueous": Path(reference_root) / "decoys" / f"{aqueous_set}-pheat-torsion",
        "decoys_membrane": Path(reference_root) / "decoys" / f"{membrane_set}-pheat-torsion",
        "tables_aqueous": Path(reference_root) / "tables" / aqueous_set,
        "tables_membrane": Path(reference_root) / "tables" / membrane_set,
        "features_aqueous": Path(reference_root) / "features" / artifact_version / "aqueous-features.jsonl",
        "features_membrane": Path(reference_root) / "features" / artifact_version / "membrane-features.jsonl",
        "model_aqueous": Path(reference_root) / "models" / artifact_version / "pheat-ml-linear-aqueous.json",
        "model_membrane": Path(reference_root) / "models" / artifact_version / "pheat-ml-linear-membrane.json",
        "validation_aqueous": Path(reference_root) / "validation" / artifact_version / "aqueous-validation.json",
        "validation_membrane": Path(reference_root) / "validation" / artifact_version / "membrane-validation.json",
    }


def _unattended_output_summary(paths: Mapping[str, Path]) -> dict[str, str]:
    keys = (
        "inventory",
        "set_aqueous",
        "set_membrane",
        "decoys_aqueous",
        "decoys_membrane",
        "tables_aqueous",
        "tables_membrane",
        "features_aqueous",
        "features_membrane",
        "model_aqueous",
        "model_membrane",
        "validation_aqueous",
        "validation_membrane",
    )
    return {key: str(paths[key]) for key in keys}


def _backup_reference_outputs(reference_root: Path, artifact_version: str) -> Optional[dict[str, Any]]:
    reference_root = Path(reference_root)
    if not reference_root.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = reference_root / "backups" / f"{artifact_version}-{timestamp}"
    moved = []
    candidates = []
    for dirname in ("fetches", "inventories", "features", "models", "validation", "runs", "canary"):
        candidates.append(reference_root / dirname / artifact_version)
    for dirname in ("sets", "decoys", "tables"):
        parent = reference_root / dirname
        if parent.exists():
            candidates.extend(path for path in parent.iterdir() if path.name.endswith(f"-{artifact_version}") or artifact_version in path.name)
    for source in sorted({path for path in candidates if path.exists()}):
        destination = backup_root / source.relative_to(reference_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({"source": str(source), "destination": str(destination)})
    if not moved:
        return None
    manifest = {
        "format": "pheat.reference-backup",
        "version": 1,
        "artifact_version": artifact_version,
        "created_at": _utc_now(),
        "moved": moved,
    }
    _write_json(backup_root / "manifest.json", manifest)
    return {**manifest, "backup_root": str(backup_root)}


def _write_reference_dataset_manifest(
    reference_root: Path,
    dataset_id: str,
    *,
    artifact_version: str,
    include_payloads: bool,
    payload_urls: Sequence[str],
    local_files: Sequence[Path],
    local_dirs: Sequence[Path],
    overwrite: bool,
) -> dict[str, Any]:
    dataset = get_reference_decoy_dataset(dataset_id)
    dataset_root = Path(reference_root) / "datasets" / dataset["id"] / artifact_version
    manifest_path = dataset_root / "manifest.json"
    _ensure_output_path(manifest_path, overwrite=overwrite)
    dataset_root.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for path in local_files:
        payload_rows.append(_registered_payload(Path(path), dataset_root=dataset_root, source_kind="local-file"))
    for directory in local_dirs:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                payload_rows.append(_registered_payload(path, dataset_root=dataset_root, source_kind="local-dir"))
    if include_payloads:
        for url in payload_urls:
            payload_rows.append(_download_payload(url, dataset_root=dataset_root))
    status = "registered-local" if payload_rows else "metadata-only"
    if include_payloads and not payload_rows:
        status = "manual-acquisition-required"
    payload = {
        "format": "pheat.reference-decoy-dataset",
        "version": 1,
        "artifact_version": artifact_version,
        "provisional": artifact_version == "v0",
        "created_at": _utc_now(),
        "dataset": dataset,
        "status": status,
        "include_payloads": bool(include_payloads),
        "payloads": payload_rows,
        "local_use_only": bool(dataset.get("local_use_only", True)),
        "packageable": bool(dataset.get("packageable", False)) and all(row.get("packageable", False) for row in payload_rows),
        "manual_acquisition_required": status == "manual-acquisition-required",
    }
    _write_json(manifest_path, payload)
    return {**payload, "manifest": str(manifest_path)}


def _registered_payload(path: Path, *, dataset_root: Path, source_kind: str) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "path": _relative_or_absolute(path, dataset_root),
        "absolute_path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "registered_at": _utc_now(),
        "local_use_only": True,
        "packageable": False,
    }


def _download_payload(url: str, *, dataset_root: Path) -> dict[str, Any]:
    payload_dir = dataset_root / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = Path(parsed.path).name or hashlib.sha256(url.encode("utf-8")).hexdigest()
    destination = payload_dir / filename
    urlretrieve(url, destination)
    return {
        "source_kind": "download-url",
        "url": url,
        "path": str(destination.relative_to(dataset_root)),
        "sha256": _file_sha256(destination),
        "bytes": destination.stat().st_size,
        "downloaded_at": _utc_now(),
        "local_use_only": True,
        "packageable": False,
    }


def _filter_subset_rows(rows: Sequence[Mapping[str, Any]], subset: str) -> tuple[list[dict[str, Any]], list[str]]:
    output = []
    unknown = 0
    for row in rows:
        environment = row.get("environment")
        environment = environment if isinstance(environment, Mapping) else {}
        if environment:
            membrane = bool(environment.get("membrane"))
            aqueous = bool(environment.get("aqueous_like"))
        else:
            membrane = _row_mentions(row, "membrane")
            aqueous = _row_mentions(row, "aqueous") or _row_mentions(row, "water")
        if subset == "membrane":
            if membrane:
                output.append(dict(row))
            continue
        if membrane:
            continue
        if not aqueous:
            unknown += 1
        output.append(dict(row))
    warnings = []
    if subset == "aqueous" and unknown:
        warnings.append(f"{unknown} entries had unknown aqueous metadata and were retained for prototype selection")
    return output, warnings


def _row_mentions(row: Mapping[str, Any], needle: str) -> bool:
    text = json.dumps(row, sort_keys=True).lower()
    return needle.lower() in text


def _native_feature_samples(training_set: Path) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": _sample_id("native", entry),
            "label": "native",
            "source_type": "native",
            "source_corpus": str(training_set),
            "structure_path": str(entry.get("path") or ""),
            "pdb_id": entry.get("pdb_id"),
            "chain_id": entry.get("chain_id"),
            "split": entry.get("split", "train"),
        }
        for entry in _training_set_entries(training_set)
    ]


def _decoy_feature_samples(decoys: Optional[Path]) -> list[dict[str, Any]]:
    if decoys is None or not Path(decoys).exists():
        return []
    return [dict(row) for row in _read_jsonl(Path(decoys))]


def _reference_feature_worker(task: tuple[dict[str, Any], tuple[str, ...], str]) -> dict[str, Any]:
    from pheat.scoring import score_structure

    sample, models, domain = task
    features: dict[str, float] = {}
    errors: dict[str, str] = {}
    path = Path(str(sample.get("structure_path") or sample.get("path") or ""))
    if not path.exists():
        errors["structure"] = f"missing structure: {path}"
    else:
        try:
            if sample.get("source_type") == "native":
                structure = load_training_entry_structure(sample, domain=domain)
            else:
                structure = load_training_structure(path)
            for model in models:
                try:
                    result = score_structure(structure, model=model, domain=domain)
                    features[f"{model}_total"] = float(result.total)
                except Exception as exc:
                    errors[model] = str(exc)
        except Exception as exc:
            errors["structure"] = str(exc)
    return {
        "format": "pheat.reference-feature-row",
        "version": 1,
        "sample_id": sample.get("sample_id"),
        "label": sample.get("label"),
        "split": sample.get("split", "train"),
        "source_type": sample.get("source_type"),
        "source_corpus": sample.get("source_corpus"),
        "structure_path": str(path),
        "atom_count": len(structure.atoms) if "structure" in locals() else None,
        "residue_count": len(structure.atoms_by_residue()) if "structure" in locals() else None,
        "native_reference_id": sample.get("native_reference_id"),
        "features": features,
        "errors": errors,
        "provenance": {
            "pdb_id": sample.get("pdb_id"),
            "chain_id": sample.get("chain_id"),
            "recipe": sample.get("recipe"),
        },
    }


def _decoy_worker(task: tuple[dict[str, Any], str, str, int, int, str, int]) -> Optional[dict[str, Any]]:
    entry, recipe, output_root_text, seed, task_index, domain, attempts_per_decoy = task
    path = Path(str(entry.get("path") or ""))
    sample_id = _sample_id(recipe, entry)
    output_root = Path(output_root_text)
    if not path.exists():
        return _rejected_decoy_row(
            entry,
            recipe=recipe,
            sample_id=sample_id,
            path=path,
            domain=domain,
            reason=f"missing selected structure: {path}",
        )
    try:
        structure = load_training_entry_structure(entry, domain=domain)
        decoy, metrics, attempts = _accepted_decoy_for_recipe(
            structure,
            recipe=recipe,
            seed=seed + task_index * 1009 + _stable_int(recipe),
            attempts_per_decoy=attempts_per_decoy,
        )
    except Exception as exc:
        return _rejected_decoy_row(
            entry,
            recipe=recipe,
            sample_id=sample_id,
            path=path,
            domain=domain,
            reason=str(exc),
        )
    decoy_path = output_root / "structures" / recipe / f"{sample_id}.json"
    decoy_path.parent.mkdir(parents=True, exist_ok=True)
    write_structure_json(decoy, decoy_path)
    return {
        "format": "pheat.reference-decoy-row",
        "version": 1,
        "sample_id": sample_id,
        "status": "accepted",
        "label": "pheat-decoy",
        "source_type": "pheat-decoy",
        "source_corpus": str(path),
        "structure_path": str(decoy_path),
        "native_reference_id": _sample_id("native", entry),
        "pdb_id": entry.get("pdb_id"),
        "chain_id": entry.get("chain_id"),
        "domain": domain,
        "atom_count": len(decoy.atoms),
        "residue_count": len(decoy.atoms_by_residue()),
        "recipe": recipe,
        "split": entry.get("split", "train"),
        "attempts": attempts,
        "quality": metrics,
        "sha256": _file_sha256(decoy_path),
    }


def _accepted_decoy_for_recipe(
    structure: HeavyAtomStructure,
    *,
    recipe: str,
    seed: int,
    attempts_per_decoy: int,
) -> tuple[HeavyAtomStructure, dict[str, Any], int]:
    attempts = max(1, int(attempts_per_decoy))
    best: Optional[tuple[HeavyAtomStructure, dict[str, Any], int]] = None
    for attempt in range(1, attempts + 1):
        decoy = _perturb_structure(structure, recipe=recipe, seed=seed + attempt)
        try:
            aligned = align_structure_to_reference(structure, decoy, atom_set="ca")["aligned_target"]
            decoy = aligned
        except Exception:
            pass
        metrics = _decoy_quality_metrics(structure, decoy, recipe=recipe)
        if best is None or _decoy_quality_sort_key(metrics) < _decoy_quality_sort_key(best[1]):
            best = (decoy, metrics, attempt)
        accepted, reason = _decoy_acceptance(recipe, metrics)
        metrics["accepted"] = accepted
        metrics["acceptance_reason"] = reason
        if accepted:
            decoy.metadata = {
                **dict(decoy.metadata),
                "decoy_recipe": recipe,
                "decoy_seed": seed,
                "decoy_attempt": attempt,
                "decoy_quality": metrics,
            }
            return decoy, metrics, attempt
    if best is None:  # pragma: no cover - attempts is forced to at least one
        raise ValueError("no decoy attempt was generated")
    best_decoy, best_metrics, best_attempt = best
    accepted, reason = _decoy_acceptance(recipe, best_metrics)
    best_metrics["accepted"] = accepted
    best_metrics["acceptance_reason"] = reason
    raise ValueError(f"no accepted {recipe} decoy after {attempts} attempts; best: {reason}")


def _perturb_structure(structure: HeavyAtomStructure, *, recipe: str, seed: int) -> HeavyAtomStructure:
    if recipe in {"backbone-noise-small", "backbone-noise-medium", "chi-rotamer-random", "compactness-perturb", "mixed"}:
        return _cartesian_perturb_structure(structure, recipe=recipe, seed=seed)
    return _torsion_perturb_structure(structure, recipe=recipe, seed=seed)


def _cartesian_perturb_structure(structure: HeavyAtomStructure, *, recipe: str, seed: int) -> HeavyAtomStructure:
    decoy = copy.deepcopy(structure)
    rng = random.Random(seed)
    coords = [atom.coord for atom in decoy.atoms]
    center = (
        sum(coord[0] for coord in coords) / len(coords),
        sum(coord[1] for coord in coords) / len(coords),
        sum(coord[2] for coord in coords) / len(coords),
    ) if coords else (0.0, 0.0, 0.0)
    sigma = {
        "backbone-noise-small": 0.15,
        "backbone-noise-medium": 0.75,
        "chi-rotamer-random": 0.85,
        "compactness-perturb": 0.05,
        "mixed": 0.35,
    }[recipe]
    scale = 1.08 if recipe in {"compactness-perturb", "mixed"} else 1.0
    for atom in decoy.atoms:
        atom_name = atom.name.strip().upper()
        if recipe == "chi-rotamer-random" and atom_name in {"N", "CA", "C", "O", "OXT"}:
            continue
        atom.x = center[0] + (atom.x - center[0]) * scale + rng.gauss(0.0, sigma)
        atom.y = center[1] + (atom.y - center[1]) * scale + rng.gauss(0.0, sigma)
        atom.z = center[2] + (atom.z - center[2]) * scale + rng.gauss(0.0, sigma)
    decoy.name = f"{structure.name}:{recipe}:decoy"
    decoy.metadata = {
        **dict(decoy.metadata),
        "decoy_recipe": recipe,
        "decoy_seed": seed,
        "decoy_generation": "cartesian-coordinate-perturbation",
    }
    return decoy


def _torsion_perturb_structure(structure: HeavyAtomStructure, *, recipe: str, seed: int) -> HeavyAtomStructure:
    rng = random.Random(seed)
    residue_geometry = structure_to_residue_geometry(
        structure,
        angle_units="radians",
        stored_angles="all",
        stored_lengths="all",
    )
    residues = residue_geometry.residues
    selected_indices = _selected_torsion_indices(recipe, len(residues), rng)
    config = _torsion_recipe_config(recipe)
    perturbations: list[dict[str, Any]] = []
    for index, residue in enumerate(residues):
        if index not in selected_indices:
            continue
        if residue.phi is not None and config["backbone_sigma"] > 0.0:
            delta = rng.gauss(0.0, config["backbone_sigma"])
            residue.phi = _wrap_radians(residue.phi + delta)
            perturbations.append({"residue_index": index, "angle": "phi", "delta_radians": delta})
        if residue.psi is not None and config["backbone_sigma"] > 0.0:
            delta = rng.gauss(0.0, config["backbone_sigma"])
            residue.psi = _wrap_radians(residue.psi + delta)
            perturbations.append({"residue_index": index, "angle": "psi", "delta_radians": delta})
        if residue.chi and config["chi_sigma"] > 0.0:
            updated_chi = []
            for chi_index, chi in enumerate(residue.chi, start=1):
                delta = rng.gauss(0.0, config["chi_sigma"])
                if recipe == "sidechain-repack":
                    delta += rng.choice((-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0))
                updated_chi.append(_wrap_radians(chi + delta))
                perturbations.append(
                    {
                        "residue_index": index,
                        "angle": f"chi{chi_index}",
                        "delta_radians": delta,
                    }
                )
            residue.chi = updated_chi

    decoy = structure_from_residue_geometry(residue_geometry, name=f"{structure.name}:{recipe}:decoy")
    if config["radial_scale"] != 1.0:
        _scale_structure_about_centroid(decoy, config["radial_scale"])
    decoy.metadata = {
        **dict(decoy.metadata),
        "decoy_recipe": recipe,
        "decoy_seed": seed,
        "decoy_generation": "pheat-residue-geometry-torsion-perturbation",
        "decoy_selected_residue_count": len(selected_indices),
        "decoy_perturbations": perturbations[:200],
        "decoy_perturbation_count": len(perturbations),
    }
    return decoy


def _rejected_decoy_row(
    entry: Mapping[str, Any],
    *,
    recipe: str,
    sample_id: str,
    path: Path,
    domain: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "format": "pheat.reference-decoy-row",
        "version": 1,
        "sample_id": sample_id,
        "status": "rejected",
        "label": "pheat-decoy",
        "source_type": "pheat-decoy",
        "source_corpus": str(path),
        "structure_path": None,
        "native_reference_id": _sample_id("native", entry),
        "pdb_id": entry.get("pdb_id"),
        "chain_id": entry.get("chain_id"),
        "domain": domain,
        "recipe": recipe,
        "split": entry.get("split", "train"),
        "reject_reason": reason,
    }


def _decoy_quality_metrics(
    native: HeavyAtomStructure,
    decoy: HeavyAtomStructure,
    *,
    recipe: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"recipe": recipe}
    try:
        metrics["all_heavy_rmsd"] = float(structure_rmsd(native, decoy, atom_set="all-heavy")["value"])
    except Exception as exc:
        metrics["all_heavy_rmsd_error"] = str(exc)
    try:
        metrics["ca_rmsd"] = float(structure_rmsd(native, decoy, atom_set="ca")["value"])
    except Exception as exc:
        metrics["ca_rmsd_error"] = str(exc)
    try:
        native_rg = structure_radius_of_gyration(native, mode="unweighted", atom_set="ca")
        decoy_rg = structure_radius_of_gyration(decoy, mode="unweighted", atom_set="ca")
        metrics["native_rg"] = float(native_rg["values"]["unweighted"])
        metrics["decoy_rg"] = float(decoy_rg["values"]["unweighted"])
        metrics["rg_ratio"] = metrics["decoy_rg"] / metrics["native_rg"] if metrics["native_rg"] else None
        metrics["rg_delta"] = radius_of_gyration_delta(native_rg, decoy_rg)["values"]
    except Exception as exc:
        metrics["rg_error"] = str(exc)
    try:
        integrity = score_structure(decoy, model="pheat-geometry-integrity")
        metrics["geometry_integrity"] = float(integrity.total)
        metrics["geometry_integrity_terms"] = dict(integrity.terms)
        metrics["geometry_integrity_warnings"] = list(integrity.warnings)
    except Exception as exc:
        metrics["geometry_integrity_error"] = str(exc)
    try:
        generic = score_structure(decoy, model="generic")
        metrics["generic_total"] = float(generic.total)
    except Exception as exc:
        metrics["generic_error"] = str(exc)
    return metrics


def _decoy_acceptance(recipe: str, metrics: Mapping[str, Any]) -> tuple[bool, str]:
    if "ca_rmsd" not in metrics:
        return False, str(metrics.get("ca_rmsd_error") or "missing C-alpha RMSD")
    if "all_heavy_rmsd" not in metrics:
        return False, str(metrics.get("all_heavy_rmsd_error") or "missing all-heavy RMSD")
    ca_rmsd = float(metrics["ca_rmsd"])
    rg_ratio = metrics.get("rg_ratio")
    if rg_ratio is not None and not (0.65 <= float(rg_ratio) <= 1.50):
        return False, f"Rg ratio outside accepted range: {rg_ratio:.3f}"
    geometry_integrity = float(metrics.get("geometry_integrity") or 0.0)
    if geometry_integrity > _geometry_integrity_acceptance_limit(recipe):
        return False, f"geometry-integrity score too high: {geometry_integrity:.3f}"
    minimum, maximum = _recipe_ca_rmsd_band(recipe)
    if ca_rmsd < minimum:
        return False, f"C-alpha RMSD {ca_rmsd:.3f} below recipe band {minimum:.3f}-{maximum:.3f}"
    if ca_rmsd > maximum:
        return False, f"C-alpha RMSD {ca_rmsd:.3f} above recipe band {minimum:.3f}-{maximum:.3f}"
    return True, "accepted"


def _recipe_ca_rmsd_band(recipe: str) -> tuple[float, float]:
    bands = {
        "sidechain-repack": (0.0, 0.35),
        "torsion-near": (0.25, 2.5),
        "torsion-medium": (1.0, 6.5),
        "torsion-far": (2.0, 14.0),
        "segment-near": (0.25, 3.0),
        "segment-medium": (1.0, 7.5),
        "compactness-contract": (0.0, 7.5),
        "compactness-expand": (0.0, 7.5),
        "backbone-noise-small": (0.0, 2.0),
        "backbone-noise-medium": (0.25, 6.5),
        "chi-rotamer-random": (0.0, 0.50),
        "compactness-perturb": (0.0, 8.0),
        "mixed": (0.25, 8.0),
    }
    return bands[recipe]


def _geometry_integrity_acceptance_limit(recipe: str) -> float:
    if recipe in {"compactness-contract", "compactness-expand", "compactness-perturb", "mixed"}:
        return 250.0
    if recipe in {"backbone-noise-small", "backbone-noise-medium"}:
        return 500.0
    return 75.0


def _decoy_quality_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float]:
    geometry = float(metrics.get("geometry_integrity") or 1.0e12)
    ca_rmsd = float(metrics.get("ca_rmsd") or 1.0e12)
    return (geometry, ca_rmsd)


def _torsion_recipe_config(recipe: str) -> dict[str, float]:
    degree = math.pi / 180.0
    configs = {
        "sidechain-repack": {"backbone_sigma": 0.0, "chi_sigma": 20.0 * degree, "radial_scale": 1.0},
        "torsion-near": {"backbone_sigma": 8.0 * degree, "chi_sigma": 35.0 * degree, "radial_scale": 1.0},
        "torsion-medium": {"backbone_sigma": 22.0 * degree, "chi_sigma": 65.0 * degree, "radial_scale": 1.0},
        "torsion-far": {"backbone_sigma": 45.0 * degree, "chi_sigma": 100.0 * degree, "radial_scale": 1.0},
        "segment-near": {"backbone_sigma": 14.0 * degree, "chi_sigma": 45.0 * degree, "radial_scale": 1.0},
        "segment-medium": {"backbone_sigma": 32.0 * degree, "chi_sigma": 75.0 * degree, "radial_scale": 1.0},
        "compactness-contract": {"backbone_sigma": 18.0 * degree, "chi_sigma": 45.0 * degree, "radial_scale": 0.98},
        "compactness-expand": {"backbone_sigma": 18.0 * degree, "chi_sigma": 45.0 * degree, "radial_scale": 1.02},
    }
    return configs[recipe]


def _selected_torsion_indices(recipe: str, residue_count: int, rng: random.Random) -> set[int]:
    if residue_count <= 0:
        return set()
    if recipe in {"segment-near", "segment-medium"}:
        segment_fraction = 0.20 if recipe == "segment-near" else 0.35
        length = max(1, min(residue_count, int(round(residue_count * segment_fraction))))
        start = rng.randrange(0, max(1, residue_count - length + 1))
        return set(range(start, start + length))
    return set(range(residue_count))


def _scale_structure_about_centroid(structure: HeavyAtomStructure, scale_factor: float) -> None:
    coords = [atom.coord for atom in structure.atoms]
    if not coords:
        return
    center = (
        sum(coord[0] for coord in coords) / len(coords),
        sum(coord[1] for coord in coords) / len(coords),
        sum(coord[2] for coord in coords) / len(coords),
    )
    for atom in structure.atoms:
        atom.x = center[0] + (atom.x - center[0]) * scale_factor
        atom.y = center[1] + (atom.y - center[1]) * scale_factor
        atom.z = center[2] + (atom.z - center[2]) * scale_factor


def _wrap_radians(value: float) -> float:
    return ((float(value) + math.pi) % (2.0 * math.pi)) - math.pi


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _expand_decoy_recipes(recipes: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for recipe in recipes:
        normalized = str(recipe).strip().lower()
        if normalized in PHEAT_DECOY_PROFILES:
            expanded.extend(PHEAT_DECOY_PROFILES[normalized])
        else:
            expanded.append(_normalize_recipe(normalized))
    return list(dict.fromkeys(expanded))


def _recipe_profile_label(requested: Sequence[str], normalized: Sequence[str]) -> Optional[str]:
    requested_values = [str(item).strip().lower() for item in requested if str(item).strip()]
    if len(requested_values) == 1 and requested_values[0] in PHEAT_DECOY_PROFILES:
        return requested_values[0]
    for profile, recipes in PHEAT_DECOY_PROFILES.items():
        if tuple(normalized) == tuple(recipes):
            return profile
    return None


def _decoy_acceptance_summary(accepted: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_recipe: dict[str, dict[str, int]] = {}
    for status, rows in (("accepted", accepted), ("rejected", rejected)):
        for row in rows:
            recipe = str(row.get("recipe") or "unknown")
            by_recipe.setdefault(recipe, {"accepted": 0, "rejected": 0})
            by_recipe[recipe][status] += 1
    return {"by_recipe": by_recipe}


def _feature_validation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feature_names = sorted(
        {
            str(key)
            for row in rows
            for key in (row.get("features") or {})
            if isinstance(row.get("features"), Mapping)
        }
    )
    metrics = {}
    for feature in feature_names:
        native = _feature_values(rows, feature, label="native")
        decoys = [value for row in rows if row.get("label") != "native" for value in _row_feature_value(row, feature)]
        metrics[feature] = {
            "score_direction": "lower-is-better",
            "native": _values_summary(native),
            "decoy": _values_summary(decoys),
            "auc_lower_is_better": _pairwise_auc(native, decoys),
        }
    return metrics


def _feature_values(rows: Sequence[Mapping[str, Any]], feature: str, *, label: str) -> list[float]:
    return [value for row in rows if row.get("label") == label for value in _row_feature_value(row, feature)]


def _row_feature_value(row: Mapping[str, Any], feature: str) -> list[float]:
    features = row.get("features")
    if not isinstance(features, Mapping) or feature not in features:
        return []
    try:
        value = float(features[feature])
    except (TypeError, ValueError):
        return []
    return [value] if math.isfinite(value) else []


def _values_summary(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _pairwise_auc(native: Sequence[float], decoys: Sequence[float]) -> Optional[float]:
    if not native or not decoys:
        return None
    wins = 0.0
    total = 0
    for left in native:
        for right in decoys:
            total += 1
            if left < right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / total if total else None


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        errors = row.get("errors")
        if not isinstance(errors, Mapping):
            continue
        for key in errors:
            counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def _inventory_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "status_counts": _counts_for(rows, "status"),
        "method_counts": _counts_for(rows, "method"),
        "domain_counts": _counts_for(rows, "domain"),
    }


def _corpus_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[Mapping[str, Any]] = [
        item for item in payload.get("entries", []) if isinstance(item, Mapping)
    ] if isinstance(payload.get("entries"), list) else []
    return {
        "format": "pheat.reference-corpus-audit",
        "version": 1,
        "artifact_id": payload.get("artifact_id"),
        "artifact_version": payload.get("artifact_version"),
        "selected_count": payload.get("selected_count"),
        "holdout_count": payload.get("holdout_count"),
        "cluster_count": payload.get("cluster_count"),
        "method_counts": _counts_for(entries, "method"),
        "resolution": _resolution_summary(entries),
        "filters": payload.get("filters", {}),
    }


def _counts_for(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) if row.get(key) is not None else "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _resolution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["resolution"]) for row in rows if row.get("resolution") is not None]
    return _values_summary(values)


def _contains_unpackageable_payload(source: Path) -> bool:
    paths = sorted(source.rglob("*.json")) if source.is_dir() else [source]
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _payload_contains_unpackageable(payload):
            return True
    return False


def _payload_contains_unpackageable(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("packageable") is False or value.get("local_use_only") is True:
            return True
        return any(_payload_contains_unpackageable(item) for item in value.values())
    if isinstance(value, list):
        return any(_payload_contains_unpackageable(item) for item in value)
    return False


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(_file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


def _lineage_payload(*, stage: str, parents: Sequence[Optional[str]]) -> dict[str, Any]:
    return {
        "stage": stage,
        "parents": [parent for parent in parents if parent],
        "created_at": _utc_now(),
        "pheat_version": __version__,
    }


def _sample_id(prefix: str, entry: Mapping[str, Any]) -> str:
    pdb_id = str(entry.get("pdb_id") or Path(str(entry.get("path") or "structure")).stem).upper()
    chain_id = str(entry.get("chain_id") or "A")
    key = f"{prefix}:{pdb_id}:{chain_id}:{entry.get('path')}"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return _safe_label(f"{prefix}-{pdb_id}-{chain_id}-{suffix}")


def _normalize_recipe(recipe: str) -> str:
    normalized = str(recipe).strip().lower()
    if normalized not in PHEAT_DECOY_RECIPES:
        raise ValueError(f"decoy recipe must be one of {', '.join(PHEAT_DECOY_RECIPES)}")
    return normalized


def _normalize_subset(subset: str) -> str:
    normalized = str(subset).strip().lower()
    if normalized not in REFERENCE_SUBSETS:
        raise ValueError(f"reference subset must be one of {', '.join(REFERENCE_SUBSETS)}")
    return normalized


def _sequence_identity_label(value: Union[str, float]) -> str:
    threshold = normalize_sequence_identity_threshold(value)
    percent = threshold * 100.0
    return (str(int(percent)) if math.isclose(percent, round(percent)) else f"{percent:g}".replace(".", "p")) + "id"


def _training_set_entries(training_set: Path) -> list[dict[str, Any]]:
    path = Path(training_set)
    selected_jsonl = path / "selected.jsonl" if path.is_dir() else path
    if selected_jsonl.exists():
        return _read_jsonl(selected_jsonl)
    selected_json = path / "selected.json"
    if selected_json.exists():
        payload = json.loads(selected_json.read_text(encoding="utf-8"))
        return [dict(row) for row in payload.get("entries", []) if isinstance(row, Mapping)]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _ensure_output_path(path: Path, *, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite to replace: {path}")
    if Path(path).exists() and overwrite:
        if Path(path).is_dir():
            shutil.rmtree(path)
        else:
            Path(path).unlink()


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


def _safe_label(value: str) -> str:
    output = []
    previous_dash = False
    for char in str(value).lower():
        if char.isalnum():
            output.append(char)
            previous_dash = False
        elif not previous_dash:
            output.append("-")
            previous_dash = True
    return "".join(output).strip("-") or "artifact"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

"""Geometry target-table helpers for opt-in reconstruction refinement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import shlex
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from pheat.domains import filter_structure_for_domain, normalize_domain
from pheat.geometry import angle_degrees, distance
from pheat.models import Atom, HeavyAtomStructure, ResidueKey
from pheat.residues import CANONICAL_RESIDUES, SIDECHAIN_STEPS, SUPPORTED_RESIDUES, SidechainStep


GEOMETRY_TABLE_SET_FORMAT = "pheat.geometry-table-set"
GEOMETRY_TABLE_SET_VERSION = 1
DEFAULT_GEOMETRY_PROFILE = "fixed"
GEOMETRY_TABLE_FILENAME = "geometry-tables.json"
GEOMETRY_MODES = ("fixed", "table")
PACKAGED_GEOMETRY_DIR = ("data", "geometry")
PACKAGED_GEOMETRY_MANIFEST = "manifest.json"
DEFAULT_PHI_PSI_BIN_SIZE = 30
DEFAULT_CDL_PHI_PSI_BIN_SIZE = 10
DEFAULT_CDL_MIN_BIN_COUNT = 20
CDL_SMOOTHING_MODES = ("nearest", "kernel")
CDL_RESIDUE_CLASS_MODES = ("gly-pro-general", "canonical", "per-residue")


@dataclass
class GeometryLookup:
    """Lookup wrapper used by reconstruction to read optional geometry tables."""

    mode: str = "fixed"
    table_set: Optional[Mapping[str, Any]] = None
    profile_id: Optional[str] = None
    profile: Optional[Mapping[str, Any]] = None
    fallback_count: int = 0

    def length(
        self,
        name: str,
        default: float,
        *,
        resname: Optional[str] = None,
        phi: Optional[float] = None,
        psi: Optional[float] = None,
    ) -> float:
        value = self._target_value("lengths", name, resname=resname, phi=phi, psi=psi)
        if value is None:
            return self._fallback(default)
        return value

    def angle(
        self,
        name: str,
        default: float,
        *,
        resname: Optional[str] = None,
        phi: Optional[float] = None,
        psi: Optional[float] = None,
    ) -> float:
        value = self._target_value("angles", name, resname=resname, phi=phi, psi=psi)
        if value is None:
            return self._fallback(default)
        return value

    def sidechain_steps(self, resname: str, default: Sequence[SidechainStep]) -> Sequence[SidechainStep]:
        if self.mode != "table" or self.profile is None:
            return default
        sidechains = self.profile.get("sidechains")
        if not isinstance(sidechains, Mapping):
            self.fallback_count += 1
            return default
        residues = sidechains.get("residues")
        if not isinstance(residues, Mapping):
            self.fallback_count += 1
            return default
        payload = residues.get(resname)
        if not isinstance(payload, Mapping):
            self.fallback_count += 1
            return default
        steps = payload.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            self.fallback_count += 1
            return default
        try:
            return [_sidechain_step_from_dict(step) for step in steps if isinstance(step, Mapping)]
        except Exception:
            self.fallback_count += 1
            return default

    def metadata(self) -> dict[str, Any]:
        if self.mode == "fixed":
            return {"mode": "fixed"}
        table_label = None
        if self.table_set is not None:
            table_label = (
                self.table_set.get("artifact_label")
                or self.table_set.get("artifact_id")
                or self.table_set.get("format")
            )
        return {
            "mode": "table",
            "profile": self.profile_id,
            "table": table_label,
            "fallback_count": self.fallback_count,
        }

    def _target_value(
        self,
        group: str,
        name: str,
        *,
        resname: Optional[str] = None,
        phi: Optional[float] = None,
        psi: Optional[float] = None,
    ) -> Optional[float]:
        if self.mode != "table" or self.profile is None:
            return None
        backbone = self.profile.get("backbone")
        if not isinstance(backbone, Mapping):
            return None
        if phi is not None and psi is not None:
            bin_size = _profile_phi_psi_bin_size(self.profile)
            value = _residue_class_bin_target(backbone, resname, group, name, phi, psi, bin_size)
            if value is not None:
                return value
            bins = backbone.get("phi_psi_bins")
            if isinstance(bins, Mapping):
                bin_key = f"phi:{_angle_bin(phi, size=bin_size)}|psi:{_angle_bin(psi, size=bin_size)}"
                value = _numeric_target(bins.get(bin_key), group, name)
                if value is not None:
                    return value
        if resname:
            residues = backbone.get("residues")
            residue_payload = residues.get(resname) if isinstance(residues, Mapping) else None
            value = _numeric_target(residue_payload, group, name)
            if value is not None:
                return value
        defaults = backbone.get("defaults")
        return _numeric_target(defaults, group, name)

    def _fallback(self, default: float) -> float:
        if self.mode == "table":
            self.fallback_count += 1
        return default


def load_geometry_table_set(value: Optional[Union[str, Path, Mapping[str, Any]]]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, str) and _looks_like_packaged_table_name(value):
        payload = load_packaged_geometry_table(value)
    else:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("geometry table set JSON must contain an object")
    validate_geometry_table_set(payload)
    return payload


def list_packaged_geometry_tables() -> list[dict[str, Any]]:
    manifest = _load_packaged_geometry_manifest()
    tables = manifest.get("tables", [])
    if not isinstance(tables, list):
        return []
    return [dict(table) for table in tables if isinstance(table, Mapping)]


def load_packaged_geometry_table(name: str) -> dict[str, Any]:
    table = _packaged_geometry_table_record(name)
    filename = str(table["filename"])
    text = _packaged_geometry_dir().joinpath(filename).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Packaged geometry table {name!r} must contain a JSON object")
    validate_geometry_table_set(payload)
    return payload


def validate_geometry_table_set(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != GEOMETRY_TABLE_SET_FORMAT:
        raise ValueError(f"geometry table set requires format {GEOMETRY_TABLE_SET_FORMAT!r}")
    if payload.get("version") != GEOMETRY_TABLE_SET_VERSION:
        raise ValueError(f"geometry table set requires version {GEOMETRY_TABLE_SET_VERSION!r}")
    if not isinstance(payload.get("profiles"), Mapping):
        raise ValueError("geometry table set requires a profiles object")
    if not payload.get("default_profile"):
        raise ValueError("geometry table set requires default_profile")


def _load_packaged_geometry_manifest() -> dict[str, Any]:
    text = _packaged_geometry_dir().joinpath(PACKAGED_GEOMETRY_MANIFEST).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Packaged geometry manifest must contain a JSON object")
    return payload


def _packaged_geometry_table_record(name: str) -> Mapping[str, Any]:
    aliases = {name}
    for table in list_packaged_geometry_tables():
        table_aliases = {
            str(table.get("id") or ""),
            str(table.get("artifact_id") or ""),
            str(table.get("artifact_label") or ""),
            str(table.get("filename") or ""),
        }
        if aliases & table_aliases:
            return table
    known = ", ".join(table["id"] for table in list_packaged_geometry_tables() if table.get("id"))
    raise ValueError(f"Unknown packaged geometry table {name!r}. Known tables: {known}")


def _looks_like_packaged_table_name(value: str) -> bool:
    path = Path(value)
    if path.exists():
        return False
    return not any(separator in value for separator in ("/", "\\")) and not value.endswith(".json")


def _packaged_geometry_dir() -> Any:
    path = resources.files("pheat")
    for part in PACKAGED_GEOMETRY_DIR:
        path = path.joinpath(part)
    return path


def write_geometry_table_set(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    validate_geometry_table_set(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def geometry_profile_payload(
    table_set: Mapping[str, Any],
    profile: Optional[str] = None,
) -> Mapping[str, Any]:
    profiles = table_set.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("geometry table set requires a profiles object")
    selected = profile or str(table_set.get("default_profile") or "")
    payload = profiles.get(selected)
    if not isinstance(payload, Mapping):
        known = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Unknown geometry table profile '{selected}'. Known profiles: {known}")
    return payload


def geometry_profile_ids(table_set: Mapping[str, Any]) -> list[str]:
    profiles = table_set.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("geometry table set requires a profiles object")
    return sorted(str(key) for key in profiles)


def geometry_default_profile(table_set: Mapping[str, Any]) -> str:
    return str(table_set.get("default_profile") or DEFAULT_GEOMETRY_PROFILE)


def geometry_lookup(
    *,
    geometry_mode: Optional[str] = None,
    geometry_table: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    geometry_profile: Optional[str] = None,
) -> GeometryLookup:
    table_set = load_geometry_table_set(geometry_table)
    mode = (geometry_mode or ("table" if table_set is not None else "fixed")).strip().lower()
    if mode not in GEOMETRY_MODES:
        raise ValueError(f"geometry_mode must be one of {', '.join(GEOMETRY_MODES)}")
    if mode == "fixed":
        return GeometryLookup(mode="fixed")
    if table_set is None:
        raise ValueError("geometry_mode='table' requires geometry_table")
    profile_id = geometry_profile or geometry_default_profile(table_set)
    return GeometryLookup(
        mode="table",
        table_set=table_set,
        profile_id=profile_id,
        profile=geometry_profile_payload(table_set, profile_id),
    )


def describe_geometry_tables(table_set: Union[str, Path, Mapping[str, Any]]) -> dict[str, Any]:
    payload = load_geometry_table_set(table_set)
    assert payload is not None
    profile_payloads = payload.get("profiles", {})
    profiles = {}
    if isinstance(profile_payloads, Mapping):
        for profile_id, profile in profile_payloads.items():
            if not isinstance(profile, Mapping):
                continue
            backbone = profile.get("backbone") if isinstance(profile.get("backbone"), Mapping) else {}
            sidechains = profile.get("sidechains") if isinstance(profile.get("sidechains"), Mapping) else {}
            phi_psi_bins = backbone.get("phi_psi_bins", {}) if isinstance(backbone, Mapping) else {}
            profiles[str(profile_id)] = {
                "has_backbone": bool(backbone),
                "has_sidechains": bool(sidechains),
                "phi_psi_bin_count": len(phi_psi_bins) if isinstance(phi_psi_bins, Mapping) else 0,
                "residue_count": len(sidechains.get("residues", {})) if isinstance(sidechains, Mapping) else 0,
            }
    return {
        "format": payload["format"],
        "version": payload["version"],
        "artifact_id": payload.get("artifact_id"),
        "artifact_version": payload.get("artifact_version"),
        "default_profile": payload.get("default_profile"),
        "profiles": profiles,
    }


def validate_geometry_tables(table_set: Union[str, Path, Mapping[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        payload = load_geometry_table_set(table_set)
        assert payload is not None
        for profile_id in geometry_profile_ids(payload):
            profile = geometry_profile_payload(payload, profile_id)
            if not isinstance(profile.get("backbone"), Mapping) and not isinstance(
                profile.get("sidechains"),
                Mapping,
            ):
                warnings.append(f"{profile_id}: profile has no backbone or sidechain targets")
            _validate_sidechain_steps(profile, profile_id, warnings)
        return {
            "ok": not warnings,
            "profile_count": len(geometry_profile_ids(payload)),
            "profiles": geometry_profile_ids(payload),
            "warnings": warnings,
        }
    except Exception as exc:
        return {"ok": False, "profile_count": 0, "profiles": [], "warnings": [str(exc)]}


def build_backbone_geometry_tables(
    training_set: Path,
    *,
    output_root: Path,
    domain: str = "protein-heavy",
    max_entries: Optional[int] = None,
    table_set_id: Optional[str] = None,
    table_set_version: str = "v1",
    command_args: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    from pheat.training import _entries_sha256, _source_corpus_summary, _training_set_entries
    from pheat.training import _training_set_manifest, load_training_structure

    entries = _training_set_entries(training_set)
    if max_entries is not None:
        entries = entries[:max_entries]
    domain = normalize_domain(domain)
    length_stats: dict[str, _Stats] = {}
    angle_stats: dict[str, _Stats] = {}
    residue_length_stats: dict[str, dict[str, _Stats]] = {}
    residue_angle_stats: dict[str, dict[str, _Stats]] = {}
    phi_psi_bins: dict[str, dict[str, dict[str, _Stats]]] = {}
    loaded_count = 0
    warnings: list[str] = []

    for entry in entries:
        path = Path(str(entry.get("path") or ""))
        if not path.exists():
            warnings.append(f"missing selected structure: {path}")
            continue
        try:
            structure = load_training_structure(path)
            filtered, _coverage = filter_structure_for_domain(structure, domain=domain)
            _accumulate_backbone_geometry(
                filtered,
                length_stats,
                angle_stats,
                residue_length_stats,
                residue_angle_stats,
                phi_psi_bins,
            )
            loaded_count += 1
        except Exception as exc:
            warnings.append(f"{path}: {exc}")

    corpus_manifest = _training_set_manifest(training_set)
    source_corpus = _source_corpus_summary(training_set, corpus_manifest, entries)
    artifact_id = table_set_id or f"{source_corpus.get('artifact_id') or domain}-geometry"
    payload = _base_geometry_table_payload(
        artifact_id=artifact_id,
        artifact_version=table_set_version,
        default_profile="backbone",
        command_args=command_args,
        metadata={
            "training_set": str(training_set),
            "entry_count": len(entries),
            "loaded_entry_count": loaded_count,
            "entry_sha256": _entries_sha256(entries),
            "source_corpus": source_corpus,
            "domain": domain,
            "source": "local-structure-statistics",
            "warnings": warnings,
        },
        profiles={
            "backbone": {
                "metadata": {
                    "kind": "backbone",
                    "source": "local-structure-statistics",
                    "entry_count": len(entries),
                    "loaded_entry_count": loaded_count,
                },
                "backbone": {
                    "defaults": {
                        "lengths": _stats_means(length_stats),
                        "angles": _stats_means(angle_stats),
                        "observations": {
                            "lengths": _stats_observations(length_stats),
                            "angles": _stats_observations(angle_stats),
                        },
                    },
                    "residues": _finalize_residue_backbone_stats(
                        residue_length_stats,
                        residue_angle_stats,
                    ),
                    "phi_psi_bin_size": DEFAULT_PHI_PSI_BIN_SIZE,
                    "phi_psi_bins": _finalize_bin_stats(phi_psi_bins),
                },
                "sidechains": {"residues": {}},
            }
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / GEOMETRY_TABLE_FILENAME
    write_geometry_table_set(output, payload)
    return {**payload, "output": str(output)}


def build_cdl_geometry_tables(
    training_set: Path,
    *,
    output_root: Path,
    domain: str = "protein-heavy",
    max_entries: Optional[int] = None,
    table_set_id: Optional[str] = None,
    table_set_version: str = "v1",
    phi_psi_bin_size: int = DEFAULT_CDL_PHI_PSI_BIN_SIZE,
    min_bin_count: int = DEFAULT_CDL_MIN_BIN_COUNT,
    smoothing: str = "nearest",
    residue_classes: str = "gly-pro-general",
    command_args: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Build a PHEAT-owned conformation-dependent backbone geometry table.

    The generated profile is inspired by CDL-style phi/psi conditioned geometry,
    but is calculated entirely from a caller-selected local training set. It does
    not vendor or reproduce Phenix/CCTBX CDL parameter tables.
    """

    from pheat.training import _entries_sha256, _source_corpus_summary, _training_set_entries
    from pheat.training import _training_set_manifest, load_training_structure

    if phi_psi_bin_size <= 0 or 360 % phi_psi_bin_size:
        raise ValueError("--phi-psi-bin-size must be a positive divisor of 360")
    if min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1")
    if smoothing not in CDL_SMOOTHING_MODES:
        raise ValueError(f"--smoothing must be one of {', '.join(CDL_SMOOTHING_MODES)}")
    if residue_classes not in CDL_RESIDUE_CLASS_MODES:
        raise ValueError(f"--residue-classes must be one of {', '.join(CDL_RESIDUE_CLASS_MODES)}")

    entries = _training_set_entries(training_set)
    if max_entries is not None:
        entries = entries[:max_entries]
    domain = normalize_domain(domain)
    length_stats: dict[str, _Stats] = {}
    angle_stats: dict[str, _Stats] = {}
    residue_length_stats: dict[str, dict[str, _Stats]] = {}
    residue_angle_stats: dict[str, dict[str, _Stats]] = {}
    phi_psi_bins: dict[str, dict[str, dict[str, _Stats]]] = {}
    residue_class_bins: dict[str, dict[str, dict[str, dict[str, _Stats]]]] = {}
    loaded_count = 0
    warnings: list[str] = []
    if smoothing == "kernel":
        warnings.append(
            "kernel smoothing is recorded in the table metadata; current runtime lookup uses nearest binned targets"
        )

    for entry in entries:
        path = Path(str(entry.get("path") or ""))
        if not path.exists():
            warnings.append(f"missing selected structure: {path}")
            continue
        try:
            structure = load_training_structure(path)
            filtered, _coverage = filter_structure_for_domain(structure, domain=domain)
            _accumulate_backbone_geometry(
                filtered,
                length_stats,
                angle_stats,
                residue_length_stats,
                residue_angle_stats,
                phi_psi_bins,
                phi_psi_bin_size=phi_psi_bin_size,
                residue_class_bins=residue_class_bins,
                residue_class_mode=residue_classes,
            )
            loaded_count += 1
        except Exception as exc:
            warnings.append(f"{path}: {exc}")

    corpus_manifest = _training_set_manifest(training_set)
    source_corpus = _source_corpus_summary(training_set, corpus_manifest, entries)
    artifact_id = table_set_id or f"{source_corpus.get('artifact_id') or domain}-cdl-geometry"
    profile_id = "backbone-cdl"
    payload = _base_geometry_table_payload(
        artifact_id=artifact_id,
        artifact_version=table_set_version,
        default_profile=profile_id,
        command_args=command_args,
        metadata={
            "training_set": str(training_set),
            "entry_count": len(entries),
            "loaded_entry_count": loaded_count,
            "entry_sha256": _entries_sha256(entries),
            "source_corpus": source_corpus,
            "domain": domain,
            "source": "local-conformation-dependent-statistics",
            "phi_psi_bin_size": phi_psi_bin_size,
            "min_bin_count": min_bin_count,
            "smoothing": smoothing,
            "residue_classes": residue_classes,
            "warnings": warnings,
        },
        profiles={
            profile_id: {
                "metadata": {
                    "kind": "backbone-cdl",
                    "source": "local-conformation-dependent-statistics",
                    "entry_count": len(entries),
                    "loaded_entry_count": loaded_count,
                    "phi_psi_bin_size": phi_psi_bin_size,
                    "min_bin_count": min_bin_count,
                    "smoothing": smoothing,
                    "residue_classes": residue_classes,
                },
                "backbone": {
                    "defaults": {
                        "lengths": _stats_means(length_stats),
                        "angles": _stats_means(angle_stats),
                        "observations": {
                            "lengths": _stats_observations(length_stats),
                            "angles": _stats_observations(angle_stats),
                        },
                    },
                    "residues": _finalize_residue_backbone_stats(
                        residue_length_stats,
                        residue_angle_stats,
                    ),
                    "phi_psi_bin_size": phi_psi_bin_size,
                    "phi_psi_bins": _finalize_bin_stats(phi_psi_bins, min_count=min_bin_count),
                    "residue_classes": {
                        "mode": residue_classes,
                        "phi_psi_bin_size": phi_psi_bin_size,
                        "smoothing": smoothing,
                        "classes": _finalize_residue_class_bins(residue_class_bins, min_count=min_bin_count),
                    },
                },
                "sidechains": {"residues": {}},
            }
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / GEOMETRY_TABLE_FILENAME
    write_geometry_table_set(output, payload)
    return {**payload, "output": str(output)}


def import_cdl_geometry_tables(
    input_path: Path,
    *,
    output_root: Path,
    table_set_id: str = "imported-cdl-geometry",
    table_set_version: str = "v1",
    source_license: Optional[str] = None,
    command_args: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Import a JSON CDL-like table into PHEAT's geometry-table-set format."""

    raw = json.loads(_read_text_maybe_gzip(input_path))
    if not isinstance(raw, Mapping):
        raise ValueError("CDL import input must contain a JSON object")
    input_sha256 = _file_sha256(input_path)
    if raw.get("format") == GEOMETRY_TABLE_SET_FORMAT:
        payload = dict(raw)
        validate_geometry_table_set(payload)
        metadata = dict(payload.get("metadata") or {})
        metadata["imported_from"] = str(input_path)
        metadata["imported_sha256"] = input_sha256
        if source_license:
            metadata["source_license"] = source_license
        payload["metadata"] = metadata
    else:
        bins = raw.get("phi_psi_bins") or raw.get("bins")
        if not isinstance(bins, Mapping):
            raise ValueError("CDL import JSON requires a phi_psi_bins or bins object")
        phi_psi_bin_size = _optional_int(raw.get("phi_psi_bin_size")) or DEFAULT_CDL_PHI_PSI_BIN_SIZE
        residue_class_mode = str(raw.get("residue_classes") or "imported")
        profile_id = "backbone-cdl"
        payload = _base_geometry_table_payload(
            artifact_id=table_set_id,
            artifact_version=table_set_version,
            default_profile=profile_id,
            command_args=command_args,
            metadata={
                "source": "imported-cdl-json",
                "input": str(input_path),
                "input_sha256": input_sha256,
                "source_license": source_license,
                "phi_psi_bin_size": phi_psi_bin_size,
                "residue_classes": residue_class_mode,
            },
            profiles={
                profile_id: {
                    "metadata": {
                        "kind": "backbone-cdl",
                        "source": "imported-cdl-json",
                        "phi_psi_bin_size": phi_psi_bin_size,
                        "residue_classes": residue_class_mode,
                    },
                    "backbone": {
                        "defaults": _normalize_target_payload(raw.get("defaults")),
                        "residues": dict(raw.get("residues") or {}),
                        "phi_psi_bin_size": phi_psi_bin_size,
                        "phi_psi_bins": _normalize_imported_bins(bins),
                    },
                    "sidechains": {"residues": {}},
                }
            },
        )

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / GEOMETRY_TABLE_FILENAME
    write_geometry_table_set(output, payload)
    return {**payload, "output": str(output)}


def build_sidechain_ccd_geometry_tables(
    ccd_dir: Optional[Path] = None,
    *,
    output_root: Path,
    ccd_full: Optional[Path] = None,
    ccd_bcif_dir: Optional[Path] = None,
    residues: Optional[Sequence[str]] = None,
    table_set_id: str = "ccd-sidechain-geometry",
    table_set_version: str = "v1",
    command_args: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    if ccd_full is None and ccd_dir is None and ccd_bcif_dir is None:
        raise ValueError("One of ccd_full, ccd_dir, or ccd_bcif_dir is required")
    selected = [residue.strip().upper() for residue in (residues or sorted(SUPPORTED_RESIDUES))]
    full_components = _load_selected_ccd_components(ccd_full, selected) if ccd_full is not None else {}
    using_bcif_only = ccd_full is None and ccd_dir is None and ccd_bcif_dir is not None
    residue_payloads = {}
    warnings = []
    if using_bcif_only:
        warnings.append(
            "CCD BinaryCIF atom/bond subsets are connectivity-only for PHEAT geometry; "
            "using PHEAT template lengths and angles"
        )
        _validate_ccd_bcif_subset_paths(ccd_bcif_dir)
    for resname in selected:
        ccd: Mapping[str, Any]
        if ccd_full is not None:
            ccd = full_components.get(resname, {})
            if not ccd:
                warnings.append(f"{resname}: CCD component not found in {ccd_full}")
                continue
        elif ccd_dir is not None:
            try:
                ccd = _load_ccd_component(ccd_dir, resname)
            except FileNotFoundError:
                warnings.append(f"{resname}: CCD CIF not found")
                continue
        else:
            ccd = {}
        steps = []
        for step in SIDECHAIN_STEPS.get(resname, []):
            steps.append(_sidechain_step_to_dict(_step_with_ccd_geometry(step, ccd)))
        residue_payloads[resname] = {
            "source": _sidechain_geometry_source(ccd_full, ccd_dir, ccd_bcif_dir),
            "placement_order_source": "pheat-template-order",
            "steps": steps,
        }

    payload = _base_geometry_table_payload(
        artifact_id=table_set_id,
        artifact_version=table_set_version,
        default_profile="ccd-sidechains",
        command_args=command_args,
        metadata={
            "ccd_dir": str(ccd_dir) if ccd_dir is not None else None,
            "ccd_full": str(ccd_full) if ccd_full is not None else None,
            "ccd_bcif_dir": str(ccd_bcif_dir) if ccd_bcif_dir is not None else None,
            "source": _sidechain_geometry_source(ccd_full, ccd_dir, ccd_bcif_dir),
            "residue_count": len(residue_payloads),
            "warnings": warnings,
        },
        profiles={
            "ccd-sidechains": {
                "metadata": {
                    "kind": "sidechains",
                    "source": _sidechain_geometry_source(ccd_full, ccd_dir, ccd_bcif_dir),
                    "placement_order_source": "pheat-template-order",
                },
                "backbone": {},
                "sidechains": {"residues": residue_payloads},
            }
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / GEOMETRY_TABLE_FILENAME
    write_geometry_table_set(output, payload)
    return {**payload, "output": str(output)}


@dataclass
class _Stats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def stdev(self) -> float:
        if self.count <= 1:
            return 0.0
        variance = max(0.0, (self.total_sq - self.total * self.total / self.count) / (self.count - 1))
        return math.sqrt(variance)


def _base_geometry_table_payload(
    *,
    artifact_id: str,
    artifact_version: str,
    default_profile: str,
    command_args: Optional[Sequence[str]],
    metadata: Mapping[str, Any],
    profiles: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_label = f"{artifact_id}-{artifact_version}" if artifact_version else artifact_id
    return {
        "format": GEOMETRY_TABLE_SET_FORMAT,
        "version": GEOMETRY_TABLE_SET_VERSION,
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "artifact_label": artifact_label,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pheat_version": _pheat_version(),
        "default_profile": default_profile,
        "metadata": {
            **dict(metadata),
            "provenance": {
                "pheat_version": _pheat_version(),
                "command_args": list(command_args) if command_args is not None else None,
            },
        },
        "profiles": dict(profiles),
    }


def _accumulate_backbone_geometry(
    structure: HeavyAtomStructure,
    length_stats: dict[str, _Stats],
    angle_stats: dict[str, _Stats],
    residue_length_stats: dict[str, dict[str, _Stats]],
    residue_angle_stats: dict[str, dict[str, _Stats]],
    phi_psi_bins: dict[str, dict[str, dict[str, _Stats]]],
    *,
    phi_psi_bin_size: int = DEFAULT_PHI_PSI_BIN_SIZE,
    residue_class_bins: Optional[dict[str, dict[str, dict[str, dict[str, _Stats]]]]] = None,
    residue_class_mode: str = "gly-pro-general",
) -> None:
    lookup = structure.atom_lookup()
    keys = structure.residue_keys()
    for index, key in enumerate(keys):
        atoms = {name: lookup.get((key, name)) for name in ("N", "CA", "C", "O", "CB")}
        resname = key[3]
        _add_length("N-CA", atoms["N"], atoms["CA"], length_stats, residue_length_stats, resname)
        _add_length("CA-C", atoms["CA"], atoms["C"], length_stats, residue_length_stats, resname)
        _add_length("C-O", atoms["C"], atoms["O"], length_stats, residue_length_stats, resname)
        _add_length("CA-CB", atoms["CA"], atoms["CB"], length_stats, residue_length_stats, resname)
        _add_angle("N-CA-C", atoms["N"], atoms["CA"], atoms["C"], angle_stats, residue_angle_stats, resname)
        _add_angle("CA-C-O", atoms["CA"], atoms["C"], atoms["O"], angle_stats, residue_angle_stats, resname)
        _add_angle(
            "N-CA-CB",
            atoms["N"],
            atoms["CA"],
            atoms["CB"],
            angle_stats,
            residue_angle_stats,
            resname,
        )
        if index + 1 < len(keys) and _continues(keys[index], keys[index + 1]):
            next_key = keys[index + 1]
            next_atoms = {name: lookup.get((next_key, name)) for name in ("N", "CA", "C")}
            _add_length(
                "C-N",
                atoms["C"],
                next_atoms["N"],
                length_stats,
                residue_length_stats,
                resname,
            )
            _add_angle(
                "CA-C-N",
                atoms["CA"],
                atoms["C"],
                next_atoms["N"],
                angle_stats,
                residue_angle_stats,
                resname,
            )
            _add_angle(
                "C-N-CA",
                atoms["C"],
                next_atoms["N"],
                next_atoms["CA"],
                angle_stats,
                residue_angle_stats,
                next_key[3],
            )
    _accumulate_phi_psi_bins(
        structure,
        phi_psi_bins,
        phi_psi_bin_size=phi_psi_bin_size,
        residue_class_bins=residue_class_bins,
        residue_class_mode=residue_class_mode,
    )


def _accumulate_phi_psi_bins(
    structure: HeavyAtomStructure,
    phi_psi_bins: dict[str, dict[str, dict[str, _Stats]]],
    *,
    phi_psi_bin_size: int = DEFAULT_PHI_PSI_BIN_SIZE,
    residue_class_bins: Optional[dict[str, dict[str, dict[str, dict[str, _Stats]]]]] = None,
    residue_class_mode: str = "gly-pro-general",
) -> None:
    from pheat.residue_geometry import structure_to_residue_geometry

    geometry = structure_to_residue_geometry(
        structure,
        angle_units="degrees",
        stored_angles="all",
        stored_lengths="backbone",
    )
    for residue in geometry.residues:
        if residue.phi is None or residue.psi is None:
            continue
        bin_key = f"phi:{_angle_bin(residue.phi, size=phi_psi_bin_size)}|psi:{_angle_bin(residue.psi, size=phi_psi_bin_size)}"
        _accumulate_residue_bin_targets(phi_psi_bins.setdefault(bin_key, {}), residue)
        if residue_class_bins is not None:
            class_id = _residue_class_id(residue.name, residue_class_mode)
            class_bins = residue_class_bins.setdefault(class_id, {})
            _accumulate_residue_bin_targets(class_bins.setdefault(bin_key, {}), residue)


def _accumulate_residue_bin_targets(bucket: dict[str, dict[str, _Stats]], residue: Any) -> None:
    angle_bucket = bucket.setdefault("angles", {})
    if residue.tau is not None:
        angle_bucket.setdefault("N-CA-C", _Stats()).add(residue.tau)
    if residue.theta is not None:
        angle_bucket.setdefault("CA-C-N", _Stats()).add(residue.theta)
    length_bucket = bucket.setdefault("lengths", {})
    for name, value in sorted(residue.bond_lengths.items()):
        length_bucket.setdefault(name, _Stats()).add(float(value))


def _add_length(
    name: str,
    first: Optional[Atom],
    second: Optional[Atom],
    defaults: dict[str, _Stats],
    residues: dict[str, dict[str, _Stats]],
    resname: str,
) -> None:
    if first is None or second is None:
        return
    value = distance(first.coord, second.coord)
    defaults.setdefault(name, _Stats()).add(value)
    residues.setdefault(resname, {}).setdefault(name, _Stats()).add(value)


def _add_angle(
    name: str,
    first: Optional[Atom],
    second: Optional[Atom],
    third: Optional[Atom],
    defaults: dict[str, _Stats],
    residues: dict[str, dict[str, _Stats]],
    resname: str,
) -> None:
    if first is None or second is None or third is None:
        return
    try:
        value = angle_degrees(first.coord, second.coord, third.coord)
    except ValueError:
        return
    defaults.setdefault(name, _Stats()).add(value)
    residues.setdefault(resname, {}).setdefault(name, _Stats()).add(value)


def _continues(left: ResidueKey, right: ResidueKey) -> bool:
    return left[0] == right[0] and left[1] + 1 == right[1]


def _angle_bin(value: float, *, size: int = 30) -> str:
    shifted = (float(value) + 180.0) % 360.0
    start = int(math.floor(shifted / size) * size - 180)
    return f"{start}:{start + size}"


def _stats_means(stats: Mapping[str, _Stats], *, min_count: int = 1) -> dict[str, float]:
    return {
        key: value.mean
        for key, value in sorted(stats.items())
        if value.count and value.count >= min_count
    }


def _stats_observations(stats: Mapping[str, _Stats], *, min_count: int = 1) -> dict[str, dict[str, float]]:
    return {
        key: {"count": value.count, "mean": value.mean, "stdev": value.stdev}
        for key, value in sorted(stats.items())
        if value.count and value.count >= min_count
    }


def _finalize_residue_backbone_stats(
    residue_lengths: Mapping[str, Mapping[str, _Stats]],
    residue_angles: Mapping[str, Mapping[str, _Stats]],
) -> dict[str, Any]:
    residues = {}
    for resname in sorted(set(residue_lengths) | set(residue_angles)):
        residues[resname] = {
            "lengths": _stats_means(residue_lengths.get(resname, {})),
            "angles": _stats_means(residue_angles.get(resname, {})),
            "observations": {
                "lengths": _stats_observations(residue_lengths.get(resname, {})),
                "angles": _stats_observations(residue_angles.get(resname, {})),
            },
        }
    return residues


def _finalize_bin_stats(
    phi_psi_bins: Mapping[str, Mapping[str, Mapping[str, _Stats]]],
    *,
    min_count: int = 1,
) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for key, values in sorted(phi_psi_bins.items()):
        lengths = _stats_means(values.get("lengths", {}), min_count=min_count)
        angles = _stats_means(values.get("angles", {}), min_count=min_count)
        if not lengths and not angles:
            continue
        payload: dict[str, Any] = {
            "observations": {
                "lengths": _stats_observations(values.get("lengths", {}), min_count=min_count),
                "angles": _stats_observations(values.get("angles", {}), min_count=min_count),
            }
        }
        if lengths:
            payload["lengths"] = lengths
        if angles:
            payload["angles"] = angles
        finalized[key] = payload
    return finalized


def _finalize_residue_class_bins(
    residue_class_bins: Mapping[str, Mapping[str, Mapping[str, Mapping[str, _Stats]]]],
    *,
    min_count: int,
) -> dict[str, Any]:
    finalized = {}
    for class_id, bins in sorted(residue_class_bins.items()):
        finalized_bins = _finalize_bin_stats(bins, min_count=min_count)
        if finalized_bins:
            finalized[class_id] = {"phi_psi_bins": finalized_bins}
    return finalized


def _profile_phi_psi_bin_size(profile: Optional[Mapping[str, Any]]) -> int:
    if not isinstance(profile, Mapping):
        return DEFAULT_PHI_PSI_BIN_SIZE
    metadata = profile.get("metadata")
    if isinstance(metadata, Mapping):
        value = _optional_int(metadata.get("phi_psi_bin_size"))
        if value:
            return value
    backbone = profile.get("backbone")
    if isinstance(backbone, Mapping):
        value = _optional_int(backbone.get("phi_psi_bin_size"))
        if value:
            return value
    return DEFAULT_PHI_PSI_BIN_SIZE


def _residue_class_bin_target(
    backbone: Mapping[str, Any],
    resname: Optional[str],
    group: str,
    name: str,
    phi: float,
    psi: float,
    bin_size: int,
) -> Optional[float]:
    if not resname:
        return None
    class_payload = backbone.get("residue_classes")
    if not isinstance(class_payload, Mapping):
        return None
    classes = class_payload.get("classes")
    if not isinstance(classes, Mapping):
        return None
    mode = str(class_payload.get("mode") or "gly-pro-general")
    class_id = _residue_class_id(resname, mode)
    selected = classes.get(class_id)
    if not isinstance(selected, Mapping):
        return None
    bins = selected.get("phi_psi_bins")
    if not isinstance(bins, Mapping):
        return None
    bin_key = f"phi:{_angle_bin(phi, size=bin_size)}|psi:{_angle_bin(psi, size=bin_size)}"
    return _numeric_target(bins.get(bin_key), group, name)


def _residue_class_id(resname: str, mode: str) -> str:
    token = str(resname).strip().upper()
    if mode == "gly-pro-general":
        if token == "GLY":
            return "GLY"
        if token == "PRO":
            return "PRO"
        return "GENERAL"
    if mode == "canonical":
        return token if token in CANONICAL_RESIDUES else "OTHER"
    return token


def _normalize_target_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"lengths": {}, "angles": {}}
    normalized: dict[str, Any] = {}
    lengths = _numeric_mapping(payload.get("lengths"))
    angles = _numeric_mapping(payload.get("angles"))
    if lengths:
        normalized["lengths"] = lengths
    if angles:
        normalized["angles"] = angles
    return normalized or {"lengths": {}, "angles": {}}


def _normalize_imported_bins(bins: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, payload in sorted(bins.items()):
        target = _normalize_target_payload(payload)
        if target.get("lengths") or target.get("angles"):
            normalized[str(key)] = target
    return normalized


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, raw in sorted(value.items()):
        number = _optional_float(raw)
        if number is not None:
            result[str(key)] = number
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_target(
    payload: Optional[object],
    group: str,
    name: str,
) -> Optional[float]:
    if not isinstance(payload, Mapping):
        return None
    values = payload.get(group)
    if not isinstance(values, Mapping):
        return None
    value = values.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sidechain_step_from_dict(payload: Mapping[str, Any]) -> SidechainStep:
    dihedral = payload.get("dihedral", 0.0)
    if isinstance(dihedral, list) and len(dihedral) == 2:
        dihedral = (str(dihedral[0]), float(dihedral[1]))
    return SidechainStep(
        atom=str(payload["atom"]),
        parent=str(payload["parent"]),
        anchor=str(payload["anchor"]),
        previous=str(payload["previous"]),
        length=float(payload["length"]),
        angle=float(payload["angle"]),
        dihedral=dihedral,
        element=str(payload["element"]) if payload.get("element") is not None else None,
    )


def _sidechain_step_to_dict(step: SidechainStep) -> dict[str, Any]:
    dihedral: object = step.dihedral
    if isinstance(dihedral, tuple):
        dihedral = [dihedral[0], dihedral[1]]
    return {
        "atom": step.atom,
        "parent": step.parent,
        "anchor": step.anchor,
        "previous": step.previous,
        "length": step.length,
        "angle": step.angle,
        "dihedral": dihedral,
        "element": step.element or "",
    }


def _validate_sidechain_steps(profile: Mapping[str, Any], profile_id: str, warnings: list[str]) -> None:
    sidechains = profile.get("sidechains")
    if not isinstance(sidechains, Mapping):
        return
    residues = sidechains.get("residues")
    if not isinstance(residues, Mapping):
        return
    for resname, payload in residues.items():
        if not isinstance(payload, Mapping):
            warnings.append(f"{profile_id}:{resname}: sidechain payload is not an object")
            continue
        steps = payload.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            warnings.append(f"{profile_id}:{resname}: steps is not a list")
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                warnings.append(f"{profile_id}:{resname}: step {index} is not an object")
                continue
            missing = {
                "atom",
                "parent",
                "anchor",
                "previous",
                "length",
                "angle",
                "dihedral",
            } - set(step)
            if missing:
                warnings.append(f"{profile_id}:{resname}: step {index} missing {sorted(missing)}")


def _load_ccd_component(ccd_dir: Path, resname: str) -> dict[str, Any]:
    candidates = [ccd_dir / f"{resname}.cif", ccd_dir / f"{resname}.cif.gz"]
    for path in candidates:
        if path.exists():
            return _parse_ccd_cif(path)
    raise FileNotFoundError(resname)


def _load_selected_ccd_components(ccd_full: Path, selected: Sequence[str]) -> dict[str, dict[str, Any]]:
    wanted = {residue.upper() for residue in selected}
    components: dict[str, dict[str, Any]] = {}
    for component_id, block in _iter_cif_data_blocks(ccd_full):
        if component_id in wanted:
            components[component_id] = _parse_ccd_cif_text(block)
            if set(components) == wanted:
                break
    return components


def _iter_cif_data_blocks(path: Path) -> Iterable[tuple[str, str]]:
    current_id: Optional[str] = None
    current_lines: list[str] = []
    with _open_text_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith("data_"):
                if current_id is not None:
                    yield current_id, "".join(current_lines)
                current_id = line[5:].strip().upper()
                current_lines = [line]
            elif current_id is not None:
                current_lines.append(line)
    if current_id is not None:
        yield current_id, "".join(current_lines)


def _parse_ccd_cif(path: Path) -> dict[str, Any]:
    text = _read_text_maybe_gzip(path)
    return _parse_ccd_cif_text(text)


def _parse_ccd_cif_text(text: str) -> dict[str, Any]:
    atoms: dict[str, dict[str, Any]] = {}
    bonds: dict[frozenset[str], float] = {}
    for loop in _parse_cif_loops(text):
        labels = loop["labels"]
        rows = loop["rows"]
        if "_chem_comp_atom.atom_id" in labels:
            for row in rows:
                record = dict(zip(labels, row))
                atom_id = _clean_cif_value(record.get("_chem_comp_atom.atom_id"))
                if not atom_id:
                    continue
                atoms[atom_id] = {
                    "element": _clean_cif_value(record.get("_chem_comp_atom.type_symbol")),
                    "coord": _coord_from_record(record),
                }
        if "_chem_comp_bond.atom_id_1" in labels and "_chem_comp_bond.atom_id_2" in labels:
            for row in rows:
                record = dict(zip(labels, row))
                atom_a = _clean_cif_value(record.get("_chem_comp_bond.atom_id_1"))
                atom_b = _clean_cif_value(record.get("_chem_comp_bond.atom_id_2"))
                value = _optional_float(record.get("_chem_comp_bond.value_dist"))
                if atom_a and atom_b and value is not None:
                    bonds[frozenset((atom_a, atom_b))] = value
    return {"atoms": atoms, "bonds": bonds}


def _open_text_maybe_gzip(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_text_maybe_gzip(path: Path) -> str:
    with _open_text_maybe_gzip(path) as handle:
        return handle.read()


def _sidechain_geometry_source(
    ccd_full: Optional[Path],
    ccd_dir: Optional[Path],
    ccd_bcif_dir: Optional[Path],
) -> str:
    if ccd_full is not None:
        return "wwpdb-ccd-full"
    if ccd_dir is not None:
        return "wwpdb-ccd-component-cif"
    if ccd_bcif_dir is not None:
        return "rcsb-ccd-bcif-connectivity-only"
    return "pheat-template-default"


def _validate_ccd_bcif_subset_paths(ccd_bcif_dir: Optional[Path]) -> None:
    if ccd_bcif_dir is None:
        return
    missing = [
        filename
        for filename in ("cca.bcif", "ccb.bcif")
        if not (ccd_bcif_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing CCD BinaryCIF subset files: {', '.join(missing)}")


def _parse_cif_loops(text: str) -> Iterable[dict[str, Any]]:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        labels: list[str] = []
        rows: list[list[str]] = []
        while index < len(lines):
            stripped = lines[index].strip()
            index += 1
            if not stripped:
                continue
            if stripped.startswith("_"):
                labels.append(stripped)
                continue
            if stripped.startswith("#"):
                yield {"labels": labels, "rows": rows}
                break
            index -= 1
            break
        else:
            yield {"labels": labels, "rows": rows}
            break

        if not labels or (index <= len(lines) and lines[index - 1].strip().startswith("#")):
            continue

        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("#"):
                index += 1
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break
            tokens = shlex.split(stripped)
            if len(tokens) == len(labels):
                rows.append(tokens)
            index += 1
        yield {"labels": labels, "rows": rows}


def _coord_from_record(record: Mapping[str, str]) -> Optional[tuple[float, float, float]]:
    for prefix in ("model_Cartn", "pdbx_model_Cartn"):
        x = _optional_float(record.get(f"_chem_comp_atom.{prefix}_x"))
        y = _optional_float(record.get(f"_chem_comp_atom.{prefix}_y"))
        z = _optional_float(record.get(f"_chem_comp_atom.{prefix}_z"))
        if x is not None and y is not None and z is not None:
            return (x, y, z)
    return None


def _step_with_ccd_geometry(step: SidechainStep, ccd: Mapping[str, Any]) -> SidechainStep:
    atoms = ccd.get("atoms", {})
    bonds = ccd.get("bonds", {})
    length = step.length
    angle = step.angle
    element = step.element
    if isinstance(bonds, Mapping):
        bond_value = bonds.get(frozenset((step.parent, step.atom)))
        if isinstance(bond_value, (int, float)):
            length = float(bond_value)
    if isinstance(atoms, Mapping):
        atom_payload = atoms.get(step.atom)
        if isinstance(atom_payload, Mapping) and atom_payload.get("element"):
            element = str(atom_payload["element"])
        coords = [
            _ccd_coord(atoms, step.anchor),
            _ccd_coord(atoms, step.parent),
            _ccd_coord(atoms, step.atom),
        ]
        if all(coord is not None for coord in coords):
            try:
                angle = angle_degrees(coords[0], coords[1], coords[2])  # type: ignore[arg-type]
            except ValueError:
                pass
        parent = _ccd_coord(atoms, step.parent)
        atom = _ccd_coord(atoms, step.atom)
        if parent is not None and atom is not None and not isinstance(
            bonds.get(frozenset((step.parent, step.atom))) if isinstance(bonds, Mapping) else None,
            (int, float),
        ):
            length = distance(parent, atom)
    return SidechainStep(
        step.atom,
        step.parent,
        step.anchor,
        step.previous,
        length,
        angle,
        step.dihedral,
        element,
    )


def _ccd_coord(atoms: Mapping[str, Any], atom: str) -> Optional[tuple[float, float, float]]:
    payload = atoms.get(atom)
    if isinstance(payload, Mapping):
        coord = payload.get("coord")
        if isinstance(coord, tuple) and len(coord) == 3:
            return coord
    return None


def _clean_cif_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text in {".", "?"} else text


def _optional_float(value: object) -> Optional[float]:
    text = _clean_cif_value(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_int(value: object) -> Optional[int]:
    text = _clean_cif_value(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _pheat_version() -> str:
    try:
        return version("pheat")
    except PackageNotFoundError:
        return "0.0.0+unknown"

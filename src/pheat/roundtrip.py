"""Roundtrip comparison helpers shared by examples, CLI, and web UI."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from pheat.metrics import (
    align_structure_to_reference,
    atoms_by_rmsd_key,
    normalize_rmsd_atom_set,
    radius_of_gyration_delta,
    structure_radius_of_gyration,
    structure_rmsd,
)
from pheat.mmcif import structure_to_mmcif_string
from pheat.models import Atom, HeavyAtomStructure
from pheat.pdbio import structure_to_pdb_string
from pheat.residue_geometry import (
    ANGLE_UNITS,
    structure_from_residue_geometry,
    structure_to_residue_geometry,
)
from pheat.scoring import score_structure, supported_models

ANGLE_NAMES = ("omega", "tau", "theta")
CHI_LIMITS = (None, 1, 2)
SCORE_MODELS = ("generic", "pheat-dfire", "pheat-goap", "heavy-mm", "openmm-prepared")


@dataclass(frozen=True)
class RoundtripCaseSpec:
    """Options for one heavy-atom to residue-geometry to heavy-atom comparison."""

    case_id: str
    stored_angles: tuple[str, ...] = ()
    max_chi: Optional[int] = None
    angle_units: str = "radians"
    include_terminal_oxt: bool = False
    geometry_mode: Optional[str] = None
    geometry_table: Optional[Union[str, Path, Mapping[str, Any]]] = None
    geometry_profile: Optional[str] = None


def single_roundtrip_case_spec(
    *,
    angle_units: str = "radians",
    include_terminal_oxt: bool = False,
    geometry_mode: Optional[str] = None,
    geometry_table: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    geometry_profile: Optional[str] = None,
) -> RoundtripCaseSpec:
    return RoundtripCaseSpec(
        case_id="single",
        angle_units=_normalize_angle_units(angle_units),
        include_terminal_oxt=include_terminal_oxt,
        geometry_mode=geometry_mode,
        geometry_table=geometry_table,
        geometry_profile=geometry_profile,
    )


def configurable_roundtrip_case_spec(
    *,
    stored_angles: Optional[object] = None,
    max_chi: Optional[int] = None,
    angle_units: str = "radians",
    include_terminal_oxt: bool = False,
    geometry_mode: Optional[str] = None,
    geometry_table: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    geometry_profile: Optional[str] = None,
) -> RoundtripCaseSpec:
    normalized_angles = normalize_stored_angles(stored_angles)
    normalized_max_chi = normalize_max_chi(max_chi)
    return RoundtripCaseSpec(
        case_id=f"{angles_label(normalized_angles)}__{chi_label(normalized_max_chi)}",
        stored_angles=normalized_angles,
        max_chi=normalized_max_chi,
        angle_units=_normalize_angle_units(angle_units),
        include_terminal_oxt=include_terminal_oxt,
        geometry_mode=geometry_mode,
        geometry_table=geometry_table,
        geometry_profile=geometry_profile,
    )


def combinatorial_roundtrip_case_specs(
    *,
    angle_units: str = "radians",
    include_terminal_oxt: bool = False,
    geometry_mode: Optional[str] = None,
    geometry_table: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    geometry_profile: Optional[str] = None,
) -> list[RoundtripCaseSpec]:
    normalized_angle_units = _normalize_angle_units(angle_units)
    specs = []
    for stored_angles in angle_subsets():
        for max_chi in CHI_LIMITS:
            specs.append(
                RoundtripCaseSpec(
                    case_id=f"{angles_label(stored_angles)}__{chi_label(max_chi)}",
                    stored_angles=stored_angles,
                    max_chi=max_chi,
                    angle_units=normalized_angle_units,
                    include_terminal_oxt=include_terminal_oxt,
                    geometry_mode=geometry_mode,
                    geometry_table=geometry_table,
                    geometry_profile=geometry_profile,
                )
            )
    return specs


def run_roundtrip_cases(
    original: HeavyAtomStructure,
    case_specs: Sequence[RoundtripCaseSpec],
    *,
    score_models: Optional[Sequence[str]] = None,
    alignment_atom_set: str = "all-heavy",
) -> dict[str, Any]:
    """Run one or more residue-geometry roundtrips against a source structure."""

    models = normalize_score_models(score_models)
    original_scores = {model: score_payload(original, model) for model in models}
    original_radius_of_gyration = radius_of_gyration_payload(original)
    cases = [
        run_roundtrip_case(
            original,
            spec,
            original_scores=original_scores,
            original_radius_of_gyration=original_radius_of_gyration,
            score_models=models,
            alignment_atom_set=alignment_atom_set,
        )
        for spec in case_specs
    ]
    return {
        "input_name": original.name,
        "original_atom_count": len(original.atoms),
        "original_residue_count": len(original.residue_keys()),
        "score_models": list(models),
        "alignment_atom_set": normalize_rmsd_atom_set(alignment_atom_set),
        "original_scores": original_scores,
        "original_radius_of_gyration": original_radius_of_gyration,
        "case_count": len(cases),
        "cases": cases,
    }


def run_roundtrip_case(
    original: HeavyAtomStructure,
    spec: RoundtripCaseSpec,
    *,
    original_scores: Optional[Mapping[str, Mapping[str, Any]]] = None,
    original_radius_of_gyration: Optional[Mapping[str, Any]] = None,
    score_models: Optional[Sequence[str]] = None,
    alignment_atom_set: str = "all-heavy",
) -> dict[str, Any]:
    models = normalize_score_models(score_models)
    before_scores = dict(original_scores or {model: score_payload(original, model) for model in models})
    before_rg = dict(original_radius_of_gyration or radius_of_gyration_payload(original))
    residue_geometry = structure_to_residue_geometry(
        original,
        angle_units=spec.angle_units,
        stored_angles=spec.stored_angles,
        max_chi=spec.max_chi,
    )
    reconstructed = structure_from_residue_geometry(
        residue_geometry,
        include_terminal_oxt=spec.include_terminal_oxt,
        geometry_mode=spec.geometry_mode,
        geometry_table=spec.geometry_table,
        geometry_profile=spec.geometry_profile,
    )
    alignment = align_reconstructed_to_original(
        original,
        reconstructed,
        alignment_atom_set=alignment_atom_set,
    )
    scores = score_comparison(
        before_scores,
        alignment["reconstructed_aligned"],
        score_models=models,
    )
    reconstructed_rg = radius_of_gyration_payload(alignment["reconstructed_aligned"])
    return {
        "case_id": spec.case_id,
        "stored_angles": list(spec.stored_angles),
        "max_chi": spec.max_chi,
        "angle_units": spec.angle_units,
        "include_terminal_oxt": spec.include_terminal_oxt,
        "reconstruction_geometry": reconstructed.metadata.get("reconstruction_geometry"),
        "rmsd": {
            "all_heavy": alignment["all_heavy_rmsd"],
            "backbone": alignment["backbone_rmsd"],
            "c_alpha": alignment["c_alpha_rmsd"],
            "alignment_atom_set": alignment["alignment_atom_set"],
            "matched_heavy_atoms": alignment["matched_heavy_atoms"],
            "matched_backbone_atoms": alignment["matched_backbone_atoms"],
            "matched_c_alpha_atoms": alignment["matched_c_alpha_atoms"],
            "matched_alignment_atoms": alignment["matched_alignment_atoms"],
            "unmatched_original_atoms": alignment["unmatched_original_atoms"],
            "unmatched_reconstructed_atoms": alignment["unmatched_reconstructed_atoms"],
        },
        "scores": scores,
        "radius_of_gyration": {
            "reconstructed": reconstructed_rg,
            "delta": radius_of_gyration_delta(before_rg, reconstructed_rg),
        },
        "artifacts": {
            "residue_geometry_json": residue_geometry.to_json(),
            "reconstructed_heavy_json": reconstructed.to_json(),
            "original_aligned_pdb": structure_to_pdb_string(
                alignment["original_aligned"],
                allow_chain_truncation=True,
            ),
            "reconstructed_aligned_pdb": structure_to_pdb_string(
                alignment["reconstructed_aligned"],
                allow_chain_truncation=True,
            ),
            "original_aligned_mmcif": structure_to_mmcif_string(alignment["original_aligned"]),
            "reconstructed_aligned_mmcif": structure_to_mmcif_string(alignment["reconstructed_aligned"]),
        },
    }


def align_reconstructed_to_original(
    original: HeavyAtomStructure,
    reconstructed: HeavyAtomStructure,
    *,
    alignment_atom_set: str = "all-heavy",
) -> dict[str, Any]:
    normalized_alignment_atom_set = normalize_rmsd_atom_set(alignment_atom_set)
    original_atoms = atoms_by_roundtrip_key(original)
    reconstructed_atoms = atoms_by_roundtrip_key(reconstructed)
    common_keys = sorted(set(original_atoms) & set(reconstructed_atoms))
    if not common_keys:
        raise ValueError("No common atoms found for roundtrip alignment.")

    alignment = align_structure_to_reference(
        original,
        reconstructed,
        atom_set=normalized_alignment_atom_set,
    )
    all_heavy_rmsd = structure_rmsd(
        original,
        reconstructed,
        atom_set="all-heavy",
        alignment_atom_set="all-heavy",
    )
    backbone_rmsd = structure_rmsd(
        original,
        reconstructed,
        atom_set="backbone",
        alignment_atom_set="backbone",
    )
    c_alpha_rmsd = structure_rmsd(
        original,
        reconstructed,
        atom_set="ca",
        alignment_atom_set="ca",
    )
    original_aligned = copy_structure(original, name=f"{original.name}:aligned_reference")
    reconstructed_aligned = alignment["aligned_target"]
    reconstructed_aligned.name = f"{reconstructed.name}:kabsch_aligned_to_original"
    original_aligned.metadata = {
        **original_aligned.metadata,
        "alignment_role": "reference",
        "alignment_target": original.name,
        "alignment_atom_set": normalized_alignment_atom_set,
    }
    reconstructed_aligned.metadata = {
        **reconstructed_aligned.metadata,
        "alignment": "kabsch_to_original",
        "alignment_atom_set": normalized_alignment_atom_set,
        "matched_heavy_atoms": len(common_keys),
        "matched_backbone_atoms": backbone_rmsd["matched_atoms"],
        "matched_c_alpha_atoms": c_alpha_rmsd["matched_atoms"],
        "matched_alignment_atoms": alignment["matched_atoms"],
    }
    return {
        "original_aligned": original_aligned,
        "reconstructed_aligned": reconstructed_aligned,
        "all_heavy_rmsd": all_heavy_rmsd["value"],
        "backbone_rmsd": backbone_rmsd["value"],
        "c_alpha_rmsd": c_alpha_rmsd["value"],
        "alignment_atom_set": normalized_alignment_atom_set,
        "matched_heavy_atoms": len(common_keys),
        "matched_backbone_atoms": backbone_rmsd["matched_atoms"],
        "matched_c_alpha_atoms": c_alpha_rmsd["matched_atoms"],
        "matched_alignment_atoms": alignment["matched_atoms"],
        "unmatched_original_atoms": len(set(original_atoms) - set(reconstructed_atoms)),
        "unmatched_reconstructed_atoms": len(set(reconstructed_atoms) - set(original_atoms)),
    }


def score_comparison(
    original_scores: Mapping[str, Mapping[str, Any]],
    reconstructed: HeavyAtomStructure,
    *,
    score_models: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, Any]]:
    scores = {}
    for model in normalize_score_models(score_models):
        before = original_scores[model]
        after = score_payload(reconstructed, model)
        comparison: dict[str, Any] = {
            "reconstructed": after,
        }
        if before["status"] == "ok" and after["status"] == "ok":
            comparison["delta"] = after["total"] - before["total"]
            comparison["units"] = after["units"]
        else:
            comparison["delta"] = None
            comparison["units"] = after.get("units") or before.get("units")
        scores[model] = comparison
    return scores


def score_payload(structure: HeavyAtomStructure, model: str) -> dict[str, Any]:
    try:
        result = score_structure(structure, model=model).to_dict()
    except Exception as exc:
        return {
            "model": model,
            "status": "unavailable",
            "error": str(exc),
            "total": None,
            "units": None,
            "terms": {},
            "warnings": [],
        }
    result["status"] = "ok"
    return result


def radius_of_gyration_payload(structure: HeavyAtomStructure) -> dict[str, Any]:
    """Return the standard roundtrip Rg payload for an atom structure."""

    try:
        return structure_radius_of_gyration(structure, mode="both")
    except Exception as exc:
        return {
            "atom_count": len(structure.atoms),
            "atom_set": "all-heavy",
            "mode": "both",
            "status": "unavailable",
            "error": str(exc),
            "units": "angstrom",
            "values": {},
        }


def write_roundtrip_artifacts(
    result: Mapping[str, Any],
    output_root: Union[str, Path],
    *,
    include_mmcif: bool = False,
) -> dict[str, Any]:
    """Write roundtrip outputs and return a summary with relative artifact paths."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "residue_geometry": output_root / "residue_geometry",
        "heavy": output_root / "heavy",
        "pdb": output_root / "pdb",
        "metrics": output_root / "metrics",
    }
    if include_mmcif:
        subdirs["mmcif"] = output_root / "mmcif"
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    cases = []
    for case in result["cases"]:
        case_id = safe_case_id(case["case_id"])
        paths = {
            "residue_geometry": subdirs["residue_geometry"] / f"{case_id}.residue-geometry.json",
            "reconstructed_heavy": subdirs["heavy"] / f"{case_id}.reconstructed.heavy.json",
            "original_aligned_pdb": subdirs["pdb"] / f"{case_id}.original_aligned.pdb",
            "reconstructed_aligned_pdb": subdirs["pdb"] / f"{case_id}.reconstructed_aligned.pdb",
            "metrics": subdirs["metrics"] / f"{case_id}.metrics.json",
        }
        if include_mmcif:
            paths["original_aligned_mmcif"] = subdirs["mmcif"] / f"{case_id}.original_aligned.cif"
            paths["reconstructed_aligned_mmcif"] = (
                subdirs["mmcif"] / f"{case_id}.reconstructed_aligned.cif"
            )
        artifacts = case["artifacts"]
        paths["residue_geometry"].write_text(artifacts["residue_geometry_json"] + "\n", encoding="utf-8")
        paths["reconstructed_heavy"].write_text(artifacts["reconstructed_heavy_json"] + "\n", encoding="utf-8")
        paths["original_aligned_pdb"].write_text(artifacts["original_aligned_pdb"], encoding="utf-8")
        paths["reconstructed_aligned_pdb"].write_text(
            artifacts["reconstructed_aligned_pdb"],
            encoding="utf-8",
        )
        if include_mmcif:
            paths["original_aligned_mmcif"].write_text(
                artifacts["original_aligned_mmcif"],
                encoding="utf-8",
            )
            paths["reconstructed_aligned_mmcif"].write_text(
                artifacts["reconstructed_aligned_mmcif"],
                encoding="utf-8",
            )

        summary_case = {key: value for key, value in case.items() if key != "artifacts"}
        summary_case["paths"] = {
            name: path.relative_to(output_root).as_posix()
            for name, path in paths.items()
        }
        paths["metrics"].write_text(_json_dumps(summary_case), encoding="utf-8")
        cases.append(summary_case)

    summary = {key: value for key, value in result.items() if key != "cases"}
    summary["cases"] = cases
    summary_path = output_root / "summary.json"
    summary["paths"] = {"summary": summary_path.relative_to(output_root).as_posix()}
    summary_path.write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def normalize_stored_angles(value: Optional[object]) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    elif isinstance(value, Iterable):
        parts = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        raise ValueError("stored angles must be blank, a comma-separated string, or an iterable.")
    if not parts:
        return ()
    if "all" in parts:
        if len(parts) > 1:
            raise ValueError("'all' cannot be combined with individual stored angles.")
        return ANGLE_NAMES
    invalid = [part for part in parts if part not in ANGLE_NAMES]
    if invalid:
        raise ValueError(f"Unsupported stored angles: {', '.join(invalid)}")
    ordered = tuple(angle for angle in ANGLE_NAMES if angle in set(parts))
    return ordered


def normalize_max_chi(value: Optional[object]) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().lower() in {"all", "none"}:
        return None
    if isinstance(value, bool):
        raise ValueError("max_chi must be a non-negative integer or blank for all.")
    normalized = int(str(value))
    if normalized < 0:
        raise ValueError("max_chi must be a non-negative integer or blank for all.")
    return normalized


def normalize_score_models(score_models: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not score_models:
        return SCORE_MODELS
    supported = set(supported_models())
    models = tuple(str(model).strip().lower() for model in score_models if str(model).strip())
    invalid = [model for model in models if model not in supported]
    if invalid:
        raise ValueError(f"Unsupported scoring models: {', '.join(invalid)}")
    if not models:
        return ("generic",)
    return models


def angle_subsets() -> Iterable[tuple[str, ...]]:
    for size in range(len(ANGLE_NAMES) + 1):
        yield from itertools.combinations(ANGLE_NAMES, size)


def angles_label(stored_angles: Sequence[str]) -> str:
    if not stored_angles:
        return "angles-none"
    if tuple(stored_angles) == ANGLE_NAMES:
        return "angles-all"
    return "angles-" + "_".join(stored_angles)


def chi_label(max_chi: Optional[int]) -> str:
    return "chi-all" if max_chi is None else f"chi-{max_chi}"


def atoms_by_roundtrip_key(structure: HeavyAtomStructure) -> dict[tuple[Any, str], Atom]:
    return atoms_by_rmsd_key(structure, atom_set="all-heavy")


def copy_structure(structure: HeavyAtomStructure, *, name: str) -> HeavyAtomStructure:
    return HeavyAtomStructure(
        atoms=[replace(atom) for atom in structure.atoms],
        name=name,
        metadata=dict(structure.metadata),
        disulfide_bonds=list(structure.disulfide_bonds),
    )


def copy_structure_with_coords(
    structure: HeavyAtomStructure,
    coords_by_key: Mapping[tuple[Any, str], Sequence[float]],
    *,
    name: str,
) -> HeavyAtomStructure:
    atoms = []
    for atom in structure.atoms:
        key = (atom.residue_key, atom.name.strip().upper())
        coord = coords_by_key.get(key, atom.coord)
        atoms.append(
            replace(
                atom,
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
            )
        )
    return HeavyAtomStructure(
        atoms=atoms,
        name=name,
        metadata=dict(structure.metadata),
        disulfide_bonds=list(structure.disulfide_bonds),
    )


def safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "case"


def _normalize_angle_units(angle_units: str) -> str:
    normalized = str(angle_units).strip().lower()
    if normalized not in ANGLE_UNITS:
        raise ValueError(f"angle_units must be one of {', '.join(ANGLE_UNITS)}")
    return normalized


def _json_dumps(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

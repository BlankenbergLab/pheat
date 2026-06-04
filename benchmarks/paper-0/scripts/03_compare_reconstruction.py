from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from _common import base_result, load_structure, package_available, runtime_executable, selected_manifest_rows, write_rows
from pheat.metrics import align_structure_to_reference, atoms_by_rmsd_key, structure_rmsd
from pheat.models import Atom, HeavyAtomStructure
from pheat.pdbio import write_pdb
from pheat.residue_geometry import structure_from_residue_geometry, structure_to_residue_geometry


COMPARATORS = ("pheat", "biopython-internal_coords", "peptidebuilder", "pulchra")
RMSD_ATOM_SETS = ("ca", "backbone", "all-heavy")
BACKBONE_OR_TERMINAL_ATOMS = {"N", "CA", "C", "O", "OXT"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare reconstruction paths for manifest structures.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparators", default=",".join(COMPARATORS))
    parser.add_argument("--pulchra", help="Optional path to PULCHRA executable.")
    parser.add_argument(
        "--comparison-output-root",
        help="Directory for PHEAT/PULCHRA reduced/artifact reconstruction outputs.",
    )
    args = parser.parse_args(argv)

    rows = []
    comparators = [item.strip().lower() for item in args.comparators.split(",") if item.strip()]
    comparison_output_root = (
        Path(args.comparison_output_root)
        if args.comparison_output_root
        else Path(args.output).parent / "reconstruction"
    )
    for manifest_row in selected_manifest_rows(args.manifest):
        for comparator in comparators:
            rows.append(_run_comparator(manifest_row, comparator, pulchra=args.pulchra))
        rows.extend(
            _run_pheat_pulchra_comparisons(
                manifest_row,
                output_root=comparison_output_root,
                pulchra=args.pulchra,
            )
        )
    write_rows(args.output, rows)
    return 0


def _run_comparator(row: Mapping[str, Any], comparator: str, *, pulchra: str | None) -> dict[str, Any]:
    result = base_result(row, comparator=comparator, benchmark="reconstruction")
    start = time.perf_counter()
    result.update(
        {
            "reconstruction_success": False,
            "failure_reason": None,
            "ca_rmsd": None,
            "backbone_rmsd": None,
            "all_heavy_rmsd": None,
            "sidechain_heavy_rmsd": None,
            "bond_length_mae": None,
            "bond_angle_mae": None,
            "chi1_circular_error_deg": None,
            "chi2_circular_error_deg": None,
            "ring_closure_failure": None,
            "disulfide_preservation": None,
        }
    )
    try:
        if comparator == "pheat":
            structure = load_structure(str(row["source_path"]), str(row.get("source_format") or ""))
            geometry = structure_to_residue_geometry(
                structure,
                stored_angles=("omega", "tau", "theta"),
                stored_lengths="all",
            )
            rebuilt = structure_from_residue_geometry(geometry)
            result.update(
                {
                    "comparator_status": "ok",
                    "reconstruction_success": True,
                    "ca_rmsd": _rmsd_value(structure, rebuilt, "ca"),
                    "backbone_rmsd": _rmsd_value(structure, rebuilt, "backbone"),
                    "all_heavy_rmsd": _rmsd_value(structure, rebuilt, "all-heavy"),
                    "disulfide_preservation": len(structure.disulfide_bonds) == len(rebuilt.disulfide_bonds),
                }
            )
        elif comparator == "biopython-internal_coords":
            result.update(_optional_python_status("Bio.PDB.internal_coords"))
        elif comparator == "peptidebuilder":
            result.update(_optional_python_status("PeptideBuilder"))
        elif comparator == "pulchra":
            executable = pulchra or runtime_executable("pulchra")
            if executable:
                result.update(
                    {
                        "comparator_status": "installed_not_run",
                        "failure_reason": (
                            "summary compatibility row only; see reconstruction_comparison rows "
                            "for reduced-input PULCHRA runs"
                        ),
                    }
                )
            else:
                result.update({"comparator_status": "not_installed", "failure_reason": "PULCHRA executable not configured"})
        else:
            result.update({"comparator_status": "unknown_comparator", "failure_reason": comparator})
    except Exception as exc:
        result.update({"comparator_status": "error", "reconstruction_success": False, "failure_reason": str(exc)})
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def _optional_python_status(module: str) -> dict[str, Any]:
    if not package_available(module):
        return {
            "comparator_status": "not_installed",
            "reconstruction_success": False,
            "failure_reason": "optional reconstruction comparator is not installed",
        }
    return {
        "comparator_status": "installed_not_run",
        "reconstruction_success": False,
        "failure_reason": "adapter scaffold present; comparable input/output contract still requires paper-scale implementation",
    }


def _rmsd_value(reference: Any, target: Any, atom_set: str) -> float | None:
    try:
        return float(structure_rmsd(reference, target, atom_set=atom_set)["value"])
    except Exception:
        return None


def _run_pheat_pulchra_comparisons(
    manifest_row: Mapping[str, Any],
    *,
    output_root: Path,
    pulchra: str | None,
) -> list[dict[str, Any]]:
    structure = load_structure(str(manifest_row["source_path"]), str(manifest_row.get("source_format") or ""))
    record_dir = output_root / _safe_path_part(str(manifest_row.get("record_id") or "record"))
    return [
        _run_comparison_task(
            manifest_row,
            structure=structure,
            output_dir=record_dir / "reduced-input",
            comparison_task="reduced-input",
            pheat_input_contract="ca-only-derived-residue-geometry",
            pulchra_input_contract="ca-only",
            pulchra=pulchra,
        ),
        _run_comparison_task(
            manifest_row,
            structure=structure,
            output_dir=record_dir / "artifact-assisted",
            comparison_task="artifact-assisted",
            pheat_input_contract="pheat-residue-geometry-artifact",
            pulchra_input_contract="ca-only",
            pulchra=pulchra,
        ),
    ]


def _run_comparison_task(
    manifest_row: Mapping[str, Any],
    *,
    structure: HeavyAtomStructure,
    output_dir: Path,
    comparison_task: str,
    pheat_input_contract: str,
    pulchra_input_contract: str,
    pulchra: str | None,
) -> dict[str, Any]:
    result = base_result(manifest_row, comparator="pheat-vs-pulchra", benchmark="reconstruction_comparison")
    start = time.perf_counter()
    result.update(
        {
            "comparison_task": comparison_task,
            "pheat_input_contract": pheat_input_contract,
            "pulchra_input_contract": pulchra_input_contract,
            "reconstruction_success": False,
            "pheat_status": "not_run",
            "pulchra_status": "not_run",
            "failure_reason": None,
            "reduced_input_pdb": None,
            "pheat_artifact_path": None,
            "pheat_output_pdb": None,
            "pulchra_output_pdb": None,
        }
    )

    pheat_structure: HeavyAtomStructure | None = None
    pulchra_structure: HeavyAtomStructure | None = None
    failure_reasons: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    ca_input_path = output_dir / "input-ca-only.pdb"
    pheat_artifact_path = output_dir / "pheat-residue-geometry.json"
    pheat_output_path = output_dir / "pheat-rebuilt.pdb"
    pulchra_output_path = output_dir / "pulchra-rebuilt.pdb"
    result.update(
        {
            "reduced_input_pdb": str(ca_input_path),
            "pheat_artifact_path": str(pheat_artifact_path),
            "pheat_output_pdb": str(pheat_output_path),
            "pulchra_output_pdb": str(pulchra_output_path),
        }
    )

    try:
        ca_structure = _ca_only_structure(structure)
        write_pdb(
            ca_structure,
            ca_input_path,
            allow_chain_truncation=True,
            remarks=[
                f"paper-0 {comparison_task} reduced input",
                "Contains CA atoms only; generated from the benchmark reference structure.",
            ],
        )
    except Exception as exc:
        result["comparator_status"] = "error"
        result["failure_reason"] = f"failed to create CA-only input: {exc}"
        result["runtime_seconds"] = time.perf_counter() - start
        return result

    try:
        source_for_pheat = ca_structure if comparison_task == "reduced-input" else structure
        geometry = structure_to_residue_geometry(
            source_for_pheat,
            stored_angles=("omega", "tau", "theta"),
            stored_lengths="all",
        )
        pheat_artifact_path.write_text(geometry.to_json(indent=2) + "\n", encoding="utf-8")
        pheat_structure = structure_from_residue_geometry(geometry)
        write_pdb(
            pheat_structure,
            pheat_output_path,
            allow_chain_truncation=True,
            remarks=[
                f"paper-0 {comparison_task} PHEAT reconstruction",
                f"input contract: {pheat_input_contract}",
            ],
        )
        result["pheat_status"] = "ok"
    except Exception as exc:
        result["pheat_status"] = "error"
        failure_reasons.append(f"PHEAT {comparison_task}: {exc}")

    executable = pulchra or runtime_executable("pulchra")
    if not executable:
        result["pulchra_status"] = "not_installed"
        failure_reasons.append("PULCHRA executable not configured")
    else:
        try:
            _run_pulchra(
                executable=executable,
                input_path=ca_input_path,
                output_path=pulchra_output_path,
            )
            pulchra_structure = load_structure(pulchra_output_path, "pdb")
            result["pulchra_status"] = "ok"
        except Exception as exc:
            result["pulchra_status"] = "error"
            failure_reasons.append(f"PULCHRA {comparison_task}: {exc}")

    if pheat_structure is not None:
        _add_rmsd_fields(result, "pheat_vs_original", structure, pheat_structure)
    if pulchra_structure is not None:
        _add_rmsd_fields(result, "pulchra_vs_original", structure, pulchra_structure)
    if pheat_structure is not None and pulchra_structure is not None:
        _add_rmsd_fields(result, "pheat_vs_pulchra", pheat_structure, pulchra_structure)

    result["reconstruction_success"] = result["pheat_status"] == "ok" and result["pulchra_status"] == "ok"
    result["comparator_status"] = "ok" if result["reconstruction_success"] else "error"
    result["failure_reason"] = "; ".join(failure_reasons) if failure_reasons else None
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def _run_pulchra(*, executable: str, input_path: Path, output_path: Path) -> None:
    completed = subprocess.run(
        [executable, str(input_path.resolve()), "-O", str(output_path.resolve())],
        cwd=output_path.parent,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(detail)
    if not output_path.exists():
        raise RuntimeError(f"PULCHRA did not create {output_path}")


def _ca_only_structure(structure: HeavyAtomStructure) -> HeavyAtomStructure:
    ca_atoms = [
        replace(atom, serial=index)
        for index, atom in enumerate(structure.atoms, start=1)
        if atom.name.strip().upper() == "CA"
    ]
    if not ca_atoms:
        raise ValueError("source structure contains no CA atoms")
    return HeavyAtomStructure(
        atoms=ca_atoms,
        name=f"{structure.name}:ca-only",
        metadata={
            **structure.metadata,
            "paper_0_reduced_input": "ca-only",
            "source_name": structure.name,
        },
        disulfide_bonds=[],
    )


def _add_rmsd_fields(
    result: dict[str, Any],
    prefix: str,
    reference: HeavyAtomStructure,
    target: HeavyAtomStructure,
) -> None:
    for atom_set in RMSD_ATOM_SETS:
        field = atom_set.replace("-", "_")
        try:
            payload = structure_rmsd(reference, target, atom_set=atom_set)
            result[f"{prefix}_{field}_rmsd"] = float(payload["value"])
            result[f"{prefix}_{field}_matched_atoms"] = int(payload["matched_atoms"])
            result[f"{prefix}_{field}_unmatched_reference_atoms"] = int(payload["unmatched_reference_atoms"])
            result[f"{prefix}_{field}_unmatched_target_atoms"] = int(payload["unmatched_target_atoms"])
        except Exception as exc:
            result[f"{prefix}_{field}_rmsd"] = None
            result[f"{prefix}_{field}_matched_atoms"] = 0
            result[f"{prefix}_{field}_rmsd_error"] = str(exc)

    sidechain_payload = _sidechain_heavy_rmsd(reference, target)
    result[f"{prefix}_sidechain_heavy_rmsd"] = sidechain_payload["value"]
    result[f"{prefix}_sidechain_heavy_matched_atoms"] = sidechain_payload["matched_atoms"]
    result[f"{prefix}_sidechain_heavy_unmatched_reference_atoms"] = sidechain_payload["unmatched_reference_atoms"]
    result[f"{prefix}_sidechain_heavy_unmatched_target_atoms"] = sidechain_payload["unmatched_target_atoms"]


def _sidechain_heavy_rmsd(reference: HeavyAtomStructure, target: HeavyAtomStructure) -> dict[str, Any]:
    reference_atoms = _sidechain_heavy_atoms_by_key(reference)
    target_atoms = _sidechain_heavy_atoms_by_key(_align_or_return(reference, target))
    common_keys = sorted(set(reference_atoms) & set(target_atoms))
    if not common_keys:
        return {
            "value": None,
            "matched_atoms": 0,
            "unmatched_reference_atoms": len(reference_atoms),
            "unmatched_target_atoms": len(target_atoms),
        }
    squared_distances = []
    for key in common_keys:
        reference_atom = reference_atoms[key]
        target_atom = target_atoms[key]
        squared_distances.append(
            (reference_atom.x - target_atom.x) ** 2
            + (reference_atom.y - target_atom.y) ** 2
            + (reference_atom.z - target_atom.z) ** 2
        )
    return {
        "value": math.sqrt(sum(squared_distances) / len(squared_distances)),
        "matched_atoms": len(common_keys),
        "unmatched_reference_atoms": len(set(reference_atoms) - set(target_atoms)),
        "unmatched_target_atoms": len(set(target_atoms) - set(reference_atoms)),
    }


def _align_or_return(reference: HeavyAtomStructure, target: HeavyAtomStructure) -> HeavyAtomStructure:
    try:
        return align_structure_to_reference(reference, target, atom_set="ca")["aligned_target"]
    except Exception:
        return target


def _sidechain_heavy_atoms_by_key(structure: HeavyAtomStructure) -> dict[tuple[Any, str], Atom]:
    return {
        key: atom
        for key, atom in atoms_by_rmsd_key(structure, atom_set="all-heavy").items()
        if atom.name.strip().upper() not in BACKBONE_OR_TERMINAL_ATOMS
    }


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return safe or "record"


if __name__ == "__main__":
    raise SystemExit(main())

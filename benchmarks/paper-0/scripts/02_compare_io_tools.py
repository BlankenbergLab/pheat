from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

from _common import (
    base_result,
    load_structure,
    package_available,
    selected_manifest_rows,
    source_format_from_path,
    structure_counts,
    write_rows,
)


COMPARATORS = ("pheat", "biopython", "gemmi", "mdanalysis", "mdtraj")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare PHEAT structure I/O with optional tools.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparators", default=",".join(COMPARATORS))
    parser.add_argument("--require-comparators", action="store_true")
    args = parser.parse_args(argv)

    rows = []
    missing_required = False
    comparators = [item.strip().lower() for item in args.comparators.split(",") if item.strip()]
    for manifest_row in selected_manifest_rows(args.manifest):
        for comparator in comparators:
            result = _run_comparator(manifest_row, comparator)
            rows.append(result)
            if args.require_comparators and comparator != "pheat" and result["comparator_status"] == "not_installed":
                missing_required = True
    write_rows(args.output, rows)
    return 1 if missing_required else 0


def _run_comparator(row: Mapping[str, Any], comparator: str) -> dict[str, Any]:
    result = base_result(row, comparator=comparator, benchmark="structure_io")
    start = time.perf_counter()
    try:
        source_path = Path(str(row.get("source_path") or ""))
        source_format = str(row.get("source_format") or source_format_from_path(source_path))
        if comparator == "pheat":
            structure = load_structure(source_path, source_format)
            result.update(structure_counts(structure))
            result.update({"comparator_status": "ok", "parse_success": True, "failure_reason": None})
        elif comparator == "biopython":
            result.update(_run_biopython(source_path, source_format))
        elif comparator == "gemmi":
            result.update(_run_gemmi(source_path))
        elif comparator == "mdanalysis":
            result.update(_run_mdanalysis(source_path))
        elif comparator == "mdtraj":
            result.update(_run_mdtraj(source_path))
        else:
            result.update({"comparator_status": "unknown_comparator", "parse_success": False, "failure_reason": comparator})
    except Exception as exc:
        result.update({"comparator_status": "error", "parse_success": False, "failure_reason": str(exc)})
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def _run_biopython(source_path: Path, source_format: str) -> dict[str, Any]:
    if not package_available("Bio.PDB"):
        return _not_installed()
    from Bio.PDB import MMCIFParser, PDBParser

    if source_format == "pdb":
        structure = PDBParser(QUIET=True).get_structure(source_path.stem, str(source_path))
    elif source_format == "mmcif":
        structure = MMCIFParser(QUIET=True).get_structure(source_path.stem, str(source_path))
    else:
        return {"comparator_status": "unsupported_format", "parse_success": False, "failure_reason": source_format}
    atoms = list(structure.get_atoms())
    return {
        "comparator_status": "ok",
        "parse_success": True,
        "failure_reason": None,
        "atom_count": len(atoms),
        "protein_heavy_atom_count": None,
        "residue_count": len(list(structure.get_residues())),
        "chain_count": len(list(structure.get_chains())),
    }


def _run_gemmi(source_path: Path) -> dict[str, Any]:
    if not package_available("gemmi"):
        return _not_installed()
    import gemmi

    structure = gemmi.read_structure(str(source_path))
    atom_count = sum(1 for model in structure for chain in model for residue in chain for _atom in residue)
    residue_count = sum(1 for model in structure for chain in model for _residue in chain)
    chain_count = sum(1 for model in structure for _chain in model)
    return {
        "comparator_status": "ok",
        "parse_success": True,
        "failure_reason": None,
        "atom_count": atom_count,
        "protein_heavy_atom_count": None,
        "residue_count": residue_count,
        "chain_count": chain_count,
    }


def _run_mdanalysis(source_path: Path) -> dict[str, Any]:
    if not package_available("MDAnalysis"):
        return _not_installed()
    import MDAnalysis as mda

    universe = mda.Universe(str(source_path))
    return {
        "comparator_status": "ok",
        "parse_success": True,
        "failure_reason": None,
        "atom_count": int(universe.atoms.n_atoms),
        "protein_heavy_atom_count": int(universe.select_atoms("protein and not name H*").n_atoms),
        "residue_count": int(universe.residues.n_residues),
        "chain_count": len(set(universe.atoms.chainIDs)) if hasattr(universe.atoms, "chainIDs") else None,
    }


def _run_mdtraj(source_path: Path) -> dict[str, Any]:
    if not package_available("mdtraj"):
        return _not_installed()
    import mdtraj as md

    trajectory = md.load(str(source_path))
    topology = trajectory.topology
    return {
        "comparator_status": "ok",
        "parse_success": True,
        "failure_reason": None,
        "atom_count": topology.n_atoms,
        "protein_heavy_atom_count": sum(1 for atom in topology.atoms if atom.residue.is_protein and atom.element.symbol != "H"),
        "residue_count": topology.n_residues,
        "chain_count": topology.n_chains,
    }


def _not_installed() -> dict[str, Any]:
    return {
        "comparator_status": "not_installed",
        "parse_success": False,
        "failure_reason": "optional comparator is not installed",
        "atom_count": None,
        "protein_heavy_atom_count": None,
        "residue_count": None,
        "chain_count": None,
    }


if __name__ == "__main__":
    raise SystemExit(main())

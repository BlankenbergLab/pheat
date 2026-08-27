"""Optional all-atom preparation helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence, cast

from pheat.models import Atom, HeavyAtomStructure

HYDROGEN_POLICIES = ("drop", "preserve", "generate")


def normalize_hydrogen_policy(value: str | None) -> str:
    normalized = str(value or "drop").strip().lower()
    if normalized not in HYDROGEN_POLICIES:
        raise ValueError(f"hydrogen policy must be one of {', '.join(HYDROGEN_POLICIES)}")
    return normalized


def is_hydrogen_element(element: str) -> bool:
    return element.strip().upper() in {"H", "D", "T"}


def generate_hydrogens(structure: HeavyAtomStructure) -> HeavyAtomStructure:
    """Return a copy of ``structure`` with hydrogens added by OpenMM Modeller.

    This is an optional dependency path because chemically reasonable hydrogen
    placement depends on a force-field topology and protonation assumptions.
    """

    try:
        from openmm import Platform, app, unit
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Hydrogen generation requires OpenMM. Install PHEAT with `.[all]` "
            "or use a conda environment that provides OpenMM."
        ) from exc

    from pheat.pdbio import structure_to_pdb_string

    try:
        with tempfile.TemporaryDirectory(prefix="pheat-hydrogens-") as tmpdir:
            pdb_path = Path(tmpdir) / "input.pdb"
            pdb_path.write_text(
                structure_to_pdb_string(structure, allow_chain_truncation=True),
                encoding="utf-8",
            )
            pdb = app.PDBFile(str(pdb_path))
            modeller = app.Modeller(pdb.topology, pdb.positions)
            forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            platform = Platform.getPlatformByName("Reference")
            modeller.addHydrogens(forcefield, pH=7.0, platform=platform)
            return _structure_from_openmm_topology(
                modeller.topology,
                modeller.positions.value_in_unit(unit.angstrom),
                source=structure,
            )
    except Exception as exc:  # pragma: no cover - input chemistry dependent
        raise RuntimeError(f"OpenMM could not generate hydrogens for this structure: {exc}") from exc


def _structure_from_openmm_topology(
    topology,
    positions: Sequence[object],
    *,
    source: HeavyAtomStructure,
) -> HeavyAtomStructure:
    atoms: list[Atom] = []
    source_metadata = dict(source.metadata)
    source_metadata["hydrogen_policy"] = "generated"
    source_metadata.setdefault("warnings", [])
    source_metadata["warnings"] = list(source_metadata["warnings"]) + [
        "hydrogens were generated with OpenMM Modeller at pH 7.0"
    ]

    for atom_index, openmm_atom in enumerate(topology.atoms()):
        residue = openmm_atom.residue
        chain = residue.chain
        coord = cast(Sequence[Any], positions[atom_index])
        element = getattr(getattr(openmm_atom, "element", None), "symbol", "") or ""
        atoms.append(
            Atom(
                name=str(openmm_atom.name),
                element=str(element).upper(),
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                resname=str(residue.name).upper(),
                chain_id=str(chain.id or "A"),
                resseq=_openmm_resseq(residue, atom_index),
                record_name="ATOM",
                serial=atom_index + 1,
                model=1,
                metadata={"hydrogen_generator": "openmm-modeller"},
            )
        )

    return HeavyAtomStructure(
        atoms=atoms,
        name=source.name,
        metadata=source_metadata,
        disulfide_bonds=list(source.disulfide_bonds),
        atom_scope="all",
    )


def _openmm_resseq(residue, fallback_index: int) -> int:
    try:
        return int(residue.id)
    except Exception:
        return fallback_index + 1

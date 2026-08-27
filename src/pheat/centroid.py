"""Reduced side-chain centroid representations."""

from __future__ import annotations

from typing import Dict, Iterable, List

from pheat.geometry import centroid
from pheat.models import Atom, Centroid, CentroidStructure, HeavyAtomStructure

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
TERMINAL_GROUPS = {
    "near": {"CB", "CG", "CG1", "CG2", "OG", "OG1", "SG"},
    "mid": {"CD", "CD1", "CD2", "ND1", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "NE", "NE1"},
    "far": {"CE3", "CZ", "CZ2", "CZ3", "NZ", "OE1", "OE2", "NE2", "NH1", "NH2", "OH", "CH2"},
}


def to_centroid_structure(structure: HeavyAtomStructure, *, mode: str = "single") -> CentroidStructure:
    """Convert side-chain heavy atoms into one or multiple residue centroids."""

    if mode not in {"single", "multi"}:
        raise ValueError("Centroid mode must be 'single' or 'multi'.")

    centroids: List[Centroid] = []
    for key, atoms in structure.atoms_by_residue().items():
        sidechain = [atom for atom in atoms if atom.name.strip().upper() not in BACKBONE_ATOMS]
        if not sidechain:
            continue
        if mode == "single":
            centroids.append(_make_centroid("SC", sidechain))
        else:
            for group_name, group_atoms in _group_sidechain_atoms(sidechain).items():
                if group_atoms:
                    centroids.append(_make_centroid(f"SC_{group_name.upper()}", group_atoms))

    return CentroidStructure(
        centroids=centroids,
        name=structure.name,
        metadata={"source": "atom_structure", "mode": mode},
    )


def _group_sidechain_atoms(atoms: Iterable[Atom]) -> Dict[str, List[Atom]]:
    grouped: Dict[str, List[Atom]] = {"near": [], "mid": [], "far": []}
    for atom in atoms:
        atom_name = atom.name.strip().upper()
        placed = False
        for group_name, names in TERMINAL_GROUPS.items():
            if atom_name in names:
                grouped[group_name].append(atom)
                placed = True
                break
        if not placed:
            grouped["mid"].append(atom)
    return grouped


def _make_centroid(name: str, atoms: List[Atom]) -> Centroid:
    x, y, z = centroid([atom.coord for atom in atoms])
    first = atoms[0]
    return Centroid(
        name=name,
        x=x,
        y=y,
        z=z,
        resname=first.resname,
        chain_id=first.chain_id,
        resseq=first.resseq,
        atom_names=[atom.name for atom in atoms],
        metadata={"atom_count": len(atoms)},
    )

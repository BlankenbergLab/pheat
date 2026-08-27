"""Connectivity helpers for optional atom-structure bond storage."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from pheat.geometry import distance
from pheat.models import Atom, Bond, HeavyAtomStructure, ResidueKey
from pheat.residues import SIDECHAIN_STEPS

BOND_STORAGE_MODES = ("none", "declared", "template", "all")

_BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"))
_AROMATIC_RING_CLOSURES = {
    "PHE": (("CE2", "CZ"),),
    "TYR": (("CE2", "CZ"),),
    "PTR": (("CE2", "CZ"),),
    "HIS": (("CE1", "NE2"),),
    "TRP": (("NE1", "CE2"), ("CZ3", "CH2")),
}
_SPECIAL_RING_CLOSURES = {
    "PRO": (("N", "CD"),),
    "HYP": (("N", "CD"),),
    "PCA": (("N", "CD"),),
    "PYL": (("CA2", "CG2"),),
}


def normalize_bond_storage(value: str | None) -> str:
    normalized = str(value or "none").strip().lower()
    if normalized not in BOND_STORAGE_MODES:
        raise ValueError(f"bond storage must be one of {', '.join(BOND_STORAGE_MODES)}")
    return normalized


def structure_with_bonds(
    structure: HeavyAtomStructure,
    *,
    mode: str | None = "none",
    declared_pairs: Iterable[tuple[int, int]] = (),
) -> HeavyAtomStructure:
    """Return ``structure`` with optional declared/template bond records attached.

    Bond indices reference the zero-based position of each atom in
    ``structure.atoms``. Lengths are measured from the current coordinates so the
    bond list documents the actual stored geometry, not ideal template values.
    """

    normalized = normalize_bond_storage(mode)
    if normalized == "none":
        return HeavyAtomStructure(
            atoms=list(structure.atoms),
            name=structure.name,
            metadata=dict(structure.metadata),
            disulfide_bonds=list(structure.disulfide_bonds),
            atom_scope=structure.atom_scope,
        )

    pairs: list[tuple[int, int, str]] = []
    if normalized in {"declared", "all"}:
        pairs.extend((first, second, "declared") for first, second in declared_pairs)
        pairs.extend(_disulfide_pairs(structure))
    if normalized in {"template", "all"}:
        pairs.extend(_template_pairs(structure))

    bonds = [
        _bond_from_pair(structure.atoms, first, second, source=source)
        for first, second, source in pairs
        if _valid_pair(first, second, len(structure.atoms))
    ]
    return HeavyAtomStructure(
        atoms=list(structure.atoms),
        name=structure.name,
        metadata=dict(structure.metadata),
        disulfide_bonds=list(structure.disulfide_bonds),
        bonds=bonds,
        atom_scope=structure.atom_scope,
    )


def declared_pairs_from_serials(
    atoms: Sequence[Atom],
    serial_pairs: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    index_by_serial = {int(atom.serial): index for index, atom in enumerate(atoms) if atom.serial is not None}
    pairs = []
    for serial_1, serial_2 in serial_pairs:
        index_1 = index_by_serial.get(serial_1)
        index_2 = index_by_serial.get(serial_2)
        if index_1 is None or index_2 is None:
            continue
        pairs.append((index_1, index_2))
    return pairs


def _bond_from_pair(atoms: Sequence[Atom], index_1: int, index_2: int, *, source: str) -> Bond:
    atom_1 = atoms[index_1]
    atom_2 = atoms[index_2]
    return Bond(
        atom_index_1=index_1,
        atom_index_2=index_2,
        source=source,
        length=distance(atom_1.coord, atom_2.coord),
        metadata={
            "atom_1": _atom_label(atom_1),
            "atom_2": _atom_label(atom_2),
        },
    )


def _template_pairs(structure: HeavyAtomStructure) -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []
    atom_index = {id(atom): index for index, atom in enumerate(structure.atoms)}
    residues = list(structure.atoms_by_residue().items())
    previous_key: ResidueKey | None = None
    previous_atoms: Mapping[str, Atom] | None = None

    for key, atoms in residues:
        resname = key[3].strip().upper()
        by_name = _atoms_by_name(atoms)
        for name_1, name_2 in _BACKBONE_BONDS:
            pairs.extend(_local_pair(by_name, atom_index, name_1, name_2, "template"))
        if resname != "GLY":
            pairs.extend(_local_pair(by_name, atom_index, "CA", "CB", "template"))
        for step in SIDECHAIN_STEPS.get(resname, []):
            pairs.extend(_local_pair(by_name, atom_index, step.parent, step.atom, "template"))
        for name_1, name_2 in _AROMATIC_RING_CLOSURES.get(resname, ()):
            pairs.extend(_local_pair(by_name, atom_index, name_1, name_2, "template"))
        for name_1, name_2 in _SPECIAL_RING_CLOSURES.get(resname, ()):
            pairs.extend(_local_pair(by_name, atom_index, name_1, name_2, "template"))

        if previous_key is not None and previous_atoms is not None:
            if _continues_previous(previous_key, key):
                pairs.extend(_cross_pair(previous_atoms, by_name, atom_index, "C", "N", "template"))
        previous_key = key
        previous_atoms = by_name

    return pairs


def _disulfide_pairs(structure: HeavyAtomStructure) -> list[tuple[int, int, str]]:
    atom_index = {id(atom): index for index, atom in enumerate(structure.atoms)}
    pairs = []
    for bond in structure.disulfide_bonds:
        atom_1 = _find_residue_atom(structure.atoms, bond.residue_ref_1, "SG")
        atom_2 = _find_residue_atom(structure.atoms, bond.residue_ref_2, "SG")
        if atom_1 is not None and atom_2 is not None:
            pairs.append((atom_index[id(atom_1)], atom_index[id(atom_2)], "disulfide"))
    return pairs


def _atoms_by_name(atoms: Sequence[Atom]) -> dict[str, Atom]:
    by_name: dict[str, Atom] = {}
    for atom in atoms:
        by_name.setdefault(atom.name.strip().upper(), atom)
    return by_name


def _local_pair(
    atoms: Mapping[str, Atom],
    atom_index: Mapping[int, int],
    name_1: str,
    name_2: str,
    source: str,
) -> list[tuple[int, int, str]]:
    atom_1 = atoms.get(name_1)
    atom_2 = atoms.get(name_2)
    if atom_1 is None or atom_2 is None:
        return []
    return [(atom_index[id(atom_1)], atom_index[id(atom_2)], source)]


def _cross_pair(
    atoms_1: Mapping[str, Atom],
    atoms_2: Mapping[str, Atom],
    atom_index: Mapping[int, int],
    name_1: str,
    name_2: str,
    source: str,
) -> list[tuple[int, int, str]]:
    atom_1 = atoms_1.get(name_1)
    atom_2 = atoms_2.get(name_2)
    if atom_1 is None or atom_2 is None:
        return []
    return [(atom_index[id(atom_1)], atom_index[id(atom_2)], source)]


def _continues_previous(previous_key: ResidueKey, key: ResidueKey) -> bool:
    previous_chain, previous_resseq, previous_icode, _previous_resname, previous_record = previous_key
    chain, resseq, icode, _resname, record = key
    if previous_chain != chain or previous_record != "ATOM" or record != "ATOM":
        return False
    if int(resseq) == int(previous_resseq) + 1:
        return True
    if int(resseq) == int(previous_resseq):
        prev_ic = (previous_icode or "").strip()
        cur_ic = (icode or "").strip()
        if prev_ic == "" and cur_ic == "A":
            return True
        if len(prev_ic) == 1 and len(cur_ic) == 1 and ord(cur_ic) == ord(prev_ic) + 1:
            return True
    return False


def _find_residue_atom(
    atoms: Sequence[Atom],
    residue_ref: tuple[str, int, str],
    atom_name: str,
) -> Atom | None:
    chain_id, resseq, icode = residue_ref
    for atom in atoms:
        if atom.name.strip().upper() != atom_name:
            continue
        if (atom.chain_id or "") != (chain_id or ""):
            continue
        if int(atom.resseq) != int(resseq):
            continue
        if (atom.icode or "") != (icode or ""):
            continue
        return atom
    return None


def _atom_label(atom: Atom) -> str:
    return (
        f"{atom.chain_id}:{atom.resseq}{atom.icode or ''}:"
        f"{atom.resname.strip().upper()}:{atom.name.strip().upper()}"
    )


def _valid_pair(index_1: int, index_2: int, atom_count: int) -> bool:
    return index_1 != index_2 and 0 <= index_1 < atom_count and 0 <= index_2 < atom_count

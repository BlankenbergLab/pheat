"""Optional solvent-accessible surface-area support.

The default PHEAT scoring path remains dependency-light.  SASA-backed burial is
available when FreeSASA is installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from pheat.geometry import distance
from pheat.models import HeavyAtomStructure, ResidueKey

SASA_BACKENDS = ("auto", "freesasa")
BURIAL_METHODS = ("contacts", "sasa")


def residue_burial(
    structure: HeavyAtomStructure,
    *,
    method: str = "contacts",
    backend: str = "auto",
) -> tuple[dict[ResidueKey, float], dict[str, object]]:
    """Return residue burial values normalized so larger means more buried."""

    normalized_method = _normalize_choice(method, BURIAL_METHODS, "burial method")
    normalized_backend = _normalize_choice(backend, SASA_BACKENDS, "SASA backend")
    if normalized_method == "contacts":
        return contact_density_burial(structure), {
            "burial_method": "contacts",
            "sasa_backend": None,
        }

    sasa_by_residue, used_backend = residue_sasa(
        structure,
        backend=normalized_backend,
    )
    if not sasa_by_residue:
        return {}, {
            "burial_method": "sasa",
            "sasa_backend": used_backend,
        }
    max_sasa = max(sasa_by_residue.values()) or 1.0
    burial = {
        key: max(0.0, 1.0 - float(value) / max_sasa)
        for key, value in sasa_by_residue.items()
    }
    return burial, {
        "burial_method": "sasa",
        "sasa_backend": used_backend,
        "sasa_units": "A^2",
        "sasa_residue_max": max_sasa,
    }


def contact_density_burial(
    structure: HeavyAtomStructure,
    *,
    cutoff: float = 8.0,
) -> dict[ResidueKey, float]:
    residues = structure.atoms_by_residue()
    keys = list(residues)
    counts = {key: 0.0 for key in keys}
    if not keys:
        return counts
    contacts: set[tuple[ResidueKey, ResidueKey]] = set()
    cell_size = cutoff
    cutoff_squared = cutoff * cutoff
    indexed_atoms = [
        (key, atom)
        for key, atoms in residues.items()
        for atom in atoms
    ]
    grid: dict[tuple[int, int, int], list[tuple[ResidueKey, object]]] = {}
    for key, atom in indexed_atoms:
        coord = getattr(atom, "coord")
        grid.setdefault(_spatial_cell(coord, cell_size), []).append((key, atom))
    for key_a, atom_a in indexed_atoms:
        coord_a = getattr(atom_a, "coord")
        cell = _spatial_cell(coord_a, cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for key_b, atom_b in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []):
                        if key_a >= key_b:
                            continue
                        if key_a[0] == key_b[0] and abs(key_a[1] - key_b[1]) <= 1:
                            continue
                        coord_b = getattr(atom_b, "coord")
                        delta = (
                            coord_a[0] - coord_b[0],
                            coord_a[1] - coord_b[1],
                            coord_a[2] - coord_b[2],
                        )
                        if delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] <= cutoff_squared:
                            contacts.add((key_a, key_b))
    for key_a, key_b in contacts:
        counts[key_a] += 1.0
        counts[key_b] += 1.0
    max_count = max(counts.values()) if counts else 0.0
    if max_count <= 0.0:
        return counts
    return {key: value / max_count for key, value in counts.items()}


def residue_sasa(
    structure: HeavyAtomStructure,
    *,
    backend: str = "auto",
) -> tuple[dict[ResidueKey, float], str]:
    # Currently only freesasa is implemented; the backend parameter is accepted
    # for forward-compatibility but "auto" always resolves to freesasa.
    _normalize_choice(backend, SASA_BACKENDS, "SASA backend")
    try:
        return _residue_sasa_freesasa(structure), "freesasa"
    except Exception as exc:
        raise RuntimeError(
            "FreeSASA is required for SASA burial. Install `pheat[scientific]`, "
            "`pheat[training]`, or `pheat[all]` with FreeSASA available."
        ) from exc


def _residue_sasa_freesasa(structure: HeavyAtomStructure) -> dict[ResidueKey, float]:
    import freesasa  # type: ignore[import-untyped]

    from pheat.pdbio import structure_to_pdb_string

    with tempfile.TemporaryDirectory(prefix="pheat-sasa-") as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        pdb_path.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        sasa_structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(sasa_structure)

    totals: dict[tuple[str, str, str], float] = {}
    for index in range(sasa_structure.nAtoms()):
        chain = str(sasa_structure.chainLabel(index)).strip() or "A"
        resnum = str(sasa_structure.residueNumber(index)).strip()
        resname = str(sasa_structure.residueName(index)).strip().upper()
        totals[(chain, resnum, resname)] = totals.get((chain, resnum, resname), 0.0) + float(
            result.atomArea(index)
        )

    mapped = {}
    for key in structure.residue_keys():
        chain, resseq, _icode, resname, _record = key
        mapped[key] = totals.get((chain[:1] or "A", str(resseq), resname), 0.0)
    return mapped


def _min_atom_distance(
    atoms_a: Sequence[object],
    atoms_b: Sequence[object],
) -> float:
    minimum = float("inf")
    for atom_a in atoms_a:
        coord_a = getattr(atom_a, "coord")
        for atom_b in atoms_b:
            minimum = min(minimum, distance(coord_a, getattr(atom_b, "coord")))
    return minimum


def _spatial_cell(coord: tuple[float, float, float], cell_size: float) -> tuple[int, int, int]:
    return (int(coord[0] // cell_size), int(coord[1] // cell_size), int(coord[2] // cell_size))


def _normalize_choice(value: str, choices: Sequence[str], label: str) -> str:
    normalized = str(value or choices[0]).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)}")
    return normalized

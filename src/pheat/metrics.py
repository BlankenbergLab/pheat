"""General structure metrics that do not belong to a scoring backend."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pheat.geometry import apply_kabsch_transform, kabsch_rmsd, kabsch_transform, radius_of_gyration
from pheat.models import Atom, HeavyAtomStructure

RADIUS_OF_GYRATION_RESULT_FORMAT = "pheat.radius-of-gyration-result"
RADIUS_OF_GYRATION_RESULT_VERSION = 1
RADIUS_OF_GYRATION_UNITS = "angstrom"
RMSD_RESULT_FORMAT = "pheat.rmsd-result"
RMSD_RESULT_VERSION = 1
RMSD_UNITS = "angstrom"
RMSD_ATOM_SETS = ("all-heavy", "all-atoms", "backbone", "ca")
SAME_AS_RMSD_ALIGNMENT = "same-as-rmsd"
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
ATOMIC_MASS_SOURCE_ID = "ciaaw-standard-atomic-weights-2024"

# Representative relative atomic masses for common heavy atoms found in PDB
# protein, modified-residue, and small heterogen records. Values are the usual
# conventional representative values derived from CIAAW standard atomic weights
# and cross-checked against the NIST Atomic Weights and Isotopic Compositions
# reference database. Interval elements use a conventional midpoint or rounded
# tabulated value suitable for center-of-mass weighting, not analytical chemistry.
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403162,
    "NA": 22.98976928,
    "MG": 24.305,
    "P": 30.973761998,
    "S": 32.06,
    "CL": 35.45,
    "K": 39.0983,
    "CA": 40.078,
    "FE": 55.845,
    "ZN": 65.38,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90447,
}
FALLBACK_ELEMENT = "C"
FALLBACK_MASS = ATOMIC_MASSES[FALLBACK_ELEMENT]


def structure_radius_of_gyration(
    structure: HeavyAtomStructure,
    *,
    mode: str = "both",
    atom_set: str = "all-heavy",
) -> Dict[str, Any]:
    """Return unweighted and/or mass-weighted radius of gyration for a structure."""

    normalized_mode = _normalize_rg_mode(mode)
    normalized_atom_set = normalize_rmsd_atom_set(atom_set)
    atoms = [atom for atom in structure.atoms if atom_in_rmsd_atom_set(atom, normalized_atom_set)]
    if not atoms:
        raise ValueError(f"No {normalized_atom_set} atoms found for radius of gyration.")
    coords = [atom.coord for atom in atoms]
    values: Dict[str, float] = {}
    payload: Dict[str, Any] = {
        "atom_count": len(atoms),
        "atom_set": normalized_atom_set,
        "mode": normalized_mode,
        "units": RADIUS_OF_GYRATION_UNITS,
        "values": values,
    }

    if normalized_mode in {"unweighted", "both"}:
        values["unweighted"] = radius_of_gyration(coords)
    if normalized_mode in {"mass-weighted", "both"}:
        masses, unknown_elements = atomic_masses_for_atoms(atoms)
        values["mass_weighted"] = radius_of_gyration(coords, weights=masses)
        payload["mass_source"] = ATOMIC_MASS_SOURCE_ID
        payload["unknown_elements"] = unknown_elements

    return payload


def radius_of_gyration_result(
    structure: HeavyAtomStructure,
    *,
    input_name: str,
    mode: str = "both",
    atom_set: str = "all-heavy",
) -> Dict[str, Any]:
    """Return the canonical CLI JSON payload for radius-of-gyration calculation."""

    return {
        "format": RADIUS_OF_GYRATION_RESULT_FORMAT,
        "version": RADIUS_OF_GYRATION_RESULT_VERSION,
        "input": input_name,
        "radius_of_gyration": structure_radius_of_gyration(
            structure,
            mode=mode,
            atom_set=atom_set,
        ),
    }


def structure_rmsd_result(
    reference: HeavyAtomStructure,
    target: HeavyAtomStructure,
    *,
    reference_name: str,
    target_name: str,
    atom_set: str = "all-heavy",
    alignment_atom_set: str = SAME_AS_RMSD_ALIGNMENT,
    aligned_target: Optional[HeavyAtomStructure] = None,
) -> Dict[str, Any]:
    """Return the canonical CLI JSON payload for structure RMSD calculation."""

    return {
        "format": RMSD_RESULT_FORMAT,
        "version": RMSD_RESULT_VERSION,
        "reference": reference_name,
        "target": target_name,
        "rmsd": structure_rmsd(
            reference,
            target,
            atom_set=atom_set,
            alignment_atom_set=alignment_atom_set,
            aligned_target=aligned_target,
        ),
    }


def structure_rmsd(
    reference: HeavyAtomStructure,
    target: HeavyAtomStructure,
    *,
    atom_set: str = "all-heavy",
    alignment_atom_set: str = SAME_AS_RMSD_ALIGNMENT,
    aligned_target: Optional[HeavyAtomStructure] = None,
) -> Dict[str, Any]:
    """Return Kabsch RMSD for matched atoms from two atom structures."""

    normalized_atom_set = normalize_rmsd_atom_set(atom_set)
    normalized_alignment_atom_set = normalize_alignment_atom_set(
        alignment_atom_set,
        rmsd_atom_set=normalized_atom_set,
    )
    measurement_target = aligned_target or align_structure_to_reference(
        reference,
        target,
        atom_set=normalized_alignment_atom_set,
    )["aligned_target"]
    reference_atoms = atoms_by_rmsd_key(reference, atom_set=normalized_atom_set)
    target_atoms = atoms_by_rmsd_key(measurement_target, atom_set=normalized_atom_set)
    common_keys = sorted(set(reference_atoms) & set(target_atoms))
    if not common_keys:
        raise ValueError(f"No common {normalized_atom_set} atoms found for RMSD.")

    reference_coords = [reference_atoms[key].coord for key in common_keys]
    target_coords = [target_atoms[key].coord for key in common_keys]
    return {
        "atom_set": normalized_atom_set,
        "alignment_atom_set": normalized_alignment_atom_set,
        "aligned_target_provided": aligned_target is not None,
        "matched_atoms": len(common_keys),
        "unmatched_reference_atoms": len(set(reference_atoms) - set(target_atoms)),
        "unmatched_target_atoms": len(set(target_atoms) - set(reference_atoms)),
        "units": RMSD_UNITS,
        "value": kabsch_rmsd(reference_coords, target_coords, aligned_target=target_coords),
    }


def align_structure_to_reference(
    reference: HeavyAtomStructure,
    target: HeavyAtomStructure,
    *,
    atom_set: str = "all-heavy",
) -> Dict[str, Any]:
    """Kabsch-align a target structure to a reference using a selected atom set."""

    normalized_atom_set = normalize_rmsd_atom_set(atom_set)
    reference_atoms = atoms_by_rmsd_key(reference, atom_set=normalized_atom_set)
    target_atoms = atoms_by_rmsd_key(target, atom_set=normalized_atom_set)
    common_keys = sorted(set(reference_atoms) & set(target_atoms))
    if not common_keys:
        raise ValueError(f"No common {normalized_atom_set} atoms found for alignment.")

    reference_coords = [reference_atoms[key].coord for key in common_keys]
    target_coords = [target_atoms[key].coord for key in common_keys]
    transform = kabsch_transform(reference_coords, target_coords)
    aligned_coords = apply_kabsch_transform([atom.coord for atom in target.atoms], transform)
    aligned_atoms = [
        replace(atom, x=coord[0], y=coord[1], z=coord[2])
        for atom, coord in zip(target.atoms, aligned_coords)
    ]
    aligned_target = HeavyAtomStructure(
        atoms=aligned_atoms,
        name=f"{target.name}:kabsch_aligned_to_reference",
        metadata={
            **target.metadata,
            "alignment": "kabsch_to_reference",
            "alignment_atom_set": normalized_atom_set,
            "matched_alignment_atoms": len(common_keys),
        },
        disulfide_bonds=list(target.disulfide_bonds),
    )
    return {
        "aligned_target": aligned_target,
        "transform": transform,
        "atom_set": normalized_atom_set,
        "matched_atoms": len(common_keys),
        "unmatched_reference_atoms": len(set(reference_atoms) - set(target_atoms)),
        "unmatched_target_atoms": len(set(target_atoms) - set(reference_atoms)),
    }


def atoms_by_rmsd_key(
    structure: HeavyAtomStructure,
    *,
    atom_set: str = "all-heavy",
) -> dict[tuple[Any, str], Atom]:
    """Return atoms keyed by residue and atom name for a named RMSD atom set."""

    normalized_atom_set = normalize_rmsd_atom_set(atom_set)
    return {
        (atom.residue_key, atom.name.strip().upper()): atom
        for atom in structure.atoms
        if atom_in_rmsd_atom_set(atom, normalized_atom_set)
    }


def atom_in_rmsd_atom_set(atom: Atom, atom_set: str) -> bool:
    atom_name = atom.name.strip().upper()
    if atom_set == "all-heavy":
        return atom.element.strip().upper() not in {"H", "D", "T"}
    if atom_set == "all-atoms":
        return True
    if atom_set == "backbone":
        return atom_name in BACKBONE_ATOMS
    if atom_set == "ca":
        return atom_name == "CA"
    raise ValueError(f"Unsupported RMSD atom set: {atom_set}")


def normalize_rmsd_atom_set(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "all": "all-heavy",
        "allheavy": "all-heavy",
        "heavy": "all-heavy",
        "all-atom": "all-atoms",
        "c-alpha": "ca",
        "calpha": "ca",
        "c-alpha-only": "ca",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RMSD_ATOM_SETS:
        raise ValueError(f"RMSD atom set must be one of {', '.join(RMSD_ATOM_SETS)}")
    return normalized


def normalize_alignment_atom_set(value: str, *, rmsd_atom_set: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == SAME_AS_RMSD_ALIGNMENT:
        return normalize_rmsd_atom_set(rmsd_atom_set)
    return normalize_rmsd_atom_set(normalized)


def atomic_masses_for_atoms(atoms: List[Atom]) -> tuple[List[float], List[str]]:
    """Return per-atom masses and sorted unknown element symbols."""

    masses: List[float] = []
    unknown_elements = set()
    for atom in atoms:
        element = normalized_element(atom.element)
        mass = ATOMIC_MASSES.get(element)
        if mass is None:
            unknown_elements.add(element)
            mass = FALLBACK_MASS
        masses.append(mass)
    return masses, sorted(unknown_elements)


def normalized_element(element: object) -> str:
    """Normalize a PDB element symbol for lookup in the compact mass table."""

    return str(element or "").strip().upper() or "UNKNOWN"


def radius_of_gyration_delta(
    original: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return reconstructed-minus-original Rg deltas for shared value keys."""

    original_values = original.get("values", {})
    reconstructed_values = reconstructed.get("values", {})
    delta_values = {
        key: float(reconstructed_values[key]) - float(original_values[key])
        for key in sorted(set(original_values) & set(reconstructed_values))
    }
    return {
        "units": reconstructed.get("units") or original.get("units") or RADIUS_OF_GYRATION_UNITS,
        "values": delta_values,
    }


def _normalize_rg_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized == "mass_weighted":
        normalized = "mass-weighted"
    if normalized not in {"unweighted", "mass-weighted", "both"}:
        raise ValueError("radius-of-gyration mode must be one of unweighted, mass-weighted, both")
    return normalized


PAIRWISE_RMSD_MATRIX_FORMAT = "pheat.pairwise-rmsd-matrix"
PAIRWISE_RMSD_MATRIX_VERSION = 1


def pairwise_rmsd_matrix(
    structures: "Sequence[HeavyAtomStructure] | Sequence[Sequence[Sequence[float]]]",
    *,
    atom_set: str = "ca",
):
    """Return the all-vs-all symmetric Kabsch RMSD matrix for an ensemble.

    ``structures`` may be either a sequence of :class:`HeavyAtomStructure`
    instances (in which case ``atom_set`` selects the comparison atoms via
    :func:`structure_rmsd`) or a sequence of raw coordinate arrays/lists of
    shape ``(N, 3)`` (in which case ``atom_set`` is ignored and the comparison
    runs over the coordinates directly).

    The returned object is an ``ndarray`` of shape ``(M, M)`` with a zero
    diagonal and symmetric off-diagonal entries.  The function requires
    NumPy because the underlying Kabsch SVD step relies on it.
    """

    from pheat.geometry import _require_numpy_for_kabsch
    from pheat.geometry import kabsch_rmsd as _kabsch_rmsd

    np = _require_numpy_for_kabsch()
    items: list[Any] = list(structures)
    n = len(items)
    matrix = np.zeros((n, n), dtype=float)
    if n < 2:
        return matrix

    def _is_pheat_structure(item: Any) -> bool:
        return hasattr(item, "atoms") and hasattr(item, "metadata")

    use_structures = _is_pheat_structure(items[0])
    for i in range(n):
        for j in range(i + 1, n):
            if use_structures:
                if not isinstance(items[i], HeavyAtomStructure) or not isinstance(
                    items[j], HeavyAtomStructure
                ):
                    raise TypeError("pairwise structure RMSD requires HeavyAtomStructure items")
                rmsd_value = float(
                    structure_rmsd(
                        items[i],
                        items[j],
                        atom_set=atom_set,
                    )["value"]
                )
            else:
                # Coordinate arrays/lists; numpy arrays are converted to lists
                # because pheat.geometry.kabsch_rmsd accepts sequences of 3-tuples.
                reference = np.asarray(items[i], dtype=float).tolist()
                target = np.asarray(items[j], dtype=float).tolist()
                rmsd_value = float(_kabsch_rmsd(reference, target))
            matrix[i, j] = rmsd_value
            matrix[j, i] = rmsd_value
    return matrix


def ensemble_rmsd_stats(matrix) -> Dict[str, float]:
    """Return raw avg/max/min-nonzero summary statistics for a pairwise RMSD matrix.

    The verdict layer (e.g. STABLE/FLEXIBLE/UNSTABLE) is intentionally not
    included here because reasonable thresholds depend on the atom set,
    structure size, and study context.  Callers should layer their own
    classification on top of these raw stats.
    """

    from pheat.geometry import _require_numpy_for_kabsch

    np = _require_numpy_for_kabsch()
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("ensemble RMSD matrix must be a square 2-D array")
    n = arr.shape[0]
    if n < 2:
        return {
            "avg_pairwise_rmsd": 0.0,
            "max_pairwise_rmsd": 0.0,
            "min_nonzero_rmsd": 0.0,
            "ensemble_size": int(n),
        }
    upper = arr[np.triu_indices(n, k=1)]
    avg = float(np.mean(upper))
    mx = float(np.max(upper))
    nonzero = upper[upper > 0]
    mn = float(np.min(nonzero)) if len(nonzero) else 0.0
    return {
        "avg_pairwise_rmsd": avg,
        "max_pairwise_rmsd": mx,
        "min_nonzero_rmsd": mn,
        "ensemble_size": int(n),
    }

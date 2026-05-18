"""Structure-domain filtering for PHEAT scoring and training.

PHEAT's native representation is heavy-atom oriented.  The default scoring
domain is therefore protein heavy atoms, while broader domains are opt-in for
models that can use them honestly.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from pheat.models import Atom, DisulfideBond, HeavyAtomStructure
from pheat.residues import SUPPORTED_RESIDUES


SCORING_DOMAINS = ("protein-heavy", "all-heavy", "full")


def filter_structure_for_domain(
    structure: HeavyAtomStructure,
    domain: str = "protein-heavy",
) -> Tuple[HeavyAtomStructure, dict[str, Any]]:
    """Return a structure filtered to a scoring/training domain plus coverage.

    ``full`` keeps every atom currently present in the PHEAT structure object.
    PDB/mmCIF/BCIF readers still canonicalize to heavy atoms by default, so
    hydrogen counts are usually reported as already dropped in source metadata.
    """

    normalized = normalize_domain(domain)
    kept_atoms = []
    ignored = {
        "hydrogen_atom_count": 0,
        "nonprotein_atom_count": 0,
        "unsupported_residue_atom_count": 0,
    }

    for atom in structure.atoms:
        element = atom.element.strip().upper()
        resname = atom.resname.strip().upper()
        is_hydrogen = element in {"H", "D", "T"}
        is_supported_protein = resname in SUPPORTED_RESIDUES

        if normalized == "protein-heavy":
            if is_hydrogen:
                ignored["hydrogen_atom_count"] += 1
                continue
            if not is_supported_protein:
                ignored["nonprotein_atom_count"] += 1
                continue
        elif normalized == "all-heavy":
            if is_hydrogen:
                ignored["hydrogen_atom_count"] += 1
                continue

        kept_atoms.append(atom)

    filtered = HeavyAtomStructure(
        atoms=list(kept_atoms),
        name=structure.name,
        metadata=dict(structure.metadata),
        disulfide_bonds=_filter_disulfides(structure.disulfide_bonds, kept_atoms),
    )
    coverage = _coverage_payload(structure, filtered, normalized, ignored)
    return filtered, coverage


def normalize_domain(domain: str) -> str:
    normalized = str(domain or "protein-heavy").strip().lower()
    if normalized not in SCORING_DOMAINS:
        raise ValueError(f"domain must be one of {', '.join(SCORING_DOMAINS)}")
    return normalized


def merge_coverage_metadata(metadata: Mapping[str, Any], coverage: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    payload["domain"] = coverage.get("domain")
    payload["coverage"] = dict(coverage)
    return payload


def _coverage_payload(
    original: HeavyAtomStructure,
    filtered: HeavyAtomStructure,
    domain: str,
    ignored: Mapping[str, int],
) -> dict[str, Any]:
    original_residues = original.residue_keys()
    filtered_residues = filtered.residue_keys()
    supported_residues = [
        key for key in filtered_residues if key[3].strip().upper() in SUPPORTED_RESIDUES
    ]
    unsupported_residue_names = sorted(
        {
            key[3].strip().upper()
            for key in filtered_residues
            if key[3].strip().upper() not in SUPPORTED_RESIDUES
        }
    )
    source_dropped_hydrogens = int(original.metadata.get("dropped_hydrogen_count") or 0)
    total_atoms = len(original.atoms)
    scored_atoms = len(filtered.atoms)
    return {
        "domain": domain,
        "input_atom_scope": original.atom_scope,
        "scored_atom_scope": filtered.atom_scope,
        "input_atom_count": total_atoms,
        "scored_atom_count": scored_atoms,
        "scored_atom_fraction": (scored_atoms / total_atoms) if total_atoms else 0.0,
        "input_residue_count": len(original_residues),
        "scored_residue_count": len(filtered_residues),
        "protein_residue_count": len(supported_residues),
        "unsupported_residue_count": len(unsupported_residue_names),
        "unsupported_residues": unsupported_residue_names,
        "ignored_hydrogen_atom_count": int(ignored.get("hydrogen_atom_count", 0)),
        "source_dropped_hydrogen_count": source_dropped_hydrogens,
        "ignored_nonprotein_atom_count": int(ignored.get("nonprotein_atom_count", 0)),
        "ignored_unsupported_residue_atom_count": int(
            ignored.get("unsupported_residue_atom_count", 0)
        ),
    }


def _filter_disulfides(
    disulfide_bonds: list[DisulfideBond],
    kept_atoms: list[Atom],
) -> list[DisulfideBond]:
    kept_refs = {(atom.chain_id or "", int(atom.resseq), atom.icode or "") for atom in kept_atoms}
    return [
        bond
        for bond in disulfide_bonds
        if bond.residue_ref_1 in kept_refs and bond.residue_ref_2 in kept_refs
    ]

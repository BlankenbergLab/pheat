"""Input-contract metadata for PHEAT scoring models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SCORE_CONTRACT_VERSION = 1

_COMMON_ATOM_STRUCTURE = {
    "version": SCORE_CONTRACT_VERSION,
    "input_structure_type": "atom-structure",
    "accepted_atom_scope": ["heavy", "all"],
}

_CONTRACTS: dict[str, dict[str, Any]] = {
    "generic": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.generic.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy", "full"],
        "required_atoms": [],
        "uses_hydrogens": "optional",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-dfire": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-dfire.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["protein-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-goap": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-goap.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["protein-heavy-atoms", "residue-orientation-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-goap-physical": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-goap-physical.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["protein-heavy-atoms", "residue-orientation-atoms", "physical-integrity-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-mj": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-mj.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["residue-representative-points"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-hydropathy": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-hydropathy.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["protein-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-hydropathy-physical": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-hydropathy-physical.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["protein-heavy-atoms", "physical-integrity-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-backbone": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-backbone.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["backbone-n-ca-c"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "derived-torsions",
    },
    "pheat-backbone-physical": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-backbone-physical.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["backbone-n-ca-c", "physical-integrity-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "derived-torsions",
    },
    "pheat-rotamer": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-rotamer.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["sidechain-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "derived-torsions",
    },
    "pheat-hbond": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-hbond.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["polar-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-rg": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-rg.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["configured-rg-atom-set"],
        "uses_hydrogens": "no",
        "uses_table": True,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-ml-linear": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-ml-linear.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy"],
        "required_atoms": ["configured-feature-inputs"],
        "uses_hydrogens": "depends-on-features",
        "uses_table": True,
        "burial_dependent": "depends-on-features",
        "coordinate_or_geometry_basis": "feature-vector",
    },
    "pheat-physics": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-physics.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["heavy-atoms", "backbone-n-ca-c-o", "sidechain-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": False,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates-and-derived-torsions",
    },
    "pheat-coarse-protein-folding-v1": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-coarse-protein-folding-v1.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["heavy-atoms", "backbone-n-ca-c-o", "sidechain-heavy-atoms"],
        "uses_hydrogens": "inferred-backbone-hydrogen",
        "uses_table": False,
        "burial_dependent": True,
        "coordinate_or_geometry_basis": "coordinates-and-optional-decoded-torsions",
    },
    "pheat-geometry-integrity": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-geometry-integrity.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["backbone-n-ca-c-o", "peptide-link-c-n", "ca-chirality-atoms"],
        "uses_hydrogens": "no",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-physical-integrity": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-physical-integrity.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["heavy-atoms", "backbone-n-ca-c-o", "peptide-link-c-n", "ca-chirality-atoms"],
        "uses_hydrogens": "no",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "heavy-mm": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.heavy-mm.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "pheat-heavy-mm-physical": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.pheat-heavy-mm-physical.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy"],
        "required_atoms": ["heavy-atoms", "physical-integrity-heavy-atoms"],
        "uses_hydrogens": "no",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "coordinates",
    },
    "openmm-prepared": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.openmm-prepared.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy", "full"],
        "required_atoms": ["openmm-supported-residues"],
        "uses_hydrogens": "prepared-internally",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "prepared-openmm",
    },
    "ambertools-sander": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.ambertools-sander.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy", "full"],
        "required_atoms": ["ambertools-supported-residues"],
        "uses_hydrogens": "prepared-internally",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "prepared-ambertools",
    },
    "gromacs-mdrun": {
        **_COMMON_ATOM_STRUCTURE,
        "id": "pheat.score-contract.gromacs-mdrun.v1",
        "default_domain": "protein-heavy",
        "compatible_domains": ["protein-heavy", "all-heavy", "full"],
        "required_atoms": ["gromacs-pdb2gmx-supported-residues"],
        "uses_hydrogens": "prepared-internally",
        "uses_table": False,
        "burial_dependent": False,
        "coordinate_or_geometry_basis": "prepared-gromacs",
    },
}


def normalize_score_model_id(model: str) -> str:
    """Normalize supported scorer aliases to canonical model IDs."""

    normalized = str(model).strip().lower()
    return "heavy-mm" if normalized == "heavy_mm" else normalized


def score_input_contract(model: str) -> dict[str, Any]:
    """Return a copy of the input contract for a scorer."""

    normalized = normalize_score_model_id(model)
    if normalized not in _CONTRACTS:
        known = ", ".join(sorted(_CONTRACTS))
        raise ValueError(f"Unknown scoring model contract {model!r}. Known contracts: {known}")
    return deepcopy(_CONTRACTS[normalized])


def score_input_contracts() -> dict[str, dict[str, Any]]:
    """Return copies of all score input contracts keyed by model ID."""

    return {model: deepcopy(contract) for model, contract in sorted(_CONTRACTS.items())}

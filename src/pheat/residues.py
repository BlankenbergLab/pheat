"""Residue metadata and approximate side-chain construction templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

CANONICAL_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# SEC and PYL are represented in PDB sequence data with one-letter codes U and O.
# Other supported modified residues keep their CCD three-letter component IDs so a
# one-letter protein sequence never silently loses the modification.
EXTENDED_ONE_TO_THREE: Dict[str, str] = {
    "U": "SEC",
    "O": "PYL",
}
ONE_TO_THREE: Dict[str, str] = {
    **CANONICAL_ONE_TO_THREE,
    **EXTENDED_ONE_TO_THREE,
}
THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}
CANONICAL_RESIDUES = set(CANONICAL_ONE_TO_THREE.values())

MODIFIED_RESIDUE_PARENTS: Dict[str, Optional[str]] = {
    "SEC": "CYS",
    "PYL": None,
    "MSE": "MET",
    "HYP": "PRO",
    "LYZ": "LYS",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
    "PCA": "GLN",
}
SUPPORTED_RESIDUES = CANONICAL_RESIDUES | set(MODIFIED_RESIDUE_PARENTS)

# HYL is the common biochemical abbreviation for hydroxylysine, but the wwPDB
# Chemical Component Dictionary component ID used in coordinate files is LYZ.
RESIDUE_ALIASES = {
    "HYL": "LYZ",
}


def one_to_three(value: str) -> str:
    token = RESIDUE_ALIASES.get(value.strip().upper(), value.strip().upper())
    if len(token) == 3:
        if token not in SUPPORTED_RESIDUES:
            raise ValueError(f"Unsupported residue '{value}'.")
        return token
    if token not in ONE_TO_THREE:
        raise ValueError(f"Unsupported residue '{value}'.")
    return ONE_TO_THREE[token]


def three_to_one(value: str) -> str:
    token = RESIDUE_ALIASES.get(value.strip().upper(), value.strip().upper())
    if token not in THREE_TO_ONE:
        raise ValueError(f"Unsupported residue '{value}'.")
    return THREE_TO_ONE[token]


def guess_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return "X"
    if stripped[0].isdigit() and len(stripped) > 1:
        stripped = stripped[1:]
    two_letter = stripped[:2].strip().upper()
    if two_letter in {"FE", "ZN", "MG", "MN", "CA", "CU", "CO", "NI", "CL", "BR", "NA", "SE", "K"}:
        return two_letter
    return stripped[0].upper()


DihedralSpec = Union[float, str, Tuple[str, float]]


@dataclass(frozen=True)
class SidechainStep:
    """Internal-coordinate recipe for placing one side-chain heavy atom."""

    atom: str
    parent: str
    anchor: str
    previous: str
    length: float
    angle: float
    dihedral: DihedralSpec
    element: Optional[str] = None


# Fallback rotamer-like angles in degrees. These are PHEAT heuristics, not a
# vendored Dunbrack rotamer table; see source manifest entry dunbrack-rotamers
# for the intended future source if exact rotamer data is added.
DEFAULT_CHIS = [-60.0, -60.0, -90.0, 180.0, 0.0, 0.0]


# Side-chain topology follows standard PDB atom naming for canonical amino acids
# and wwPDB CCD component atom names/connectivity for supported modified residues
# (source manifest entries: engh-huber-1991 and wwpdb-ccd). Bond lengths/angles
# are rounded ideal geometry values, not a force-field parameter table. Branches
# and planar groups use fixed offsets such as ("chi2", 180.0) to avoid requiring
# extra residue-geometry inputs.
SIDECHAIN_STEPS: Dict[str, List[SidechainStep]] = {
    "ALA": [],
    "ARG": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.53, 113.0, "chi2"),
        SidechainStep("NE", "CD", "CG", "CB", 1.46, 112.0, "chi3", "N"),
        SidechainStep("CZ", "NE", "CD", "CG", 1.33, 124.0, "chi4", "C"),
        SidechainStep("NH1", "CZ", "NE", "CD", 1.33, 120.0, 0.0, "N"),
        SidechainStep("NH2", "CZ", "NE", "CD", 1.33, 120.0, 180.0, "N"),
    ],
    "ASN": [
        SidechainStep("CG", "CB", "CA", "N", 1.52, 112.0, "chi1", "C"),
        SidechainStep("OD1", "CG", "CB", "CA", 1.23, 120.0, "chi2", "O"),
        SidechainStep("ND2", "CG", "CB", "CA", 1.33, 120.0, ("chi2", 180.0), "N"),
    ],
    "ASP": [
        SidechainStep("CG", "CB", "CA", "N", 1.52, 112.0, "chi1", "C"),
        SidechainStep("OD1", "CG", "CB", "CA", 1.25, 120.0, "chi2", "O"),
        SidechainStep("OD2", "CG", "CB", "CA", 1.25, 120.0, ("chi2", 180.0), "O"),
    ],
    "CYS": [SidechainStep("SG", "CB", "CA", "N", 1.81, 114.0, "chi1", "S")],
    "SEC": [SidechainStep("SE", "CB", "CA", "N", 1.96, 114.0, "chi1", "SE")],
    "GLN": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.52, 112.0, "chi2", "C"),
        SidechainStep("OE1", "CD", "CG", "CB", 1.23, 120.0, "chi3", "O"),
        SidechainStep("NE2", "CD", "CG", "CB", 1.33, 120.0, ("chi3", 180.0), "N"),
    ],
    "GLU": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.52, 112.0, "chi2", "C"),
        SidechainStep("OE1", "CD", "CG", "CB", 1.25, 120.0, "chi3", "O"),
        SidechainStep("OE2", "CD", "CG", "CB", 1.25, 120.0, ("chi3", 180.0), "O"),
    ],
    "GLY": [],
    "HIS": [
        SidechainStep("CG", "CB", "CA", "N", 1.50, 113.0, "chi1", "C"),
        SidechainStep("ND1", "CG", "CB", "CA", 1.38, 126.0, "chi2", "N"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.35, 126.0, ("chi2", 180.0), "C"),
        SidechainStep("CE1", "ND1", "CG", "CD2", 1.32, 108.0, 0.0, "C"),
        SidechainStep("NE2", "CD2", "CG", "ND1", 1.35, 108.0, 0.0, "N"),
    ],
    "ILE": [
        SidechainStep("CG1", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CG2", "CB", "CA", "N", 1.53, 113.0, ("chi1", 120.0)),
        SidechainStep("CD1", "CG1", "CB", "CA", 1.53, 113.0, "chi2"),
    ],
    "LEU": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD1", "CG", "CB", "CA", 1.53, 113.0, "chi2"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.53, 113.0, ("chi2", 120.0)),
    ],
    "LYS": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.53, 113.0, "chi2"),
        SidechainStep("CE", "CD", "CG", "CB", 1.53, 113.0, "chi3"),
        SidechainStep("NZ", "CE", "CD", "CG", 1.49, 112.0, "chi4", "N"),
    ],
    "LYZ": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.53, 113.0, "chi2"),
        SidechainStep("CE", "CD", "CG", "CB", 1.53, 113.0, "chi3"),
        SidechainStep("NZ", "CE", "CD", "CG", 1.49, 112.0, "chi4", "N"),
        SidechainStep("OH", "CD", "CG", "CB", 1.41, 109.5, ("chi3", -120.0), "O"),
    ],
    "MET": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("SD", "CG", "CB", "CA", 1.81, 112.0, "chi2", "S"),
        SidechainStep("CE", "SD", "CG", "CB", 1.79, 100.0, "chi3"),
    ],
    "MSE": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("SE", "CG", "CB", "CA", 1.96, 112.0, "chi2", "SE"),
        SidechainStep("CE", "SE", "CG", "CB", 1.96, 100.0, "chi3"),
    ],
    "PHE": [
        SidechainStep("CG", "CB", "CA", "N", 1.50, 113.0, "chi1", "C"),
        SidechainStep("CD1", "CG", "CB", "CA", 1.39, 120.0, "chi2"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.39, 120.0, ("chi2", 180.0)),
        SidechainStep("CE1", "CD1", "CG", "CD2", 1.39, 120.0, 0.0),
        SidechainStep("CE2", "CD2", "CG", "CD1", 1.39, 120.0, 0.0),
        SidechainStep("CZ", "CE1", "CD1", "CD2", 1.39, 120.0, 0.0),
    ],
    "PRO": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 104.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.50, 104.0, "chi2"),
    ],
    "HYP": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 104.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.50, 104.0, "chi2"),
        SidechainStep("OD1", "CG", "CB", "CA", 1.41, 109.5, ("chi2", -120.0), "O"),
    ],
    "PCA": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 104.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.52, 104.0, "chi2", "C"),
        SidechainStep("OE", "CD", "CG", "CB", 1.23, 120.0, "chi3", "O"),
    ],
    "PYL": [
        SidechainStep("CG", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CD", "CG", "CB", "CA", 1.53, 113.0, "chi2"),
        SidechainStep("CE", "CD", "CG", "CB", 1.53, 113.0, "chi3"),
        SidechainStep("NZ", "CE", "CD", "CG", 1.45, 112.0, "chi4", "N"),
        SidechainStep("C2", "NZ", "CE", "CD", 1.33, 120.0, "chi5", "C"),
        SidechainStep("O2", "C2", "NZ", "CE", 1.23, 120.0, 0.0, "O"),
        SidechainStep("CA2", "C2", "NZ", "CE", 1.52, 120.0, 180.0, "C"),
        SidechainStep("N2", "CA2", "C2", "NZ", 1.47, 110.0, "chi6", "N"),
        SidechainStep("CE2", "N2", "CA2", "C2", 1.30, 110.0, 0.0, "C"),
        SidechainStep("CD2", "CE2", "N2", "CA2", 1.50, 110.0, 0.0, "C"),
        SidechainStep("CG2", "CD2", "CE2", "N2", 1.53, 110.0, 0.0, "C"),
        SidechainStep("CB2", "CG2", "CD2", "CE2", 1.53, 113.0, 120.0, "C"),
    ],
    "SER": [SidechainStep("OG", "CB", "CA", "N", 1.41, 111.0, "chi1", "O")],
    "SEP": [
        SidechainStep("OG", "CB", "CA", "N", 1.41, 111.0, "chi1", "O"),
        SidechainStep("P", "OG", "CB", "CA", 1.61, 120.0, "chi2", "P"),
        SidechainStep("O1P", "P", "OG", "CB", 1.48, 109.5, "chi3", "O"),
        SidechainStep("O2P", "P", "OG", "CB", 1.57, 109.5, ("chi3", 120.0), "O"),
        SidechainStep("O3P", "P", "OG", "CB", 1.57, 109.5, ("chi3", -120.0), "O"),
    ],
    "THR": [
        SidechainStep("OG1", "CB", "CA", "N", 1.43, 111.0, "chi1", "O"),
        SidechainStep("CG2", "CB", "CA", "N", 1.53, 113.0, ("chi1", 120.0)),
    ],
    "TPO": [
        SidechainStep("OG1", "CB", "CA", "N", 1.43, 111.0, "chi1", "O"),
        SidechainStep("CG2", "CB", "CA", "N", 1.53, 113.0, ("chi1", 120.0)),
        SidechainStep("P", "OG1", "CB", "CA", 1.61, 120.0, "chi2", "P"),
        SidechainStep("O1P", "P", "OG1", "CB", 1.48, 109.5, "chi3", "O"),
        SidechainStep("O2P", "P", "OG1", "CB", 1.57, 109.5, ("chi3", 120.0), "O"),
        SidechainStep("O3P", "P", "OG1", "CB", 1.57, 109.5, ("chi3", -120.0), "O"),
    ],
    "TRP": [
        SidechainStep("CG", "CB", "CA", "N", 1.50, 113.0, "chi1", "C"),
        SidechainStep("CD1", "CG", "CB", "CA", 1.37, 126.0, "chi2"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.43, 126.0, ("chi2", 180.0)),
        SidechainStep("NE1", "CD1", "CG", "CD2", 1.38, 108.0, 0.0, "N"),
        SidechainStep("CE2", "CD2", "CG", "CD1", 1.40, 108.0, 0.0),
        SidechainStep("CE3", "CD2", "CE2", "CG", 1.40, 120.0, 0.0),
        SidechainStep("CZ2", "CE2", "CD2", "CG", 1.40, 120.0, 0.0),
        SidechainStep("CZ3", "CE3", "CD2", "CE2", 1.40, 120.0, 0.0),
        SidechainStep("CH2", "CZ2", "CE2", "CD2", 1.40, 120.0, 0.0),
    ],
    "TYR": [
        SidechainStep("CG", "CB", "CA", "N", 1.50, 113.0, "chi1", "C"),
        SidechainStep("CD1", "CG", "CB", "CA", 1.39, 120.0, "chi2"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.39, 120.0, ("chi2", 180.0)),
        SidechainStep("CE1", "CD1", "CG", "CD2", 1.39, 120.0, 0.0),
        SidechainStep("CE2", "CD2", "CG", "CD1", 1.39, 120.0, 0.0),
        SidechainStep("CZ", "CE1", "CD1", "CD2", 1.39, 120.0, 0.0),
        SidechainStep("OH", "CZ", "CE1", "CD1", 1.36, 120.0, 180.0, "O"),
    ],
    "PTR": [
        SidechainStep("CG", "CB", "CA", "N", 1.50, 113.0, "chi1", "C"),
        SidechainStep("CD1", "CG", "CB", "CA", 1.39, 120.0, "chi2"),
        SidechainStep("CD2", "CG", "CB", "CA", 1.39, 120.0, ("chi2", 180.0)),
        SidechainStep("CE1", "CD1", "CG", "CD2", 1.39, 120.0, 0.0),
        SidechainStep("CE2", "CD2", "CG", "CD1", 1.39, 120.0, 0.0),
        SidechainStep("CZ", "CE1", "CD1", "CD2", 1.39, 120.0, 0.0),
        SidechainStep("OH", "CZ", "CE1", "CD1", 1.36, 120.0, 180.0, "O"),
        SidechainStep("P", "OH", "CZ", "CE1", 1.61, 120.0, "chi3", "P"),
        SidechainStep("O1P", "P", "OH", "CZ", 1.48, 109.5, "chi4", "O"),
        SidechainStep("O2P", "P", "OH", "CZ", 1.57, 109.5, ("chi4", 120.0), "O"),
        SidechainStep("O3P", "P", "OH", "CZ", 1.57, 109.5, ("chi4", -120.0), "O"),
    ],
    "VAL": [
        SidechainStep("CG1", "CB", "CA", "N", 1.53, 113.0, "chi1"),
        SidechainStep("CG2", "CB", "CA", "N", 1.53, 113.0, ("chi1", 120.0)),
    ],
}

SIDECHAIN_STEPS["HYL"] = SIDECHAIN_STEPS["LYZ"]


def sidechain_atom_names(resname: str) -> List[str]:
    if resname == "ALA":
        return ["CB"]
    if resname == "GLY":
        return []
    return ["CB"] + [step.atom for step in SIDECHAIN_STEPS.get(resname, [])]

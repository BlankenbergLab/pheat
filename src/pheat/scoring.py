"""Deterministic approximate heavy-atom scoring backends."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pheat.domains import filter_structure_for_domain, merge_coverage_metadata
from pheat.geometry import cross, dihedral_degrees, distance, dot, normalize, place_atom, sub
from pheat.metrics import RMSD_ATOM_SETS, normalize_rmsd_atom_set, structure_radius_of_gyration
from pheat.models import Atom, EnergyResult, HeavyAtomStructure, ResidueKey
from pheat.residue_geometry import ANGLE_CA_C_O, C_N, C_O, CA_C, N_CA, PRO_N_CD
from pheat.residues import CANONICAL_RESIDUES, SIDECHAIN_STEPS, SUPPORTED_RESIDUES, THREE_TO_ONE
from pheat.sasa import SASA_BACKENDS, residue_burial
from pheat.score_contracts import normalize_score_model_id, score_input_contract
from pheat.score_tables import default_profile, load_score_table_set, profile_ids, table_for_model

# Common van der Waals radii used for broad contact detection. These are
# Bondi-style approximate radii (source manifest: bondi-1964-vdw-radii) plus a
# few practical ion values for PDB heterogens; they are not force-field sigmas.
VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "SE": 1.90,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
    "NA": 2.27,
    "K": 2.75,
}

# The remaining built-in scoring constants are PHEAT heuristics for deterministic
# relative comparisons. They are intentionally not original DFIRE, GOAP, AMBER,
# or OpenMM parameter tables; the PHEAT DFIRE/GOAP names mark inspired local
# approximations instead of redistributed external statistical-potential data.
ELEMENT_CONTACT_WEIGHT = {
    "C": -0.10,
    "N": -0.06,
    "O": -0.08,
    "S": -0.12,
    "SE": -0.12,
    "P": -0.05,
}

ATOM_CHARGES = {
    "OD1": -0.45,
    "OD2": -0.45,
    "OE1": -0.45,
    "OE2": -0.45,
    "NZ": 0.45,
    "NH1": 0.35,
    "NH2": 0.35,
    "NE": 0.20,
    "ND1": 0.10,
    "NE2": 0.10,
    "P": 0.60,
    "O1P": -0.45,
    "O2P": -0.45,
    "O3P": -0.45,
    "O": -0.35,
    "N": -0.20,
    "C": 0.30,
}

PHEAT_PHYSICS_CHARGE_PROFILES = (
    "pheat-default",
    "protein-coarse-charge-v1",
)

_PHEAT_PHYSICS_COMMON_CHARGES = {
    "OXT": -1.0,
    "NZ": 1.0,
    "NH1": 0.5,
    "NH2": 0.5,
    "OD1": -0.5,
    "OD2": -0.5,
    "OE1": -0.5,
    "OE2": -0.5,
    "ND2": 0.5,
    "NE2": 0.5,
    "SG": -0.1,
    "SD": -0.1,
    "ND1": -0.4,
}
_PHEAT_PHYSICS_CHARGE_TABLES = {
    "protein-coarse-charge-v1": {
        "N": -0.42,
        "H": 0.00,
        "CA": 0.00,
        "C": 0.60,
        "O": -0.57,
        "OG": -0.2,
        "HG": 0.0,
        "OG1": -0.2,
        "HG1": 0.0,
        "OH": -0.1,
        "HH": 0.0,
        "NE1": -0.1,
        "HE1": 0.0,
    },
}
GENERIC_CONTACT_CUTOFF = 6.0
PHEAT_DISTANCE_CONTACT_CUTOFF = 15.0
HEAVY_MM_NONBONDED_CUTOFF = 15.0
PHYSICAL_INTEGRITY_CLASH_CUTOFF = 4.0
PHYSICAL_INTEGRITY_CLASH_FRACTION = 0.75
PHYSICAL_INTEGRITY_SHORT_CONTACT_A = 0.70
PHYSICAL_INTEGRITY_CLASH_WEIGHT = 100.0
PHYSICAL_INTEGRITY_SHORT_CONTACT_WEIGHT = 100000.0
PHYSICAL_INTEGRITY_NONFINITE_PENALTY = 1000000000.0
PHYSICAL_INTEGRITY_WEIGHT_DEFAULT = 1000.0
GOAP_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT = 0.005
HYDROPATHY_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT = 0.0001
BACKBONE_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT = 0.001
COARSE_OMEGA_WINDOW_MIN_DEG = 170.0
COARSE_OMEGA_WINDOW_MAX_DEG = 190.0
COARSE_OMEGA_WINDOW_SCALE_DEFAULT = 25.0
COARSE_OMEGA_SCALE_DEFAULT = 1.0
COARSE_HARD_CLASH_MIN_A_DEFAULT = 1.20
COARSE_HARD_CLASH_SCALE_DEFAULT = 5000.0
COARSE_HARD_CLASH_POWER_DEFAULT = 4.0
COARSE_HARD_CLASH_14_SCALE_DEFAULT = 0.25
COARSE_LJ_TYPE_PARAMS = {
    "H": (1.20, 0.0157), "H_polar": (1.05, 0.0157),
    "C_backbone": (1.75, 0.0700), "C_carbonyl": (1.70, 0.0860),
    "C_aliphatic": (1.90, 0.1094), "C_aromatic": (1.85, 0.1200),
    "N_backbone": (1.65, 0.1700), "N_sidechain": (1.65, 0.1700),
    "O_carbonyl": (1.60, 0.2100), "O_hydroxyl": (1.55, 0.1700),
    "O_carboxyl": (1.60, 0.2100), "S_sulfur": (2.00, 0.2500),
    "X": (1.75, 0.1000),
}
COARSE_CHI_CAP_BY_RESIDUE = {
    "CYS": 1,
    "ASP": 1,
    "GLU": 2,
    "SER": 1,
    "THR": 1,
    "VAL": 1,
    "ILE": 2,
    "LEU": 1,
    "MET": 3,
    "LYS": 4,
    "ARG": 4,
}
OPENMM_PREPARATION_SEED = 20260514

RESIDUE_HYDROPHOBICITY = {
    "ALA": -0.20,
    "VAL": -0.45,
    "ILE": -0.50,
    "LEU": -0.50,
    "MET": -0.35,
    "PHE": -0.55,
    "TRP": -0.55,
    "TYR": -0.25,
    "PRO": -0.20,
    "CYS": -0.25,
    "GLY": -0.05,
    "SER": 0.05,
    "THR": 0.05,
    "ASN": 0.12,
    "GLN": 0.12,
    "ASP": 0.25,
    "GLU": 0.25,
    "LYS": 0.20,
    "ARG": 0.20,
    "HIS": 0.10,
}

KYTE_DOOLITTLE_HYDROPATHY = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}


SUPPORTED_MODELS = (
    "generic",
    "pheat-dfire",
    "pheat-goap",
    "pheat-goap-physical",
    "pheat-mj",
    "pheat-hydropathy",
    "pheat-hydropathy-physical",
    "pheat-backbone",
    "pheat-backbone-physical",
    "pheat-rotamer",
    "pheat-hbond",
    "pheat-rg",
    "pheat-ml-linear",
    "pheat-physics",
    "pheat-custom-energy-v1",
    "pheat-geometry-integrity",
    "pheat-physical-integrity",
    "heavy-mm",
    "pheat-heavy-mm-physical",
    "openmm-prepared",
    "ambertools-sander",
    "gromacs-mdrun",
)

IMPLEMENTATION_NATIVE_PHEAT = "native-pheat"
IMPLEMENTATION_EXTERNAL_PYTHON = "external-python"
IMPLEMENTATION_EXTERNAL_EXECUTABLE = "external-executable"


def _native_model_metadata(units: str) -> dict[str, Any]:
    return {
        "units": units,
        "requires": [],
        "implementation_origin": IMPLEMENTATION_NATIVE_PHEAT,
        "implementation_backend": "pheat",
        "implementation_summary": "Implemented entirely inside PHEAT with bundled Python code and packaged tables when used.",
    }


MODEL_METADATA = {
    "generic": _native_model_metadata("arbitrary"),
    "pheat-dfire": _native_model_metadata("arbitrary"),
    "pheat-goap": _native_model_metadata("arbitrary"),
    "pheat-goap-physical": _native_model_metadata("arbitrary"),
    "pheat-mj": _native_model_metadata("arbitrary"),
    "pheat-hydropathy": _native_model_metadata("arbitrary"),
    "pheat-hydropathy-physical": _native_model_metadata("arbitrary"),
    "pheat-backbone": _native_model_metadata("arbitrary"),
    "pheat-backbone-physical": _native_model_metadata("arbitrary"),
    "pheat-rotamer": _native_model_metadata("arbitrary"),
    "pheat-hbond": _native_model_metadata("arbitrary"),
    "pheat-rg": _native_model_metadata("arbitrary"),
    "pheat-ml-linear": _native_model_metadata("arbitrary"),
    "pheat-physics": _native_model_metadata("arbitrary"),
    "pheat-custom-energy-v1": _native_model_metadata("arbitrary"),
    "pheat-geometry-integrity": _native_model_metadata("arbitrary"),
    "pheat-physical-integrity": _native_model_metadata("arbitrary"),
    "heavy-mm": _native_model_metadata("arbitrary"),
    "pheat-heavy-mm-physical": _native_model_metadata("arbitrary"),
    "openmm-prepared": {
        "units": "kJ/mol",
        "requires": ["openmm"],
        "optional_requires": ["pdbfixer"],
        "implementation_origin": IMPLEMENTATION_EXTERNAL_PYTHON,
        "implementation_backend": "openmm",
        "implementation_summary": "Uses optional OpenMM/PDBFixer Python packages for preparation and force-field scoring.",
    },
    "ambertools-sander": {
        "units": "kcal/mol",
        "requires": ["executable:tleap", "executable:sander"],
        "optional_requires": ["openmm", "pdbfixer"],
        "implementation_origin": IMPLEMENTATION_EXTERNAL_EXECUTABLE,
        "implementation_backend": "ambertools",
        "implementation_summary": "Shells out to user-installed AmberTools tleap and sander executables.",
    },
    "gromacs-mdrun": {
        "units": "kJ/mol",
        "requires": ["executable:gmx"],
        "optional_requires": ["openmm", "pdbfixer"],
        "implementation_origin": IMPLEMENTATION_EXTERNAL_EXECUTABLE,
        "implementation_backend": "gromacs",
        "implementation_summary": "Shells out to a user-installed GROMACS gmx executable.",
    },
}

PREPARE_MODES = ("auto", "never", "write")
AMBER_SOLVENT_MODES = ("vacuum", "gb")
DEFAULT_AMBER_FORCEFIELD = "leaprc.protein.ff14SB"
GROMACS_RUN_MODES = ("rerun", "minimize", "minimize-rerun")
GROMACS_PREFLIGHT_MODES = ("strict", "warn", "off")
GROMACS_WATER_MODELS = ("auto", "none", "opc", "opc3", "tip3p", "spc", "spce", "tip4p")
DEFAULT_GROMACS_FORCEFIELD = "amber19sb"
DEFAULT_GROMACS_WATER = "auto"
DEFAULT_GROMACS_BOX_DISTANCE_NM = 1.0
PREP_CACHE_MODES = ("off", "readwrite", "readonly", "refresh")


@dataclass(frozen=True)
class GromacsRunSettings:
    """Configurable GROMACS run/preprocessing settings for external scoring."""

    minimize_steps: int = 500
    emtol: float = 1000.0
    emstep: float = 0.01
    box_distance_nm: float = DEFAULT_GROMACS_BOX_DISTANCE_NM
    cutoff_nm: float = 1.0
    coulombtype: str = "PME"
    vdwtype: str = "Cut-off"
    nstlist: int = 10
    pbc: str = "xyz"
    comm_mode: str = "Linear"
    mdrun_flags: tuple[str, ...] = field(default_factory=tuple)
    grompp_maxwarn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimize_steps": self.minimize_steps,
            "emtol": self.emtol,
            "emstep": self.emstep,
            "box_distance_nm": self.box_distance_nm,
            "cutoff_nm": self.cutoff_nm,
            "coulombtype": self.coulombtype,
            "vdwtype": self.vdwtype,
            "nstlist": self.nstlist,
            "pbc": self.pbc,
            "comm_mode": self.comm_mode,
            "mdrun_flags": list(self.mdrun_flags),
            "grompp_maxwarn": self.grompp_maxwarn,
        }

PHEAT_RG_DEFAULTS: Dict[str, Any] = {
    "atom_set": "ca",
    "mode": "unweighted",
    "a": 2.2,
    "b": 1.0 / 3.0,
    "sigma_fraction": 0.25,
    "fitted": False,
}


def supported_models() -> List[str]:
    """Return every scoring model ID recognized by PHEAT."""

    return list(SUPPORTED_MODELS)


def available_models() -> List[str]:
    """Return scoring model IDs runnable in the current Python environment."""

    return [capability["model"] for capability in model_capabilities() if capability["available"]]


def model_capabilities() -> List[dict[str, Any]]:
    """Return supported scoring models with current runtime availability metadata."""

    capabilities = []
    for model in SUPPORTED_MODELS:
        metadata = MODEL_METADATA[model]
        requirements = list(metadata["requires"])
        optional_requirements = list(metadata.get("optional_requires", []))
        missing = _missing_optional_requirements(requirements)
        missing_optional = _missing_optional_requirements(optional_requirements)
        capabilities.append(
            {
                "model": model,
                "supported": True,
                "available": not missing,
                "requires": requirements,
                "optional_requires": optional_requirements,
                "units": metadata["units"],
                "implementation": _implementation_metadata_for_model(model),
                "input_contract": score_input_contract(model),
                "reason": None if not missing else "missing dependency: " + "; ".join(missing),
                "optional_status": None
                if not missing_optional
                else "missing optional enhancement: " + "; ".join(missing_optional),
            }
        )
    return capabilities


def score_model_option_specs(model: Optional[str] = None) -> Union[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return structured option metadata for PHEAT scoring models."""

    if model is not None:
        normalized = normalize_score_model_id(model)
        if normalized not in SUPPORTED_MODELS:
            known = ", ".join(supported_models())
            raise ValueError(f"Unknown scoring model '{model}'. Supported models: {known}")
        return deepcopy(_score_model_option_specs(normalized))
    return {item: deepcopy(_score_model_option_specs(item)) for item in SUPPORTED_MODELS}


def validate_scoring_options(
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    require_executables: bool = False,
) -> dict[str, Any]:
    """Validate scorer options without running structure preparation or scoring."""

    errors: list[str] = []
    warnings: list[str] = []
    normalized_model = normalize_score_model_id(model)
    if normalized_model not in SUPPORTED_MODELS:
        return {
            "ok": False,
            "model": normalized_model,
            "options": {},
            "errors": [f"unsupported scoring model: {model}"],
            "warnings": warnings,
            "require_executables": bool(require_executables),
        }

    specs = _score_model_option_specs(normalized_model)
    specs_by_name = {str(spec["name"]): spec for spec in specs}
    provided = dict(options or {})
    normalized_options: dict[str, Any] = {}
    for name, value in provided.items():
        if name not in specs_by_name:
            errors.append(f"unknown option for {normalized_model}: {name}")
            continue
        normalized_options[name] = _normalize_scoring_option_value(specs_by_name[name], value, errors)

    domain = str(normalized_options.get("domain", "protein-heavy"))
    compatible_domains = score_input_contract(normalized_model).get("compatible_domains", [])
    if domain not in compatible_domains:
        warnings.append(
            f"{normalized_model} input contract is calibrated for domains "
            f"{', '.join(str(item) for item in compatible_domains)}; requested domain {domain}"
        )

    if normalized_model in {"ambertools-sander", "gromacs-mdrun"}:
        _validate_external_scoring_option_semantics(
            normalized_model,
            normalized_options,
            require_executables=require_executables,
            errors=errors,
            warnings=warnings,
        )

    return {
        "ok": not errors,
        "model": normalized_model,
        "options": normalized_options,
        "errors": errors,
        "warnings": warnings,
        "require_executables": bool(require_executables),
    }


def _implementation_metadata_for_model(model: str) -> dict[str, Any]:
    metadata = MODEL_METADATA[normalize_score_model_id(model)]
    origin = str(metadata["implementation_origin"])
    return {
        "origin": origin,
        "backend": str(metadata["implementation_backend"]),
        "native_pheat": origin == IMPLEMENTATION_NATIVE_PHEAT,
        "external": origin != IMPLEMENTATION_NATIVE_PHEAT,
        "summary": str(metadata["implementation_summary"]),
    }


def _missing_optional_requirements(requirements: Sequence[str]) -> list[str]:
    missing = []
    for requirement in requirements:
        if requirement.startswith("executable:"):
            executable = requirement.split(":", 1)[1]
            if shutil.which(executable) is None:
                missing.append(f"{executable} executable was not found on PATH")
            continue
        try:
            importlib.import_module(requirement)
        except Exception as exc:  # pragma: no cover - depends on optional user environment
            missing.append(f"{requirement} ({exc})")
    return missing


def _score_model_option_specs(model: str) -> list[dict[str, Any]]:
    contract = score_input_contract(model)
    specs = [
        _option_spec(
            "domain",
            "string",
            "protein-heavy",
            "Atom/residue domain to score.",
            choices=["protein-heavy", "all-heavy", "full"],
        )
    ]
    if bool(contract.get("uses_table")):
        specs.extend(
            [
                _option_spec("table_set", "path-or-mapping", None, "Score table set path or loaded mapping."),
                _option_spec("profile", "string", None, "Score table profile to use."),
            ]
        )
    if model in {"pheat-hydropathy", "pheat-hydropathy-physical"}:
        specs.extend(
            [
                _option_spec(
                    "burial_method",
                    "string",
                    None,
                    "Residue burial method for hydropathy scoring.",
                    choices=["contacts", "sasa"],
                ),
                _option_spec(
                    "sasa_backend",
                    "string",
                    "auto",
                    "SASA backend used when burial_method is sasa.",
                    choices=list(SASA_BACKENDS),
                ),
            ]
        )
    if model == "pheat-hydropathy-physical":
        specs.append(
            _option_spec(
                "physical_integrity_weight",
                "float",
                HYDROPATHY_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
                "Weight applied to the PHEAT physical-integrity penalty in the guarded hydropathy objective.",
            )
        )
    if model == "pheat-backbone-physical":
        specs.append(
            _option_spec(
                "physical_integrity_weight",
                "float",
                BACKBONE_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
                "Weight applied to the PHEAT physical-integrity penalty in the guarded backbone objective.",
            )
        )
    if model == "pheat-goap-physical":
        specs.append(
            _option_spec(
                "physical_integrity_weight",
                "float",
                GOAP_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
                "Weight applied to the PHEAT physical-integrity penalty in the guarded GOAP objective.",
            )
        )
    if model == "pheat-heavy-mm-physical":
        specs.append(
            _option_spec(
                "physical_integrity_weight",
                "float",
                PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
                "Weight applied to the PHEAT physical-integrity penalty.",
            )
        )
    if model in {"pheat-physics", "pheat-custom-energy-v1"}:
        specs.extend(
            [
                _option_spec(
                    "charge_profile",
                    "string",
                    "protein-coarse-charge-v1",
                    "PHEAT heuristic atom-name charge profile used by the native physical scorer; this is not an external all-atom force field.",
                    choices=list(PHEAT_PHYSICS_CHARGE_PROFILES),
                ),
                _option_spec("hydrophobic_gamma", "float", 15.0, "Hydrophobic burial penalty scale."),
                _option_spec(
                    "end_to_end_weight",
                    "float",
                    8.0 if model == "pheat-custom-energy-v1" else 50.0,
                    "End-to-end compactness/restraint weight.",
                ),
                _option_spec("end_to_end_target", "float", None, "Optional target CA-to-CA end-to-end distance in Angstroms."),
                _option_spec("end_to_end_slack", "float", None, "Optional dead-band before end-to-end restraint is penalized."),
                _option_spec("hbond_weight", "float", 1.0, "Hydrogen-bond/contact proxy term weight."),
                _option_spec("electrostatic_weight", "float", 1.0, "Electrostatic term weight."),
                _option_spec("disulfide_weight", "float", 1.0, "Disulfide favorability/overload term weight."),
                _option_spec("vdw_weight", "float", 1.0, "VDW/steric term weight."),
                _option_spec("rotamer_weight", "float", 1.0, "Side-chain rotamer plausibility term weight."),
                _option_spec("backbone_weight", "float", 1.0, "Backbone torsion plausibility term weight."),
                _option_spec("pi_stacking_weight", "float", 1.0, "Aromatic/pi-stacking term weight."),
                _option_spec("geometry_integrity_weight", "float", 1.0, "Geometry-integrity term weight."),
                _option_spec("adjacent_heavy_steric_weight", "float", 1.0, "Adjacent-residue local heavy-atom steric penalty weight."),
            ]
        )
        if model == "pheat-physics":
            specs.append(_option_spec("physical_integrity_weight", "float", 0.0, "Additional physical-integrity barrier weight."))
        if model == "pheat-custom-energy-v1":
            specs.extend(
                [
                    _option_spec("decoded_torsions", "mapping", None, "Optional decoded torsion angles in radians keyed as residueIndex_angleName, for example 3_phi or 3_chi1."),
                _option_spec("hydrophobic_burial_denominator", "float", 35.0, "Neighbor-count denominator matching the QTF coarse objective."),
                _option_spec("hydrophobic_burial_scale", "float", 0.7, "Scale matching the QTF coarse objective's SASA factor."),
                    _option_spec("use_end_to_end_constraint", "boolean", True, "Whether to apply the end-to-end compactness restraint."),
                    _option_spec("end_to_end_scale", "float", 1.0, "Additional scale multiplier for the end-to-end compactness restraint."),
                    _option_spec("omega_weight", "float", COARSE_OMEGA_SCALE_DEFAULT, "Omega preference weight used for decoded torsion scoring."),
                    _option_spec("omega_window_scale", "float", COARSE_OMEGA_WINDOW_SCALE_DEFAULT, "Penalty scale for omega values outside the trans window."),
                    _option_spec("hard_clash_min_A", "float", COARSE_HARD_CLASH_MIN_A_DEFAULT, "Minimum heavy-atom distance before the coarse hard-clash wall activates."),
                    _option_spec("hard_clash_scale", "float", COARSE_HARD_CLASH_SCALE_DEFAULT, "Scale factor for the coarse hard-clash wall."),
                    _option_spec("hard_clash_power", "float", COARSE_HARD_CLASH_POWER_DEFAULT, "Power used by the coarse hard-clash wall."),
                    _option_spec("hard_clash_14_scale", "float", COARSE_HARD_CLASH_14_SCALE_DEFAULT, "Relative scale applied to graph-distance-3 pairs in the coarse hard-clash wall."),
                ]
            )
    if model == "ambertools-sander":
        specs.extend(
            [
                _option_spec("prepare", "string", "auto", "Preparation mode before scoring.", choices=list(PREPARE_MODES)),
                _option_spec("prepared_output", "path", None, "Optional prepared coordinate output path."),
                _option_spec("amber_forcefield", "string", DEFAULT_AMBER_FORCEFIELD, "AmberTools tleap force-field source."),
                _option_spec(
                    "amber_solvent",
                    "string",
                    "vacuum",
                    "AmberTools solvent treatment.",
                    choices=list(AMBER_SOLVENT_MODES),
                ),
                _option_spec("ambertools_work_dir", "path", None, "Directory for AmberTools intermediate files."),
                _option_spec("keep_ambertools_files", "boolean", False, "Keep AmberTools intermediate files."),
                _option_spec("external_timeout_seconds", "float", None, "Timeout for external commands in seconds."),
                _option_spec("prep_cache_dir", "path", None, "Optional preparation cache directory."),
                _option_spec("prep_cache_mode", "string", "off", "Preparation cache mode.", choices=list(PREP_CACHE_MODES)),
            ]
        )
    if model == "gromacs-mdrun":
        specs.extend(
            [
                _option_spec("prepare", "string", "auto", "Preparation mode before scoring.", choices=list(PREPARE_MODES)),
                _option_spec("prepared_output", "path", None, "Optional prepared coordinate output path."),
                _option_spec("gromacs_forcefield", "string", DEFAULT_GROMACS_FORCEFIELD, "GROMACS pdb2gmx force-field name."),
                _option_spec(
                    "gromacs_water",
                    "string",
                    DEFAULT_GROMACS_WATER,
                    "GROMACS water model.",
                    choices=list(GROMACS_WATER_MODELS),
                ),
                _option_spec("gromacs_solvate", "boolean", False, "Build a solvated box before scoring."),
                _option_spec(
                    "gromacs_run_mode",
                    "string",
                    "rerun",
                    "GROMACS run mode.",
                    choices=list(GROMACS_RUN_MODES),
                ),
                _option_spec(
                    "gromacs_preflight",
                    "string",
                    "strict",
                    "Physical-integrity preflight mode before GROMACS execution.",
                    choices=list(GROMACS_PREFLIGHT_MODES),
                ),
                _option_spec("gromacs_work_dir", "path", None, "Directory for GROMACS intermediate files."),
                _option_spec("keep_gromacs_files", "boolean", False, "Keep GROMACS intermediate files."),
                _option_spec("gromacs_metrics", "boolean", False, "Attempt additional GROMACS metrics such as gyrate."),
                _option_spec("gromacs_run_settings", "mapping", None, "GROMACS MDP and mdrun settings."),
                _option_spec("external_timeout_seconds", "float", None, "Timeout for external commands in seconds."),
                _option_spec("prep_cache_dir", "path", None, "Optional preparation cache directory."),
                _option_spec("prep_cache_mode", "string", "off", "Preparation cache mode.", choices=list(PREP_CACHE_MODES)),
            ]
        )
    return specs


def _option_spec(
    name: str,
    value_type: str,
    default: Any,
    description: str,
    *,
    choices: Optional[Sequence[Any]] = None,
    required: bool = False,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "type": value_type,
        "default": default,
        "required": bool(required),
        "description": description,
    }
    if choices is not None:
        payload["choices"] = list(choices)
    return payload


def _normalize_scoring_option_value(spec: Mapping[str, Any], value: Any, errors: list[str]) -> Any:
    name = str(spec["name"])
    value_type = str(spec["type"])
    normalized: Any
    try:
        if value is None:
            normalized = None
        elif value_type == "string":
            normalized = str(value).strip()
        elif value_type == "path":
            normalized = str(Path(value))
        elif value_type == "path-or-mapping":
            normalized = dict(value) if isinstance(value, Mapping) else str(Path(value))
        elif value_type == "boolean":
            normalized = _normalize_bool_option(value, name)
        elif value_type == "float":
            normalized = float(value)
        elif value_type == "mapping":
            if name == "gromacs_run_settings":
                normalized = _normalize_gromacs_run_settings(value).to_dict()
            elif isinstance(value, Mapping):
                normalized = dict(value)
            else:
                raise ValueError(f"{name} must be a mapping")
        else:
            normalized = value
        choices = spec.get("choices")
        if choices is not None and normalized is not None and normalized not in choices:
            errors.append(f"{name} must be one of {', '.join(str(item) for item in choices)}")
        return normalized
    except Exception as exc:
        errors.append(str(exc))
        return value


def _normalize_bool_option(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean")


def _validate_external_scoring_option_semantics(
    model: str,
    options: Mapping[str, Any],
    *,
    require_executables: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    _collect_validation_error(errors, lambda: _normalize_prepare_mode(str(options.get("prepare", "auto"))))
    _collect_validation_error(errors, lambda: _normalize_external_timeout(options.get("external_timeout_seconds")))
    cache_mode = _collect_validation_value(
        errors,
        lambda: _normalize_prep_cache_mode(str(options.get("prep_cache_mode", "off"))),
    )
    prep_cache_dir = options.get("prep_cache_dir")
    if cache_mode and cache_mode != "off" and prep_cache_dir is None:
        errors.append("prep_cache_mode requires prep_cache_dir unless mode is off")
    if cache_mode == "readonly" and prep_cache_dir is not None and not Path(str(prep_cache_dir)).exists():
        errors.append(f"readonly prep cache does not exist: {prep_cache_dir}")

    if model == "ambertools-sander":
        _collect_validation_error(errors, lambda: _normalize_amber_solvent(str(options.get("amber_solvent", "vacuum"))))
        if require_executables:
            for executable in ("tleap", "sander"):
                if shutil.which(executable) is None:
                    errors.append(f"{executable} executable was not found on PATH")
        if cache_mode and cache_mode != "off":
            warnings.append("AmberTools prep cache is conservative and does not skip tleap topology generation")

    if model == "gromacs-mdrun":
        _collect_validation_error(errors, lambda: _normalize_gromacs_run_mode(str(options.get("gromacs_run_mode", "rerun"))))
        _collect_validation_error(errors, lambda: _normalize_gromacs_preflight(str(options.get("gromacs_preflight", "strict"))))
        _collect_validation_error(
            errors,
            lambda: _effective_gromacs_water(
                str(options.get("gromacs_water", DEFAULT_GROMACS_WATER)),
                solvate=bool(options.get("gromacs_solvate", False)),
            ),
        )
        if "gromacs_run_settings" in options:
            _collect_validation_error(errors, lambda: _normalize_gromacs_run_settings(options.get("gromacs_run_settings")))
        if require_executables:
            gmx = shutil.which("gmx")
            if gmx is None:
                errors.append("gmx executable was not found on PATH")
            else:
                installed = _installed_gromacs_forcefields(gmx, Path.cwd())
                forcefield = _collect_validation_value(
                    errors,
                    lambda: _normalize_gromacs_forcefield(str(options.get("gromacs_forcefield", DEFAULT_GROMACS_FORCEFIELD))),
                )
                if installed and forcefield is not None and forcefield not in installed:
                    errors.append(
                        f"GROMACS force field '{forcefield}' was not found in the active GROMACS installation. "
                        f"Installed force fields: {', '.join(installed)}."
                    )


def validate_external_scoring_options(
    *,
    model: str,
    prepare: str = "auto",
    amber_solvent: str = "vacuum",
    gromacs_forcefield: str = DEFAULT_GROMACS_FORCEFIELD,
    gromacs_water: str = DEFAULT_GROMACS_WATER,
    gromacs_solvate: bool = False,
    gromacs_run_mode: str = "rerun",
    gromacs_preflight: str = "strict",
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Union[str, Path]] = None,
    prep_cache_mode: str = "off",
    require_executables: bool = True,
) -> dict[str, Any]:
    """Validate external scorer options without running preparation or scoring."""

    errors: list[str] = []
    warnings: list[str] = []
    normalized_model = normalize_score_model_id(model)
    installed_forcefields: list[str] = []
    settings_payload: Optional[dict[str, Any]] = None

    if normalized_model not in SUPPORTED_MODELS:
        errors.append(f"unsupported scoring model: {model}")
        return {
            "ok": False,
            "model": normalized_model,
            "errors": errors,
            "warnings": warnings,
        }
    if normalized_model not in {"ambertools-sander", "gromacs-mdrun"}:
        return {
            "ok": True,
            "model": normalized_model,
            "external": False,
            "errors": errors,
            "warnings": warnings,
        }

    _collect_validation_error(errors, lambda: _normalize_prepare_mode(prepare))
    _collect_validation_error(errors, lambda: _normalize_external_timeout(external_timeout_seconds))
    cache_mode = _collect_validation_value(errors, lambda: _normalize_prep_cache_mode(prep_cache_mode))
    if cache_mode and cache_mode != "off" and prep_cache_dir is None:
        errors.append("--prep-cache-mode requires --prep-cache-dir unless mode is off")
    if cache_mode == "readonly" and prep_cache_dir is not None and not Path(prep_cache_dir).exists():
        errors.append(f"readonly prep cache does not exist: {prep_cache_dir}")

    if normalized_model == "ambertools-sander":
        _collect_validation_error(errors, lambda: _normalize_amber_solvent(amber_solvent))
        if require_executables:
            for executable in ("tleap", "sander"):
                if shutil.which(executable) is None:
                    errors.append(f"{executable} executable was not found on PATH")
        if cache_mode and cache_mode != "off":
            warnings.append("AmberTools prep cache is conservative and does not skip tleap topology generation")

    if normalized_model == "gromacs-mdrun":
        _collect_validation_error(errors, lambda: _normalize_gromacs_run_mode(gromacs_run_mode))
        preflight_mode = _collect_validation_value(errors, lambda: _normalize_gromacs_preflight(gromacs_preflight))
        _collect_validation_error(
            errors,
            lambda: _effective_gromacs_water(gromacs_water, solvate=gromacs_solvate),
        )
        settings = _collect_validation_value(
            errors,
            lambda: _normalize_gromacs_run_settings(gromacs_run_settings),
        )
        if settings is not None:
            settings_payload = settings.to_dict()
        gmx = shutil.which("gmx")
        if require_executables and gmx is None:
            errors.append("gmx executable was not found on PATH")
        if gmx is not None:
            installed_forcefields = _installed_gromacs_forcefields(gmx, Path.cwd())
            forcefield = _collect_validation_value(
                errors,
                lambda: _normalize_gromacs_forcefield(gromacs_forcefield),
            )
            if installed_forcefields and forcefield is not None and forcefield not in installed_forcefields:
                errors.append(
                    f"GROMACS force field '{forcefield}' was not found in the active GROMACS installation. "
                    f"Installed force fields: {', '.join(installed_forcefields)}."
                )

    return {
        "ok": not errors,
        "model": normalized_model,
        "external": True,
        "errors": errors,
        "warnings": warnings,
        "installed_gromacs_forcefields": installed_forcefields,
        "gromacs_preflight": preflight_mode if normalized_model == "gromacs-mdrun" else None,
        "gromacs_run_settings": settings_payload,
        "prep_cache_mode": cache_mode,
        "prep_cache_dir": str(prep_cache_dir) if prep_cache_dir is not None else None,
        "external_timeout_seconds": external_timeout_seconds,
    }


def _collect_validation_error(errors: list[str], callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception as exc:
        errors.append(str(exc))


def _collect_validation_value(errors: list[str], callback: Callable[[], Any]) -> Any:
    try:
        return callback()
    except Exception as exc:
        errors.append(str(exc))
        return None


def score_structure(
    structure: HeavyAtomStructure,
    *,
    model: str = "generic",
    domain: str = "protein-heavy",
    table_set: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    profile: Optional[str] = None,
    burial_method: Optional[str] = None,
    sasa_backend: str = "auto",
    prepare: str = "auto",
    prepared_output: Optional[Union[str, Path]] = None,
    amber_forcefield: str = DEFAULT_AMBER_FORCEFIELD,
    amber_solvent: str = "vacuum",
    ambertools_work_dir: Optional[Union[str, Path]] = None,
    keep_ambertools_files: bool = False,
    gromacs_forcefield: str = DEFAULT_GROMACS_FORCEFIELD,
    gromacs_water: str = DEFAULT_GROMACS_WATER,
    gromacs_solvate: bool = False,
    gromacs_run_mode: str = "rerun",
    gromacs_preflight: str = "strict",
    gromacs_work_dir: Optional[Union[str, Path]] = None,
    keep_gromacs_files: bool = False,
    gromacs_metrics: bool = False,
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Union[str, Path]] = None,
    prep_cache_mode: str = "off",
    physical_integrity_weight: Optional[float] = None,
    charge_profile: str = "protein-coarse-charge-v1",
    hydrophobic_gamma: float = 15.0,
    hydrophobic_burial_denominator: float = 35.0,
    hydrophobic_burial_scale: float = 0.7,
    end_to_end_weight: float = 8.0,
    end_to_end_target: Optional[float] = None,
    end_to_end_slack: Optional[float] = None,
    hbond_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    disulfide_weight: float = 1.0,
    vdw_weight: float = 1.0,
    rotamer_weight: float = 1.0,
    backbone_weight: float = 1.0,
    pi_stacking_weight: float = 1.0,
    geometry_integrity_weight: float = 1.0,
    adjacent_heavy_steric_weight: float = 1.0,
    omega_weight: float = COARSE_OMEGA_SCALE_DEFAULT,
    omega_window_scale: float = COARSE_OMEGA_WINDOW_SCALE_DEFAULT,
    hard_clash_min_A: float = COARSE_HARD_CLASH_MIN_A_DEFAULT,
    hard_clash_scale: float = COARSE_HARD_CLASH_SCALE_DEFAULT,
    hard_clash_power: float = COARSE_HARD_CLASH_POWER_DEFAULT,
    hard_clash_14_scale: float = COARSE_HARD_CLASH_14_SCALE_DEFAULT,
    decoded_torsions: Optional[Mapping[str, Any]] = None,
    use_end_to_end_constraint: bool = True,
    end_to_end_scale: float = 1.0,
    status_stream: Optional[Any] = None,
) -> EnergyResult:
    """Score an atom structure with an explicitly named backend."""

    normalized = normalize_score_model_id(model)
    filtered, coverage = filter_structure_for_domain(structure, domain=domain)
    tables = load_score_table_set(table_set)
    contract = score_input_contract(normalized)
    contract_warnings = _score_contract_warnings(normalized, contract, structure, filtered, coverage)
    if normalized == "generic":
        return _with_coverage(_score_generic(filtered), coverage, contract=contract, contract_warnings=contract_warnings)
    if normalized == "pheat-dfire":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_dfire(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-goap":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_goap(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-goap-physical":
        table = table_for_model(tables, "pheat-goap", profile=profile)
        weight = (
            GOAP_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT
            if physical_integrity_weight is None
            else float(physical_integrity_weight)
        )
        return _with_coverage(
            _score_pheat_goap_physical(filtered, table=table, physical_integrity_weight=weight),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-mj":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_mj(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-hydropathy":
        table = table_for_model(tables, normalized, profile=profile)
        method = burial_method or _table_burial_method(table) or "contacts"
        return _with_coverage(
            _score_pheat_hydropathy(filtered, table=table, burial_method=method, sasa_backend=sasa_backend),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-hydropathy-physical":
        table = table_for_model(tables, "pheat-hydropathy", profile=profile)
        method = burial_method or _table_burial_method(table) or "contacts"
        weight = (
            HYDROPATHY_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT
            if physical_integrity_weight is None
            else float(physical_integrity_weight)
        )
        return _with_coverage(
            _score_pheat_hydropathy_physical(
                filtered,
                table=table,
                burial_method=method,
                sasa_backend=sasa_backend,
                physical_integrity_weight=weight,
            ),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-backbone":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_backbone(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-backbone-physical":
        table = table_for_model(tables, "pheat-backbone", profile=profile)
        weight = (
            BACKBONE_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT
            if physical_integrity_weight is None
            else float(physical_integrity_weight)
        )
        return _with_coverage(
            _score_pheat_backbone_physical(filtered, table=table, physical_integrity_weight=weight),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-rotamer":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_rotamer(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-hbond":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_hbond(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-rg":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_rg(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-ml-linear":
        table = table_for_model(tables, normalized, profile=profile)
        return _with_coverage(
            _score_pheat_ml_linear(filtered, table=table),
            coverage,
            tables,
            profile,
            table=table,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-physics":
        return _with_coverage(
            _score_pheat_physics(
                filtered,
                charge_profile=charge_profile,
                hydrophobic_gamma=hydrophobic_gamma,
                end_to_end_weight=end_to_end_weight,
                end_to_end_target=end_to_end_target,
                end_to_end_slack=end_to_end_slack,
                hbond_weight=hbond_weight,
                electrostatic_weight=electrostatic_weight,
                disulfide_weight=disulfide_weight,
                vdw_weight=vdw_weight,
                rotamer_weight=rotamer_weight,
                backbone_weight=backbone_weight,
                pi_stacking_weight=pi_stacking_weight,
                geometry_integrity_weight=geometry_integrity_weight,
                physical_integrity_weight=0.0 if physical_integrity_weight is None else float(physical_integrity_weight),
                adjacent_heavy_steric_weight=adjacent_heavy_steric_weight,
            ),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-custom-energy-v1":
        return _with_coverage(
            _score_pheat_coarse_protein_folding_v1(
                filtered,
                charge_profile=charge_profile,
                hydrophobic_gamma=hydrophobic_gamma,
                hydrophobic_burial_denominator=hydrophobic_burial_denominator,
                hydrophobic_burial_scale=hydrophobic_burial_scale,
                end_to_end_weight=end_to_end_weight,
                end_to_end_target=end_to_end_target,
                end_to_end_slack=end_to_end_slack,
                hbond_weight=hbond_weight,
                electrostatic_weight=electrostatic_weight,
                disulfide_weight=disulfide_weight,
                vdw_weight=vdw_weight,
                rotamer_weight=rotamer_weight,
                backbone_weight=backbone_weight,
                pi_stacking_weight=pi_stacking_weight,
                geometry_integrity_weight=geometry_integrity_weight,
                adjacent_heavy_steric_weight=adjacent_heavy_steric_weight,
                omega_weight=omega_weight,
                omega_window_scale=omega_window_scale,
                hard_clash_min_A=hard_clash_min_A,
                hard_clash_scale=hard_clash_scale,
                hard_clash_power=hard_clash_power,
                hard_clash_14_scale=hard_clash_14_scale,
                decoded_torsions=decoded_torsions,
                use_end_to_end_constraint=use_end_to_end_constraint,
                end_to_end_scale=end_to_end_scale,
            ),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-geometry-integrity":
        return _with_coverage(
            _score_pheat_geometry_integrity(filtered),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "pheat-physical-integrity":
        return _with_coverage(
            _score_pheat_physical_integrity(filtered),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "heavy-mm":
        return _with_coverage(_score_heavy_mm(filtered), coverage, contract=contract, contract_warnings=contract_warnings)
    if normalized == "pheat-heavy-mm-physical":
        weight = (
            PHYSICAL_INTEGRITY_WEIGHT_DEFAULT
            if physical_integrity_weight is None
            else float(physical_integrity_weight)
        )
        return _with_coverage(
            _score_pheat_heavy_mm_physical(filtered, physical_integrity_weight=weight),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "openmm-prepared":
        return _with_coverage(
            _score_openmm_prepared(filtered),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "ambertools-sander":
        return _with_coverage(
            _score_ambertools_sander(
                filtered,
                prepare=prepare,
                prepared_output=Path(prepared_output) if prepared_output is not None else None,
                amber_forcefield=amber_forcefield,
                amber_solvent=amber_solvent,
                ambertools_work_dir=Path(ambertools_work_dir) if ambertools_work_dir is not None else None,
                keep_ambertools_files=keep_ambertools_files,
                external_timeout_seconds=external_timeout_seconds,
                prep_cache_dir=Path(prep_cache_dir) if prep_cache_dir is not None else None,
                prep_cache_mode=prep_cache_mode,
            ),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "gromacs-mdrun":
        return _with_coverage(
            _score_gromacs_mdrun(
                filtered,
                prepare=prepare,
                prepared_output=Path(prepared_output) if prepared_output is not None else None,
                gromacs_forcefield=gromacs_forcefield,
                gromacs_water=gromacs_water,
                gromacs_solvate=gromacs_solvate,
                gromacs_run_mode=gromacs_run_mode,
                gromacs_preflight=gromacs_preflight,
                gromacs_work_dir=Path(gromacs_work_dir) if gromacs_work_dir is not None else None,
                keep_gromacs_files=keep_gromacs_files,
                gromacs_metrics=gromacs_metrics,
                gromacs_run_settings=gromacs_run_settings,
                external_timeout_seconds=external_timeout_seconds,
                prep_cache_dir=Path(prep_cache_dir) if prep_cache_dir is not None else None,
                prep_cache_mode=prep_cache_mode,
                status_stream=status_stream,
            ),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    raise ValueError(f"Unknown scoring model '{model}'. Supported models: {', '.join(supported_models())}")


def score_structure_profiles(
    structure: HeavyAtomStructure,
    *,
    model: str,
    table_set: Union[str, Path, Mapping[str, Any]],
    profiles: Optional[Sequence[str]] = None,
    domain: str = "protein-heavy",
    burial_method: Optional[str] = None,
    sasa_backend: str = "auto",
) -> dict[str, Any]:
    tables = load_score_table_set(table_set)
    if tables is None:
        raise ValueError("profile scoring requires a score table set")
    selected_profiles = list(profiles or profile_ids(tables))
    return {
        "format": "pheat.profile-score-results",
        "version": 1,
        "model": model,
        "domain": domain,
        "default_profile": default_profile(tables),
        "profiles": {
            profile: score_structure(
                structure,
                model=model,
                domain=domain,
                table_set=tables,
                profile=profile,
                burial_method=burial_method,
                sasa_backend=sasa_backend,
            ).to_dict()
            for profile in selected_profiles
        },
    }


def _score_generic(structure: HeavyAtomStructure) -> EnergyResult:
    """Element-contact smoke score for broad PDB coverage."""

    steric = 0.0
    contact = 0.0
    warnings = _unsupported_warnings(structure)

    for atom_a, atom_b in _nonlocal_pairs(structure.atoms, cutoff=GENERIC_CONTACT_CUTOFF):
        dist = max(distance(atom_a.coord, atom_b.coord), 1e-6)
        radius_sum = _radius(atom_a) + _radius(atom_b)
        if dist < 0.75 * radius_sum:
            steric += (0.75 * radius_sum - dist) ** 2 * 25.0
        if dist < 6.0:
            contact += _element_weight(atom_a) * _element_weight(atom_b) / dist

    return EnergyResult(
        model="generic",
        total=steric + contact,
        units="arbitrary",
        terms={"steric": steric, "contact": contact},
        warnings=warnings,
        citations=["pheat-generic-element-contact-v1"],
        metadata={"description": "element-based steric/contact score for broad PDB coverage"},
    )


def _score_pheat_dfire(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """PHEAT distance-contact heuristic inspired by DFIRE."""

    pair_term = 0.0
    burial_term = 0.0
    warnings = _unsupported_warnings(structure)

    for atom_a, atom_b in _nonlocal_pairs(structure.atoms, cutoff=PHEAT_DISTANCE_CONTACT_CUTOFF):
        if atom_a.resname not in CANONICAL_RESIDUES or atom_b.resname not in CANONICAL_RESIDUES:
            continue
        dist = distance(atom_a.coord, atom_b.coord)
        if 2.0 <= dist <= 15.0:
            bin_index = min(int(dist), 14)
            residue_bias = RESIDUE_HYDROPHOBICITY.get(atom_a.resname, 0.0) + RESIDUE_HYDROPHOBICITY.get(
                atom_b.resname, 0.0
            )
            atom_bias = _element_weight(atom_a) + _element_weight(atom_b)
            table_weight = _table_pair_bin_weight(table, atom_a.resname, atom_b.resname, bin_index)
            pair_term += table_weight * (residue_bias + atom_bias) / (bin_index * bin_index)
    for residue_atoms in structure.atoms_by_residue().values():
        sidechain_atoms = [
            atom for atom in residue_atoms if atom.name.strip().upper() not in {"N", "CA", "C", "O"}
        ]
        if len(sidechain_atoms) >= 3:
            burial_term -= 0.02 * len(sidechain_atoms)

    warnings.append(
        "pheat-dfire scorer uses a bundled PHEAT distance-contact approximation, not the original DFIRE table"
    )
    return EnergyResult(
        model="pheat-dfire",
        total=pair_term + burial_term,
        units="arbitrary",
        terms={"distance_statistical": pair_term, "sidechain_burial": burial_term},
        warnings=warnings,
        citations=["zhou-zhou-2002-dfire", "pheat-dfire-like-v1"],
        metadata={
            "description": "PHEAT DFIRE-inspired heavy-atom contact approximation",
            "table_set": _table_source(table),
        },
    )


def _score_heavy_mm(structure: HeavyAtomStructure) -> EnergyResult:
    """Heavy-atom-only Lennard-Jones/electrostatic approximation."""

    lj = 0.0
    electrostatic = 0.0
    bonded = _bonded_backbone_penalty(structure)
    warnings = _unsupported_warnings(structure)

    for atom_a, atom_b in _nonlocal_pairs(structure.atoms, cutoff=HEAVY_MM_NONBONDED_CUTOFF):
        dist = max(distance(atom_a.coord, atom_b.coord), 1e-6)
        sigma = 0.5 * (_radius(atom_a) + _radius(atom_b))
        epsilon = 0.05 if "C" in {atom_a.element.upper(), atom_b.element.upper()} else 0.03
        sr6 = (sigma / dist) ** 6
        lj += 4.0 * epsilon * (sr6 * sr6 - sr6)
        electrostatic += 3.0 * _charge(atom_a) * _charge(atom_b) / dist

    warnings.append("heavy-mm is a heavy-atoms-only approximate score, not an all-atom force-field energy")
    warnings.append(f"heavy-mm uses a {HEAVY_MM_NONBONDED_CUTOFF:g} A nonbonded cutoff for corpus-scale scoring")
    return EnergyResult(
        model="heavy-mm",
        total=lj + electrostatic + bonded,
        units="arbitrary",
        terms={"lennard_jones": lj, "electrostatic": electrostatic, "bonded_geometry": bonded},
        warnings=warnings,
        citations=["pheat-heavy-mm-v1"],
        metadata={"description": "approximate heavy-atom MM-style score"},
    )


def _score_pheat_physical_integrity(structure: HeavyAtomStructure) -> EnergyResult:
    """PHEAT-native finite-coordinate, local-geometry, and nonbonded clash guard."""

    geometry = _score_pheat_geometry_integrity(structure)
    steric_clash = 0.0
    short_contact = 0.0
    nonfinite_coordinates = 0.0
    clash_count = 0
    short_contact_count = 0
    nonfinite_atom_count = 0
    min_nonlocal_distance: Optional[float] = None
    max_vdw_overlap = 0.0

    warnings = list(geometry.warnings)
    for atom in structure.atoms:
        if not all(math.isfinite(float(value)) for value in atom.coord):
            nonfinite_atom_count += 1
    if nonfinite_atom_count:
        nonfinite_coordinates = PHYSICAL_INTEGRITY_NONFINITE_PENALTY * nonfinite_atom_count
        warnings.append(f"physical-integrity detected {nonfinite_atom_count} atoms with non-finite coordinates")
    else:
        for atom_a, atom_b in _nonlocal_pairs(structure.atoms, cutoff=PHYSICAL_INTEGRITY_CLASH_CUTOFF):
            dist = distance(atom_a.coord, atom_b.coord)
            if not math.isfinite(dist):
                nonfinite_coordinates += PHYSICAL_INTEGRITY_NONFINITE_PENALTY
                warnings.append("physical-integrity detected a non-finite interatomic distance")
                continue
            min_nonlocal_distance = (
                dist
                if min_nonlocal_distance is None
                else min(min_nonlocal_distance, dist)
            )
            radius_sum = _radius(atom_a) + _radius(atom_b)
            clash_distance = PHYSICAL_INTEGRITY_CLASH_FRACTION * radius_sum
            if dist < clash_distance:
                overlap = clash_distance - dist
                steric_clash += PHYSICAL_INTEGRITY_CLASH_WEIGHT * overlap * overlap
                clash_count += 1
                max_vdw_overlap = max(max_vdw_overlap, overlap)
            if dist < PHYSICAL_INTEGRITY_SHORT_CONTACT_A:
                residual = PHYSICAL_INTEGRITY_SHORT_CONTACT_A - dist
                short_contact += PHYSICAL_INTEGRITY_SHORT_CONTACT_WEIGHT * residual * residual
                short_contact_count += 1

    if clash_count:
        warnings.append(f"physical-integrity detected {clash_count} nonlocal heavy-atom clashes")
    if short_contact_count:
        warnings.append(f"physical-integrity detected {short_contact_count} severe short nonlocal contacts")

    terms = {
        "geometry_integrity": float(geometry.total),
        "steric_clash": float(steric_clash),
        "short_contact": float(short_contact),
        "nonfinite_coordinates": float(nonfinite_coordinates),
    }
    return EnergyResult(
        model="pheat-physical-integrity",
        total=float(sum(terms.values())),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-physical-integrity-v1", *geometry.citations],
        metadata={
            "description": "PHEAT coordinate physical-integrity guard with geometry and nonbonded clash penalties",
            "score_direction": "lower-is-better",
            "geometry_integrity_terms": dict(geometry.terms),
            "checked_counts": {
                "nonlocal_pairs_cutoff_a": PHYSICAL_INTEGRITY_CLASH_CUTOFF,
                "clash_count": clash_count,
                "short_contact_count": short_contact_count,
                "nonfinite_atom_count": nonfinite_atom_count,
            },
            "constants": {
                "clash_cutoff_a": PHYSICAL_INTEGRITY_CLASH_CUTOFF,
                "clash_fraction_of_vdw_sum": PHYSICAL_INTEGRITY_CLASH_FRACTION,
                "short_contact_a": PHYSICAL_INTEGRITY_SHORT_CONTACT_A,
                "clash_weight": PHYSICAL_INTEGRITY_CLASH_WEIGHT,
                "short_contact_weight": PHYSICAL_INTEGRITY_SHORT_CONTACT_WEIGHT,
                "nonfinite_penalty": PHYSICAL_INTEGRITY_NONFINITE_PENALTY,
            },
            "min_nonlocal_distance": min_nonlocal_distance,
            "max_vdw_overlap": max_vdw_overlap,
        },
    )


def _finite_score_value(value: float, *, fallback: float = 0.0) -> tuple[float, bool]:
    resolved = float(value)
    if math.isfinite(resolved):
        return resolved, True
    return float(fallback), False


def _score_pheat_goap_physical(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
    physical_integrity_weight: float = GOAP_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
) -> EnergyResult:
    """GOAP-inspired objective with a physical-integrity guard term."""

    weight = float(physical_integrity_weight)
    if weight < 0.0:
        raise ValueError("physical_integrity_weight must be non-negative")

    goap = _score_pheat_goap(structure, table=table)
    physical = _score_pheat_physical_integrity(structure)
    goap_total, goap_finite = _finite_score_value(goap.total)
    weighted_physical = weight * float(physical.total)
    terms = {
        "goap": goap_total,
        "physical_integrity": float(physical.total),
        "weighted_physical_integrity": weighted_physical,
    }
    for key, value in goap.terms.items():
        terms[f"goap_{key}"] = _finite_score_value(value)[0]
    for key, value in physical.terms.items():
        terms[f"physical_integrity_{key}"] = float(value)

    warnings = list(goap.warnings) + list(physical.warnings)
    if not goap_finite:
        warnings.append("GOAP component was non-finite; composite objective used zero GOAP contribution plus physical penalty")

    physical_counts = (physical.metadata or {}).get("checked_counts") or {}
    return EnergyResult(
        model="pheat-goap-physical",
        total=float(goap_total + weighted_physical),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-goap-physical-v1", *goap.citations, *physical.citations],
        metadata={
            "description": "PHEAT GOAP-inspired score with a finite physical-integrity penalty guard",
            "score_direction": "lower-is-better",
            "physical_integrity_weight": weight,
            "clash_count": physical_counts.get("clash_count"),
            "short_contact_count": physical_counts.get("short_contact_count"),
            "nonfinite_atom_count": physical_counts.get("nonfinite_atom_count"),
            "min_nonlocal_distance": physical.metadata.get("min_nonlocal_distance"),
            "max_vdw_overlap": physical.metadata.get("max_vdw_overlap"),
            "component_models": {
                "goap": goap.to_dict(),
                "physical_integrity": physical.to_dict(),
            },
        },
    )


def _score_pheat_heavy_mm_physical(
    structure: HeavyAtomStructure,
    *,
    physical_integrity_weight: float = PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
) -> EnergyResult:
    """Heavy-MM objective with a finite physical-integrity barrier."""

    weight = float(physical_integrity_weight)
    if weight < 0.0:
        raise ValueError("physical_integrity_weight must be non-negative")

    heavy = _score_heavy_mm(structure)
    physical = _score_pheat_physical_integrity(structure)
    heavy_total, heavy_finite = _finite_score_value(heavy.total)
    terms = {
        "heavy_mm": heavy_total,
        "physical_integrity": float(physical.total),
        "weighted_physical_integrity": weight * float(physical.total),
    }
    for key, value in heavy.terms.items():
        terms[f"heavy_mm_{key}"] = _finite_score_value(value)[0]
    for key, value in physical.terms.items():
        terms[f"physical_integrity_{key}"] = float(value)

    warnings = list(heavy.warnings) + list(physical.warnings)
    if not heavy_finite:
        warnings.append("heavy-mm component was non-finite; composite objective used zero heavy-mm contribution plus physical penalty")

    return EnergyResult(
        model="pheat-heavy-mm-physical",
        total=float(heavy_total + terms["weighted_physical_integrity"]),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-heavy-mm-physical-v1", *heavy.citations, *physical.citations],
        metadata={
            "description": "PHEAT heavy-MM score with a finite physical-integrity penalty barrier",
            "score_direction": "lower-is-better",
            "physical_integrity_weight": weight,
            "component_models": {
                "heavy_mm": heavy.to_dict(),
                "physical_integrity": physical.to_dict(),
            },
        },
    )


def _score_pheat_goap(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """PHEAT orientation heuristic layered on the PHEAT DFIRE-inspired score."""

    base = _score_pheat_dfire(structure, table=table)
    orientation = 0.0
    residue_vectors = _residue_orientation_vectors(structure)
    for key_a, key_b in itertools.combinations(residue_vectors, 2):
        if key_a[0] == key_b[0] and abs(key_a[1] - key_b[1]) <= 1:
            continue
        ca_a, vector_a = residue_vectors[key_a]
        ca_b, vector_b = residue_vectors[key_b]
        dist = distance(ca_a, ca_b)
        if 3.0 <= dist <= 12.0:
            orientation -= 0.05 * abs(dot(vector_a, vector_b)) / dist

    warnings = list(base.warnings)
    warnings.append(
        "pheat-goap scorer uses a bundled PHEAT orientation approximation, not the original GOAP tables"
    )
    return EnergyResult(
        model="pheat-goap",
        total=base.total + orientation,
        units="arbitrary",
        terms={**base.terms, "orientation": orientation},
        warnings=warnings,
        citations=["zhou-skolnick-2011-goap", "pheat-goap-like-v1"],
        metadata={
            "description": "PHEAT GOAP-inspired orientation-aware heavy-atom approximation",
            "table_set": _table_source(table),
        },
    )


def _score_pheat_mj(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Residue-contact score using a PHEAT MJ-style matrix when provided."""

    contact = 0.0
    warnings = _unsupported_warnings(structure)
    points = _residue_representative_points(structure)
    keys = list(points)
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            if key_a[0] == key_b[0] and abs(key_a[1] - key_b[1]) <= 2:
                continue
            dist = distance(points[key_a], points[key_b])
            if dist <= 6.5:
                pair = _residue_pair_key(key_a[3], key_b[3])
                contact += _table_pair_weight(table, pair) * _mj_pair_bias(key_a[3], key_b[3]) / max(dist, 1.0)
    warnings.append(
        "pheat-mj is a PHEAT-generated Miyazawa-Jernigan-style residue contact score, not a redistributed original MJ table"
    )
    return EnergyResult(
        model="pheat-mj",
        total=contact,
        units="arbitrary",
        terms={"residue_contact": contact},
        warnings=warnings,
        citations=["miyazawa-jernigan-contact-potentials", "pheat-mj-like-v1"],
        metadata={
            "description": "PHEAT MJ-style residue contact approximation",
            "table_set": _table_source(table),
        },
    )


def _score_pheat_hydropathy(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
    burial_method: str = "contacts",
    sasa_backend: str = "auto",
) -> EnergyResult:
    """Hydrophobic/hydrophilic burial compatibility score."""

    warnings = _unsupported_warnings(structure)
    burial, burial_metadata = residue_burial(structure, method=burial_method, backend=sasa_backend)
    scale = _table_hydropathy_scale(table)
    compatibility = 0.0
    hydrophobic_contact = 0.0
    hydrophilic_mismatch = 0.0
    for key in structure.residue_keys():
        value = float(scale.get(key[3], 0.0))
        buried = float(burial.get(key, 0.0))
        compatibility -= value * (buried - 0.5)
        if value > 0.0:
            hydrophobic_contact -= value * buried
        elif value < 0.0:
            hydrophilic_mismatch += abs(value) * buried
    return EnergyResult(
        model="pheat-hydropathy",
        total=compatibility + hydrophobic_contact + hydrophilic_mismatch,
        units="arbitrary",
        terms={
            "burial_compatibility": compatibility,
            "hydrophobic_contact": hydrophobic_contact,
            "hydrophilic_mismatch": hydrophilic_mismatch,
        },
        warnings=warnings,
        citations=["kyte-doolittle-hydropathy", "pheat-hydropathy-v1"],
        metadata={
            "description": "hydrophobic/hydrophilic burial compatibility score",
            "hydropathy_scale": "Kyte-Doolittle",
            "table_set": _table_source(table),
            **burial_metadata,
        },
    )


def _score_pheat_backbone(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Backbone geometry plausibility from extracted residue geometry."""

    from pheat.residue_geometry import structure_to_residue_geometry

    geometry = structure_to_residue_geometry(structure, angle_units="degrees", stored_angles="all")
    ramachandran = 0.0
    omega = 0.0
    for residue in geometry.residues:
        if residue.phi is not None and residue.psi is not None:
            ramachandran += _angle_outlier_penalty(residue.phi, residue.psi)
        if residue.omega is not None:
            omega += min(abs(abs(residue.omega) - 180.0), abs(residue.omega)) / 180.0
    return EnergyResult(
        model="pheat-backbone",
        total=ramachandran + omega,
        units="arbitrary",
        terms={"ramachandran": ramachandran, "omega": omega},
        warnings=_unsupported_warnings(structure),
        citations=["pheat-backbone-v1"],
        metadata={
            "description": "PHEAT backbone torsion plausibility score",
            "table_set": _table_source(table),
        },
    )


def _score_pheat_hydropathy_physical(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
    burial_method: str = "contacts",
    sasa_backend: str = "auto",
    physical_integrity_weight: float = HYDROPATHY_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
) -> EnergyResult:
    """Hydropathy objective with a light physical-integrity guard."""

    weight = float(physical_integrity_weight)
    if weight < 0.0:
        raise ValueError("physical_integrity_weight must be non-negative")

    hydropathy = _score_pheat_hydropathy(
        structure,
        table=table,
        burial_method=burial_method,
        sasa_backend=sasa_backend,
    )
    physical = _score_pheat_physical_integrity(structure)
    hydropathy_total, hydropathy_finite = _finite_score_value(hydropathy.total)
    weighted_physical = weight * float(physical.total)
    terms = {
        "hydropathy": hydropathy_total,
        "physical_integrity": float(physical.total),
        "weighted_physical_integrity": weighted_physical,
    }
    for key, value in hydropathy.terms.items():
        terms[f"hydropathy_{key}"] = _finite_score_value(value)[0]
    for key, value in physical.terms.items():
        terms[f"physical_integrity_{key}"] = float(value)

    warnings = list(hydropathy.warnings) + list(physical.warnings)
    if not hydropathy_finite:
        warnings.append("hydropathy component was non-finite; composite objective used zero hydropathy contribution plus physical penalty")

    physical_counts = (physical.metadata or {}).get("checked_counts") or {}
    return EnergyResult(
        model="pheat-hydropathy-physical",
        total=float(hydropathy_total + weighted_physical),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-hydropathy-physical-v1", *hydropathy.citations, *physical.citations],
        metadata={
            "description": "PHEAT hydropathy score with a finite physical-integrity penalty guard",
            "score_direction": "lower-is-better",
            "physical_integrity_weight": weight,
            "clash_count": physical_counts.get("clash_count"),
            "short_contact_count": physical_counts.get("short_contact_count"),
            "nonfinite_atom_count": physical_counts.get("nonfinite_atom_count"),
            "min_nonlocal_distance": physical.metadata.get("min_nonlocal_distance"),
            "max_vdw_overlap": physical.metadata.get("max_vdw_overlap"),
            "component_models": {
                "hydropathy": hydropathy.to_dict(),
                "physical_integrity": physical.to_dict(),
            },
        },
    )


def _score_pheat_backbone_physical(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
    physical_integrity_weight: float = BACKBONE_PHYSICAL_INTEGRITY_WEIGHT_DEFAULT,
) -> EnergyResult:
    """Backbone torsion objective with a physical-integrity guard."""

    weight = float(physical_integrity_weight)
    if weight < 0.0:
        raise ValueError("physical_integrity_weight must be non-negative")

    backbone = _score_pheat_backbone(structure, table=table)
    physical = _score_pheat_physical_integrity(structure)
    backbone_total, backbone_finite = _finite_score_value(backbone.total)
    weighted_physical = weight * float(physical.total)
    terms = {
        "backbone": backbone_total,
        "physical_integrity": float(physical.total),
        "weighted_physical_integrity": weighted_physical,
    }
    for key, value in backbone.terms.items():
        terms[f"backbone_{key}"] = _finite_score_value(value)[0]
    for key, value in physical.terms.items():
        terms[f"physical_integrity_{key}"] = float(value)

    warnings = list(backbone.warnings) + list(physical.warnings)
    if not backbone_finite:
        warnings.append("backbone component was non-finite; composite objective used zero backbone contribution plus physical penalty")

    physical_counts = (physical.metadata or {}).get("checked_counts") or {}
    return EnergyResult(
        model="pheat-backbone-physical",
        total=float(backbone_total + weighted_physical),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-backbone-physical-v1", *backbone.citations, *physical.citations],
        metadata={
            "description": "PHEAT backbone score with a finite physical-integrity penalty guard",
            "score_direction": "lower-is-better",
            "physical_integrity_weight": weight,
            "clash_count": physical_counts.get("clash_count"),
            "short_contact_count": physical_counts.get("short_contact_count"),
            "nonfinite_atom_count": physical_counts.get("nonfinite_atom_count"),
            "min_nonlocal_distance": physical.metadata.get("min_nonlocal_distance"),
            "max_vdw_overlap": physical.metadata.get("max_vdw_overlap"),
            "component_models": {
                "backbone": backbone.to_dict(),
                "physical_integrity": physical.to_dict(),
            },
        },
    )


def _score_pheat_rotamer(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Side-chain chi-angle rotamer plausibility score."""

    from pheat.residue_geometry import structure_to_residue_geometry

    geometry = structure_to_residue_geometry(structure, angle_units="degrees")
    rotamer = 0.0
    chi_count = 0
    for residue in geometry.residues:
        for chi in residue.chi:
            chi_count += 1
            rotamer += min(abs(_angle_delta(chi, target)) for target in (-60.0, 60.0, 180.0)) / 180.0
    return EnergyResult(
        model="pheat-rotamer",
        total=rotamer,
        units="arbitrary",
        terms={"rotamer_outlier": rotamer, "chi_count": float(chi_count)},
        warnings=_unsupported_warnings(structure),
        citations=["dunbrack-rotamers", "pheat-rotamer-v1"],
        metadata={
            "description": "PHEAT side-chain chi rotamer plausibility score",
            "table_set": _table_source(table),
        },
    )


def _score_pheat_hbond(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Heavy-atom donor/acceptor contact geometry score."""

    hbond = 0.0
    buried_polar = 0.0
    polar_atoms = [atom for atom in structure.atoms if atom.element.strip().upper() in {"N", "O", "S", "SE"}]
    contacted = set()
    for index, atom_a in enumerate(polar_atoms):
        for atom_b in polar_atoms[index + 1 :]:
            if atom_a.chain_id == atom_b.chain_id and abs(atom_a.resseq - atom_b.resseq) <= 1:
                continue
            dist = distance(atom_a.coord, atom_b.coord)
            if 2.4 <= dist <= 3.6:
                hbond -= 1.0 - abs(dist - 2.9) / 1.2
                contacted.add(id(atom_a))
                contacted.add(id(atom_b))
    burial, burial_metadata = residue_burial(structure, method="contacts")
    for atom in polar_atoms:
        if id(atom) not in contacted:
            buried_polar += 0.1 * float(burial.get(atom.residue_key, 0.0))
    return EnergyResult(
        model="pheat-hbond",
        total=hbond + buried_polar,
        units="arbitrary",
        terms={"heavy_atom_hbond": hbond, "buried_polar": buried_polar},
        warnings=_unsupported_warnings(structure)
        + ["pheat-hbond infers donor/acceptor behavior from heavy atoms; protonation is ambiguous"],
        citations=["pheat-hbond-v1"],
        metadata={
            "description": "PHEAT heavy-atom hydrogen-bond/contact score",
            "table_set": _table_source(table),
            **burial_metadata,
        },
    )


def _score_pheat_rg(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Radius-of-gyration compactness penalty with configurable coefficients."""

    config = _pheat_rg_config(table)
    atom_set = str(config["atom_set"])
    mode = str(config["mode"])
    rg_payload = structure_radius_of_gyration(structure, mode=mode, atom_set=atom_set)
    value_key = "mass_weighted" if mode == "mass-weighted" else mode
    observed_rg = float(rg_payload["values"][value_key])
    residue_count = len(structure.residue_keys())
    if residue_count <= 0:
        raise ValueError("pheat-rg requires at least one residue.")

    a = float(config["a"])
    b = float(config["b"])
    sigma_fraction = float(config["sigma_fraction"])
    expected_rg = a * (float(residue_count) ** b)
    sigma = max(sigma_fraction * expected_rg, 1e-12)
    rg_z = (observed_rg - expected_rg) / sigma
    penalty = rg_z * rg_z

    warnings = _unsupported_warnings(structure)
    if atom_set == "ca" and int(rg_payload["atom_count"]) != residue_count:
        warnings.append(
            "pheat-rg CA compactness used fewer CA atoms than residues; missing backbone atoms affect the score"
        )
    if not bool(config["fitted"]):
        warnings.append(
            "pheat-rg uses placeholder expected-Rg coefficients; fit coefficients from an in-domain corpus "
            "before interpreting the score as a calibrated potential"
        )

    return EnergyResult(
        model="pheat-rg",
        total=penalty,
        units="arbitrary",
        terms={
            "rg_compactness_penalty": penalty,
            "observed_rg": observed_rg,
            "expected_rg": expected_rg,
            "rg_z": rg_z,
            "sigma": sigma,
            "residue_count": float(residue_count),
            "rg_atom_count": float(rg_payload["atom_count"]),
        },
        warnings=warnings,
        citations=["pheat-rg-placeholder-v1"],
        metadata={
            "description": "PHEAT expected-radius-of-gyration compactness penalty",
            "table_set": _table_source(table),
            "atom_set": atom_set,
            "mode": mode,
            "fitted": bool(config["fitted"]),
            "coefficients": {
                "a": a,
                "b": b,
                "sigma_fraction": sigma_fraction,
            },
            "formula": "score=((observed_rg - a * residue_count ** b) / (sigma_fraction * expected_rg)) ** 2",
            "note": "This is a shape/compactness penalty, not thermodynamic free energy.",
        },
    )


def _score_pheat_ml_linear(
    structure: HeavyAtomStructure,
    *,
    table: Optional[Mapping[str, Any]] = None,
) -> EnergyResult:
    """Lightweight linear feature-combination score."""

    if table is None:
        warnings = ["pheat-ml-linear has no trained table set; using deterministic baseline weights"]
        coefficients = {"generic_total": 0.25, "pheat-dfire_total": 0.5, "heavy-mm_total": 0.25}
        intercept = 0.0
    else:
        warnings = []
        coefficients = dict(table.get("coefficients") or {})
        intercept = float(table.get("intercept") or 0.0)
    feature_results = {
        "generic_total": _score_generic(structure).total,
        "pheat-dfire_total": _score_pheat_dfire(structure).total,
        "heavy-mm_total": _score_heavy_mm(structure).total,
    }
    linear = intercept
    for name, weight in coefficients.items():
        linear += float(weight) * float(feature_results.get(name, 0.0))
    return EnergyResult(
        model="pheat-ml-linear",
        total=linear,
        units="arbitrary",
        terms={"linear": linear},
        warnings=warnings + _unsupported_warnings(structure),
        citations=["pheat-ml-linear-v1"],
        metadata={
            "description": "lightweight linear PHEAT feature combination",
            "table_set": _table_source(table),
            "features": feature_results,
        },
    )


def _score_pheat_physics(
    structure: HeavyAtomStructure,
    *,
    charge_profile: str = "protein-coarse-charge-v1",
    hydrophobic_gamma: float = 15.0,
    hydrophobic_burial_denominator: float = 15.0,
    hydrophobic_burial_scale: float = 1.0,
    end_to_end_weight: float = 50.0,
    end_to_end_target: Optional[float] = None,
    end_to_end_slack: Optional[float] = None,
    hbond_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    disulfide_weight: float = 1.0,
    vdw_weight: float = 1.0,
    rotamer_weight: float = 1.0,
    backbone_weight: float = 1.0,
    pi_stacking_weight: float = 1.0,
    geometry_integrity_weight: float = 1.0,
    physical_integrity_weight: float = 0.0,
    adjacent_heavy_steric_weight: float = 1.0,
) -> EnergyResult:
    """General PHEAT physical objective for coarse-to-fine structure scoring."""

    residues = structure.residue_keys()
    charge_profile = _normalize_pheat_physics_charge_profile(charge_profile)
    terminal_residue_keys = _terminal_residue_keys_by_chain(residues)
    atoms_by_residue = structure.atoms_by_residue()
    ca_atoms = _ca_atoms_for_residues(atoms_by_residue, residues)

    hydrophobic_burial = 0.0
    if structure.atoms:
        for atom in structure.atoms:
            if atom.element.strip().upper() == "H":
                continue
            scale = KYTE_DOOLITTLE_HYDROPATHY.get(atom.resname.strip().upper(), 0.0)
            if scale <= 0.0:
                continue
            neighbors = 0.0
            for other in structure.atoms:
                if other is atom or other.element.strip().upper() == "H":
                    continue
                d = distance(atom.coord, other.coord)
                neighbors += 1.0 / (1.0 + math.exp(d - 6.0))
            burial = min(max(neighbors / 15.0, 0.0), 1.0)
            hydrophobic_burial += float(hydrophobic_gamma) * scale * (1.0 - burial)

    end_to_end = 0.0
    target = float(end_to_end_target) if end_to_end_target is not None else 4.5 + 0.40 * max(0, len(residues) - 5)
    slack = float(end_to_end_slack) if end_to_end_slack is not None else 1.5 + 0.05 * len(residues)
    if len(ca_atoms) >= 2 and float(end_to_end_weight) != 0.0:
        observed = distance(ca_atoms[0].coord, ca_atoms[-1].coord)
        deviation = max(0.0, abs(observed - target) - slack)
        end_to_end = float(end_to_end_weight) * deviation * deviation

    hbond = _score_pheat_hbond(structure).total
    rotamer = _score_pheat_rotamer(structure).total
    backbone = _score_pheat_backbone(structure).total
    geometry = _score_pheat_geometry_integrity(structure)
    physical = _score_pheat_physical_integrity(structure)
    pi_stacking = _pheat_physics_pi_stacking(structure)
    electrostatic = 0.0
    disulfide = 0.0
    vdw = 0.0
    for atom_a, atom_b in _nonlocal_pairs(structure.atoms, cutoff=HEAVY_MM_NONBONDED_CUTOFF):
        dist = max(distance(atom_a.coord, atom_b.coord), 1e-6)
        electrostatic += (
            _pheat_physics_charge(atom_a, charge_profile=charge_profile, terminal_residue_keys=terminal_residue_keys)
            * _pheat_physics_charge(atom_b, charge_profile=charge_profile, terminal_residue_keys=terminal_residue_keys)
            / dist
        )
        radius_sum = _radius(atom_a) + _radius(atom_b)
        if dist < radius_sum:
            term = (radius_sum / (dist + 0.1)) ** 12
            vdw += 0.1 * (50.0 + math.log(term - 49.0) if term > 50.0 else term)
        if atom_a.resname.strip().upper() == "CYS" and atom_b.resname.strip().upper() == "CYS":
            if atom_a.name.strip().upper() == "SG" and atom_b.name.strip().upper() == "SG":
                disulfide -= 25.0 * math.exp(-((dist - 2.05) ** 2) / 0.5) if dist < 3.0 else 0.0

    adjacent_heavy_sterics = _pheat_physics_adjacent_heavy_sterics(structure)
    raw_terms = {
        "hydrophobic_burial": float(hydrophobic_burial),
        "end_to_end": float(end_to_end),
        "hbond": float(hbond),
        "electrostatic": float(electrostatic),
        "disulfide": float(disulfide),
        "vdw_repulsion": float(vdw),
        "rotamer": float(rotamer),
        "backbone": float(backbone),
        "pi_stacking": float(pi_stacking),
        "geometry_integrity": float(geometry.total),
        "physical_integrity": float(physical.total),
        "adjacent_heavy_sterics": float(adjacent_heavy_sterics),
    }
    weights = {
        "hydrophobic_burial": 1.0,
        "end_to_end": 1.0,
        "hbond": float(hbond_weight),
        "electrostatic": float(electrostatic_weight),
        "disulfide": float(disulfide_weight),
        "vdw_repulsion": float(vdw_weight),
        "rotamer": float(rotamer_weight),
        "backbone": float(backbone_weight),
        "pi_stacking": float(pi_stacking_weight),
        "geometry_integrity": float(geometry_integrity_weight),
        "physical_integrity": float(physical_integrity_weight),
        "adjacent_heavy_sterics": float(adjacent_heavy_steric_weight),
    }
    weighted_terms = {f"weighted_{key}": raw_terms[key] * weights[key] for key in raw_terms}
    total = float(sum(weighted_terms.values()))
    return EnergyResult(
        model="pheat-physics",
        total=total,
        units="arbitrary",
        terms={**raw_terms, **weighted_terms},
        warnings=_unsupported_warnings(structure) + list(geometry.warnings) + list(physical.warnings),
        citations=["pheat-physics-v1", *geometry.citations, *physical.citations],
        metadata={
            "description": "General PHEAT physical structure objective with configurable term weights",
            "score_direction": "lower-is-better",
            "charge_profile": charge_profile,
            "end_to_end_target": target,
            "end_to_end_slack": slack,
            "weights": weights,
            "component_models": {
                "geometry_integrity": geometry.to_dict(),
                "physical_integrity": physical.to_dict(),
            },
        },
    )


def _terminal_residue_keys_by_chain(residues: Sequence[ResidueKey]) -> set[ResidueKey]:
    """Return the first and last residue key in each chain."""

    termini: set[ResidueKey] = set()
    by_chain: dict[str, list[ResidueKey]] = {}
    for key in residues:
        by_chain.setdefault(key[0], []).append(key)
    for chain_residues in by_chain.values():
        if chain_residues:
            termini.add(chain_residues[0])
            termini.add(chain_residues[-1])
    return termini


def _ca_atoms_for_residues(
    atoms_by_residue: Mapping[ResidueKey, Sequence[Atom]],
    residues: Sequence[ResidueKey],
) -> list[Atom]:
    ca_atoms: list[Atom] = []
    for key in residues:
        ca_atom = next((atom for atom in atoms_by_residue[key] if atom.name.strip().upper() == "CA"), None)
        if ca_atom is not None:
            ca_atoms.append(ca_atom)
    return ca_atoms


def _score_pheat_coarse_protein_folding_v1(
    structure: HeavyAtomStructure,
    *,
    charge_profile: str = "protein-coarse-charge-v1",
    hydrophobic_gamma: float = 15.0,
    hydrophobic_burial_denominator: float = 35.0,
    hydrophobic_burial_scale: float = 0.7,
    end_to_end_weight: float = 8.0,
    end_to_end_target: Optional[float] = None,
    end_to_end_slack: Optional[float] = None,
    hbond_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    disulfide_weight: float = 1.0,
    vdw_weight: float = 1.0,
    rotamer_weight: float = 1.0,
    backbone_weight: float = 1.0,
    pi_stacking_weight: float = 1.0,
    geometry_integrity_weight: float = 1.0,
    adjacent_heavy_steric_weight: float = 1.0,
    omega_weight: float = COARSE_OMEGA_SCALE_DEFAULT,
    omega_window_scale: float = COARSE_OMEGA_WINDOW_SCALE_DEFAULT,
    hard_clash_min_A: float = COARSE_HARD_CLASH_MIN_A_DEFAULT,
    hard_clash_scale: float = COARSE_HARD_CLASH_SCALE_DEFAULT,
    hard_clash_power: float = COARSE_HARD_CLASH_POWER_DEFAULT,
    hard_clash_14_scale: float = COARSE_HARD_CLASH_14_SCALE_DEFAULT,
    decoded_torsions: Optional[Mapping[str, Any]] = None,
    use_end_to_end_constraint: bool = True,
    end_to_end_scale: float = 1.0,
) -> EnergyResult:
    """Coarse protein folding objective matching the legacy staged folding terms."""

    nonfinite = _coarse_nonfinite_coordinate_count(structure)
    if nonfinite:
        penalty = float(1e6 + 1e3 * nonfinite)
        return EnergyResult(
            model="pheat-custom-energy-v1",
            total=penalty,
            units="arbitrary",
            terms={"non_finite_penalty": penalty, "total": penalty},
            warnings=[f"non-finite atom coordinates detected: {nonfinite}"],
            citations=["pheat-custom-energy-v1"],
            metadata={"description": "coarse protein folding objective", "score_direction": "lower-is-better"},
        )

    charge_profile = _normalize_pheat_physics_charge_profile(charge_profile)
    residues = structure.residue_keys()
    residue_index = {key: index for index, key in enumerate(residues)}
    atoms_by_residue = structure.atoms_by_residue()
    atom_to_res = [residue_index.get(atom.residue_key, 0) for atom in structure.atoms]
    terminal_residue_keys = _terminal_residue_keys_by_chain(residues)
    target = float(end_to_end_target) if end_to_end_target is not None else 4.5 + 0.40 * max(0, len(residues) - 5)
    slack = float(end_to_end_slack) if end_to_end_slack is not None else 1.5 + 0.05 * len(residues)
    torsions, ignored_torsions = _normalize_decoded_torsions(decoded_torsions, residue_keys=residues)
    warnings = _unsupported_warnings(structure)
    if ignored_torsions:
        warnings.append(f"ignored non-numeric or non-finite decoded torsion values: {ignored_torsions}")

    terms: dict[str, float] = {
        "end_to_end": 0.0,
        "hydrophobic_burial": 0.0,
        "hbond": 0.0,
        "electrostatic": 0.0,
        "disulfide": 0.0,
        "steric": 0.0,
        "rotamer": 0.0,
        "aromatic": 0.0,
        "ramachandran": 0.0,
        "omega": 0.0,
        "omega_window_penalty": 0.0,
        "hard_clash": 0.0,
        "geometry_integrity": 0.0,
        "adjacent_heavy_sterics": 0.0,
    }

    ca_atoms = _ca_atoms_for_residues(atoms_by_residue, residues)
    if len(ca_atoms) >= 2 and use_end_to_end_constraint and float(end_to_end_weight) != 0.0:
        dist_ends = distance(ca_atoms[0].coord, ca_atoms[-1].coord)
        deviation = max(0.0, abs(dist_ends - target) - slack)
        terms["end_to_end"] = float(end_to_end_scale) * float(end_to_end_weight) * deviation * deviation

    terms["hydrophobic_burial"] = _coarse_hydrophobic_burial_term(
        structure,
        hydrophobic_gamma=float(hydrophobic_gamma),
        burial_denominator=float(hydrophobic_burial_denominator),
        burial_scale=float(hydrophobic_burial_scale),
    )
    terms["hbond"] = float(hbond_weight) * _coarse_hbond_term(structure, residue_index=residue_index)
    terms["electrostatic"] = float(electrostatic_weight) * _coarse_electrostatic_term(
        structure,
        atom_to_res=atom_to_res,
        charge_profile=charge_profile,
        terminal_residue_keys=terminal_residue_keys,
    )
    terms["disulfide"] = float(disulfide_weight) * _coarse_disulfide_term(structure)
    terms["steric"] = float(vdw_weight) * _coarse_vdw_term(structure)
    terms["rotamer"] = float(rotamer_weight) * _coarse_rotamer_term(residues, torsions)
    terms["aromatic"] = float(pi_stacking_weight) * _pheat_physics_pi_stacking(structure)
    terms["ramachandran"] = float(backbone_weight) * _coarse_ramachandran_term(residues, torsions)
    terms["omega"] = float(omega_weight) * _coarse_omega_term(residues, torsions)
    terms["omega_window_penalty"] = float(omega_window_scale) * _coarse_omega_window_penalty(residues, torsions)
    hard_clash_value, hard_clash_diagnostics = _coarse_hard_clash_term(
        structure,
        hard_clash_min_A=float(hard_clash_min_A),
        hard_clash_scale=float(hard_clash_scale),
        hard_clash_power=float(hard_clash_power),
        hard_clash_14_scale=float(hard_clash_14_scale),
    )
    terms["hard_clash"] = hard_clash_value
    geometry_total, geometry_terms = _coarse_geometry_integrity_term(structure)
    terms["geometry_integrity"] = float(geometry_integrity_weight) * geometry_total
    adjacent = _pheat_physics_adjacent_heavy_sterics(structure)
    terms["adjacent_heavy_sterics"] = float(adjacent_heavy_steric_weight) * float(adjacent)

    total = float(sum(terms.values()))
    terms["total"] = total
    weights = {
        "end_to_end": float(end_to_end_weight) * float(end_to_end_scale),
        "hydrophobic_burial": float(hydrophobic_gamma) * float(hydrophobic_burial_scale),
        "hbond": float(hbond_weight),
        "electrostatic": float(electrostatic_weight),
        "disulfide": float(disulfide_weight),
        "steric": float(vdw_weight),
        "rotamer": float(rotamer_weight),
        "ramachandran": float(backbone_weight),
        "aromatic": float(pi_stacking_weight),
        "geometry_integrity": float(geometry_integrity_weight),
        "adjacent_heavy_sterics": float(adjacent_heavy_steric_weight),
        "omega": float(omega_weight),
        "omega_window_penalty": float(omega_window_scale),
        "hard_clash_min_A": float(hard_clash_min_A),
        "hard_clash_scale": float(hard_clash_scale),
        "hard_clash_power": float(hard_clash_power),
        "hard_clash_14_scale": float(hard_clash_14_scale),
    }
    return EnergyResult(
        model="pheat-custom-energy-v1",
        total=total,
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-custom-energy-v1"],
        metadata={
            "description": "Coarse staged protein folding objective with compactness, burial, contact, torsion, and geometry terms",
            "score_direction": "lower-is-better",
            "charge_profile": charge_profile,
            "end_to_end_target": target,
            "end_to_end_slack": slack,
            "use_end_to_end_constraint": bool(use_end_to_end_constraint),
            "end_to_end_scale": float(end_to_end_scale),
            "hydrophobic_gamma": float(hydrophobic_gamma),
            "hydrophobic_burial_denominator": float(hydrophobic_burial_denominator),
            "hydrophobic_burial_scale": float(hydrophobic_burial_scale),
            "end_to_end_weight": float(end_to_end_weight),
            **hard_clash_diagnostics,
            "weights": weights,
            "decoded_torsion_count": len(torsions),
            "ignored_decoded_torsion_count": ignored_torsions,
            "geometry_terms": geometry_terms,
        },
    )


def _coarse_geometry_integrity_term(structure: HeavyAtomStructure) -> tuple[float, dict[str, float]]:
    """Match QTF main's chirality and peptide-planarity guardrails."""

    atoms_by_residue = structure.atoms_by_residue()
    residues = structure.residue_keys()
    terms = {"pro_ring": 0.0, "chirality": 0.0, "planarity": 0.0}
    for index, residue_key in enumerate(residues):
        atoms = {atom.name.strip().upper(): atom for atom in atoms_by_residue[residue_key]}
        if all(name in atoms for name in ("CA", "N", "C", "CB")):
            ca = atoms["CA"].coord
            volume = dot(
                cross(sub(atoms["N"].coord, ca), sub(atoms["C"].coord, ca)),
                sub(atoms["CB"].coord, ca),
            )
            if volume < 1.0:
                terms["chirality"] += 50.0 * (1.0 - volume) ** 2
        if index >= len(residues) - 1 or residues[index + 1][0] != residue_key[0]:
            continue
        next_atoms = {
            atom.name.strip().upper(): atom
            for atom in atoms_by_residue[residues[index + 1]]
        }
        if not all(name in atoms for name in ("CA", "C")) or not all(
            name in next_atoms for name in ("N", "CA")
        ):
            continue
        b1 = sub(atoms["C"].coord, atoms["CA"].coord)
        b2 = sub(next_atoms["N"].coord, atoms["C"].coord)
        b3 = sub(next_atoms["CA"].coord, next_atoms["N"].coord)
        n1 = normalize(cross(b1, b2), fallback=(0.0, 0.0, 0.0))
        n2 = normalize(cross(b2, b3), fallback=(0.0, 0.0, 0.0))
        if n1 == (0.0, 0.0, 0.0) or n2 == (0.0, 0.0, 0.0):
            continue
        twist_penalty = 1.0 - abs(dot(n1, n2))
        if twist_penalty > 0.05:
            terms["planarity"] += 20.0 * twist_penalty
    return float(sum(terms.values())), terms


def _coarse_nonfinite_coordinate_count(structure: HeavyAtomStructure) -> int:
    count = 0
    for atom in structure.atoms:
        for value in atom.coord:
            if not math.isfinite(float(value)):
                count += 1
    return count


def _normalize_decoded_torsions(
    decoded_torsions: Optional[Mapping[str, Any]],
    *,
    residue_keys: Optional[Sequence[tuple[str, int, str, str, str]]] = None,
) -> tuple[dict[str, float], int]:
    if not decoded_torsions:
        return {}, 0
    torsions: dict[str, float] = {}
    allowed_chi_by_residue = _allowed_decoded_chi_by_residue(residue_keys or [])
    ignored = 0
    for key, value in decoded_torsions.items():
        normalized_key = str(key).strip().lower()
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            ignored += 1
            continue
        if not math.isfinite(normalized):
            ignored += 1
            continue
        if not _is_supported_decoded_torsion_key(normalized_key, allowed_chi_by_residue=allowed_chi_by_residue):
            ignored += 1
            continue
        torsions[normalized_key] = normalized
    return torsions, ignored


def _decoded_torsion(torsions: Mapping[str, float], residue_index: int, name: str) -> Optional[float]:
    key = f"{int(residue_index)}_{name.strip().lower()}"
    value = torsions.get(key)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _is_supported_decoded_torsion_key(
    key: str,
    *,
    allowed_chi_by_residue: Optional[Mapping[int, set[str]]] = None,
) -> bool:
    if "_" not in key:
        return False
    residue_index, angle_name = key.split("_", 1)
    if not residue_index.isdigit():
        return False
    res_idx = int(residue_index)
    angle_name = angle_name.strip().lower()
    if angle_name in {"phi", "psi", "omega", "tau", "theta"}:
        return True
    if angle_name.startswith("chi") and angle_name[3:].isdigit():
        chi_index = int(angle_name[3:])
        if not 1 <= chi_index <= 4:
            return False
        if allowed_chi_by_residue is None:
            return True
        return angle_name in allowed_chi_by_residue.get(res_idx, set())
    return False


def _allowed_decoded_chi_by_residue(
    residue_keys: Sequence[tuple[str, int, str, str, str]],
) -> dict[int, set[str]]:
    allowed: dict[int, set[str]] = {}
    for index, residue_key in enumerate(residue_keys):
        resname = residue_key[3].strip().upper()
        chi_names: set[str] = set()
        for step in SIDECHAIN_STEPS.get(resname, []):
            dihedral = step.dihedral
            if isinstance(dihedral, str) and dihedral.startswith("chi"):
                chi_names.add(dihedral.lower())
            elif isinstance(dihedral, tuple) and dihedral and isinstance(dihedral[0], str) and dihedral[0].startswith("chi"):
                chi_names.add(dihedral[0].lower())
        cap = COARSE_CHI_CAP_BY_RESIDUE.get(resname)
        if cap is not None:
            chi_names = {
                name
                for name in chi_names
                if name.startswith("chi") and name[3:].isdigit() and int(name[3:]) <= cap
            }
        allowed[index] = chi_names
    return allowed


def _coarse_residue_one_letter(residue_key: tuple[str, int, str, str, str]) -> str:
    resname = residue_key[3].strip().upper()
    return THREE_TO_ONE.get(resname, resname[:1])


def _sigmoid_contact(distance_value: float, midpoint: float = 6.0) -> float:
    x = max(-60.0, min(60.0, float(distance_value) - midpoint))
    return 1.0 / (1.0 + math.exp(x))


def _coarse_hydrophobic_burial_term(
    structure: HeavyAtomStructure,
    *,
    hydrophobic_gamma: float,
    burial_denominator: float = 15.0,
    burial_scale: float = 1.0,
) -> float:
    hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TYR", "TRP", "PRO", "CYS"}
    hydrophobic_atoms = []
    for atom in structure.atoms:
        element = atom.element.strip().upper()
        if element in {"H", "D", "T"}:
            continue
        name = atom.name.strip().upper()
        resname = atom.resname.strip().upper()
        if resname not in hydrophobic_residues:
            continue
        if (name.startswith("C") and name not in {"C", "CA"}) or element == "S":
            hydrophobic_atoms.append(atom)
    if not hydrophobic_atoms:
        return 0.0
    energy = 0.0
    heavy_atoms = [atom for atom in structure.atoms if atom.element.strip().upper() not in {"H", "D", "T"}]
    for atom in hydrophobic_atoms:
        neighbor_count = 0.0
        for other in heavy_atoms:
            if other is atom:
                continue
            neighbor_count += _sigmoid_contact(distance(atom.coord, other.coord), midpoint=6.0)
        denominator = max(float(burial_denominator), 1e-9)
        burial_fraction = min(max(neighbor_count / denominator, 0.0), 1.0)
        energy += float(burial_scale) * float(hydrophobic_gamma) * 30.0 * (1.0 - burial_fraction)
    return float(energy)


def _coarse_hbond_term(
    structure: HeavyAtomStructure,
    *,
    residue_index: Mapping[tuple[str, int, str, str, str], int],
) -> float:
    lookup = structure.atom_lookup()
    n_atoms = [atom for atom in structure.atoms if atom.name.strip().upper() == "N"]
    o_atoms = [atom for atom in structure.atoms if atom.name.strip().upper() == "O"]
    energy = 0.0
    residue_keys = structure.residue_keys()
    for n_atom in n_atoms:
        key = n_atom.residue_key
        res_d = residue_index.get(key)
        if res_d is None:
            continue
        ca_atom = lookup.get((key, "CA"))
        prev_c = None
        if res_d > 0:
            prev_key = residue_keys[res_d - 1]
            if prev_key[0] == key[0]:
                prev_c = lookup.get((prev_key, "C"))
        if prev_c is None or ca_atom is None:
            pos_h = (n_atom.x, n_atom.y, n_atom.z + 1.0)
            pos_n = n_atom.coord
        else:
            v_nc = normalize(sub(prev_c.coord, n_atom.coord))
            v_nca = normalize(sub(ca_atom.coord, n_atom.coord))
            summed = (v_nc[0] + v_nca[0], v_nc[1] + v_nca[1], v_nc[2] + v_nca[2])
            v_h = normalize((-summed[0], -summed[1], -summed[2]), fallback=(0.0, 0.0, 1.0))
            pos_h = (n_atom.x + v_h[0] * 1.01, n_atom.y + v_h[1] * 1.01, n_atom.z + v_h[2] * 1.01)
            pos_n = n_atom.coord
        v_hn = normalize(sub(pos_n, pos_h))
        for o_atom in o_atoms:
            o_res = residue_index.get(o_atom.residue_key)
            if o_res is None or abs(o_res - res_d) < 2:
                continue
            d_ho = distance(o_atom.coord, pos_h)
            if d_ho >= 3.5:
                continue
            v_ho = normalize(sub(o_atom.coord, pos_h))
            angle_cos = dot(v_ho, v_hn)
            if angle_cos >= -0.4:
                continue
            radial_term = math.exp(-((d_ho - 2.0) ** 2) / 0.5)
            angular_term = (abs(angle_cos) - 0.4) * 2.0
            energy += -37.5 * radial_term * angular_term
    return float(energy)


def _coarse_electrostatic_term(
    structure: HeavyAtomStructure,
    *,
    atom_to_res: Sequence[int],
    charge_profile: str,
    terminal_residue_keys: set[tuple[str, int, str, str, str]],
) -> float:
    energy = 0.0
    charges = [
        _pheat_physics_charge(atom, charge_profile=charge_profile, terminal_residue_keys=terminal_residue_keys)
        for atom in structure.atoms
    ]
    for i, atom_a in enumerate(structure.atoms[:-1]):
        q_a = charges[i]
        if abs(q_a) <= 0.0001:
            continue
        for j in range(i + 1, len(structure.atoms)):
            if abs(atom_to_res[i] - atom_to_res[j]) < 2:
                continue
            q_b = charges[j]
            product = q_a * q_b
            if abs(product) <= 0.0001:
                continue
            r = max(distance(atom_a.coord, structure.atoms[j].coord), 1.0)
            energy += 332.0637 * product / (4.0 * r)
    return float(energy)


def _coarse_omega_term(
    residue_keys: Sequence[tuple[str, int, str, str, str]],
    torsions: Mapping[str, float],
) -> float:
    omega_min = math.radians(COARSE_OMEGA_WINDOW_MIN_DEG)
    omega_max = math.radians(COARSE_OMEGA_WINDOW_MAX_DEG)
    omega_center = math.pi
    omega_half_width = 0.5 * (omega_max - omega_min)
    energy = 0.0
    for index, _ in enumerate(residue_keys[:-1]):
        omega = _decoded_torsion(torsions, index, "omega")
        if omega is None:
            continue
        val = float(omega)
        if not math.isfinite(val):
            continue
        if -omega_max <= val <= -omega_min:
            val = (2.0 * math.pi) + val
        val = min(max(val, omega_min), omega_max)
        delta = val - omega_center
        energy += (delta / max(omega_half_width, 1e-9)) ** 2
    return float(energy)


def _coarse_omega_window_penalty(
    residue_keys: Sequence[tuple[str, int, str, str, str]],
    torsions: Mapping[str, float],
) -> float:
    omega_min = math.radians(COARSE_OMEGA_WINDOW_MIN_DEG)
    omega_max = math.radians(COARSE_OMEGA_WINDOW_MAX_DEG)
    omega_half_width = 0.5 * (omega_max - omega_min)
    energy = 0.0
    for index, _ in enumerate(residue_keys[:-1]):
        omega = _decoded_torsion(torsions, index, "omega")
        if omega is None:
            continue
        val = float(omega)
        if not math.isfinite(val):
            continue
        if omega_min <= val <= omega_max or -math.pi <= val <= -omega_min:
            continue
        if 0.0 <= val < omega_min:
            deviation = omega_min - val
        elif val > math.pi:
            deviation = min(abs(val - math.pi), abs(val - omega_max))
        elif -omega_min < val < 0.0:
            deviation = omega_min + val
        else:
            deviation = abs(abs(val) - math.pi)
        energy += float((deviation / max(omega_half_width, 1e-9)) ** 2)
    return float(energy)


def _coarse_hard_clash_term(
    structure: HeavyAtomStructure,
    *,
    hard_clash_min_A: float,
    hard_clash_scale: float,
    hard_clash_power: float,
    hard_clash_14_scale: float,
) -> tuple[float, dict[str, float]]:
    graph_dist = _coarse_graph_distances_up_to_three(structure)
    atoms = structure.atoms
    energy = 0.0
    min_dist = math.inf
    count = 0
    for i, atom_a in enumerate(atoms[:-1]):
        if atom_a.element.strip().upper() in {"H", "D", "T"}:
            continue
        for j in range(i + 1, len(atoms)):
            atom_b = atoms[j]
            if atom_b.element.strip().upper() in {"H", "D", "T"}:
                continue
            path = graph_dist.get((i, j), 99)
            if path <= 2:
                continue
            pair_scale = float(hard_clash_14_scale) if path == 3 else 1.0
            dist = distance(atom_a.coord, atom_b.coord)
            shortfall = max(0.0, float(hard_clash_min_A) - dist)
            if shortfall <= 0.0:
                continue
            min_dist = min(min_dist, float(dist))
            count += 1
            denom = max(float(hard_clash_min_A), 1e-6)
            energy += float(hard_clash_scale) * pair_scale * (shortfall / denom) ** float(hard_clash_power)
    diagnostics = {
        "hard_clash_min_dist": float(min_dist) if math.isfinite(min_dist) else 0.0,
        "hard_clash_count": float(count),
    }
    return float(energy), diagnostics


def _coarse_disulfide_term(structure: HeavyAtomStructure) -> float:
    sg_atoms = [
        atom
        for atom in structure.atoms
        if atom.name.strip().upper() == "SG" and atom.resname.strip().upper() in {"CYS", "CYX"}
    ]
    if len(sg_atoms) <= 1:
        return 0.0
    strengths = [[0.0 for _ in sg_atoms] for _ in sg_atoms]
    energy = 0.0
    for i, atom_a in enumerate(sg_atoms[:-1]):
        for j in range(i + 1, len(sg_atoms)):
            dist = distance(atom_a.coord, sg_atoms[j].coord)
            strength = math.exp(-((dist - 2.05) ** 2) / 0.5) if dist < 3.0 else 0.0
            strengths[i][j] = strength
            strengths[j][i] = strength
            energy -= 25.0 * strength
    for row in strengths:
        overload = sum(row) - 1.0
        if overload > 0.1:
            energy += 40.0 * overload * overload
    return float(energy)


def _coarse_vdw_term(structure: HeavyAtomStructure) -> float:
    graph_dist = _coarse_graph_distances_up_to_three(structure)
    energy = 0.0
    atoms = structure.atoms
    for i, atom_a in enumerate(atoms[:-1]):
        if atom_a.element.strip().upper() in {"H", "D", "T"}:
            continue
        for j in range(i + 1, len(atoms)):
            atom_b = atoms[j]
            if atom_b.element.strip().upper() in {"H", "D", "T"}:
                continue
            path = graph_dist.get((i, j), 99)
            if path <= 2:
                continue
            scale_factor = 0.35 if path == 3 else 1.0
            dist = distance(atom_a.coord, atom_b.coord)
            radius_a, epsilon_a = COARSE_LJ_TYPE_PARAMS[_coarse_lj_type(atom_a)]
            radius_b, epsilon_b = COARSE_LJ_TYPE_PARAMS[_coarse_lj_type(atom_b)]
            contact_distance = 0.95 * (radius_a + radius_b)
            sigma = contact_distance / (2.0 ** (1.0 / 6.0))
            r = max(dist, 1.2)
            sr6 = (sigma / r) ** 6
            lj = 4.0 * math.sqrt(epsilon_a * epsilon_b) * (sr6 * sr6 - sr6)
            repulsive = max(lj, 0.0)
            if repulsive > 25.0:
                repulsive = 25.0 + math.log1p(repulsive - 25.0)
            attractive = min(max(lj, -2.5), 0.0)
            energy += scale_factor * (0.01 * repulsive + 0.1 * attractive)
    return float(energy)


def _coarse_lj_type(atom: Atom) -> str:
    name = atom.name.strip().upper()
    element = atom.element.strip().upper()
    resname = atom.resname.strip().upper()
    if element == "H" or name.startswith("H"):
        return "H_polar" if name in {"H", "HN", "HG", "HG1", "HH", "HE1", "HE2"} else "H"
    if element == "S" or name.startswith("S"):
        return "S_sulfur"
    if element == "O":
        if name in {"O", "OXT"}:
            return "O_carbonyl"
        if name in {"OD1", "OD2", "OE1", "OE2"}:
            return "O_carboxyl"
        return "O_hydroxyl"
    if element == "N":
        return "N_backbone" if name == "N" else "N_sidechain"
    if element == "C":
        if name == "C":
            return "C_carbonyl"
        if name == "CA":
            return "C_backbone"
        if resname in {"PHE", "TYR", "TRP", "HIS"} and name in {
            "CG", "CD1", "CD2", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
        }:
            return "C_aromatic"
        return "C_aliphatic"
    return "X"


def _coarse_graph_distances_up_to_three(structure: HeavyAtomStructure) -> dict[tuple[int, int], int]:
    adjacency: list[set[int]] = [set() for _ in structure.atoms]
    index_by_residue_atom = {
        (atom.residue_key, atom.name.strip().upper()): index
        for index, atom in enumerate(structure.atoms)
    }

    def add_bond(first: Optional[int], second: Optional[int]) -> None:
        if first is None or second is None or first == second:
            return
        if not (0 <= first < len(adjacency) and 0 <= second < len(adjacency)):
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    for bond in structure.bonds:
        add_bond(int(bond.atom_index_1), int(bond.atom_index_2))

    def atom_index(residue_key: ResidueKey, name: str) -> Optional[int]:
        return index_by_residue_atom.get((residue_key, name))

    residues = structure.residue_keys()
    for residue_key in residues:
        add_bond(atom_index(residue_key, "N"), atom_index(residue_key, "CA"))
        add_bond(atom_index(residue_key, "CA"), atom_index(residue_key, "C"))
        add_bond(atom_index(residue_key, "C"), atom_index(residue_key, "O"))
        add_bond(atom_index(residue_key, "C"), atom_index(residue_key, "OXT"))
        for step in SIDECHAIN_STEPS.get(residue_key[3].strip().upper(), []):
            add_bond(
                atom_index(residue_key, step.parent.strip().upper()),
                atom_index(residue_key, step.atom.strip().upper()),
            )
    for left, right in zip(residues, residues[1:]):
        if left[0] == right[0] and int(right[1]) - int(left[1]) <= 1:
            add_bond(index_by_residue_atom.get((left, "C")), index_by_residue_atom.get((right, "N")))

    distances: dict[tuple[int, int], int] = {}
    for start in range(len(adjacency)):
        frontier = {start}
        visited = {start}
        depth = 0
        while frontier and depth < 3:
            depth += 1
            next_frontier = set()
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    pair = (start, neighbor) if start < neighbor else (neighbor, start)
                    distances[pair] = min(distances.get(pair, depth), depth)
                    next_frontier.add(neighbor)
            frontier = next_frontier
    return distances


def _coarse_rotamer_term(
    residue_keys: Sequence[tuple[str, int, str, str, str]],
    torsions: Mapping[str, float],
) -> float:
    energy = 0.0
    for index, key in enumerate(residue_keys):
        chi = _decoded_torsion(torsions, index, "chi1")
        res = _coarse_residue_one_letter(key)
        if chi is not None:
            if res in {"V", "I", "T"}:
                d_trans = _wrapped_angle_delta(chi, math.pi) ** 2
                d_gplus = _wrapped_angle_delta(chi, -1.0471975512) ** 2
                energy += -3.0 * (math.exp(-d_trans / 0.5) + math.exp(-d_gplus / 0.5))
            elif res == "P":
                d_down = _wrapped_angle_delta(chi, -0.5) ** 2
                d_up = _wrapped_angle_delta(chi, 0.5) ** 2
                energy += 10.0 * min(d_down, d_up)
            elif res in {"W", "F", "Y", "H"}:
                d_trans = _wrapped_angle_delta(chi, math.pi) ** 2
                d_gplus = _wrapped_angle_delta(chi, -1.0471975512) ** 2
                d_gminus = _wrapped_angle_delta(chi, 1.0471975512) ** 2
                energy += -2.0 * (
                    math.exp(-d_trans / 0.45)
                    + 0.8 * math.exp(-d_gplus / 0.45)
                    + 0.8 * math.exp(-d_gminus / 0.45)
                )
            else:
                energy += 1.0 * (1.0 + math.cos(3.0 * chi))

        centers = (-1.0471975512, 1.0471975512, math.pi)
        for chi_index in (2, 3, 4):
            higher_chi = _decoded_torsion(torsions, index, f"chi{chi_index}")
            if higher_chi is None:
                continue
            denominator = 0.35 if chi_index == 2 and res in {"W", "F", "Y", "H"} else 0.50
            wells = sum(
                math.exp(-(_wrapped_angle_delta(higher_chi, center) ** 2) / denominator)
                for center in centers
            )
            energy += (-1.5 if chi_index == 2 and res in {"W", "F", "Y", "H"} else -0.75) * wells
    return float(energy)


def _wrapped_angle_delta(first: float, second: float) -> float:
    return (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi


def _coarse_ramachandran_term(
    residue_keys: Sequence[tuple[str, int, str, str, str]],
    torsions: Mapping[str, float],
) -> float:
    energy = 0.0
    for index, key in enumerate(residue_keys):
        phi = _decoded_torsion(torsions, index, "phi")
        psi = _decoded_torsion(torsions, index, "psi")
        if phi is None or psi is None:
            continue
        res = _coarse_residue_one_letter(key)
        d_helix = (phi - (-1.0)) ** 2 + (psi - (-0.8)) ** 2
        d_sheet = (phi - (-2.3)) ** 2 + (psi - 2.4) ** 2
        if res == "G":
            d_helix_l = (phi - 1.0) ** 2 + (psi - 0.8) ** 2
            d_sheet_l = (phi - 2.3) ** 2 + (psi - (-2.4)) ** 2
            energy += -3.0 * math.exp(-min(d_helix, d_sheet, d_helix_l, d_sheet_l) / 0.6)
        else:
            d_forbidden = (phi - (-2.0)) ** 2 + (psi - 1.0) ** 2
            energy += -3.0 * math.exp(-d_helix / 0.6) - 3.0 * math.exp(-d_sheet / 0.6) + 5.0 * math.exp(-d_forbidden / 1.0)
    return float(energy)

def _normalize_pheat_physics_charge_profile(charge_profile: str) -> str:
    normalized = str(charge_profile or "protein-coarse-charge-v1").strip().lower()
    if normalized not in PHEAT_PHYSICS_CHARGE_PROFILES:
        known = ", ".join(PHEAT_PHYSICS_CHARGE_PROFILES)
        raise ValueError(f"Unknown pheat-physics charge_profile '{charge_profile}'. Choose from: {known}")
    return normalized

def _pheat_physics_charge(
    atom: Atom,
    *,
    charge_profile: str,
    terminal_residue_keys: set[tuple[str, int, str, str, str]],
) -> float:
    profile = _normalize_pheat_physics_charge_profile(charge_profile)
    if profile == "pheat-default":
        return _charge(atom)

    name = atom.name.strip().upper()
    if atom.residue_key in terminal_residue_keys and name in {"N", "CA", "C", "O", "OXT", "H1", "H2", "H3", "H"}:
        return 0.0

    charges = dict(_PHEAT_PHYSICS_COMMON_CHARGES)
    charges.update(_PHEAT_PHYSICS_CHARGE_TABLES[profile])
    q = float(charges.get(name, 0.0))
    if atom.resname.strip().upper() in {"HIS", "H"}:
        if name == "NE2":
            q = 0.0
        if name == "ND1":
            q = -0.4
    return q


def _pheat_physics_adjacent_heavy_sterics(structure: HeavyAtomStructure) -> float:
    atoms_by_residue = structure.atoms_by_residue()
    residues = structure.residue_keys()
    energy = 0.0
    for left_key, right_key in zip(residues, residues[1:]):
        if left_key[0] != right_key[0]:
            continue
        if int(right_key[1]) - int(left_key[1]) != 1:
            continue
        for atom_a in atoms_by_residue[left_key]:
            if atom_a.element.strip().upper() == "H":
                continue
            for atom_b in atoms_by_residue[right_key]:
                if atom_b.element.strip().upper() == "H":
                    continue
                if atom_a.name.strip().upper() == "C" and atom_b.name.strip().upper() == "N":
                    continue
                radius_a = COARSE_LJ_TYPE_PARAMS[_coarse_lj_type(atom_a)][0]
                radius_b = COARSE_LJ_TYPE_PARAMS[_coarse_lj_type(atom_b)][0]
                threshold = max(1.35, 0.55 * (radius_a + radius_b))
                dist = distance(atom_a.coord, atom_b.coord)
                if dist < threshold:
                    energy += 10.0 * ((threshold - dist) / 0.5) ** 2
    return float(energy)


def _pheat_physics_pi_stacking(structure: HeavyAtomStructure) -> float:
    rings = []
    for key, atoms in structure.atoms_by_residue().items():
        atom_map = {atom.name.strip().upper(): atom for atom in atoms}
        resname = key[3]
        if resname not in {"PHE", "TYR", "TRP"}:
            continue
        ring_names = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
        ring_atoms = [atom_map[name] for name in ring_names if name in atom_map]
        if len(ring_atoms) < 3:
            continue
        centroid = tuple(sum(atom.coord[i] for atom in ring_atoms) / len(ring_atoms) for i in range(3))
        v1 = sub(ring_atoms[1].coord, ring_atoms[0].coord)
        v2 = sub(ring_atoms[2].coord, ring_atoms[0].coord)
        normal = normalize(cross(v1, v2))
        rings.append((centroid, normal))
    energy = 0.0
    for index, (center_a, normal_a) in enumerate(rings[:-1]):
        for center_b, normal_b in rings[index + 1:]:
            dist = distance(center_a, center_b)
            if dist > 7.0:
                continue
            alignment = abs(dot(normal_a, normal_b))
            if alignment < 0.3 and 4.5 < dist < 6.0:
                energy -= 4.0 * math.exp(-((dist - 5.0) ** 2))
            elif alignment > 0.8 and 3.4 < dist < 4.5:
                energy -= 5.0 * math.exp(-((dist - 3.8) ** 2))
    return float(energy)


GEOMETRY_INTEGRITY_HUBER_DELTA = 0.25
GEOMETRY_INTEGRITY_BOND_TOLERANCE = 0.03
GEOMETRY_INTEGRITY_LINK_TOLERANCE = 0.04
GEOMETRY_INTEGRITY_PROLINE_TOLERANCE = 0.10
GEOMETRY_INTEGRITY_PLANARITY_TOLERANCE_DEGREES = 20.0
GEOMETRY_INTEGRITY_CHIRALITY_MIN_VOLUME = 1.0
GEOMETRY_INTEGRITY_BACKBONE_BOND_WEIGHT = 20.0
GEOMETRY_INTEGRITY_PEPTIDE_LINK_WEIGHT = 30.0
GEOMETRY_INTEGRITY_PROLINE_RING_WEIGHT = 50.0
GEOMETRY_INTEGRITY_CHIRALITY_WEIGHT = 50.0
GEOMETRY_INTEGRITY_PLANARITY_WEIGHT = 20.0


def _score_pheat_geometry_integrity(structure: HeavyAtomStructure) -> EnergyResult:
    """Coordinate-geometry plausibility score for PHEAT atom structures."""

    terms = {
        "backbone_bond_lengths": 0.0,
        "peptide_c_n_links": 0.0,
        "ca_chirality": 0.0,
        "peptide_planarity": 0.0,
        "proline_ring_closure": 0.0,
    }
    counts = {key: 0 for key in terms}
    skipped = {key: 0 for key in terms}
    residues = structure.atoms_by_residue()
    residue_keys = structure.residue_keys()
    atom_maps = {
        key: {atom.name.strip().upper(): atom for atom in atoms}
        for key, atoms in residues.items()
    }

    for key in residue_keys:
        atoms = atom_maps[key]
        for first, second, expected in (
            ("N", "CA", N_CA),
            ("CA", "C", CA_C),
            ("C", "O", C_O),
        ):
            if first in atoms and second in atoms:
                terms["backbone_bond_lengths"] += _geometry_huber_distance_penalty(
                    distance(atoms[first].coord, atoms[second].coord),
                    expected,
                    tolerance=GEOMETRY_INTEGRITY_BOND_TOLERANCE,
                )
                counts["backbone_bond_lengths"] += 1
            else:
                skipped["backbone_bond_lengths"] += 1

        if key[3] not in {"GLY"} and all(name in atoms for name in ("N", "CA", "C", "CB")):
            ca = atoms["CA"].coord
            volume = dot(cross(sub(atoms["N"].coord, ca), sub(atoms["C"].coord, ca)), sub(atoms["CB"].coord, ca))
            if volume < GEOMETRY_INTEGRITY_CHIRALITY_MIN_VOLUME:
                terms["ca_chirality"] += GEOMETRY_INTEGRITY_CHIRALITY_WEIGHT * _huber_loss(
                    GEOMETRY_INTEGRITY_CHIRALITY_MIN_VOLUME - volume,
                    GEOMETRY_INTEGRITY_HUBER_DELTA,
                )
            counts["ca_chirality"] += 1
        elif key[3] not in {"GLY"}:
            skipped["ca_chirality"] += 1

        if key[3] in {"PRO", "HYP"}:
            if "N" in atoms and "CD" in atoms:
                terms["proline_ring_closure"] += _geometry_huber_distance_penalty(
                    distance(atoms["N"].coord, atoms["CD"].coord),
                    PRO_N_CD,
                    tolerance=GEOMETRY_INTEGRITY_PROLINE_TOLERANCE,
                    weight=GEOMETRY_INTEGRITY_PROLINE_RING_WEIGHT,
                )
                counts["proline_ring_closure"] += 1
            else:
                skipped["proline_ring_closure"] += 1

    for left, right in zip(residue_keys, residue_keys[1:]):
        if left[0] != right[0]:
            continue
        if right[1] - left[1] > 1:
            continue
        left_atoms = atom_maps[left]
        right_atoms = atom_maps[right]
        if "C" in left_atoms and "N" in right_atoms:
            terms["peptide_c_n_links"] += _geometry_huber_distance_penalty(
                distance(left_atoms["C"].coord, right_atoms["N"].coord),
                C_N,
                tolerance=GEOMETRY_INTEGRITY_LINK_TOLERANCE,
                weight=GEOMETRY_INTEGRITY_PEPTIDE_LINK_WEIGHT,
            )
            counts["peptide_c_n_links"] += 1
        else:
            skipped["peptide_c_n_links"] += 1
        if all(name in left_atoms for name in ("CA", "C")) and all(name in right_atoms for name in ("N", "CA")):
            try:
                omega = dihedral_degrees(
                    left_atoms["CA"].coord,
                    left_atoms["C"].coord,
                    right_atoms["N"].coord,
                    right_atoms["CA"].coord,
                )
                planar_deviation = min(abs(omega), abs(abs(omega) - 180.0))
                residual = max(0.0, planar_deviation - GEOMETRY_INTEGRITY_PLANARITY_TOLERANCE_DEGREES) / 20.0
                terms["peptide_planarity"] += GEOMETRY_INTEGRITY_PLANARITY_WEIGHT * _huber_loss(
                    residual,
                    GEOMETRY_INTEGRITY_HUBER_DELTA,
                )
                counts["peptide_planarity"] += 1
            except ValueError:
                skipped["peptide_planarity"] += 1
        else:
            skipped["peptide_planarity"] += 1

    warnings = _unsupported_warnings(structure)
    for term, count in skipped.items():
        if count:
            warnings.append(f"pheat-geometry-integrity skipped {count} {term} checks because atoms were missing")

    return EnergyResult(
        model="pheat-geometry-integrity",
        total=sum(terms.values()),
        units="arbitrary",
        terms=terms,
        warnings=warnings,
        citations=["pheat-geometry-integrity-v1", "engh-huber-1991"],
        metadata={
            "description": "PHEAT coordinate-geometry integrity score with robust Huber penalties",
            "score_direction": "lower-is-better",
            "checked_counts": counts,
            "skipped_counts": skipped,
            "constants": {
                "n_ca": N_CA,
                "ca_c": CA_C,
                "c_o": C_O,
                "c_n": C_N,
                "pro_n_cd": PRO_N_CD,
                "huber_delta": GEOMETRY_INTEGRITY_HUBER_DELTA,
                "bond_tolerance": GEOMETRY_INTEGRITY_BOND_TOLERANCE,
                "peptide_link_tolerance": GEOMETRY_INTEGRITY_LINK_TOLERANCE,
                "proline_ring_tolerance": GEOMETRY_INTEGRITY_PROLINE_TOLERANCE,
                "planarity_tolerance_degrees": GEOMETRY_INTEGRITY_PLANARITY_TOLERANCE_DEGREES,
                "chirality_min_volume": GEOMETRY_INTEGRITY_CHIRALITY_MIN_VOLUME,
                "weights": {
                    "backbone_bond_lengths": GEOMETRY_INTEGRITY_BACKBONE_BOND_WEIGHT,
                    "peptide_c_n_links": GEOMETRY_INTEGRITY_PEPTIDE_LINK_WEIGHT,
                    "ca_chirality": GEOMETRY_INTEGRITY_CHIRALITY_WEIGHT,
                    "peptide_planarity": GEOMETRY_INTEGRITY_PLANARITY_WEIGHT,
                    "proline_ring_closure": GEOMETRY_INTEGRITY_PROLINE_RING_WEIGHT,
                },
                "planarity_target": "cis-or-trans",
            },
            "note": (
                "Deterministic diagnostic defaults; future trained profiles should calibrate "
                "normalization or thresholds rather than replace PHEAT ideal geometry definitions."
            ),
        },
    )


def _geometry_huber_distance_penalty(
    observed: float,
    expected: float,
    *,
    tolerance: float,
    weight: float = GEOMETRY_INTEGRITY_BACKBONE_BOND_WEIGHT,
) -> float:
    residual = max(0.0, abs(float(observed) - float(expected)) - float(tolerance))
    return float(weight) * _huber_loss(residual, GEOMETRY_INTEGRITY_HUBER_DELTA)


def _huber_loss(value: float, delta: float) -> float:
    magnitude = abs(float(value))
    threshold = float(delta)
    if magnitude <= threshold:
        return magnitude * magnitude
    return 2.0 * threshold * magnitude - threshold * threshold


def _score_ambertools_sander(
    structure: HeavyAtomStructure,
    *,
    prepare: str = "auto",
    prepared_output: Optional[Path] = None,
    amber_forcefield: str = DEFAULT_AMBER_FORCEFIELD,
    amber_solvent: str = "vacuum",
    ambertools_work_dir: Optional[Path] = None,
    keep_ambertools_files: bool = False,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Path] = None,
    prep_cache_mode: str = "off",
) -> EnergyResult:
    """Run an AmberTools ``sander`` single-point energy through a prepared PDB."""

    prepare_mode = _normalize_prepare_mode(prepare)
    solvent_mode = _normalize_amber_solvent(amber_solvent)
    timeout = _normalize_external_timeout(external_timeout_seconds)
    cache_mode = _normalize_prep_cache_mode(prep_cache_mode)
    tleap = shutil.which("tleap")
    sander = shutil.which("sander")
    if tleap is None or sander is None:
        missing = [name for name, path in [("tleap", tleap), ("sander", sander)] if path is None]
        raise RuntimeError(
            "ambertools-sander scoring requires AmberTools executables on PATH: "
            + ", ".join(missing)
        )

    if prepare_mode == "write" and prepared_output is None:
        raise ValueError("--prepare write requires --prepared-output")

    cache_metadata = _external_prep_cache_metadata(
        backend="ambertools",
        structure=structure,
        cache_dir=prep_cache_dir,
        cache_mode=cache_mode,
        settings={
            "prepare": prepare_mode,
            "amber_forcefield": amber_forcefield,
            "amber_solvent": solvent_mode,
        },
        reusable=False,
        disabled_reason="AmberTools cache does not skip tleap because inpcrd coordinates are candidate-specific",
    )

    with _ambertools_workspace(ambertools_work_dir, keep_ambertools_files) as work_dir:
        prepared_pdb = work_dir / "prepared.pdb"
        preparation_warnings, preparation_metadata = _prepare_pdb_for_forcefield(
            structure,
            prepared_pdb,
            prepare=prepare_mode,
        )
        if prepared_output is not None:
            prepared_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prepared_pdb, prepared_output)
            preparation_metadata["prepared_output"] = str(prepared_output)

        leap_input = _write_tleap_input(
            work_dir,
            amber_forcefield=amber_forcefield,
            amber_pbradii=_amber_pbradii_for_solvent(solvent_mode),
        )
        leap_log = work_dir / "leap.log"
        try:
            leap_result = _run_subprocess([tleap, "-f", str(leap_input.name)], cwd=work_dir, timeout=timeout)
        except RuntimeError as exc:
            raise RuntimeError(_external_command_error_with_file_tail(exc, leap_log, "leap.log")) from exc
        prmtop = work_dir / "system.prmtop"
        inpcrd = work_dir / "system.inpcrd"
        if not prmtop.exists() or not inpcrd.exists():
            raise RuntimeError(
                "AmberTools tleap did not produce system.prmtop/system.inpcrd. "
                + _subprocess_tail(leap_result)
            )

        sander_input = _write_sander_input(work_dir, solvent=solvent_mode)
        sander_output = work_dir / "sander.out"
        try:
            sander_result = _run_subprocess(
                [
                    sander,
                    "-O",
                    "-i",
                    str(sander_input.name),
                    "-o",
                    str(sander_output.name),
                    "-p",
                    str(prmtop.name),
                    "-c",
                    str(inpcrd.name),
                ],
                cwd=work_dir,
                timeout=timeout,
            )
        except RuntimeError as exc:
            raise RuntimeError(_external_command_error_with_file_tail(exc, sander_output, "sander.out")) from exc
        if not sander_output.exists():
            raise RuntimeError("AmberTools sander did not produce sander.out. " + _subprocess_tail(sander_result))
        terms = parse_sander_energy_terms(sander_output.read_text(encoding="utf-8", errors="replace"))

        metadata = {
            "description": "AmberTools sander single-point AMBER molecular mechanics score",
            "preparation": preparation_metadata,
            "amber_forcefield": amber_forcefield,
            "amber_solvent": solvent_mode,
            "amber_pbradii": _amber_pbradii_for_solvent(solvent_mode),
            "executables": {"tleap": tleap, "sander": sander},
            "command_lines": {
                "tleap": [tleap, "-f", str(leap_input.name)],
                "sander": [
                    sander,
                    "-O",
                    "-i",
                    str(sander_input.name),
                    "-o",
                    str(sander_output.name),
                    "-p",
                    str(prmtop.name),
                    "-c",
                    str(inpcrd.name),
                ],
            },
            "work_dir": str(work_dir),
            "kept_work_dir": bool(ambertools_work_dir is not None or keep_ambertools_files),
            "input_atom_scope": structure.atom_scope,
            "external_timeout_seconds": timeout,
            "prep_cache": cache_metadata,
        }

    total = terms.get("Etot") or terms.get("EPtot") or terms.get("ENERGY")
    if total is None:
        raise RuntimeError("Could not parse Etot, EPtot, or NSTEP ENERGY from AmberTools sander output.")
    return EnergyResult(
        model="ambertools-sander",
        total=float(total),
        units="kcal/mol",
        terms={key.lower(): value for key, value in terms.items()},
        warnings=preparation_warnings
        + [
            "ambertools-sander is a single-point molecular mechanics potential energy, not folding free energy",
            "compare only matched structures prepared with the same force field and solvent settings",
        ],
        citations=["ambertools", "amber-ff14sb"],
        metadata=metadata,
    )


def parse_sander_energy_terms(output_text: str) -> dict[str, float]:
    """Parse energy terms from AmberTools ``sander`` output text."""

    terms: dict[str, float] = {}
    nstep_energy = re.search(
        r"NSTEP\s+ENERGY\b[^\n]*\n\s*\d+\s+(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        output_text,
    )
    if nstep_energy:
        terms["ENERGY"] = float(nstep_energy.group(1))
    value_pattern = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    for label in [
        "Etot",
        "EPtot",
        "EKtot",
        "BOND",
        "ANGLE",
        "DIHED",
        "VDWAALS",
        "EEL",
        "HBOND",
        "1-4 VDW",
        "1-4 EEL",
        "RESTRAINT",
    ]:
        prefix = r"(?<![A-Za-z0-9-])" if label.startswith("1-4") else r"(?:^\s*|\s{2,})"
        matches = re.findall(rf"{prefix}{re.escape(label)}\s*=\s*{value_pattern}", output_text, flags=re.MULTILINE)
        if matches:
            terms[label] = float(matches[-1])
    return terms


def prepare_gromacs_structure(
    structure: HeavyAtomStructure,
    *,
    prepare: str = "auto",
    output: Union[str, Path],
    topology_output: Optional[Union[str, Path]] = None,
    gromacs_forcefield: str = DEFAULT_GROMACS_FORCEFIELD,
    gromacs_water: str = DEFAULT_GROMACS_WATER,
    gromacs_solvate: bool = False,
    gromacs_work_dir: Optional[Union[str, Path]] = None,
    keep_gromacs_files: bool = False,
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Union[str, Path]] = None,
    prep_cache_mode: str = "off",
    status_stream: Optional[Any] = None,
) -> dict[str, Any]:
    """Prepare a GROMACS coordinate/topology pair without running an energy calculation."""

    gmx = _require_gromacs_executable()
    timeout = _normalize_external_timeout(external_timeout_seconds)
    settings = _normalize_gromacs_run_settings(gromacs_run_settings)
    cache_mode = _normalize_prep_cache_mode(prep_cache_mode)
    output_path = Path(output)
    topology_path = Path(topology_output) if topology_output is not None else None
    with _gromacs_workspace(
        Path(gromacs_work_dir) if gromacs_work_dir is not None else None,
        keep_gromacs_files,
    ) as work_dir:
        prepared = _prepare_gromacs_system(
            structure,
            work_dir,
            gmx=gmx,
            prepare=prepare,
            gromacs_forcefield=gromacs_forcefield,
            gromacs_water=gromacs_water,
            gromacs_solvate=gromacs_solvate,
            settings=settings,
            timeout=timeout,
            prep_cache_dir=Path(prep_cache_dir) if prep_cache_dir is not None else None,
            prep_cache_mode=cache_mode,
            status_stream=status_stream,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prepared["coordinate_path"], output_path)
        if topology_path is not None:
            topology_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prepared["topology_path"], topology_path)
        return {
            "format": "pheat.gromacs-preparation",
            "version": 1,
            "coordinate_output": str(output_path),
            "topology_output": str(topology_path) if topology_path is not None else None,
            "warnings": prepared["warnings"],
            "metadata": {
                "implementation": _implementation_metadata_for_model("gromacs-mdrun"),
                **prepared["metadata"],
                "work_dir": str(work_dir),
                "kept_work_dir": bool(gromacs_work_dir is not None or keep_gromacs_files),
                "external_timeout_seconds": timeout,
            },
        }


def _gromacs_physical_integrity_preflight(
    structure: HeavyAtomStructure,
    *,
    settings: GromacsRunSettings,
    run_mode: str,
    mode: str = "strict",
) -> Optional[EnergyResult]:
    """Return a blocking or warning GROMACS preflight result for unsafe coordinates."""

    normalized_mode = _normalize_gromacs_preflight(mode)
    physical = _score_pheat_physical_integrity(structure)
    counts = dict((physical.metadata or {}).get("checked_counts") or {})
    short_contact_count = int(counts.get("short_contact_count") or 0)
    nonfinite_atom_count = int(counts.get("nonfinite_atom_count") or 0)
    min_distance = physical.metadata.get("min_nonlocal_distance")
    blocking_failures: list[str] = []
    unsafe_failures: list[str] = []
    if nonfinite_atom_count > 0:
        blocking_failures.append(f"{nonfinite_atom_count} atoms have non-finite coordinates")
    if normalized_mode != "off":
        if short_contact_count > 0:
            unsafe_failures.append(f"{short_contact_count} severe nonlocal contacts are below {PHYSICAL_INTEGRITY_SHORT_CONTACT_A:g} A")
        if min_distance is not None and float(min_distance) < PHYSICAL_INTEGRITY_SHORT_CONTACT_A:
            unsafe_failures.append(
                f"minimum nonlocal heavy-atom distance is {float(min_distance):.4g} A "
                f"(< {PHYSICAL_INTEGRITY_SHORT_CONTACT_A:g} A)"
            )
    failures = [*blocking_failures, *unsafe_failures]
    if not failures:
        return None

    status = "unavailable" if blocking_failures or normalized_mode == "strict" else "warning"
    reason = "; ".join(failures)
    action = (
        "GROMACS mdrun preflight skipped execution because physical-integrity checks failed"
        if status == "unavailable"
        else "GROMACS mdrun preflight warning: execution continued despite physical-integrity findings"
    )
    return EnergyResult(
        model="gromacs-mdrun",
        total=0.0,
        units="unavailable" if status == "unavailable" else "warning",
        terms={},
        warnings=[
            action,
            reason,
            *physical.warnings,
        ],
        citations=["pheat-gromacs-preflight-v1", *physical.citations],
        metadata={
            "description": "GROMACS mdrun preflight physical-integrity guard",
            "status": status,
            "mode": normalized_mode,
            "reason": reason,
            "preflight": physical.to_dict(),
            "gromacs_run_mode": run_mode,
            "gromacs_run_settings": settings.to_dict(),
            "input_atom_scope": structure.atom_scope,
        },
    )


def _score_gromacs_mdrun(
    structure: HeavyAtomStructure,
    *,
    prepare: str = "auto",
    prepared_output: Optional[Path] = None,
    gromacs_forcefield: str = DEFAULT_GROMACS_FORCEFIELD,
    gromacs_water: str = DEFAULT_GROMACS_WATER,
    gromacs_solvate: bool = False,
    gromacs_run_mode: str = "rerun",
    gromacs_preflight: str = "strict",
    gromacs_work_dir: Optional[Path] = None,
    keep_gromacs_files: bool = False,
    gromacs_metrics: bool = False,
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Path] = None,
    prep_cache_mode: str = "off",
    status_stream: Optional[Any] = None,
) -> EnergyResult:
    """Run a GROMACS topology-prepared energy check for validation/reranking."""

    run_mode = _normalize_gromacs_run_mode(gromacs_run_mode)
    preflight_mode = _normalize_gromacs_preflight(gromacs_preflight)
    settings = _normalize_gromacs_run_settings(gromacs_run_settings)
    timeout = _normalize_external_timeout(external_timeout_seconds)
    cache_mode = _normalize_prep_cache_mode(prep_cache_mode)
    preflight = _gromacs_physical_integrity_preflight(
        structure,
        settings=settings,
        run_mode=run_mode,
        mode=preflight_mode,
    )
    preflight_warning = None
    if preflight is not None:
        if (preflight.metadata or {}).get("status") == "unavailable":
            return preflight
        preflight_warning = preflight
    gmx = _require_gromacs_executable()
    if _normalize_prepare_mode(prepare) == "write" and prepared_output is None:
        raise ValueError("--prepare write requires --prepared-output")

    with _gromacs_workspace(gromacs_work_dir, keep_gromacs_files) as work_dir:
        version = _gromacs_version(gmx, work_dir)
        prepared = _prepare_gromacs_system(
            structure,
            work_dir,
            gmx=gmx,
            prepare=prepare,
            gromacs_forcefield=gromacs_forcefield,
            gromacs_water=gromacs_water,
            gromacs_solvate=gromacs_solvate,
            settings=settings,
            timeout=timeout,
            prep_cache_dir=prep_cache_dir,
            prep_cache_mode=cache_mode,
            status_stream=status_stream,
        )
        energy = _run_gromacs_energy_workflow(
            gmx,
            work_dir,
            coordinate_path=prepared["coordinate_path"],
            topology_path=prepared["topology_path"],
            run_mode=run_mode,
            metrics=gromacs_metrics,
            settings=settings,
            timeout=timeout,
            status_stream=status_stream,
        )
        final_coordinate = energy["final_coordinate_path"]
        if prepared_output is not None:
            prepared_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_coordinate, prepared_output)
            prepared["metadata"]["prepared_output"] = str(prepared_output)

        metadata = {
            "description": "GROMACS mdrun force-field validation/reranking score",
            "preparation": prepared["metadata"],
            "gromacs_version": version,
            "gromacs_forcefield": prepared["metadata"]["gromacs_forcefield"],
            "gromacs_water": prepared["metadata"]["gromacs_water"],
            "gromacs_solvate": prepared["metadata"]["gromacs_solvate"],
            "gromacs_run_mode": run_mode,
            "gromacs_preflight": {
                "mode": preflight_mode,
                "status": "ok" if preflight_warning is None else "warning",
                "result": None if preflight_warning is None else preflight_warning.to_dict(),
            },
            "gromacs_run_settings": settings.to_dict(),
            "executables": {"gmx": gmx},
            "command_lines": {
                **prepared["command_lines"],
                **energy["command_lines"],
            },
            "work_dir": str(work_dir),
            "kept_work_dir": bool(gromacs_work_dir is not None or keep_gromacs_files),
            "input_atom_scope": structure.atom_scope,
            "external_timeout_seconds": timeout,
        }
        if energy["metrics"]:
            metadata["gromacs_metrics"] = energy["metrics"]

    terms = dict(energy["terms"])
    total = terms["potential"] if "potential" in terms else terms.get("total_energy")
    if total is None:
        raise RuntimeError("Could not parse Potential or Total-Energy from GROMACS energy output.")
    return EnergyResult(
        model="gromacs-mdrun",
        total=float(total),
        units="kJ/mol",
        terms=terms,
        warnings=(preflight_warning.warnings if preflight_warning is not None else [])
        + prepared["warnings"]
        + energy["warnings"]
        + [
            "gromacs-mdrun is a force-field potential-energy validation/reranking score, not folding free energy",
            "compare only structures prepared with the same force field, water, solvation, and run-mode settings",
        ],
        citations=_gromacs_citations(str(metadata["gromacs_forcefield"])),
        metadata=metadata,
    )


def _with_forcefield_terminal_oxt(structure: HeavyAtomStructure) -> tuple[HeavyAtomStructure, int]:
    """Return a copy with missing protein-chain terminal OXT atoms completed."""
    completed = deepcopy(structure)
    residues = completed.atoms_by_residue()
    last_by_chain: dict[str, Any] = {}
    for key in completed.residue_keys():
        if key[4] == "ATOM":
            last_by_chain[key[0]] = key

    added = 0
    next_serial = max((atom.serial or 0 for atom in completed.atoms), default=0) + 1
    for key in last_by_chain.values():
        atoms = {atom.name.strip().upper(): atom for atom in residues.get(key, [])}
        if "OXT" in atoms or not {"N", "CA", "C", "O"}.issubset(atoms):
            continue
        n_atom, ca_atom, c_atom, o_atom = (atoms[name] for name in ("N", "CA", "C", "O"))
        o_dihedral = dihedral_degrees(n_atom.coord, ca_atom.coord, c_atom.coord, o_atom.coord)
        coord = place_atom(
            n_atom.coord,
            ca_atom.coord,
            c_atom.coord,
            length=C_O,
            angle_degrees=ANGLE_CA_C_O,
            dihedral_degrees=o_dihedral + 180.0,
        )
        completed.atoms.append(
            Atom(
                name="OXT",
                element="O",
                x=coord[0],
                y=coord[1],
                z=coord[2],
                resname=c_atom.resname,
                chain_id=c_atom.chain_id,
                resseq=c_atom.resseq,
                icode=c_atom.icode,
                record_name=c_atom.record_name,
                serial=next_serial,
            )
        )
        next_serial += 1
        added += 1
    return completed, added


def _score_openmm_prepared(structure: HeavyAtomStructure) -> EnergyResult:
    """Explicit OpenMM path that prepares a force-field-ready structure before scoring."""

    prepared_structure, terminal_oxt_added = _with_forcefield_terminal_oxt(structure)
    try:
        from openmm import Platform, VerletIntegrator, app, unit
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "openmm-prepared scoring requires the optional Miniforge environment with OpenMM/PDBFixer."
        ) from exc

    from pheat.pdbio import structure_to_pdb_string

    try:
        with tempfile.TemporaryDirectory(prefix="pheat-openmm-") as tmpdir:
            pdb_path = Path(tmpdir) / "input.pdb"
            # OpenMM receives a temporary PDB because this path prepares a force-field
            # topology internally; chain IDs do not affect the energy calculation.
            pdb_path.write_text(
                structure_to_pdb_string(prepared_structure, allow_chain_truncation=True),
                encoding="utf-8",
            )
            forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            platform = Platform.getPlatformByName("Reference")

            topology, positions, preparation_warnings, preparation_metadata = _prepare_openmm_topology(
                str(pdb_path),
                app,
                forcefield,
                platform,
            )
            system = forcefield.createSystem(
                topology,
                nonbondedMethod=app.NoCutoff,
                constraints=None,
                ignoreExternalBonds=True,
            )
            integrator = VerletIntegrator(0.001)
            context = app.Simulation(topology, system, integrator, platform).context
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception as exc:  # pragma: no cover - parameterization depends on input chemistry
        raise RuntimeError(f"OpenMM could not prepare/score this structure: {exc}") from exc

    return EnergyResult(
        model="openmm-prepared",
        total=float(energy),
        units="kJ/mol",
        terms={"potential_energy": float(energy)},
        warnings=(
            ([f"added {terminal_oxt_added} missing terminal OXT atom(s) internally for force-field preparation"]
             if terminal_oxt_added else [])
            + preparation_warnings
        ),
        citations=["openmm", "amber-ff14sb"],
        metadata={
            "description": "OpenMM AMBER score after internal structure preparation",
            "terminal_oxt_added": terminal_oxt_added,
            **preparation_metadata,
        },
    )


def _prepare_openmm_topology(
    pdb_path: str,
    app,
    forcefield,
    platform,
) -> tuple[object, object, List[str], Dict[str, object]]:
    """Prepare a heavy-atom PDB for OpenMM while leaving the source structure untouched."""

    try:
        from pdbfixer import PDBFixer
    except Exception:  # pragma: no cover - depends on optional conda package
        pdb = app.PDBFile(pdb_path)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        _with_python_random_seed(
            OPENMM_PREPARATION_SEED,
            lambda: modeller.addHydrogens(forcefield, pH=7.0, platform=platform),
        )
        return (
            modeller.topology,
            modeller.positions,
            [
                "hydrogens were added internally for OpenMM scoring; coordinates in the input structure were not modified"
            ],
            {"preparation": "openmm-modeller", "preparation_seed": OPENMM_PREPARATION_SEED},
        )

    fixer = PDBFixer(filename=pdb_path, platform=platform)
    fixer.findMissingResidues()
    # Do not build unresolved residues from SEQRES records during scoring. PHEAT
    # scores the atoms supplied by the caller, with only force-field preparation
    # additions such as terminal atoms and hydrogens applied internally.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    missing_atom_count = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    missing_terminal_count = sum(len(atoms) for atoms in fixer.missingTerminals.values())
    fixer.addMissingAtoms(seed=OPENMM_PREPARATION_SEED)
    _with_python_random_seed(
        OPENMM_PREPARATION_SEED,
        lambda: fixer.addMissingHydrogens(7.0),
    )

    warnings = [
        "hydrogens were added internally for OpenMM scoring; coordinates in the input structure were not modified"
    ]
    if missing_atom_count or missing_terminal_count:
        warnings.append(
            "PDBFixer added missing heavy or terminal atoms internally for OpenMM scoring; "
            "the input structure was not modified"
        )
    return (
        fixer.topology,
        fixer.positions,
        warnings,
        {
            "preparation": "pdbfixer",
            "preparation_seed": OPENMM_PREPARATION_SEED,
            "missing_heavy_atoms_added": missing_atom_count,
            "missing_terminal_atoms_added": missing_terminal_count,
        },
    )


def _prepare_pdb_for_forcefield(
    structure: HeavyAtomStructure,
    output_pdb: Path,
    *,
    prepare: str,
) -> tuple[list[str], dict[str, object]]:
    """Write a force-field-ready PDB for AmberTools or fail with actionable context."""

    from pheat.pdbio import structure_to_pdb_string

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    input_hydrogens = _hydrogen_atom_count(structure)
    if prepare == "never":
        if input_hydrogens == 0 or structure.atom_scope != "all":
            raise RuntimeError(
                "--prepare never requires an all-atom input structure with hydrogens. "
                "Use --prepare auto to let PHEAT prepare the structure before AmberTools scoring."
            )
        output_pdb.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        return [], {
            "mode": prepare,
            "engine": "input",
            "input_atom_count": len(structure.atoms),
            "input_hydrogen_atom_count": input_hydrogens,
            "prepared_atom_count": len(structure.atoms),
            "prepared_hydrogen_atom_count": input_hydrogens,
            "hydrogens_added": 0,
        }

    prepared_structure, terminal_oxt_added = _with_forcefield_terminal_oxt(structure)
    try:
        from openmm import Platform, app
    except Exception as exc:  # pragma: no cover - depends on optional user environment
        raise RuntimeError(
            "--prepare auto/write for ambertools-sander requires OpenMM for deterministic "
            "hydrogen and missing-atom preparation."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="pheat-amber-prep-") as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        pdb_path.write_text(
            structure_to_pdb_string(prepared_structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        platform = Platform.getPlatformByName("Reference")
        topology, positions, warnings, metadata = _prepare_openmm_topology(
            str(pdb_path),
            app,
            forcefield,
            platform,
        )
        with output_pdb.open("w", encoding="utf-8") as prepared_handle:
            app.PDBFile.writeFile(topology, positions, prepared_handle)
    removed_openmm_hydrogens = _strip_pdb_hydrogens(output_pdb)

    topology_any: Any = topology
    prepared_atom_count = len(list(topology_any.atoms()))
    openmm_hydrogens = _openmm_hydrogen_atom_count(topology_any)
    metadata = {
        **metadata,
        "mode": prepare,
        "engine": "openmm-pdbfixer",
        "input_atom_count": len(structure.atoms),
        "input_hydrogen_atom_count": input_hydrogens,
        "prepared_atom_count": prepared_atom_count - removed_openmm_hydrogens,
        "prepared_hydrogen_atom_count": 0,
        "openmm_hydrogens_removed_before_tleap": removed_openmm_hydrogens,
        "hydrogens_added": 0,
        "hydrogen_assignment": "tleap-residue-templates",
        "prepared_coordinate_count": prepared_atom_count - removed_openmm_hydrogens,
        "pH": 7.0,
        "terminal_oxt_added": terminal_oxt_added,
    }
    if terminal_oxt_added:
        warnings = [
            f"added {terminal_oxt_added} missing terminal OXT atom(s) internally for force-field preparation",
            *warnings,
        ]
    if openmm_hydrogens:
        warnings = list(warnings) + [
            "OpenMM/PDBFixer hydrogens were stripped before AmberTools; tleap assigns hydrogens and atom types"
        ]
    return warnings, metadata


@contextmanager
def _ambertools_workspace(work_dir: Optional[Path], keep: bool):
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        yield work_dir
        return
    if keep:
        persistent = Path(tempfile.mkdtemp(prefix="pheat-ambertools-"))
        yield persistent
        return
    with tempfile.TemporaryDirectory(prefix="pheat-ambertools-") as temporary:
        yield Path(temporary)


@contextmanager
def _gromacs_workspace(work_dir: Optional[Path], keep: bool):
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        yield work_dir
        return
    if keep:
        persistent = Path(tempfile.mkdtemp(prefix="pheat-gromacs-"))
        yield persistent
        return
    with tempfile.TemporaryDirectory(prefix="pheat-gromacs-") as temporary:
        yield Path(temporary)


def _require_gromacs_executable() -> str:
    gmx = shutil.which("gmx")
    if gmx is None:
        raise RuntimeError("gromacs-mdrun scoring requires the GROMACS gmx executable on PATH")
    return gmx


def _validate_gromacs_forcefield_available(gmx: str, work_dir: Path, forcefield: str) -> None:
    installed = _installed_gromacs_forcefields(gmx, work_dir)
    if not installed or forcefield in installed:
        return
    installed_list = ", ".join(installed)
    message = (
        f"GROMACS force field '{forcefield}' was not found in the active GROMACS installation. "
        f"Installed force fields: {installed_list}. "
        "Install or point GMXLIB at force-field files that provide the requested name, "
        "or pass --gromacs-forcefield with one of the installed force fields."
    )
    if forcefield == DEFAULT_GROMACS_FORCEFIELD and "amber99sb-ildn" in installed:
        message += " The conda-forge GROMACS package on some platforms currently provides amber99sb-ildn."
    raise RuntimeError(message)


def _installed_gromacs_forcefields(gmx: str, work_dir: Path) -> list[str]:
    directories = [work_dir]
    gmxlibrary = os.environ.get("GMXLIB")
    if gmxlibrary:
        directories.append(Path(gmxlibrary))
    data_prefix = _gromacs_data_prefix(gmx, work_dir)
    if data_prefix is not None:
        directories.append(data_prefix / "share" / "gromacs" / "top")

    forcefields: set[str] = set()
    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_dir() and child.name.endswith(".ff") and (child / "forcefield.itp").exists():
                forcefields.add(child.name[:-3])
    return sorted(forcefields)


def _gromacs_data_prefix(gmx: str, work_dir: Path) -> Optional[Path]:
    try:
        result = _run_subprocess([gmx, "--version"], cwd=work_dir)
    except RuntimeError:  # pragma: no cover - depends on local GROMACS behavior
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Data prefix:"):
            value = stripped.split(":", 1)[1].strip()
            return Path(value) if value else None
    return None


def _gromacs_citations(forcefield: str) -> list[str]:
    citations = ["gromacs"]
    if forcefield == "amber19sb":
        citations.append("amber-ff19sb")
    elif forcefield == "amber14sb":
        citations.append("amber-ff14sb")
    return citations


def _external_prep_cache_metadata(
    *,
    backend: str,
    structure: HeavyAtomStructure,
    cache_dir: Optional[Path],
    cache_mode: str,
    settings: Mapping[str, Any],
    reusable: bool,
    disabled_reason: str,
) -> dict[str, Any]:
    mode = _normalize_prep_cache_mode(cache_mode)
    payload = {
        "backend": backend,
        "mode": mode,
        "status": "disabled",
        "reusable": bool(reusable),
        "key": None,
        "path": None,
        "reason": None,
    }
    if mode == "off":
        payload["reason"] = "cache mode is off"
        return payload
    if cache_dir is None:
        payload["reason"] = "cache directory was not provided"
        return payload
    signature = {
        "format": "pheat.external-prep-cache-key",
        "version": 1,
        "backend": backend,
        "settings": _json_safe(settings),
        "structure_topology": _structure_topology_signature(structure),
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = Path(cache_dir) / backend / key
    payload.update({"status": "enabled", "key": key, "path": str(path)})
    if not reusable:
        payload.update({"status": "disabled", "reason": disabled_reason})
    elif mode == "readonly" and not path.exists():
        payload.update({"status": "miss", "reason": "readonly cache entry was not found"})
    return payload


def _restore_gromacs_prep_cache(cache_metadata: Mapping[str, Any], topology_path: Path) -> bool:
    if cache_metadata.get("status") not in {"enabled", "miss"}:
        return False
    if cache_metadata.get("mode") == "refresh":
        return False
    cache_path = Path(str(cache_metadata.get("path") or ""))
    cached_topology = cache_path / "topol.top"
    if not cached_topology.exists():
        return False
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_topology, topology_path)
    return True


def _store_gromacs_prep_cache(cache_metadata: Mapping[str, Any], topology_path: Path) -> None:
    if cache_metadata.get("status") not in {"miss", "refresh"}:
        return
    if cache_metadata.get("mode") == "readonly":
        return
    cache_path_value = cache_metadata.get("path")
    if not cache_path_value:
        return
    cache_path = Path(str(cache_path_value))
    cache_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(topology_path, cache_path / "topol.top")
    (cache_path / "metadata.json").write_text(
        json.dumps(dict(cache_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _structure_topology_signature(structure: HeavyAtomStructure) -> dict[str, Any]:
    return {
        "atom_scope": structure.atom_scope,
        "atom_count": len(structure.atoms),
        "atoms": [
            {
                "name": atom.name.strip().upper(),
                "element": atom.element.strip().upper(),
                "resname": atom.resname.strip().upper(),
                "chain_id": atom.chain_id,
                "resseq": int(atom.resseq),
                "icode": atom.icode or "",
                "record_name": atom.record_name,
            }
            for atom in structure.atoms
        ],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _gromacs_version(gmx: str, work_dir: Path) -> str:
    try:
        result = _run_subprocess([gmx, "--version"], cwd=work_dir)
    except RuntimeError as exc:  # pragma: no cover - depends on local GROMACS behavior
        return f"unavailable ({exc})"
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("GROMACS version:"):
            return stripped.split(":", 1)[1].strip()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "unknown"


def _prepare_gromacs_system(
    structure: HeavyAtomStructure,
    work_dir: Path,
    *,
    gmx: str,
    prepare: str,
    gromacs_forcefield: str,
    gromacs_water: str,
    gromacs_solvate: bool,
    settings: GromacsRunSettings,
    timeout: Optional[float],
    prep_cache_dir: Optional[Path],
    prep_cache_mode: str,
    status_stream: Optional[Any] = None,
) -> dict[str, Any]:
    prepare_mode = _normalize_prepare_mode(prepare)
    forcefield = _normalize_gromacs_forcefield(gromacs_forcefield)
    _validate_gromacs_forcefield_available(gmx, work_dir, forcefield)
    water = _effective_gromacs_water(gromacs_water, solvate=gromacs_solvate)
    cache_metadata = _external_prep_cache_metadata(
        backend="gromacs",
        structure=structure,
        cache_dir=prep_cache_dir,
        cache_mode=prep_cache_mode,
        settings={
            "prepare": prepare_mode,
            "gromacs_forcefield": forcefield,
            "gromacs_water": water,
            "gromacs_solvate": bool(gromacs_solvate),
        },
        reusable=prepare_mode == "never" and _hydrogen_atom_count(structure) > 0,
        disabled_reason="GROMACS cache reuse requires --prepare never with hydrogens and the same atom order",
    )
    input_pdb = work_dir / "input.pdb"
    preparation_warnings, preparation_metadata = _write_pdb_for_gromacs_pdb2gmx(
        structure,
        input_pdb,
        prepare=prepare_mode,
    )

    prepared_gro = work_dir / "prepared.gro"
    topology = work_dir / "topol.top"
    command_lines: dict[str, list[str]] = {}
    cache_hit = _restore_gromacs_prep_cache(cache_metadata, topology)
    if cache_hit:
        prepared_gro = input_pdb
        cache_metadata["status"] = "hit"
    else:
        if prep_cache_mode == "readonly" and cache_metadata.get("status") == "miss":
            raise RuntimeError(f"GROMACS prep cache entry was not found: {cache_metadata.get('path')}")
        if cache_metadata["status"] == "enabled":
            cache_metadata["status"] = "refresh" if prep_cache_mode == "refresh" else "miss"
        pdb2gmx_command = [
            gmx,
            "pdb2gmx",
            "-f",
            str(input_pdb.name),
            "-o",
            str(prepared_gro.name),
            "-p",
            str(topology.name),
            "-ff",
            forcefield,
            "-water",
            water,
            "-ignh",
        ]
        _status_print(status_stream, "GROMACS prep: pdb2gmx")
        _run_subprocess(
            pdb2gmx_command,
            cwd=work_dir,
            input_text="\n",
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS pdb2gmx",
        )
        if not prepared_gro.exists() or not topology.exists():
            raise RuntimeError("GROMACS pdb2gmx did not produce prepared.gro/topol.top.")
        command_lines["pdb2gmx"] = pdb2gmx_command
        _store_gromacs_prep_cache(cache_metadata, topology)
    boxed = work_dir / "boxed.gro"
    editconf_command = [
        gmx,
        "editconf",
        "-f",
        str(prepared_gro.name),
        "-o",
        str(boxed.name),
        "-c",
        "-d",
        str(settings.box_distance_nm),
        "-bt",
        "dodecahedron",
    ]
    _status_print(status_stream, "GROMACS prep: editconf")
    _run_subprocess(
        editconf_command,
        cwd=work_dir,
        timeout=timeout,
        status_stream=status_stream,
        status_label="GROMACS editconf",
    )
    if not boxed.exists():
        raise RuntimeError("GROMACS editconf did not produce boxed.gro.")
    coordinate = boxed
    command_lines["editconf"] = editconf_command
    if gromacs_solvate:
        solvated = work_dir / "solvated.gro"
        solvate_command = [
            gmx,
            "solvate",
            "-cp",
            str(boxed.name),
            "-cs",
            "spc216.gro",
            "-o",
            str(solvated.name),
            "-p",
            str(topology.name),
        ]
        _status_print(status_stream, "GROMACS prep: solvate")
        _run_subprocess(
            solvate_command,
            cwd=work_dir,
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS solvate",
        )
        if not solvated.exists():
            raise RuntimeError("GROMACS solvate did not produce solvated.gro.")
        coordinate = solvated
        command_lines["solvate"] = solvate_command

    return {
        "coordinate_path": coordinate,
        "topology_path": topology,
        "warnings": preparation_warnings,
        "command_lines": command_lines,
        "metadata": {
            **preparation_metadata,
            "gromacs_forcefield": forcefield,
            "gromacs_water": water,
            "gromacs_water_requested": str(gromacs_water or DEFAULT_GROMACS_WATER).strip().lower(),
            "gromacs_solvate": bool(gromacs_solvate),
            "coordinate_file": str(coordinate),
            "topology_file": str(topology),
            "box_distance_nm": settings.box_distance_nm,
            "hydrogen_assignment": "gmx-pdb2gmx-forcefield-templates",
            "prep_cache": cache_metadata,
        },
    }


def _run_gromacs_energy_workflow(
    gmx: str,
    work_dir: Path,
    *,
    coordinate_path: Path,
    topology_path: Path,
    run_mode: str,
    metrics: bool,
    settings: GromacsRunSettings,
    timeout: Optional[float],
    status_stream: Optional[Any] = None,
) -> dict[str, Any]:
    command_lines: dict[str, list[str]] = {}
    warnings: list[str] = []
    final_coordinate = coordinate_path
    final_tpr = work_dir / "rerun.tpr"
    energy_edr = work_dir / "rerun.edr"

    if run_mode in {"minimize", "minimize-rerun"}:
        _status_print(status_stream, "GROMACS minimize: preparing")
        minimize_mdp = _write_gromacs_mdp(work_dir, mode="minimize", settings=settings)
        minimize_tpr = work_dir / "minimize.tpr"
        grompp_minimize = [
            gmx,
            "grompp",
            "-f",
            str(minimize_mdp.name),
            "-c",
            str(coordinate_path.name),
            "-p",
            str(topology_path.name),
            "-o",
            str(minimize_tpr.name),
        ]
        if settings.grompp_maxwarn:
            grompp_minimize.extend(["-maxwarn", str(settings.grompp_maxwarn)])
        _run_subprocess(
            grompp_minimize,
            cwd=work_dir,
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS grompp minimize",
        )
        mdrun_minimize = [gmx, "mdrun", "-deffnm", "minimize", "-s", str(minimize_tpr.name)]
        mdrun_minimize.extend(settings.mdrun_flags)
        _run_subprocess(
            mdrun_minimize,
            cwd=work_dir,
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS mdrun minimize",
        )
        minimize_log = work_dir / "minimize.log"
        minimize_text = minimize_log.read_text(encoding="utf-8", errors="replace") if minimize_log.exists() else ""
        force_matches = re.findall(
            r"Maximum force\s*=\s*([0-9.eE+\-]+)",
            minimize_text,
        )
        final_max_force = float(force_matches[-1]) if force_matches else math.nan
        if not math.isfinite(final_max_force) or final_max_force > float(settings.emtol):
            raise RuntimeError(
                "GROMACS minimization did not converge to the requested force threshold "
                f"(emtol={settings.emtol:g}, final Fmax={final_max_force:g})."
            )
        minimized = work_dir / "minimize.gro"
        if not minimized.exists():
            raise RuntimeError("GROMACS mdrun minimization did not produce minimize.gro.")
        final_coordinate = minimized
        final_tpr = minimize_tpr
        energy_edr = work_dir / "minimize.edr"
        command_lines["grompp_minimize"] = grompp_minimize
        command_lines["mdrun_minimize"] = mdrun_minimize

    if run_mode in {"rerun", "minimize-rerun"}:
        _status_print(status_stream, "GROMACS rerun: preparing")
        rerun_mdp = _write_gromacs_mdp(work_dir, mode="rerun", settings=settings)
        rerun_tpr = work_dir / "rerun.tpr"
        grompp_rerun = [
            gmx,
            "grompp",
            "-f",
            str(rerun_mdp.name),
            "-c",
            str(final_coordinate.name),
            "-p",
            str(topology_path.name),
            "-o",
            str(rerun_tpr.name),
        ]
        if settings.grompp_maxwarn:
            grompp_rerun.extend(["-maxwarn", str(settings.grompp_maxwarn)])
        _run_subprocess(
            grompp_rerun,
            cwd=work_dir,
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS grompp rerun",
        )
        mdrun_rerun = [
            gmx,
            "mdrun",
            "-deffnm",
            "rerun",
            "-s",
            str(rerun_tpr.name),
            "-rerun",
            str(final_coordinate.name),
        ]
        mdrun_rerun.extend(settings.mdrun_flags)
        _run_subprocess(
            mdrun_rerun,
            cwd=work_dir,
            timeout=timeout,
            status_stream=status_stream,
            status_label="GROMACS mdrun rerun",
        )
        final_tpr = rerun_tpr
        energy_edr = work_dir / "rerun.edr"
        command_lines["grompp_rerun"] = grompp_rerun
        command_lines["mdrun_rerun"] = mdrun_rerun

    if not energy_edr.exists():
        raise RuntimeError(f"GROMACS run did not produce {energy_edr.name}.")
    terms, energy_command, energy_warnings = _extract_gromacs_energy_terms(
        gmx,
        work_dir,
        energy_edr=energy_edr,
        timeout=timeout,
    )
    potential = terms.get("potential", terms.get("total_energy"))
    if potential is None or not math.isfinite(float(potential)) or abs(float(potential)) > 1.0e9:
        raise RuntimeError("GROMACS produced a non-finite or physically invalid potential energy.")
    command_lines["energy"] = energy_command
    warnings.extend(energy_warnings)
    gromacs_metrics: dict[str, Any] = {}
    if metrics:
        metric_payload, metric_command, metric_warnings = _run_gromacs_gyrate(
            gmx,
            work_dir,
            coordinate_path=final_coordinate,
            tpr_path=final_tpr,
            timeout=timeout,
        )
        if metric_command:
            command_lines["gyrate"] = metric_command
        warnings.extend(metric_warnings)
        gromacs_metrics.update(metric_payload)
    return {
        "terms": terms,
        "warnings": warnings,
        "command_lines": command_lines,
        "metrics": gromacs_metrics,
        "final_coordinate_path": final_coordinate,
        "final_tpr_path": final_tpr,
        "energy_edr_path": energy_edr,
    }


def _write_tleap_input(
    work_dir: Path,
    *,
    amber_forcefield: str,
    amber_pbradii: Optional[str] = None,
) -> Path:
    commands = [f"source {amber_forcefield}"]
    if amber_pbradii:
        commands.append(f"set default PBRadii {amber_pbradii}")
    commands.extend(
        [
            "mol = loadpdb prepared.pdb",
            "check mol",
            "saveamberparm mol system.prmtop system.inpcrd",
            "quit",
            "",
        ]
    )
    path = work_dir / "tleap.in"
    path.write_text("\n".join(commands), encoding="utf-8")
    return path


def _amber_pbradii_for_solvent(solvent: str) -> Optional[str]:
    if _normalize_amber_solvent(solvent) == "gb":
        return "mbondi3"
    return None


def _write_sander_input(work_dir: Path, *, solvent: str) -> Path:
    igb = "8" if solvent == "gb" else "0"
    path = work_dir / "sander.in"
    path.write_text(
        "\n".join(
            [
                "PHEAT AmberTools single-point score",
                "&cntrl",
                "  imin=1, maxcyc=0, ntb=0,",
                f"  igb={igb}, cut=999.0,",
                "/",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_pdb_for_gromacs_pdb2gmx(
    structure: HeavyAtomStructure,
    output_pdb: Path,
    *,
    prepare: str,
) -> tuple[list[str], dict[str, object]]:
    """Write the PDB that GROMACS ``pdb2gmx`` should parameterize."""

    from pheat.pdbio import structure_to_pdb_string

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    input_hydrogens = _hydrogen_atom_count(structure)
    direct_metadata = {
        "mode": prepare,
        "engine": "input",
        "input_atom_count": len(structure.atoms),
        "input_hydrogen_atom_count": input_hydrogens,
        "prepared_atom_count": len(structure.atoms),
        "prepared_hydrogen_atom_count": input_hydrogens,
    }
    if prepare == "never":
        output_pdb.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        return [], direct_metadata

    prepared_structure, terminal_oxt_added = _with_forcefield_terminal_oxt(structure)

    try:
        from openmm import Platform, app
    except Exception as exc:  # pragma: no cover - depends on optional user environment
        output_pdb.write_text(
            structure_to_pdb_string(prepared_structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        warnings = [
            "OpenMM/PDBFixer was unavailable; GROMACS pdb2gmx will prepare the supplied atoms directly"
        ]
        if terminal_oxt_added:
            warnings.insert(
                0,
                f"added {terminal_oxt_added} missing terminal OXT atom(s) internally for force-field preparation",
            )
        return warnings, {
            **direct_metadata,
            "terminal_oxt_added": terminal_oxt_added,
            "openmm_pdbfixer_unavailable": str(exc),
        }

    with tempfile.TemporaryDirectory(prefix="pheat-gmx-prep-") as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        pdb_path.write_text(
            structure_to_pdb_string(prepared_structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        platform = Platform.getPlatformByName("Reference")
        topology, positions, warnings, metadata = _prepare_openmm_topology(
            str(pdb_path),
            app,
            forcefield,
            platform,
        )
        with output_pdb.open("w", encoding="utf-8") as prepared_handle:
            app.PDBFile.writeFile(topology, positions, prepared_handle)
    removed_openmm_hydrogens = _strip_pdb_hydrogens(output_pdb)
    topology_any: Any = topology
    prepared_atom_count = len(list(topology_any.atoms()))
    metadata = {
        **metadata,
        "mode": prepare,
        "engine": "openmm-pdbfixer",
        "input_atom_count": len(structure.atoms),
        "input_hydrogen_atom_count": input_hydrogens,
        "prepared_atom_count": prepared_atom_count - removed_openmm_hydrogens,
        "prepared_hydrogen_atom_count": 0,
        "openmm_hydrogens_removed_before_pdb2gmx": removed_openmm_hydrogens,
        "prepared_coordinate_count": prepared_atom_count - removed_openmm_hydrogens,
        "pH": 7.0,
        "terminal_oxt_added": terminal_oxt_added,
    }
    terminal_warnings = (
        [f"added {terminal_oxt_added} missing terminal OXT atom(s) internally for force-field preparation"]
        if terminal_oxt_added
        else []
    )
    return terminal_warnings + list(warnings) + [
        "OpenMM/PDBFixer hydrogens were stripped before GROMACS; pdb2gmx assigns force-field hydrogens and topology"
    ], metadata


def _write_gromacs_mdp(work_dir: Path, *, mode: str, settings: GromacsRunSettings) -> Path:
    path = work_dir / f"{mode}.mdp"
    if mode == "minimize":
        lines = [
            "integrator = steep",
            f"emtol = {settings.emtol}",
            f"emstep = {settings.emstep}",
            f"nsteps = {settings.minimize_steps}",
        ]
    else:
        lines = [
            "integrator = md",
            "nsteps = 0",
            "dt = 0.001",
        ]
    path.write_text(
        "\n".join(
            [
                "; PHEAT GROMACS validation/reranking settings",
                *lines,
                "constraints = none",
                "constraint_algorithm = lincs",
                "continuation = yes",
                "cutoff-scheme = Verlet",
                f"nstlist = {settings.nstlist}",
                f"pbc = {settings.pbc}",
                f"comm-mode = {settings.comm_mode}",
                f"coulombtype = {settings.coulombtype}",
                f"vdwtype = {settings.vdwtype}",
                f"rlist = {settings.cutoff_nm}",
                f"rcoulomb = {settings.cutoff_nm}",
                f"rvdw = {settings.cutoff_nm}",
                "nstenergy = 1",
                "nstlog = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _extract_gromacs_energy_terms(
    gmx: str,
    work_dir: Path,
    *,
    energy_edr: Path,
    timeout: Optional[float],
) -> tuple[dict[str, float], list[str], list[str]]:
    xvg_path = work_dir / "energy.xvg"
    command = [gmx, "energy", "-f", str(energy_edr.name), "-o", str(xvg_path.name)]
    selection = "\n".join(
        [
            "Potential",
            "Total-Energy",
            "Bond",
            "Angle",
            "Proper-Dih.",
            "Improper-Dih.",
            "LJ-14",
            "Coulomb-14",
            "LJ-(SR)",
            "Coulomb-(SR)",
            "Disper.-corr.",
            "0",
            "",
        ]
    )
    warnings: list[str] = []
    try:
        _run_subprocess(command, cwd=work_dir, input_text=selection, timeout=timeout)
    except RuntimeError as exc:
        warnings.append(
            "GROMACS energy term extraction fell back to Potential only after broad term selection failed: "
            + str(exc)
        )
        _run_subprocess(command, cwd=work_dir, input_text="Potential\n0\n", timeout=timeout)
    if not xvg_path.exists():
        raise RuntimeError("GROMACS energy did not produce energy.xvg.")
    return parse_gromacs_xvg_terms(xvg_path.read_text(encoding="utf-8", errors="replace")), command, warnings


def _run_gromacs_gyrate(
    gmx: str,
    work_dir: Path,
    *,
    coordinate_path: Path,
    tpr_path: Path,
    timeout: Optional[float],
) -> tuple[dict[str, Any], Optional[list[str]], list[str]]:
    xvg_path = work_dir / "gyrate.xvg"
    command = [
        gmx,
        "gyrate",
        "-f",
        str(coordinate_path.name),
        "-s",
        str(tpr_path.name),
        "-o",
        str(xvg_path.name),
    ]
    try:
        _run_subprocess(command, cwd=work_dir, input_text="System\n", timeout=timeout)
    except RuntimeError as exc:
        return {}, command, [f"GROMACS gyrate metric was unavailable: {exc}"]
    if not xvg_path.exists():
        return {}, command, ["GROMACS gyrate did not produce gyrate.xvg."]
    terms = parse_gromacs_xvg_terms(xvg_path.read_text(encoding="utf-8", errors="replace"))
    return {"radius_of_gyration": terms}, command, []


def parse_gromacs_xvg_terms(output_text: str) -> dict[str, float]:
    """Parse the last frame of a GROMACS XVG series into normalized term names."""

    legends: dict[int, str] = {}
    data_rows: list[list[float]] = []
    for line in output_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        legend = re.search(r'@\s+s(\d+)\s+legend\s+"([^"]+)"', stripped)
        if legend:
            legends[int(legend.group(1))] = _normalize_gromacs_term_name(legend.group(2))
            continue
        if stripped.startswith(("#", "@")):
            continue
        try:
            data_rows.append([float(part) for part in stripped.split()])
        except ValueError:
            continue
    if not data_rows:
        return {}
    last = data_rows[-1]
    if not legends and len(last) == 2:
        return {"potential": float(last[1])}
    terms: dict[str, float] = {}
    for index, value in enumerate(last[1:]):
        name = legends.get(index, f"series_{index}")
        terms[name] = float(value)
    return terms


def _normalize_gromacs_term_name(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("(sr)", "sr").replace("(lr)", "lr")
    normalized = normalized.replace(" ", "_").replace("-", "_").replace(".", "")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if normalized in {"potential", "total_energy"}:
        return normalized
    return normalized


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    input_text: Optional[str] = None,
    timeout: Optional[float] = None,
    status_stream: Optional[Any] = None,
    status_label: Optional[str] = None,
    heartbeat_seconds: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {' '.join(command)}") from exc

    started_at = time.monotonic()
    deadline = (started_at + float(timeout)) if timeout is not None else None
    stdin_payload = input_text
    first_call = True
    label = status_label or "subprocess"
    _status_print(status_stream, f"{label} started: {' '.join(command)}")
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                proc.kill()
                stdout, stderr = proc.communicate()
                exc = subprocess.TimeoutExpired(list(command), timeout)
                exc.stdout = stdout
                exc.stderr = stderr
                raise RuntimeError(
                    f"Command timed out after {timeout} seconds: {' '.join(command)}. "
                    + _timeout_tail(exc)
                ) from exc
            wait_for = min(float(heartbeat_seconds), remaining) if status_stream is not None else remaining
        else:
            wait_for = float(heartbeat_seconds) if status_stream is not None else None
        try:
            if first_call:
                stdout, stderr = proc.communicate(input=stdin_payload, timeout=wait_for)
                first_call = False
                stdin_payload = None
            else:
                stdout, stderr = proc.communicate(timeout=wait_for)
            result = subprocess.CompletedProcess(list(command), proc.returncode or 0, stdout, stderr)
            break
        except subprocess.TimeoutExpired:
            if status_stream is not None:
                elapsed = time.monotonic() - started_at
                _status_print(status_stream, f"{label} heartbeat: {elapsed:.1f}s elapsed")
            stdin_payload = None
            continue
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}. "
            + _subprocess_tail(result)
        )
    _status_print(status_stream, f"{label} completed in {time.monotonic() - started_at:.1f}s")
    return result


def _external_command_error_with_file_tail(exc: RuntimeError, path: Path, label: str) -> str:
    message = str(exc)
    if not path.exists():
        return message
    tail = _file_tail(path)
    if not tail:
        return message
    return f"{message}\nLast lines from {label}:\n{tail}"


def _file_tail(path: Path, line_count: int = 20) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-line_count:])


def _timeout_tail(exc: subprocess.TimeoutExpired) -> str:
    parts = []
    for value in (exc.stdout, exc.stderr):
        decoded: Optional[str]
        if isinstance(value, bytes):
            decoded = value.decode("utf-8", errors="replace")
        else:
            decoded = value
        if decoded:
            parts.append(str(decoded))
    text = "\n".join(parts)
    lines = text.splitlines()
    return "\n".join(lines[-20:]) if lines else ""


def _subprocess_tail(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    lines = text.splitlines()
    return "\n".join(lines[-20:]) if lines else ""


def _status_print(status_stream: Optional[Any], message: str) -> None:
    if status_stream is None:
        return
    if hasattr(status_stream, "status"):
        status_stream.status(message)
        return
    print(message, file=status_stream, flush=True)


def _normalize_prepare_mode(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in PREPARE_MODES:
        raise ValueError(f"prepare mode must be one of {', '.join(PREPARE_MODES)}")
    return normalized


def _normalize_amber_solvent(value: str) -> str:
    normalized = str(value or "vacuum").strip().lower()
    if normalized not in AMBER_SOLVENT_MODES:
        raise ValueError(f"Amber solvent mode must be one of {', '.join(AMBER_SOLVENT_MODES)}")
    return normalized


def _normalize_external_timeout(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0.0:
        raise ValueError("external timeout must be greater than zero seconds")
    return timeout


def _normalize_prep_cache_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower()
    if normalized not in PREP_CACHE_MODES:
        raise ValueError(f"prep cache mode must be one of {', '.join(PREP_CACHE_MODES)}")
    return normalized


def _normalize_gromacs_forcefield(value: str) -> str:
    normalized = str(value or DEFAULT_GROMACS_FORCEFIELD).strip().lower()
    if not normalized:
        raise ValueError("GROMACS force field must be a non-empty pdb2gmx force-field short name")
    return normalized


def _effective_gromacs_water(value: str, *, solvate: bool) -> str:
    normalized = str(value or DEFAULT_GROMACS_WATER).strip().lower()
    if normalized not in GROMACS_WATER_MODELS:
        raise ValueError(f"GROMACS water model must be one of {', '.join(GROMACS_WATER_MODELS)}")
    if normalized == "auto":
        return "opc" if solvate else "none"
    return normalized


def _normalize_gromacs_run_mode(value: str) -> str:
    normalized = str(value or "rerun").strip().lower()
    if normalized not in GROMACS_RUN_MODES:
        raise ValueError(f"GROMACS run mode must be one of {', '.join(GROMACS_RUN_MODES)}")
    return normalized


def _normalize_gromacs_preflight(value: str) -> str:
    normalized = str(value or "strict").strip().lower()
    if normalized not in GROMACS_PREFLIGHT_MODES:
        raise ValueError(f"GROMACS preflight must be one of {', '.join(GROMACS_PREFLIGHT_MODES)}")
    return normalized


def _normalize_gromacs_run_settings(
    value: Optional[Union[GromacsRunSettings, Mapping[str, Any]]],
) -> GromacsRunSettings:
    if value is None:
        settings = GromacsRunSettings()
    elif isinstance(value, GromacsRunSettings):
        settings = value
    else:
        data = dict(value)
        if "mdrun_flags" in data and data["mdrun_flags"] is not None:
            data["mdrun_flags"] = tuple(str(item) for item in data["mdrun_flags"])
        settings = GromacsRunSettings(**data)
    if settings.minimize_steps < 0:
        raise ValueError("GROMACS minimize steps must be non-negative")
    if settings.emtol <= 0.0:
        raise ValueError("GROMACS emtol must be greater than zero")
    if settings.emstep <= 0.0:
        raise ValueError("GROMACS emstep must be greater than zero")
    if settings.box_distance_nm <= 0.0:
        raise ValueError("GROMACS box distance must be greater than zero")
    if settings.cutoff_nm <= 0.0:
        raise ValueError("GROMACS cutoff must be greater than zero")
    if settings.nstlist <= 0:
        raise ValueError("GROMACS nstlist must be greater than zero")
    if settings.grompp_maxwarn < 0:
        raise ValueError("GROMACS grompp maxwarn must be non-negative")
    if not settings.coulombtype.strip():
        raise ValueError("GROMACS coulombtype must be non-empty")
    if not settings.vdwtype.strip():
        raise ValueError("GROMACS vdwtype must be non-empty")
    if not settings.pbc.strip():
        raise ValueError("GROMACS pbc must be non-empty")
    if not settings.comm_mode.strip():
        raise ValueError("GROMACS comm-mode must be non-empty")
    flags = tuple(str(flag).strip() for flag in settings.mdrun_flags if str(flag).strip())
    return GromacsRunSettings(
        minimize_steps=int(settings.minimize_steps),
        emtol=float(settings.emtol),
        emstep=float(settings.emstep),
        box_distance_nm=float(settings.box_distance_nm),
        cutoff_nm=float(settings.cutoff_nm),
        coulombtype=str(settings.coulombtype).strip(),
        vdwtype=str(settings.vdwtype).strip(),
        nstlist=int(settings.nstlist),
        pbc=str(settings.pbc).strip(),
        comm_mode=str(settings.comm_mode).strip(),
        mdrun_flags=flags,
        grompp_maxwarn=int(settings.grompp_maxwarn),
    )


def _hydrogen_atom_count(structure: HeavyAtomStructure) -> int:
    return sum(1 for atom in structure.atoms if atom.element.strip().upper() in {"H", "D", "T"})


def _openmm_hydrogen_atom_count(topology) -> int:
    count = 0
    for atom in topology.atoms():
        element = getattr(getattr(atom, "element", None), "symbol", "") or ""
        if str(element).upper() in {"H", "D", "T"}:
            count += 1
    return count


def _strip_pdb_hydrogens(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    removed = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and _pdb_line_is_hydrogen(line):
            removed += 1
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def _pdb_line_is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element in {"H", "D", "T"}:
        return True
    atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
    return atom_name.startswith("H") or (len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == "H")


def _with_python_random_seed(seed: int, callback):
    state = random.getstate()
    random.seed(seed)
    try:
        return callback()
    finally:
        random.setstate(state)


def _nonlocal_pairs(atoms: Sequence[Atom], *, cutoff: Optional[float] = None) -> Iterable[Tuple[Atom, Atom]]:
    """Yield atom pairs while skipping local bonded-neighbor contacts."""

    if cutoff is not None:
        yield from _spatial_nonlocal_pairs(atoms, cutoff=cutoff)
        return

    for atom_a, atom_b in itertools.combinations(atoms, 2):
        if atom_a.chain_id == atom_b.chain_id and abs(atom_a.resseq - atom_b.resseq) <= 1:
            continue
        yield atom_a, atom_b


def _spatial_nonlocal_pairs(atoms: Sequence[Atom], *, cutoff: float) -> Iterable[Tuple[Atom, Atom]]:
    """Yield nonlocal atom pairs within ``cutoff`` using a simple cell list."""

    if cutoff <= 0.0:
        return
    grid: dict[tuple[int, int, int], list[tuple[int, Atom]]] = {}
    for index, atom in enumerate(atoms):
        grid.setdefault(_spatial_cell(atom.coord, cutoff), []).append((index, atom))

    cutoff_squared = cutoff * cutoff
    for index, atom_a in enumerate(atoms):
        cell = _spatial_cell(atom_a.coord, cutoff)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for other_index, atom_b in grid.get(neighbor_cell, []):
                        if other_index <= index:
                            continue
                        if atom_a.chain_id == atom_b.chain_id and abs(atom_a.resseq - atom_b.resseq) <= 1:
                            continue
                        delta = sub(atom_a.coord, atom_b.coord)
                        if dot(delta, delta) <= cutoff_squared:
                            yield atom_a, atom_b


def _spatial_cell(coord: tuple[float, float, float], cell_size: float) -> tuple[int, int, int]:
    return (int(coord[0] // cell_size), int(coord[1] // cell_size), int(coord[2] // cell_size))


def _radius(atom: Atom) -> float:
    return VDW_RADII.get(atom.element.upper(), 1.70)


def _element_weight(atom: Atom) -> float:
    return ELEMENT_CONTACT_WEIGHT.get(atom.element.upper(), -0.02)


def _charge(atom: Atom) -> float:
    return ATOM_CHARGES.get(atom.name.strip().upper(), 0.0)


def _unsupported_warnings(structure: HeavyAtomStructure) -> List[str]:
    residue_names = sorted({atom.resname for atom in structure.atoms if atom.resname not in CANONICAL_RESIDUES})
    if not residue_names:
        return []
    supported = [name for name in residue_names if name in SUPPORTED_RESIDUES]
    unsupported = [name for name in residue_names if name not in SUPPORTED_RESIDUES]
    warnings = []
    if supported:
        warnings.append(
            "modified/special residues are reconstructable but outside canonical residue statistical terms: "
            + ", ".join(supported)
        )
    if unsupported:
        warnings.append(
            "heterogens or unsupported residues handled only by generic element terms where available: "
            + ", ".join(unsupported)
        )
    return warnings


def _bonded_backbone_penalty(structure: HeavyAtomStructure) -> float:
    penalty = 0.0
    lookup = structure.atom_lookup()
    for key in structure.residue_keys():
        for first, second, target in [("N", "CA", 1.458), ("CA", "C", 1.525), ("C", "O", 1.231)]:
            atom_a = lookup.get((key, first))
            atom_b = lookup.get((key, second))
            if atom_a and atom_b:
                penalty += (distance(atom_a.coord, atom_b.coord) - target) ** 2 * 10.0
    residue_keys = structure.residue_keys()
    for left, right in zip(residue_keys, residue_keys[1:]):
        if left[0] != right[0]:
            continue
        if right[1] - left[1] > 1:
            continue
        c_atom = lookup.get((left, "C"))
        n_atom = lookup.get((right, "N"))
        if c_atom and n_atom:
            penalty += (distance(c_atom.coord, n_atom.coord) - 1.329) ** 2 * 10.0
    return penalty


def _residue_orientation_vectors(structure: HeavyAtomStructure) -> Dict[Tuple[str, int, str, str, str], Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    vectors = {}
    lookup = structure.atom_lookup()
    for key in structure.residue_keys():
        ca_atom = lookup.get((key, "CA"))
        cb_atom = lookup.get((key, "CB"))
        n_atom = lookup.get((key, "N"))
        if ca_atom and cb_atom:
            vectors[key] = (ca_atom.coord, normalize(sub(cb_atom.coord, ca_atom.coord)))
        elif ca_atom and n_atom:
            vectors[key] = (ca_atom.coord, normalize(sub(ca_atom.coord, n_atom.coord)))
    return vectors


def _with_coverage(
    result: EnergyResult,
    coverage: Mapping[str, Any],
    table_set: Optional[Mapping[str, Any]] = None,
    profile: Optional[str] = None,
    *,
    table: Optional[Mapping[str, Any]] = None,
    contract: Optional[Mapping[str, Any]] = None,
    contract_warnings: Sequence[str] = (),
) -> EnergyResult:
    result.metadata = merge_coverage_metadata(result.metadata, coverage)
    result.metadata["implementation"] = _implementation_metadata_for_model(result.model)
    if contract is not None:
        result.metadata["input_contract"] = dict(contract)
    result.warnings.extend(contract_warnings)
    if table is not None:
        result.warnings.extend(_table_contract_warnings(table, contract))
    if table_set is not None:
        result.metadata["profile"] = profile or default_profile(table_set)
    return result


def _score_contract_warnings(
    model: str,
    contract: Mapping[str, Any],
    original: HeavyAtomStructure,
    filtered: HeavyAtomStructure,
    coverage: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    domain = str(coverage.get("domain") or "")
    compatible_domains = [str(item) for item in contract.get("compatible_domains", [])]
    if compatible_domains and domain not in compatible_domains:
        warnings.append(
            f"{model} input contract is calibrated for domains {', '.join(compatible_domains)}; "
            f"requested domain {domain}"
        )
    input_scope = original.atom_scope
    accepted_scopes = [str(item) for item in contract.get("accepted_atom_scope", [])]
    if accepted_scopes and input_scope not in accepted_scopes:
        warnings.append(
            f"{model} input contract accepts atom scopes {', '.join(accepted_scopes)}; "
            f"input atom_scope is {input_scope}"
        )
    warnings.extend(_required_atom_warnings(model, contract, filtered))
    return warnings


def _required_atom_warnings(
    model: str,
    contract: Mapping[str, Any],
    structure: HeavyAtomStructure,
) -> list[str]:
    requirements = {str(item) for item in contract.get("required_atoms", [])}
    warnings: list[str] = []
    residues = structure.atoms_by_residue()
    if "backbone-n-ca-c" in requirements:
        missing = _missing_residue_atom_count(residues, {"N", "CA", "C"})
        if missing:
            warnings.append(f"{model} input contract expects N/CA/C backbone atoms; {missing} residues are incomplete")
    if "backbone-n-ca-c-o" in requirements:
        missing = _missing_residue_atom_count(residues, {"N", "CA", "C", "O"})
        if missing:
            warnings.append(f"{model} input contract expects N/CA/C/O backbone atoms; {missing} residues are incomplete")
    if "peptide-link-c-n" in requirements and not _has_any_peptide_link(residues):
        warnings.append(f"{model} input contract expects adjacent peptide C-N links; none were scored")
    if "ca-chirality-atoms" in requirements:
        chirality_count = sum(
            1
            for atoms in residues.values()
            if {atom.name.strip().upper() for atom in atoms}.issuperset({"N", "CA", "C", "CB"})
        )
        if chirality_count == 0:
            warnings.append(f"{model} input contract expects N/CA/C/CB atoms for C-alpha chirality; none were scored")
    if "residue-orientation-atoms" in requirements and not _residue_orientation_vectors(structure):
        warnings.append(f"{model} input contract expects CA plus CB or N atoms for residue orientation vectors")
    if "sidechain-heavy-atoms" in requirements:
        sidechain_count = sum(
            1
            for atoms in residues.values()
            for atom in atoms
            if atom.name.strip().upper() not in {"N", "CA", "C", "O", "OXT"}
        )
        if sidechain_count == 0:
            warnings.append(f"{model} input contract expects side-chain heavy atoms; none were scored")
    if "polar-heavy-atoms" in requirements:
        polar_count = sum(1 for atom in structure.atoms if atom.element.strip().upper() in {"N", "O", "S", "SE"})
        if polar_count == 0:
            warnings.append(f"{model} input contract expects polar heavy atoms; none were scored")
    return warnings


def _has_any_peptide_link(
    residues: Mapping[Tuple[str, int, str, str, str], Sequence[Atom]],
) -> bool:
    keys = list(residues)
    for left, right in zip(keys, keys[1:]):
        if left[0] != right[0]:
            continue
        left_names = {atom.name.strip().upper() for atom in residues[left]}
        right_names = {atom.name.strip().upper() for atom in residues[right]}
        if "C" in left_names and "N" in right_names:
            return True
    return False


def _missing_residue_atom_count(
    residues: Mapping[Tuple[str, int, str, str, str], Sequence[Atom]],
    required_atoms: set[str],
) -> int:
    missing = 0
    for atoms in residues.values():
        names = {atom.name.strip().upper() for atom in atoms}
        if required_atoms - names:
            missing += 1
    return missing


def _table_contract_warnings(
    table: Mapping[str, Any],
    contract: Optional[Mapping[str, Any]],
) -> list[str]:
    if contract is None:
        return []
    expected = str(contract.get("id") or "")
    table_contract = table.get("input_contract")
    actual = ""
    if isinstance(table_contract, Mapping):
        actual = str(table_contract.get("id") or "")
    elif table.get("input_contract_id"):
        actual = str(table.get("input_contract_id"))
    if actual and expected and actual != expected:
        return [f"score table input contract {actual} does not match scorer input contract {expected}"]
    return []


def _residue_representative_points(
    structure: HeavyAtomStructure,
) -> Dict[Tuple[str, int, str, str, str], Tuple[float, float, float]]:
    points = {}
    for key, atoms in structure.atoms_by_residue().items():
        sidechain = [
            atom.coord
            for atom in atoms
            if atom.name.strip().upper() not in {"N", "CA", "C", "O", "OXT"}
        ]
        if sidechain:
            points[key] = _centroid(sidechain)
            continue
        ca_atom = next((atom for atom in atoms if atom.name.strip().upper() == "CA"), None)
        if ca_atom is not None:
            points[key] = ca_atom.coord
    return points


def _centroid(coords: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    count = float(len(coords))
    return (
        sum(coord[0] for coord in coords) / count,
        sum(coord[1] for coord in coords) / count,
        sum(coord[2] for coord in coords) / count,
    )


def _residue_pair_key(left: str, right: str) -> str:
    return "-".join(sorted((left.strip().upper(), right.strip().upper())))


def _mj_pair_bias(left: str, right: str) -> float:
    left_value = KYTE_DOOLITTLE_HYDROPATHY.get(left.strip().upper(), 0.0)
    right_value = KYTE_DOOLITTLE_HYDROPATHY.get(right.strip().upper(), 0.0)
    return -0.05 * left_value * right_value


def _table_pair_weight(table: Optional[Mapping[str, Any]], pair: str) -> float:
    if not table:
        return 1.0
    counts = table.get("counts")
    if not isinstance(counts, Mapping):
        return 1.0
    value = counts.get(pair)
    if isinstance(value, (int, float)):
        return 1.0 + float(value) / 100.0
    return 1.0


def _table_pair_bin_weight(
    table: Optional[Mapping[str, Any]],
    left: str,
    right: str,
    bin_index: int,
) -> float:
    if not table:
        return 1.0
    counts = table.get("counts")
    if not isinstance(counts, Mapping):
        return 1.0
    pair_counts = counts.get(_residue_pair_key(left, right))
    if not isinstance(pair_counts, Mapping):
        return 1.0
    value = pair_counts.get(f"{bin_index:02d}")
    if isinstance(value, (int, float)):
        return 1.0 + float(value) / 100.0
    return 1.0


def _table_hydropathy_scale(table: Optional[Mapping[str, Any]]) -> Mapping[str, float]:
    if not table:
        return KYTE_DOOLITTLE_HYDROPATHY
    scale = table.get("hydropathy_scale")
    if not isinstance(scale, Mapping):
        return KYTE_DOOLITTLE_HYDROPATHY
    merged = dict(KYTE_DOOLITTLE_HYDROPATHY)
    for key, value in scale.items():
        try:
            merged[str(key).upper()] = float(value)
        except Exception:
            continue
    return merged


def _table_burial_method(table: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not table:
        return None
    value = table.get("burial_method")
    return str(value).lower() if value else None


def _pheat_rg_config(table: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    config = dict(PHEAT_RG_DEFAULTS)
    if table:
        for key in ("atom_set", "mode", "a", "b", "sigma_fraction", "fitted"):
            if key in table:
                config[key] = table[key]

    atom_set = normalize_rmsd_atom_set(str(config["atom_set"]))
    mode = str(config["mode"]).strip().lower().replace("_", "-")
    if mode not in {"unweighted", "mass-weighted"}:
        raise ValueError("pheat-rg mode must be one of unweighted, mass-weighted")

    try:
        a = float(config["a"])
        b = float(config["b"])
        sigma_fraction = float(config["sigma_fraction"])
    except (TypeError, ValueError) as exc:
        raise ValueError("pheat-rg coefficients a, b, and sigma_fraction must be numeric") from exc
    if a <= 0.0:
        raise ValueError("pheat-rg coefficient a must be greater than zero")
    if sigma_fraction <= 0.0:
        raise ValueError("pheat-rg sigma_fraction must be greater than zero")
    if atom_set not in RMSD_ATOM_SETS:
        raise ValueError(f"pheat-rg atom_set must be one of {', '.join(RMSD_ATOM_SETS)}")

    return {
        "atom_set": atom_set,
        "mode": mode,
        "a": a,
        "b": b,
        "sigma_fraction": sigma_fraction,
        "fitted": bool(config["fitted"]),
    }


def _table_source(table: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not table:
        return None
    return str(table.get("generated_by") or table.get("type") or "provided")


def _angle_outlier_penalty(phi: float, psi: float) -> float:
    # Broad helix/sheet-friendly wells; this is a deterministic plausibility term,
    # not a full Ramachandran probability table.
    wells = [(-60.0, -45.0), (-120.0, 130.0), (60.0, 40.0)]
    best = min(
        (_angle_delta(phi, well_phi) ** 2 + _angle_delta(psi, well_psi) ** 2) ** 0.5
        for well_phi, well_psi in wells
    )
    return best / 180.0


def _angle_delta(angle: float, target: float) -> float:
    delta = (float(angle) - float(target) + 180.0) % 360.0 - 180.0
    return delta

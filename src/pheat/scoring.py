"""Deterministic approximate heavy-atom scoring backends."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pheat.domains import filter_structure_for_domain, merge_coverage_metadata
from pheat.geometry import cross, dihedral_degrees, distance, dot, normalize, sub
from pheat.metrics import RMSD_ATOM_SETS, normalize_rmsd_atom_set, structure_radius_of_gyration
from pheat.models import Atom, EnergyResult, HeavyAtomStructure
from pheat.residue_geometry import C_N, C_O, CA_C, N_CA, PRO_N_CD
from pheat.residues import CANONICAL_RESIDUES, SUPPORTED_RESIDUES
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
GENERIC_CONTACT_CUTOFF = 6.0
PHEAT_DISTANCE_CONTACT_CUTOFF = 15.0
HEAVY_MM_NONBONDED_CUTOFF = 15.0
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
    "pheat-mj",
    "pheat-hydropathy",
    "pheat-backbone",
    "pheat-rotamer",
    "pheat-hbond",
    "pheat-rg",
    "pheat-ml-linear",
    "pheat-geometry-integrity",
    "heavy-mm",
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
    "pheat-mj": _native_model_metadata("arbitrary"),
    "pheat-hydropathy": _native_model_metadata("arbitrary"),
    "pheat-backbone": _native_model_metadata("arbitrary"),
    "pheat-rotamer": _native_model_metadata("arbitrary"),
    "pheat-hbond": _native_model_metadata("arbitrary"),
    "pheat-rg": _native_model_metadata("arbitrary"),
    "pheat-ml-linear": _native_model_metadata("arbitrary"),
    "pheat-geometry-integrity": _native_model_metadata("arbitrary"),
    "heavy-mm": _native_model_metadata("arbitrary"),
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
    if model == "pheat-hydropathy":
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
    gromacs_work_dir: Optional[Union[str, Path]] = None,
    keep_gromacs_files: bool = False,
    gromacs_metrics: bool = False,
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Union[str, Path]] = None,
    prep_cache_mode: str = "off",
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
    if normalized == "pheat-geometry-integrity":
        return _with_coverage(
            _score_pheat_geometry_integrity(filtered),
            coverage,
            contract=contract,
            contract_warnings=contract_warnings,
        )
    if normalized == "heavy-mm":
        return _with_coverage(_score_heavy_mm(filtered), coverage, contract=contract, contract_warnings=contract_warnings)
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
                gromacs_work_dir=Path(gromacs_work_dir) if gromacs_work_dir is not None else None,
                keep_gromacs_files=keep_gromacs_files,
                gromacs_metrics=gromacs_metrics,
                gromacs_run_settings=gromacs_run_settings,
                external_timeout_seconds=external_timeout_seconds,
                prep_cache_dir=Path(prep_cache_dir) if prep_cache_dir is not None else None,
                prep_cache_mode=prep_cache_mode,
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


def _score_gromacs_mdrun(
    structure: HeavyAtomStructure,
    *,
    prepare: str = "auto",
    prepared_output: Optional[Path] = None,
    gromacs_forcefield: str = DEFAULT_GROMACS_FORCEFIELD,
    gromacs_water: str = DEFAULT_GROMACS_WATER,
    gromacs_solvate: bool = False,
    gromacs_run_mode: str = "rerun",
    gromacs_work_dir: Optional[Path] = None,
    keep_gromacs_files: bool = False,
    gromacs_metrics: bool = False,
    gromacs_run_settings: Optional[Union[GromacsRunSettings, Mapping[str, Any]]] = None,
    external_timeout_seconds: Optional[float] = None,
    prep_cache_dir: Optional[Path] = None,
    prep_cache_mode: str = "off",
) -> EnergyResult:
    """Run a GROMACS topology-prepared energy check for validation/reranking."""

    run_mode = _normalize_gromacs_run_mode(gromacs_run_mode)
    settings = _normalize_gromacs_run_settings(gromacs_run_settings)
    timeout = _normalize_external_timeout(external_timeout_seconds)
    cache_mode = _normalize_prep_cache_mode(prep_cache_mode)
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
        warnings=prepared["warnings"]
        + energy["warnings"]
        + [
            "gromacs-mdrun is a force-field potential-energy validation/reranking score, not folding free energy",
            "compare only structures prepared with the same force field, water, solvation, and run-mode settings",
        ],
        citations=_gromacs_citations(str(metadata["gromacs_forcefield"])),
        metadata=metadata,
    )


def _score_openmm_prepared(structure: HeavyAtomStructure) -> EnergyResult:
    """Explicit OpenMM path that prepares a force-field-ready structure before scoring."""

    try:
        from openmm import Platform, VerletIntegrator, unit
        from openmm import app
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
                structure_to_pdb_string(structure, allow_chain_truncation=True),
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
        warnings=preparation_warnings,
        citations=["openmm", "amber-ff14sb"],
        metadata={
            "description": "OpenMM AMBER score after internal structure preparation",
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

    try:
        from openmm import Platform
        from openmm import app
    except Exception as exc:  # pragma: no cover - depends on optional user environment
        raise RuntimeError(
            "--prepare auto/write for ambertools-sander requires OpenMM for deterministic "
            "hydrogen and missing-atom preparation."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="pheat-amber-prep-") as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        pdb_path.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
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
    }
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
        _run_subprocess(pdb2gmx_command, cwd=work_dir, input_text="\n", timeout=timeout)
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
    _run_subprocess(editconf_command, cwd=work_dir, timeout=timeout)
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
        _run_subprocess(solvate_command, cwd=work_dir, timeout=timeout)
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
) -> dict[str, Any]:
    command_lines: dict[str, list[str]] = {}
    warnings: list[str] = []
    final_coordinate = coordinate_path
    final_tpr = work_dir / "rerun.tpr"
    energy_edr = work_dir / "rerun.edr"

    if run_mode in {"minimize", "minimize-rerun"}:
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
        _run_subprocess(grompp_minimize, cwd=work_dir, timeout=timeout)
        mdrun_minimize = [gmx, "mdrun", "-deffnm", "minimize", "-s", str(minimize_tpr.name)]
        mdrun_minimize.extend(settings.mdrun_flags)
        _run_subprocess(mdrun_minimize, cwd=work_dir, timeout=timeout)
        minimized = work_dir / "minimize.gro"
        if not minimized.exists():
            raise RuntimeError("GROMACS mdrun minimization did not produce minimize.gro.")
        final_coordinate = minimized
        final_tpr = minimize_tpr
        energy_edr = work_dir / "minimize.edr"
        command_lines["grompp_minimize"] = grompp_minimize
        command_lines["mdrun_minimize"] = mdrun_minimize

    if run_mode in {"rerun", "minimize-rerun"}:
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
        _run_subprocess(grompp_rerun, cwd=work_dir, timeout=timeout)
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
        _run_subprocess(mdrun_rerun, cwd=work_dir, timeout=timeout)
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

    try:
        from openmm import Platform
        from openmm import app
    except Exception as exc:  # pragma: no cover - depends on optional user environment
        output_pdb.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
            encoding="utf-8",
        )
        return [
            "OpenMM/PDBFixer was unavailable; GROMACS pdb2gmx will prepare the supplied atoms directly"
        ], {**direct_metadata, "openmm_pdbfixer_unavailable": str(exc)}

    with tempfile.TemporaryDirectory(prefix="pheat-gmx-prep-") as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        pdb_path.write_text(
            structure_to_pdb_string(structure, allow_chain_truncation=True),
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
    }
    return list(warnings) + [
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
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "check": False,
        "capture_output": True,
        "text": True,
    }
    if input_text is not None:
        kwargs["input"] = input_text
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        result = subprocess.run(list(command), **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout} seconds: {' '.join(command)}. "
            + _timeout_tail(exc)
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}. "
            + _subprocess_tail(result)
        )
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

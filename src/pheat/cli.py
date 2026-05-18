"""Thin command-line wrappers around the backend library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pheat import __version__
from pheat.archive import (
    add_download_arguments,
    add_snapshot_download_arguments,
    add_snapshot_metadata_arguments,
    add_snapshot_relocate_arguments,
    add_snapshot_verify_arguments,
    get_snapshot,
    list_snapshots,
    relocate_snapshot_manifest_from_args,
    run_download_from_args,
    run_snapshot_download_from_args,
    snapshot_metadata_from_args,
    snapshot_ids,
    snapshot_to_dict,
    verify_snapshot_from_args,
)
from pheat.bcif import load_bcif
from pheat.bonds import BOND_STORAGE_MODES, structure_with_bonds
from pheat.centroid import to_centroid_structure
from pheat.corpus import (
    SUPPORTED_CORPUS_SPEC_TEMPLATES,
    build_reference_corpus,
    init_corpus_spec,
    summarize_corpus_spec,
    validate_corpus_spec,
)
from pheat.examples import (
    fetch_example_set,
    list_example_sets,
    load_example_manifest,
    run_example_set,
)
from pheat.geometry_tables import (
    CDL_RESIDUE_CLASS_MODES,
    CDL_SMOOTHING_MODES,
    GEOMETRY_MODES,
    build_backbone_geometry_tables,
    build_cdl_geometry_tables,
    build_sidechain_ccd_geometry_tables,
    describe_geometry_tables,
    import_cdl_geometry_tables,
    list_packaged_geometry_tables,
    validate_geometry_tables,
)
from pheat.hydrogens import HYDROGEN_POLICIES, generate_hydrogens
from pheat.models import HeavyAtomStructure
from pheat.metrics import (
    RMSD_ATOM_SETS,
    SAME_AS_RMSD_ALIGNMENT,
    radius_of_gyration_result,
    structure_rmsd_result,
)
from pheat.mmcif import load_mmcif, write_mmcif
from pheat.molstar_assets import (
    DEFAULT_MOLSTAR_VERSION,
    MOLSTAR_ENV_VAR,
    install_molstar_assets,
    molstar_asset_status,
    molstar_missing_assets_message,
)
from pheat.pdbio import (
    load_heavy_json,
    load_pdb,
    write_heavy_json,
    write_pdb,
)
from pheat.reference import (
    DEFAULT_REFERENCE_ARTIFACT_VERSION,
    DEFAULT_REFERENCE_DECOY_ATTEMPTS,
    DEFAULT_REFERENCE_DECOY_PROFILE,
    DEFAULT_REFERENCE_FEATURE_MODELS,
    DEFAULT_REFERENCE_MAX_RESOLUTION,
    DEFAULT_REFERENCE_METHOD,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_REFERENCE_SEED,
    DEFAULT_REFERENCE_SEQUENCE_IDENTITY,
    DEFAULT_REFERENCE_SNAPSHOT_ID,
    REFERENCE_SUBSETS,
    audit_reference_artifact_version,
    build_reference_decoys,
    build_reference_scores,
    extract_reference_features,
    fetch_reference_inputs,
    get_reference_decoy_dataset,
    inventory_reference,
    list_reference_decoy_datasets,
    package_reference_scoring_assets,
    promote_reference_artifact,
    run_reference_unattended,
    select_reference_corpus,
    train_reference_ml,
    validate_reference_features,
)
from pheat.scoring import (
    AMBER_SOLVENT_MODES,
    DEFAULT_AMBER_FORCEFIELD,
    DEFAULT_GROMACS_FORCEFIELD,
    DEFAULT_GROMACS_WATER,
    GromacsRunSettings,
    GROMACS_RUN_MODES,
    GROMACS_WATER_MODELS,
    PREP_CACHE_MODES,
    PREPARE_MODES,
    prepare_gromacs_structure,
    score_structure,
    score_structure_profiles,
    supported_models,
    validate_external_scoring_options,
)
from pheat.sources import fetch_source, list_sources, verify_sources
from pheat.sasa import BURIAL_METHODS, SASA_BACKENDS
from pheat.training import (
    DEFAULT_TRAINING_DOMAIN,
    DEFAULT_TRAINING_MODELS,
    DEFAULT_TRAINING_ROOT,
    SUCCESSFUL_ARCHIVE_STATUSES,
    InventoryOptions,
    SelectionOptions,
    build_score_tables,
    describe_score_tables,
    describe_training_corpus,
    extract_features,
    fetch_decoy_dataset,
    get_decoy_dataset,
    inventory_snapshot,
    list_decoy_datasets,
    load_training_structure,
    normalize_model_list,
    select_corpus,
    train_linear_model,
    validate_tables,
    verify_decoy_root,
    write_snapshot_ids,
)
from pheat.residue_geometry import (
    ANGLE_UNITS,
    structure_from_residue_geometry,
    structure_to_residue_geometry,
    write_residue_geometry_json,
)
from pheat.webapp import (
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    DEFAULT_WEB_WORK_DIR,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._argv = raw_argv
    args._status_stream = _build_status_stream(args)
    try:
        _emit_startup_status(args)
        return int(args.func(args) or 0)
    except Exception as exc:
        _emit_cli_error(args, exc)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pheat",
        description="Protein Heavy-atom Energy and Analysis Toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"pheat {__version__}")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress startup and progress status output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase diagnostic output. Use twice to include tracebacks on errors.",
    )
    parser.add_argument(
        "--log",
        help="Append startup, status, progress, and error messages to this log file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pdb_to_heavy = subparsers.add_parser("pdb-to-structure", help="Convert PDB to atom-structure JSON.")
    pdb_to_heavy.add_argument("input")
    pdb_to_heavy.add_argument("-o", "--output", required=True)
    _add_hydrogen_and_bond_options(pdb_to_heavy)
    pdb_to_heavy.set_defaults(func=_cmd_pdb_to_heavy)

    mmcif_to_heavy = subparsers.add_parser("mmcif-to-structure", help="Convert mmCIF to atom-structure JSON.")
    mmcif_to_heavy.add_argument("input")
    mmcif_to_heavy.add_argument("-o", "--output", required=True)
    mmcif_to_heavy.add_argument(
        "--chain-id-source",
        choices=["auth", "label"],
        default="auth",
        help="mmCIF chain/residue identifier namespace to use. Default: auth.",
    )
    _add_hydrogen_and_bond_options(mmcif_to_heavy)
    mmcif_to_heavy.set_defaults(func=_cmd_mmcif_to_heavy)

    bcif_to_heavy = subparsers.add_parser("bcif-to-structure", help="Convert BinaryCIF to atom-structure JSON.")
    bcif_to_heavy.add_argument("input")
    bcif_to_heavy.add_argument("-o", "--output", required=True)
    bcif_to_heavy.add_argument(
        "--model",
        type=int,
        default=1,
        help="Model number to read from multi-model BinaryCIF input. Default: 1.",
    )
    _add_hydrogen_and_bond_options(bcif_to_heavy)
    bcif_to_heavy.set_defaults(func=_cmd_bcif_to_heavy)

    heavy_to_pdb = subparsers.add_parser("structure-to-pdb", help="Convert atom-structure JSON to PDB.")
    heavy_to_pdb.add_argument("input")
    heavy_to_pdb.add_argument("-o", "--output", required=True)
    heavy_to_pdb.add_argument(
        "--allow-pdb-chain-truncation",
        action="store_true",
        help="Permit truncating chain IDs to one character for legacy PDB output.",
    )
    heavy_to_pdb.set_defaults(func=_cmd_heavy_to_pdb)

    heavy_to_mmcif = subparsers.add_parser("structure-to-mmcif", help="Convert atom-structure JSON to mmCIF.")
    heavy_to_mmcif.add_argument("input")
    heavy_to_mmcif.add_argument("-o", "--output", required=True)
    heavy_to_mmcif.set_defaults(func=_cmd_heavy_to_mmcif)

    heavy_to_geometry = subparsers.add_parser(
        "structure-to-geometry",
        help="Extract residue-geometry JSON from atom-structure JSON.",
    )
    heavy_to_geometry.add_argument("input")
    heavy_to_geometry.add_argument("-o", "--output", required=True)
    heavy_to_geometry.add_argument(
        "--angle-units",
        choices=ANGLE_UNITS,
        default="radians",
        help="Angle units to write. Default: radians.",
    )
    heavy_to_geometry.add_argument(
        "--store-angles",
        default="",
        help=(
            "Comma-separated optional backbone geometry fields to store: omega, tau, "
            "theta, or all. Definitions: omega=CA(i)-C(i)-N(i+1)-CA(i+1), "
            "tau=N(i)-CA(i)-C(i), theta=CA(i)-C(i)-N(i+1). Default: omit them."
        ),
    )
    heavy_to_geometry.add_argument(
        "--max-chi",
        type=int,
        help="Maximum number of chi angles to store per residue. Default: no limit.",
    )
    heavy_to_geometry.add_argument(
        "--store-lengths",
        default="",
        help=(
            "Comma-separated optional residue bond lengths to store: all, backbone, "
            "sidechain, or explicit ATOM-ATOM keys. Default: omit them."
        ),
    )
    heavy_to_geometry.set_defaults(func=_cmd_heavy_to_geometry)

    pdb_to_geometry = subparsers.add_parser(
        "pdb-to-geometry",
        help="Convert PDB directly to residue-geometry JSON.",
    )
    pdb_to_geometry.add_argument("input")
    pdb_to_geometry.add_argument("-o", "--output", required=True)
    pdb_to_geometry.add_argument(
        "--angle-units",
        choices=ANGLE_UNITS,
        default="radians",
        help="Angle units to write. Default: radians.",
    )
    pdb_to_geometry.add_argument(
        "--store-angles",
        default="",
        help=(
            "Comma-separated optional backbone geometry fields to store: omega, tau, "
            "theta, or all. Definitions: omega=CA(i)-C(i)-N(i+1)-CA(i+1), "
            "tau=N(i)-CA(i)-C(i), theta=CA(i)-C(i)-N(i+1). Default: omit them."
        ),
    )
    pdb_to_geometry.add_argument(
        "--max-chi",
        type=int,
        help="Maximum number of chi angles to store per residue. Default: no limit.",
    )
    pdb_to_geometry.add_argument(
        "--store-lengths",
        default="",
        help=(
            "Comma-separated optional residue bond lengths to store: all, backbone, "
            "sidechain, or explicit ATOM-ATOM keys. Default: omit them."
        ),
    )
    pdb_to_geometry.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling policy while loading PDB input. Default: drop.",
    )
    pdb_to_geometry.set_defaults(func=_cmd_pdb_to_geometry)

    mmcif_to_geometry = subparsers.add_parser(
        "mmcif-to-geometry",
        help="Convert mmCIF directly to residue-geometry JSON.",
    )
    mmcif_to_geometry.add_argument("input")
    mmcif_to_geometry.add_argument("-o", "--output", required=True)
    mmcif_to_geometry.add_argument(
        "--chain-id-source",
        choices=["auth", "label"],
        default="auth",
        help="mmCIF chain/residue identifier namespace to use. Default: auth.",
    )
    mmcif_to_geometry.add_argument(
        "--angle-units",
        choices=ANGLE_UNITS,
        default="radians",
        help="Angle units to write. Default: radians.",
    )
    mmcif_to_geometry.add_argument(
        "--store-angles",
        default="",
        help=(
            "Comma-separated optional backbone geometry fields to store: omega, tau, "
            "theta, or all. Default: omit them."
        ),
    )
    mmcif_to_geometry.add_argument(
        "--max-chi",
        type=int,
        help="Maximum number of chi angles to store per residue. Default: no limit.",
    )
    mmcif_to_geometry.add_argument(
        "--store-lengths",
        default="",
        help=(
            "Comma-separated optional residue bond lengths to store: all, backbone, "
            "sidechain, or explicit ATOM-ATOM keys. Default: omit them."
        ),
    )
    mmcif_to_geometry.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling policy while loading mmCIF input. Default: drop.",
    )
    mmcif_to_geometry.set_defaults(func=_cmd_mmcif_to_geometry)

    bcif_to_geometry = subparsers.add_parser(
        "bcif-to-geometry",
        help="Convert BinaryCIF directly to residue-geometry JSON.",
    )
    bcif_to_geometry.add_argument("input")
    bcif_to_geometry.add_argument("-o", "--output", required=True)
    bcif_to_geometry.add_argument(
        "--model",
        type=int,
        default=1,
        help="Model number to read from multi-model BinaryCIF input. Default: 1.",
    )
    bcif_to_geometry.add_argument(
        "--angle-units",
        choices=ANGLE_UNITS,
        default="radians",
        help="Angle units to write. Default: radians.",
    )
    bcif_to_geometry.add_argument(
        "--store-angles",
        default="",
        help=(
            "Comma-separated optional backbone geometry fields to store: omega, tau, "
            "theta, or all. Default: omit them."
        ),
    )
    bcif_to_geometry.add_argument(
        "--max-chi",
        type=int,
        help="Maximum number of chi angles to store per residue. Default: no limit.",
    )
    bcif_to_geometry.add_argument(
        "--store-lengths",
        default="",
        help=(
            "Comma-separated optional residue bond lengths to store: all, backbone, "
            "sidechain, or explicit ATOM-ATOM keys. Default: omit them."
        ),
    )
    bcif_to_geometry.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling policy while loading BinaryCIF input. Default: drop.",
    )
    bcif_to_geometry.set_defaults(func=_cmd_bcif_to_geometry)

    geometry_to_heavy = subparsers.add_parser(
        "geometry-to-structure",
        help="Reconstruct an atom structure from residue-geometry JSON.",
    )
    geometry_to_heavy.add_argument("input")
    geometry_to_heavy.add_argument("-o", "--output", required=True)
    geometry_to_heavy.add_argument(
        "--angle-units",
        choices=ANGLE_UNITS,
        help="Override angle units to read. Default: JSON angle_units, or radians if absent.",
    )
    geometry_to_heavy.add_argument(
        "--include-terminal-oxt",
        action="store_true",
        help="Add terminal OXT atoms at chain termini during reconstruction.",
    )
    geometry_to_heavy.add_argument(
        "--geometry-mode",
        choices=GEOMETRY_MODES,
        help="Reconstruction geometry mode. Default: fixed, or table when --geometry-table is set.",
    )
    geometry_to_heavy.add_argument(
        "--geometry-table",
        help="Optional pheat.geometry-table-set JSON for opt-in table-mode reconstruction.",
    )
    geometry_to_heavy.add_argument("--geometry-profile", help="Profile ID to read from --geometry-table.")
    geometry_to_heavy.add_argument("--pdb-output", help="Also write reconstructed coordinates as PDB.")
    geometry_to_heavy.add_argument("--mmcif-output", help="Also write reconstructed coordinates as mmCIF.")
    geometry_to_heavy.add_argument(
        "--hydrogens",
        choices=["drop", "generate"],
        default="drop",
        help="Hydrogen handling after heavy-atom reconstruction. Default: drop.",
    )
    geometry_to_heavy.add_argument(
        "--store-bonds",
        choices=BOND_STORAGE_MODES,
        default="none",
        help="Optional bond records to store in atom-structure JSON. Default: none.",
    )
    geometry_to_heavy.add_argument(
        "--allow-pdb-chain-truncation",
        action="store_true",
        help="Permit truncating chain IDs to one character when --pdb-output is used.",
    )
    geometry_to_heavy.set_defaults(func=_cmd_geometry_to_heavy)

    centroid = subparsers.add_parser("centroid", help="Convert heavy atoms to side-chain centroids.")
    centroid.add_argument("input")
    centroid.add_argument("-o", "--output", required=True)
    centroid.add_argument("--mode", choices=["single", "multi"], default="single")
    centroid.set_defaults(func=_cmd_centroid)

    score = subparsers.add_parser("score", help="Score a PDB, mmCIF, or atom-structure JSON structure.")
    score.add_argument("input")
    score.add_argument("--model", choices=supported_models(), default="generic")
    score.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
        help="Atoms/residues to score. Default: protein-heavy.",
    )
    score.add_argument(
        "--table-set",
        help=(
            "Load trained PHEAT score-table JSON for table-backed models. "
            "Use packaged:<id> for bundled initial v0 assets."
        ),
    )
    score.add_argument("--profile", help="Score against one profile from --table-set.")
    score.add_argument(
        "--profiles",
        help="Comma-separated profile IDs to score side by side from --table-set.",
    )
    score.add_argument(
        "--burial-method",
        choices=BURIAL_METHODS,
        help="Burial method for hydropathy scoring. Default: table metadata or contacts.",
    )
    score.add_argument(
        "--sasa-backend",
        choices=SASA_BACKENDS,
        default="auto",
        help="SASA backend when --burial-method sasa is selected. Default: auto.",
    )
    score.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Force-field preparation mode for force-field scoring models. Default: auto.",
    )
    score.add_argument(
        "--prepared-output",
        help="Write the prepared all-atom PDB when --prepare write is used.",
    )
    score.add_argument(
        "--amber-forcefield",
        default=DEFAULT_AMBER_FORCEFIELD,
        help=f"AmberTools tleap force field to source. Default: {DEFAULT_AMBER_FORCEFIELD}.",
    )
    score.add_argument(
        "--amber-solvent",
        choices=AMBER_SOLVENT_MODES,
        default="vacuum",
        help="AmberTools sander solvent mode. Default: vacuum.",
    )
    score.add_argument(
        "--ambertools-work-dir",
        help="Directory for AmberTools intermediate files.",
    )
    score.add_argument(
        "--keep-ambertools-files",
        action="store_true",
        help="Keep AmberTools intermediate files for inspection.",
    )
    _add_external_backend_options(score)
    _add_gromacs_options(score)
    score.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling while reading coordinate input. Default: drop.",
    )
    score.add_argument("-o", "--output")
    score.set_defaults(func=_cmd_score)

    scoring = subparsers.add_parser("scoring", help="Inspect or validate scoring options.")
    scoring_sub = scoring.add_subparsers(dest="scoring_command", required=True)
    scoring_validate = scoring_sub.add_parser("validate-options")
    scoring_validate.add_argument("--model", choices=supported_models(), required=True)
    scoring_validate.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Force-field preparation mode to validate. Default: auto.",
    )
    scoring_validate.add_argument(
        "--amber-solvent",
        choices=AMBER_SOLVENT_MODES,
        default="vacuum",
        help="AmberTools sander solvent mode. Default: vacuum.",
    )
    _add_external_backend_options(scoring_validate)
    _add_gromacs_options(scoring_validate)
    scoring_validate.set_defaults(func=_cmd_scoring_validate_options)

    gromacs = subparsers.add_parser("gromacs", help="Prepare, score, or validate structures with GROMACS.")
    gromacs_sub = gromacs.add_subparsers(dest="gromacs_command", required=True)
    gromacs_prepare = gromacs_sub.add_parser("prepare")
    gromacs_prepare.add_argument("input")
    gromacs_prepare.add_argument("-o", "--output", required=True, help="Prepared GROMACS coordinate output.")
    gromacs_prepare.add_argument("--topology", required=True, help="Prepared GROMACS topology output.")
    gromacs_prepare.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Input preparation mode before pdb2gmx. Default: auto.",
    )
    _add_external_backend_options(gromacs_prepare)
    _add_gromacs_options(gromacs_prepare, include_run_mode=False, include_metrics=False)
    gromacs_prepare.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling while reading coordinate input. Default: drop.",
    )
    gromacs_prepare.set_defaults(func=_cmd_gromacs_prepare)

    gromacs_score = gromacs_sub.add_parser("score")
    gromacs_score.add_argument("input")
    gromacs_score.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
        help="Atoms/residues to score. Default: protein-heavy.",
    )
    gromacs_score.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Input preparation mode before pdb2gmx. Default: auto.",
    )
    gromacs_score.add_argument("--prepared-output", help="Write the final prepared GROMACS coordinate file.")
    _add_external_backend_options(gromacs_score)
    _add_gromacs_options(gromacs_score)
    gromacs_score.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling while reading coordinate input. Default: drop.",
    )
    gromacs_score.add_argument("-o", "--output")
    gromacs_score.set_defaults(func=_cmd_gromacs_score)

    gromacs_minimize = gromacs_sub.add_parser("minimize")
    gromacs_minimize.add_argument("input")
    gromacs_minimize.add_argument("-o", "--output", required=True, help="Minimized GROMACS coordinate output.")
    gromacs_minimize.add_argument("--score-output", help="Optional JSON score output path.")
    gromacs_minimize.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
        help="Atoms/residues to score. Default: protein-heavy.",
    )
    gromacs_minimize.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Input preparation mode before pdb2gmx. Default: auto.",
    )
    _add_external_backend_options(gromacs_minimize)
    _add_gromacs_options(gromacs_minimize, include_run_mode=False)
    gromacs_minimize.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling while reading coordinate input. Default: drop.",
    )
    gromacs_minimize.set_defaults(func=_cmd_gromacs_minimize)

    gromacs_validate = gromacs_sub.add_parser("validate")
    gromacs_validate.add_argument("input")
    gromacs_validate.add_argument("--json", dest="json_output", help="Write validation JSON to this path.")
    gromacs_validate.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
        help="Atoms/residues to score. Default: protein-heavy.",
    )
    gromacs_validate.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Input preparation mode before pdb2gmx. Default: auto.",
    )
    _add_external_backend_options(gromacs_validate)
    _add_gromacs_options(gromacs_validate)
    gromacs_validate.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling while reading coordinate input. Default: drop.",
    )
    gromacs_validate.set_defaults(func=_cmd_gromacs_validate)
    gromacs_validate_options = gromacs_sub.add_parser("validate-options")
    gromacs_validate_options.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="auto",
        help="Input preparation mode before pdb2gmx. Default: auto.",
    )
    _add_external_backend_options(gromacs_validate_options)
    _add_gromacs_options(gromacs_validate_options)
    gromacs_validate_options.set_defaults(func=_cmd_gromacs_validate_options)

    rg = subparsers.add_parser(
        "radius-of-gyration",
        aliases=["rg"],
        help="Compute unweighted and/or mass-weighted radius of gyration.",
    )
    rg.add_argument("input")
    rg.add_argument(
        "--mode",
        choices=["unweighted", "mass-weighted", "both"],
        default="both",
        help="Rg calculation mode. Default: both.",
    )
    rg.add_argument(
        "--atom-set",
        choices=RMSD_ATOM_SETS,
        default="all-heavy",
        help="Atoms to include in the Rg calculation. Default: all-heavy.",
    )
    rg.add_argument("-o", "--output")
    rg.set_defaults(func=_cmd_radius_of_gyration)

    rmsd = subparsers.add_parser("rmsd", help="Compute Kabsch RMSD between two structures.")
    rmsd.add_argument("reference")
    rmsd.add_argument("target")
    rmsd.add_argument(
        "--atom-set",
        choices=RMSD_ATOM_SETS,
        default="all-heavy",
        help="Matched atoms to measure. Default: all-heavy.",
    )
    rmsd.add_argument(
        "--alignment-atom-set",
        choices=(SAME_AS_RMSD_ALIGNMENT, *RMSD_ATOM_SETS),
        default=SAME_AS_RMSD_ALIGNMENT,
        help="Matched atoms used for Kabsch alignment. Default: same-as-rmsd.",
    )
    rmsd.add_argument("-o", "--output")
    rmsd.set_defaults(func=_cmd_rmsd)

    examples = subparsers.add_parser("examples", help="List, show, or fetch example sets.")
    examples_sub = examples.add_subparsers(dest="examples_command", required=True)
    examples_list = examples_sub.add_parser("list")
    examples_list.set_defaults(func=_cmd_examples_list)
    examples_show = examples_sub.add_parser("show")
    examples_show.add_argument("name")
    examples_show.set_defaults(func=_cmd_examples_show)
    examples_fetch = examples_sub.add_parser("fetch")
    examples_fetch.add_argument("name")
    examples_fetch.add_argument("-d", "--destination", default=".pheat-cache/examples")
    examples_fetch.set_defaults(func=_cmd_examples_fetch)
    examples_run = examples_sub.add_parser("run")
    examples_run.add_argument("name")
    examples_run.add_argument("--model", choices=supported_models(), default="generic")
    examples_run.add_argument("--fetch", action="store_true")
    examples_run.add_argument("-d", "--cache-dir", default=".pheat-cache/examples")
    examples_run.add_argument("-o", "--output")
    examples_run.set_defaults(func=_cmd_examples_run)

    sources = subparsers.add_parser("sources", help="List, fetch, or verify source data.")
    sources_sub = sources.add_subparsers(dest="sources_command", required=True)
    sources_list = sources_sub.add_parser("list")
    sources_list.set_defaults(func=_cmd_sources_list)
    sources_fetch = sources_sub.add_parser("fetch")
    sources_fetch.add_argument("source_id")
    sources_fetch.add_argument("-d", "--destination", default=".pheat-cache/sources")
    sources_fetch.set_defaults(func=_cmd_sources_fetch)
    sources_verify = sources_sub.add_parser("verify")
    sources_verify.add_argument("-d", "--cache-dir", default=".pheat-cache/sources")
    sources_verify.set_defaults(func=_cmd_sources_verify)

    geometry = subparsers.add_parser("geometry", help="Build, describe, or validate geometry tables.")
    geometry_sub = geometry.add_subparsers(dest="geometry_command", required=True)
    geometry_tables = geometry_sub.add_parser("tables", help="Build or validate geometry table sets.")
    geometry_tables_sub = geometry_tables.add_subparsers(dest="geometry_tables_command", required=True)
    geometry_backbone = geometry_tables_sub.add_parser(
        "build-backbone",
        help="Build backbone geometry targets from a selected training corpus.",
    )
    geometry_backbone.add_argument("--training-set", required=True)
    geometry_backbone.add_argument("--output-root", required=True)
    geometry_backbone.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
    )
    geometry_backbone.add_argument("--max-entries", type=int)
    geometry_backbone.add_argument("--table-set-id")
    geometry_backbone.add_argument("--table-set-version", default="v1")
    geometry_backbone.set_defaults(func=_cmd_geometry_tables_build_backbone)
    geometry_cdl = geometry_tables_sub.add_parser(
        "build-cdl",
        help="Build phi/psi-binned conformation-dependent backbone geometry targets.",
    )
    geometry_cdl.add_argument("--training-set", required=True)
    geometry_cdl.add_argument("--output-root", required=True)
    geometry_cdl.add_argument(
        "--domain",
        choices=["protein-heavy", "all-heavy", "full"],
        default=DEFAULT_TRAINING_DOMAIN,
    )
    geometry_cdl.add_argument("--max-entries", type=int)
    geometry_cdl.add_argument("--table-set-id")
    geometry_cdl.add_argument("--table-set-version", default="v1")
    geometry_cdl.add_argument("--phi-psi-bin-size", type=int, default=10)
    geometry_cdl.add_argument("--min-bin-count", type=int, default=20)
    geometry_cdl.add_argument("--smoothing", choices=CDL_SMOOTHING_MODES, default="nearest")
    geometry_cdl.add_argument(
        "--residue-classes",
        choices=CDL_RESIDUE_CLASS_MODES,
        default="gly-pro-general",
    )
    geometry_cdl.set_defaults(func=_cmd_geometry_tables_build_cdl)
    geometry_import_cdl = geometry_tables_sub.add_parser(
        "import-cdl",
        help="Import a JSON CDL-like target table into PHEAT geometry-table-set format.",
    )
    geometry_import_cdl.add_argument("--input", required=True)
    geometry_import_cdl.add_argument("--output-root", required=True)
    geometry_import_cdl.add_argument("--table-set-id", default="imported-cdl-geometry")
    geometry_import_cdl.add_argument("--table-set-version", default="v1")
    geometry_import_cdl.add_argument("--source-license")
    geometry_import_cdl.set_defaults(func=_cmd_geometry_tables_import_cdl)
    geometry_sidechain = geometry_tables_sub.add_parser(
        "build-sidechain-ccd",
        help="Build side-chain geometry targets from cached CCD component CIF files.",
    )
    geometry_sidechain.add_argument(
        "--ccd-full",
        help="Full wwPDB CCD components.cif.gz or components.cif file. Preferred for geometry.",
    )
    geometry_sidechain.add_argument("--ccd-dir", help="Directory of per-component CCD CIF/CIF.GZ files.")
    geometry_sidechain.add_argument(
        "--ccd-bcif-dir",
        help="Directory containing RCSB cca.bcif and ccb.bcif connectivity subsets.",
    )
    geometry_sidechain.add_argument("--output-root", required=True)
    geometry_sidechain.add_argument(
        "--residues",
        help="Comma-separated residue component IDs. Default: all residues supported by PHEAT.",
    )
    geometry_sidechain.add_argument("--table-set-id", default="ccd-sidechain-geometry")
    geometry_sidechain.add_argument("--table-set-version", default="v1")
    geometry_sidechain.set_defaults(func=_cmd_geometry_tables_build_sidechain_ccd)
    geometry_list = geometry_tables_sub.add_parser("list", help="List packaged geometry table sets.")
    geometry_list.set_defaults(func=_cmd_geometry_tables_list)
    geometry_describe = geometry_tables_sub.add_parser("describe")
    geometry_describe.add_argument("--table-set", required=True)
    geometry_describe.set_defaults(func=_cmd_geometry_tables_describe)
    geometry_validate = geometry_tables_sub.add_parser("validate")
    geometry_validate.add_argument("--table-set", required=True)
    geometry_validate.set_defaults(func=_cmd_geometry_tables_validate)

    archive = subparsers.add_parser("archive", help="Build local PDB archive corpora.")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    archive_download = archive_sub.add_parser(
        "download",
        help="Download or plan PDB archive coordinate corpora with provenance manifests.",
    )
    add_download_arguments(archive_download)
    archive_download.set_defaults(func=_cmd_archive_download)
    archive_snapshots = archive_sub.add_parser(
        "snapshots",
        help="List, describe, download, or verify PDB archive snapshot presets.",
    )
    snapshots_sub = archive_snapshots.add_subparsers(dest="snapshots_command", required=True)
    snapshots_list = snapshots_sub.add_parser("list", help="List known downloadable snapshots.")
    snapshots_list.add_argument("--text", action="store_true", help="Write tabular text instead of JSON.")
    snapshots_list.set_defaults(func=_cmd_archive_snapshots_list)
    snapshots_describe = snapshots_sub.add_parser("describe", help="Describe a snapshot preset.")
    snapshots_describe.add_argument("snapshot_id", choices=snapshot_ids())
    snapshots_describe.set_defaults(func=_cmd_archive_snapshots_describe)
    snapshots_download = snapshots_sub.add_parser(
        "download",
        help="Download or plan a snapshot preset.",
    )
    snapshots_download.add_argument("snapshot_id", choices=snapshot_ids())
    add_snapshot_download_arguments(snapshots_download)
    snapshots_download.set_defaults(func=_cmd_archive_snapshots_download)
    snapshots_verify = snapshots_sub.add_parser(
        "verify",
        help="Verify a local snapshot manifest against recorded SHA-256 checksums.",
    )
    snapshots_verify.add_argument("snapshot_id", choices=snapshot_ids())
    add_snapshot_verify_arguments(snapshots_verify)
    snapshots_verify.set_defaults(func=_cmd_archive_snapshots_verify)
    snapshots_metadata = snapshots_sub.add_parser(
        "metadata",
        help="Extract compact normalized metadata for a local snapshot.",
    )
    snapshots_metadata.add_argument("snapshot_id", choices=snapshot_ids())
    add_snapshot_metadata_arguments(snapshots_metadata)
    snapshots_metadata.set_defaults(func=_cmd_archive_snapshots_metadata)
    snapshots_relocate = snapshots_sub.add_parser(
        "relocate",
        help="Repair moved snapshot manifests by rewriting coordinate paths.",
    )
    snapshots_relocate.add_argument("snapshot_id", choices=snapshot_ids())
    add_snapshot_relocate_arguments(snapshots_relocate)
    snapshots_relocate.set_defaults(func=_cmd_archive_snapshots_relocate)
    snapshots_ids = snapshots_sub.add_parser(
        "ids",
        help="Write local PDB IDs represented by a snapshot files manifest.",
    )
    snapshots_ids.add_argument("snapshot_id", choices=snapshot_ids())
    snapshots_ids.add_argument("--output-root")
    snapshots_ids.add_argument("--manifest-dir")
    snapshots_ids.add_argument(
        "--status",
        action="append",
        default=[],
        help="Manifest status to include. Repeatable. Default: downloaded, present, reused, verified.",
    )
    snapshots_ids.add_argument("-o", "--output")
    snapshots_ids.set_defaults(func=_cmd_archive_snapshots_ids)

    training = subparsers.add_parser("training", help="Build PHEAT training corpora and score tables.")
    training_sub = training.add_subparsers(dest="training_command", required=True)
    training_decoys = training_sub.add_parser("decoys", help="List, describe, fetch, or verify decoy datasets.")
    decoys_sub = training_decoys.add_subparsers(dest="decoys_command", required=True)
    decoys_list = decoys_sub.add_parser("list")
    decoys_list.set_defaults(func=_cmd_training_decoys_list)
    decoys_describe = decoys_sub.add_parser("describe")
    decoys_describe.add_argument("dataset_id")
    decoys_describe.set_defaults(func=_cmd_training_decoys_describe)
    decoys_fetch = decoys_sub.add_parser("fetch")
    decoys_fetch.add_argument("dataset_id")
    decoys_fetch.add_argument("--output-root", default=str(DEFAULT_TRAINING_ROOT / "decoys"))
    decoys_fetch.add_argument("-y", "--yes", action="store_true")
    decoys_fetch.set_defaults(func=_cmd_training_decoys_fetch)
    decoys_verify = decoys_sub.add_parser("verify")
    decoys_verify.add_argument("--input-root", default=str(DEFAULT_TRAINING_ROOT / "decoys"))
    decoys_verify.set_defaults(func=_cmd_training_decoys_verify)

    training_corpus = training_sub.add_parser("corpus", help="Inventory and select local training corpora.")
    corpus_sub = training_corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_inventory = corpus_sub.add_parser("inventory")
    corpus_inventory.add_argument("--snapshot-root", required=True)
    corpus_inventory.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    corpus_inventory.add_argument("--metadata-jsonl")
    corpus_inventory.add_argument("--max-entries", type=int)
    corpus_inventory.add_argument("--workers", default="1")
    corpus_inventory.add_argument("-o", "--output", required=True)
    corpus_inventory.set_defaults(func=_cmd_training_corpus_inventory)
    corpus_select = corpus_sub.add_parser("select")
    corpus_select.add_argument("--inventory", required=True)
    corpus_select.add_argument("--output-root", required=True)
    corpus_select.add_argument("--method")
    corpus_select.add_argument("--max-resolution", type=float)
    corpus_select.add_argument("--sequence-identity", default="0.30")
    corpus_select.add_argument(
        "--sequence-clusters",
        help=(
            "Optional local sequence-cluster file. Each nonblank line is one cluster "
            "with whitespace/comma-separated PDB_chain members."
        ),
    )
    corpus_select.add_argument("--corpus-id")
    corpus_select.add_argument("--corpus-version", default="v1")
    corpus_select.add_argument("--include-file")
    corpus_select.add_argument("--exclude-file")
    corpus_select.add_argument("--holdout-file")
    corpus_select.add_argument("--canonical-only", action="store_true")
    corpus_select.add_argument("--min-length", type=int, default=50)
    corpus_select.add_argument("--max-length", type=int, default=800)
    corpus_select.set_defaults(func=_cmd_training_corpus_select)
    corpus_describe = corpus_sub.add_parser("describe")
    corpus_describe.add_argument("--training-set", required=True)
    corpus_describe.set_defaults(func=_cmd_training_corpus_describe)

    training_tables = training_sub.add_parser("tables", help="Build or validate trained score-table sets.")
    tables_sub = training_tables.add_subparsers(dest="tables_command", required=True)
    tables_build = tables_sub.add_parser("build")
    tables_build.add_argument("--training-set", required=True)
    tables_build.add_argument("--output-root", required=True)
    tables_build.add_argument("--models", default=",".join(DEFAULT_TRAINING_MODELS))
    tables_build.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    tables_build.add_argument("--burial-method", choices=list(BURIAL_METHODS) + ["both"], default="contacts")
    tables_build.add_argument("--sasa-backend", choices=SASA_BACKENDS, default="auto")
    tables_build.add_argument("--max-entries", type=int)
    tables_build.add_argument("--table-set-id")
    tables_build.add_argument("--table-set-version", default="v1")
    tables_build.add_argument("--workers", default="1")
    tables_build.set_defaults(func=_cmd_training_tables_build)
    tables_describe = tables_sub.add_parser("describe")
    tables_describe.add_argument("--table-set", required=True, help="Path or packaged:<id> table set.")
    tables_describe.set_defaults(func=_cmd_training_tables_describe)
    tables_validate = tables_sub.add_parser("validate")
    tables_validate.add_argument("--table-set", required=True, help="Path or packaged:<id> table set.")
    tables_validate.add_argument("--decoy-root")
    tables_validate.set_defaults(func=_cmd_training_tables_validate)

    training_features = training_sub.add_parser("features", help="Extract score features for ML baselines.")
    features_sub = training_features.add_subparsers(dest="features_command", required=True)
    features_extract = features_sub.add_parser("extract")
    features_extract.add_argument("--training-set", required=True)
    features_extract.add_argument("--models", default="generic,pheat-dfire,heavy-mm")
    features_extract.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    features_extract.add_argument("--workers", default="1")
    features_extract.add_argument("-o", "--output", required=True)
    features_extract.set_defaults(func=_cmd_training_features_extract)

    training_ml = training_sub.add_parser("ml", help="Train lightweight ML scoring baselines.")
    ml_sub = training_ml.add_subparsers(dest="ml_command", required=True)
    ml_train_linear = ml_sub.add_parser("train-linear")
    ml_train_linear.add_argument("--features", required=True)
    ml_train_linear.add_argument("--target", default="native_vs_decoy")
    ml_train_linear.add_argument("-o", "--output", required=True)
    ml_train_linear.set_defaults(func=_cmd_training_ml_train_linear)

    reference = subparsers.add_parser("reference", help="Build PHEAT reference corpora, decoys, tables, and ML artifacts.")
    reference_sub = reference.add_subparsers(dest="reference_command", required=True)
    reference_datasets = reference_sub.add_parser("datasets", help="List or describe reference decoy datasets.")
    reference_datasets_sub = reference_datasets.add_subparsers(dest="datasets_command", required=True)
    reference_datasets_list = reference_datasets_sub.add_parser("list")
    reference_datasets_list.set_defaults(func=_cmd_reference_datasets_list)
    reference_datasets_describe = reference_datasets_sub.add_parser("describe")
    reference_datasets_describe.add_argument("dataset_id")
    reference_datasets_describe.set_defaults(func=_cmd_reference_datasets_describe)

    reference_validate_spec = reference_sub.add_parser("validate-spec", help="Validate a PHEAT corpus spec.")
    reference_validate_spec.add_argument("corpus_spec")
    reference_validate_spec.set_defaults(func=_cmd_reference_validate_spec)

    reference_summarize_spec = reference_sub.add_parser("summarize-spec", help="Summarize a PHEAT corpus spec.")
    reference_summarize_spec.add_argument("corpus_spec")
    reference_summarize_spec.set_defaults(func=_cmd_reference_summarize_spec)

    reference_init_spec = reference_sub.add_parser("init-spec", help="Write a starter PHEAT corpus spec.")
    reference_init_spec.add_argument("--template", choices=SUPPORTED_CORPUS_SPEC_TEMPLATES, required=True)
    reference_init_spec.add_argument("--output", required=True)
    reference_init_spec.set_defaults(func=_cmd_reference_init_spec)

    reference_build = reference_sub.add_parser("build", help="Build a local/user-defined reference corpus.")
    reference_build.add_argument("--corpus-spec", required=True)
    reference_build.add_argument("--output-root", required=True)
    reference_build.add_argument("--ccd", help="Optional CSV or JSONL CCD-like component metadata table.")
    reference_build.add_argument("--dry-run", action="store_true")
    reference_build.add_argument("--overwrite", action="store_true")
    reference_build.set_defaults(func=_cmd_reference_build)

    reference_fetch = reference_sub.add_parser("fetch", help="Record/fetch reference snapshots and decoy datasets.")
    _add_reference_root_options(reference_fetch)
    reference_fetch.add_argument("--snapshot-id", choices=snapshot_ids(), default=DEFAULT_REFERENCE_SNAPSHOT_ID)
    reference_fetch.add_argument("--snapshot-root")
    reference_fetch.add_argument("--dataset", action="append", default=[])
    reference_fetch.add_argument("--include-payloads", action="store_true")
    reference_fetch.add_argument("--payload-url", action="append", default=[], help="Dataset URL mapping, DATASET=URL.")
    reference_fetch.add_argument("--local-file", action="append", default=[], help="Dataset local file mapping, DATASET=PATH.")
    reference_fetch.add_argument("--local-dir", action="append", default=[], help="Dataset local directory mapping, DATASET=PATH.")
    reference_fetch.add_argument("--workers", default="auto")
    reference_fetch.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    reference_fetch.add_argument("--overwrite", action="store_true")
    reference_fetch.set_defaults(func=_cmd_reference_fetch)

    reference_inventory = reference_sub.add_parser("inventory", help="Build a reference inventory from a local snapshot.")
    _add_reference_root_options(reference_inventory)
    reference_inventory.add_argument("--snapshot-root")
    reference_inventory.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_inventory.add_argument("--metadata-jsonl")
    reference_inventory.add_argument("--max-entries", type=int)
    reference_inventory.add_argument("--workers", default="auto")
    reference_inventory.add_argument("--overwrite", action="store_true")
    reference_inventory.add_argument("-o", "--output")
    reference_inventory.set_defaults(func=_cmd_reference_inventory)

    reference_select = reference_sub.add_parser(
        "select",
        help="Select an X-ray 30 percent ID reference subset.",
        description="Select an X-ray 30 percent ID reference subset.",
    )
    _add_reference_root_options(reference_select)
    reference_select.add_argument("--inventory", required=True)
    reference_select.add_argument("--output-root")
    reference_select.add_argument("--subset", choices=REFERENCE_SUBSETS, default="aqueous")
    reference_select.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_select.add_argument("--method", default=DEFAULT_REFERENCE_METHOD)
    reference_select.add_argument("--max-resolution", type=float, default=DEFAULT_REFERENCE_MAX_RESOLUTION)
    reference_select.add_argument("--sequence-identity", default=DEFAULT_REFERENCE_SEQUENCE_IDENTITY)
    reference_select.add_argument("--sequence-clusters")
    reference_select.add_argument("--include-file")
    reference_select.add_argument("--exclude-file")
    reference_select.add_argument("--holdout-file")
    reference_select.add_argument("--canonical-only", action="store_true")
    reference_select.add_argument("--min-length", type=int, default=50)
    reference_select.add_argument("--max-length", type=int, default=800)
    reference_select.add_argument("--overwrite", action="store_true")
    reference_select.set_defaults(func=_cmd_reference_select)

    reference_decoys = reference_sub.add_parser("build-decoys", help="Generate PHEAT-owned reference decoys.")
    _add_reference_root_options(reference_decoys)
    reference_decoys.add_argument("--training-set", required=True)
    reference_decoys.add_argument("--output-root")
    reference_decoys.add_argument("--recipes", default=DEFAULT_REFERENCE_DECOY_PROFILE, help="Comma-separated recipes or recipe profile.")
    reference_decoys.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_decoys.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    reference_decoys.add_argument("--max-entries", type=int)
    reference_decoys.add_argument("--attempts-per-decoy", type=int, default=DEFAULT_REFERENCE_DECOY_ATTEMPTS)
    reference_decoys.add_argument("--workers", default="auto")
    reference_decoys.add_argument("--overwrite", action="store_true")
    reference_decoys.set_defaults(func=_cmd_reference_build_decoys)

    reference_scores = reference_sub.add_parser("build-scores", help="Build PHEAT reference score tables.")
    _add_reference_root_options(reference_scores)
    reference_scores.add_argument("--training-set", required=True)
    reference_scores.add_argument("--output-root")
    reference_scores.add_argument("--models", default=",".join(DEFAULT_TRAINING_MODELS))
    reference_scores.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_scores.add_argument("--burial-method", choices=list(BURIAL_METHODS) + ["both"], default="both")
    reference_scores.add_argument("--sasa-backend", choices=SASA_BACKENDS, default="auto")
    reference_scores.add_argument("--max-entries", type=int)
    reference_scores.add_argument("--table-set-id")
    reference_scores.add_argument("--workers", default="auto")
    reference_scores.add_argument("--overwrite", action="store_true")
    reference_scores.set_defaults(func=_cmd_reference_build_scores)

    reference_features = reference_sub.add_parser("extract-features", help="Extract native and decoy reference features.")
    _add_reference_root_options(reference_features)
    reference_features.add_argument("--training-set", required=True)
    reference_features.add_argument("--decoys")
    reference_features.add_argument("--models", default=",".join(DEFAULT_REFERENCE_FEATURE_MODELS))
    reference_features.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_features.add_argument("--max-entries", type=int)
    reference_features.add_argument("--workers", default="auto")
    reference_features.add_argument("--overwrite", action="store_true")
    reference_features.add_argument("-o", "--output")
    reference_features.set_defaults(func=_cmd_reference_extract_features)

    reference_ml = reference_sub.add_parser("train-ml", help="Train reference ML scoring baselines.")
    _add_reference_root_options(reference_ml)
    reference_ml.add_argument("--features", required=True)
    reference_ml.add_argument("--target", default="native_vs_decoy")
    reference_ml.add_argument("--overwrite", action="store_true")
    reference_ml.add_argument("-o", "--output")
    reference_ml.set_defaults(func=_cmd_reference_train_ml)

    reference_validate = reference_sub.add_parser("validate", help="Validate reference feature rows and score separation.")
    _add_reference_root_options(reference_validate)
    reference_validate.add_argument("--features", required=True)
    reference_validate.add_argument("--overwrite", action="store_true")
    reference_validate.add_argument("-o", "--output")
    reference_validate.set_defaults(func=_cmd_reference_validate)

    reference_promote = reference_sub.add_parser("promote", help="Promote a reviewed reference artifact.")
    reference_promote.add_argument("--source", required=True)
    reference_promote.add_argument("--destination", required=True)
    reference_promote.add_argument("--note", required=True)
    reference_promote.add_argument("--overwrite", action="store_true")
    reference_promote.set_defaults(func=_cmd_reference_promote)

    reference_package = reference_sub.add_parser(
        "package-scoring-assets",
        help="Package reference score/model JSON assets as compressed bundled scoring data.",
    )
    reference_package.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    reference_package.add_argument("--artifact-version", default=DEFAULT_REFERENCE_ARTIFACT_VERSION)
    reference_package.add_argument("--destination-root")
    reference_package.add_argument("--overwrite", action="store_true")
    reference_package.set_defaults(func=_cmd_reference_package_scoring_assets)

    reference_audit = reference_sub.add_parser(
        "audit-version",
        help="Audit a reference archive for artifact-version consistency.",
    )
    reference_audit.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    reference_audit.add_argument("--artifact-version", default=DEFAULT_REFERENCE_ARTIFACT_VERSION)
    reference_audit.add_argument("-o", "--output", help="Optional path to write the audit JSON report.")
    reference_audit.set_defaults(func=_cmd_reference_audit_version)

    reference_unattended = reference_sub.add_parser("run-unattended", help="Run the full non-interactive reference build.")
    _add_reference_root_options(reference_unattended)
    reference_unattended.add_argument("--snapshot-id", choices=snapshot_ids(), default=DEFAULT_REFERENCE_SNAPSHOT_ID)
    reference_unattended.add_argument("--snapshot-root")
    reference_unattended.add_argument("--domain", choices=["protein-heavy", "all-heavy", "full"], default=DEFAULT_TRAINING_DOMAIN)
    reference_unattended.add_argument("--method", default=DEFAULT_REFERENCE_METHOD)
    reference_unattended.add_argument("--max-resolution", type=float, default=DEFAULT_REFERENCE_MAX_RESOLUTION)
    reference_unattended.add_argument("--sequence-identity", default=DEFAULT_REFERENCE_SEQUENCE_IDENTITY)
    reference_unattended.add_argument("--min-length", type=int, default=50)
    reference_unattended.add_argument("--max-length", type=int, default=800)
    reference_unattended.add_argument("--decoy-recipes", default=DEFAULT_REFERENCE_DECOY_PROFILE)
    reference_unattended.add_argument("--attempts-per-decoy", type=int, default=DEFAULT_REFERENCE_DECOY_ATTEMPTS)
    reference_unattended.add_argument("--models", default=",".join(DEFAULT_TRAINING_MODELS))
    reference_unattended.add_argument("--feature-models", default=",".join(DEFAULT_REFERENCE_FEATURE_MODELS))
    reference_unattended.add_argument("--sasa-backend", choices=SASA_BACKENDS, default="auto")
    reference_unattended.add_argument("--metadata-source", default="auto")
    reference_unattended.add_argument("--workers", default="auto")
    reference_unattended.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    reference_unattended.add_argument("--canary-entries", type=int, default=25)
    reference_unattended.add_argument("--skip-canary", action="store_true")
    reference_unattended.add_argument("--backup-existing", action="store_true")
    reference_unattended.add_argument("--dry-run", action="store_true")
    reference_unattended.add_argument("--overwrite", action="store_true")
    reference_unattended.set_defaults(func=_cmd_reference_run_unattended)

    molstar = subparsers.add_parser("molstar", help="Manage local Mol* browser assets.")
    molstar_sub = molstar.add_subparsers(dest="molstar_command", required=True)
    molstar_install = molstar_sub.add_parser("install", help="Download Mol* browser assets.")
    molstar_install.add_argument("--version", default=DEFAULT_MOLSTAR_VERSION)
    molstar_install.add_argument(
        "--destination",
        help=(
            "Directory to write assets into. Defaults to the platform-aware PHEAT cache, "
            f"or {MOLSTAR_ENV_VAR} when set."
        ),
    )
    molstar_install.add_argument("--timeout", type=float, default=60.0)
    molstar_install.add_argument("--force", action="store_true")
    molstar_install.set_defaults(func=_cmd_molstar_install)
    molstar_status = molstar_sub.add_parser("status", help="Show Mol* browser asset status.")
    molstar_status.add_argument("--version", default=DEFAULT_MOLSTAR_VERSION)
    molstar_status.add_argument(
        "--path",
        help=(
            "Directory to inspect. Defaults to the platform-aware PHEAT cache, "
            f"or {MOLSTAR_ENV_VAR} when set."
        ),
    )
    molstar_status.set_defaults(func=_cmd_molstar_status)

    web = subparsers.add_parser("web", help="Run the local PDB/mmCIF roundtrip comparison web app.")
    web.add_argument(
        "--host",
        default=DEFAULT_WEB_HOST,
        help="Bind host. Default: 127.0.0.1 local-only. Use 0.0.0.0 for all IPv4 interfaces.",
    )
    web.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    web.add_argument("--work-dir", default=DEFAULT_WEB_WORK_DIR)
    web.add_argument(
        "--molstar-vendor-dir",
        default=None,
        help=(
            "Optional directory containing molstar.js, molstar.css, and LICENSE. "
            "Defaults to PHEAT's platform-aware Mol* cache."
        ),
    )
    web.add_argument(
        "--random-port-if-taken",
        action="store_true",
        help="Use a random available fallback port if the requested port is already taken.",
    )
    web.set_defaults(func=_cmd_web)

    return parser


def _add_hydrogen_and_bond_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hydrogens",
        choices=HYDROGEN_POLICIES,
        default="drop",
        help="Hydrogen handling policy. Default: drop.",
    )
    parser.add_argument(
        "--store-bonds",
        choices=BOND_STORAGE_MODES,
        default="none",
        help="Optional bond records to store in atom-structure JSON. Default: none.",
    )


def _add_reference_root_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--artifact-version", default=DEFAULT_REFERENCE_ARTIFACT_VERSION)


def _add_external_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--external-timeout",
        type=float,
        help="Timeout in seconds for each external command. Default: no timeout.",
    )
    parser.add_argument(
        "--prep-cache-dir",
        help="Optional preparation/topology cache directory for external scorers.",
    )
    parser.add_argument(
        "--prep-cache-mode",
        choices=PREP_CACHE_MODES,
        default="off",
        help="Preparation/topology cache mode. Default: off.",
    )


def _add_gromacs_options(
    parser: argparse.ArgumentParser,
    *,
    include_run_mode: bool = True,
    include_metrics: bool = True,
) -> None:
    parser.add_argument(
        "--gromacs-forcefield",
        default=DEFAULT_GROMACS_FORCEFIELD,
        help=f"GROMACS pdb2gmx force-field short name. Default: {DEFAULT_GROMACS_FORCEFIELD}.",
    )
    parser.add_argument(
        "--gromacs-water",
        choices=GROMACS_WATER_MODELS,
        default=DEFAULT_GROMACS_WATER,
        help="GROMACS water model. Default: auto, meaning none unless --gromacs-solvate is selected.",
    )
    parser.add_argument(
        "--gromacs-solvate",
        action="store_true",
        help="Build a simple solvated GROMACS box before scoring. Default: off.",
    )
    parser.add_argument(
        "--gromacs-minimize-steps",
        type=int,
        default=500,
        help="GROMACS minimization nsteps. Default: 500.",
    )
    parser.add_argument(
        "--gromacs-emtol",
        type=float,
        default=1000.0,
        help="GROMACS minimization emtol. Default: 1000.0.",
    )
    parser.add_argument(
        "--gromacs-emstep",
        type=float,
        default=0.01,
        help="GROMACS minimization emstep. Default: 0.01.",
    )
    parser.add_argument(
        "--gromacs-box-distance",
        type=float,
        default=1.0,
        help="GROMACS box distance in nm. Default: 1.0.",
    )
    parser.add_argument(
        "--gromacs-cutoff",
        type=float,
        default=1.0,
        help="GROMACS rlist/rcoulomb/rvdw cutoff in nm. Default: 1.0.",
    )
    parser.add_argument(
        "--gromacs-coulombtype",
        default="PME",
        help="GROMACS coulombtype MDP setting. Default: PME.",
    )
    parser.add_argument(
        "--gromacs-vdwtype",
        default="Cut-off",
        help="GROMACS vdwtype MDP setting. Default: Cut-off.",
    )
    parser.add_argument(
        "--gromacs-nstlist",
        type=int,
        default=10,
        help="GROMACS nstlist MDP setting. Default: 10.",
    )
    parser.add_argument(
        "--gromacs-pbc",
        default="xyz",
        help="GROMACS pbc MDP setting. Default: xyz.",
    )
    parser.add_argument(
        "--gromacs-comm-mode",
        default="Linear",
        help="GROMACS comm-mode MDP setting. Default: Linear.",
    )
    parser.add_argument(
        "--gromacs-mdrun-flag",
        action="append",
        default=[],
        help="Additional token passed to gmx mdrun. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--gromacs-grompp-maxwarn",
        type=int,
        default=0,
        help="Add grompp -maxwarn N when N is greater than zero. Default: 0.",
    )
    if include_run_mode:
        parser.add_argument(
            "--gromacs-run-mode",
            choices=GROMACS_RUN_MODES,
            default="rerun",
            help="GROMACS scoring mode. Default: rerun.",
        )
    parser.add_argument(
        "--gromacs-work-dir",
        help="Directory for GROMACS intermediate files.",
    )
    parser.add_argument(
        "--keep-gromacs-files",
        action="store_true",
        help="Keep GROMACS intermediate files for inspection.",
    )
    if include_metrics:
        parser.add_argument(
            "--gromacs-metrics",
            action="store_true",
            help="Also attempt GROMACS validation metrics such as gmx gyrate.",
        )


class _CliStatusStream:
    def __init__(self, terminal_stream: Optional[Any], log_path: Optional[Path]) -> None:
        self.terminal_stream = terminal_stream
        self.log_path = log_path
        self._log_buffer = ""
        self._progress_width = 0
        self._progress_active = False

    def write(self, text: str) -> int:
        if self.terminal_stream is not None:
            self.terminal_stream.write(text)
        self._buffer_log_text(text)
        return len(text)

    def flush(self) -> None:
        if self.terminal_stream is not None:
            self.terminal_stream.flush()
        self._flush_log_buffer()

    def isatty(self) -> bool:
        if self.terminal_stream is None:
            return False
        isatty = getattr(self.terminal_stream, "isatty", None)
        if not callable(isatty):
            return False
        return bool(isatty())

    def status(self, message: str) -> None:
        self._end_progress_line()
        if self.terminal_stream is not None:
            print(message, file=self.terminal_stream, flush=True)
        self._write_log_line(message)

    def progress(self, message: str, *, redraw: bool, final: bool = False) -> None:
        if self.terminal_stream is not None:
            if redraw and self.isatty():
                padding = " " * max(0, self._progress_width - len(message))
                self.terminal_stream.write(f"\r{message}{padding}")
                if final:
                    self.terminal_stream.write("\n")
                    self._progress_active = False
                    self._progress_width = 0
                else:
                    self._progress_active = True
                    self._progress_width = len(message)
                self.terminal_stream.flush()
            else:
                self._end_progress_line()
                print(message, file=self.terminal_stream, flush=True)
        self._write_log_line(message)

    def log_line(self, message: str) -> None:
        self._write_log_line(message)

    def _end_progress_line(self) -> None:
        if self._progress_active and self.terminal_stream is not None:
            self.terminal_stream.write("\n")
            self.terminal_stream.flush()
        self._progress_active = False
        self._progress_width = 0

    def _buffer_log_text(self, text: str) -> None:
        if self.log_path is None:
            return
        self._log_buffer += text
        while "\n" in self._log_buffer:
            line, self._log_buffer = self._log_buffer.split("\n", 1)
            if line.strip():
                self._write_log_line(line.rstrip("\r"))

    def _flush_log_buffer(self) -> None:
        if self._log_buffer.strip():
            self._write_log_line(self._log_buffer.rstrip("\r\n"))
        self._log_buffer = ""

    def _write_log_line(self, message: str) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")


def _build_status_stream(args: argparse.Namespace) -> Optional[_CliStatusStream]:
    log_path = Path(args.log) if getattr(args, "log", None) else None
    terminal_stream = None if getattr(args, "quiet", False) else sys.stderr
    if terminal_stream is None and log_path is None:
        return None
    return _CliStatusStream(terminal_stream, log_path)


def _emit_startup_status(args: argparse.Namespace) -> None:
    stream = _status_stream(args)
    if stream is None:
        return
    lines = [f"pheat {__version__}", f"command: {_command_path(args)}"]
    if getattr(args, "verbose", 0):
        lines.append(f"verbose: {args.verbose}")
    if getattr(args, "log", None):
        lines.append(f"log: {args.log}")
    lines.extend(_command_status_lines(args))
    for line in lines:
        if hasattr(stream, "status"):
            stream.status(line)
        else:
            print(line, file=stream, flush=True)


def _status_stream(args: argparse.Namespace) -> Optional[Any]:
    if hasattr(args, "_status_stream"):
        return args._status_stream
    return _build_status_stream(args)


def _progress_callback(args: argparse.Namespace) -> Optional[Callable[[str], None]]:
    stream = _status_stream(args)
    if stream is None:
        return None

    def emit(message: str) -> None:
        if hasattr(stream, "progress"):
            stream.progress(message, redraw=False, final=False)
        else:
            print(message, file=stream, flush=True)

    return emit


def _omit_large_payloads(result: Mapping[str, Any]) -> dict[str, Any]:
    omitted: dict[str, int] = {}
    compact: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"entries", "holdout_entries"} and isinstance(value, list):
            omitted[key] = len(value)
            continue
        if key == "profiles" and isinstance(value, Mapping):
            compact[key] = {
                str(profile_id): _score_profile_stdout_summary(profile)
                for profile_id, profile in value.items()
            }
            omitted["profile_payloads"] = len(value)
            continue
        compact[key] = value
    if omitted:
        compact["omitted_from_stdout"] = omitted
    return compact


def _score_profile_stdout_summary(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        return {}
    metadata = dict(profile.get("metadata", {})) if isinstance(profile.get("metadata", {}), Mapping) else {}
    warnings = metadata.get("warnings")
    if isinstance(warnings, list) and len(warnings) > 10:
        metadata["warnings"] = warnings[:10]
        metadata["warning_count"] = len(warnings)
        metadata["warnings_omitted_from_stdout"] = len(warnings) - 10
    models = profile.get("models", {})
    return {
        "metadata": metadata,
        "models": sorted(str(model) for model in models) if isinstance(models, Mapping) else [],
    }


def _emit_cli_error(args: argparse.Namespace, exc: Exception) -> None:
    if getattr(args, "verbose", 0) >= 2:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    else:
        print(f"pheat: error: {exc}", file=sys.stderr)
    stream = _status_stream(args)
    if stream is not None and hasattr(stream, "log_line"):
        stream.log_line(f"pheat: error: {exc}")
        if getattr(args, "verbose", 0) >= 2:
            for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
                for part in line.rstrip().splitlines():
                    stream.log_line(part)


def _command_path(args: argparse.Namespace) -> str:
    parts = [str(args.command)]
    if args.command == "examples":
        parts.append(str(args.examples_command))
        if hasattr(args, "name"):
            parts.append(str(args.name))
    elif args.command == "sources":
        parts.append(str(args.sources_command))
        if hasattr(args, "source_id"):
            parts.append(str(args.source_id))
    elif args.command == "archive":
        parts.append(str(args.archive_command))
        if args.archive_command == "snapshots":
            parts.append(str(args.snapshots_command))
            if hasattr(args, "snapshot_id"):
                parts.append(str(args.snapshot_id))
    elif args.command == "geometry":
        parts.append(str(args.geometry_command))
        if args.geometry_command == "tables":
            parts.append(str(args.geometry_tables_command))
    elif args.command == "scoring":
        parts.append(str(args.scoring_command))
    elif args.command == "gromacs":
        parts.append(str(args.gromacs_command))
    elif args.command == "training":
        parts.append(str(args.training_command))
        for name in ("decoys_command", "corpus_command", "tables_command", "features_command", "ml_command"):
            if hasattr(args, name):
                parts.append(str(getattr(args, name)))
    elif args.command == "reference":
        parts.append(str(args.reference_command))
        if hasattr(args, "datasets_command"):
            parts.append(str(args.datasets_command))
    elif args.command == "molstar":
        parts.append(str(args.molstar_command))
    return " ".join(parts)


def _command_status_lines(args: argparse.Namespace) -> list[str]:
    command = str(args.command)
    if command in {
        "pdb-to-structure",
        "mmcif-to-structure",
        "bcif-to-structure",
        "structure-to-pdb",
        "structure-to-mmcif",
        "structure-to-geometry",
        "pdb-to-geometry",
        "mmcif-to-geometry",
        "bcif-to-geometry",
        "geometry-to-structure",
        "centroid",
    }:
        return _conversion_status_lines(args)
    if command == "score":
        return [
            "action: score structure",
            f"input: {args.input}",
            f"model: {args.model}",
            f"domain: {args.domain}",
            f"table set: {_optional_value(args.table_set)}",
            f"profile: {_optional_value(args.profile)}",
            f"profiles: {_optional_value(args.profiles)}",
            f"prepare: {args.prepare}",
            f"amber forcefield: {args.amber_forcefield}",
            f"amber solvent: {args.amber_solvent}",
            f"gromacs forcefield: {args.gromacs_forcefield}",
            f"gromacs water: {args.gromacs_water}",
            f"gromacs run mode: {args.gromacs_run_mode}",
            f"external timeout: {_optional_value(args.external_timeout)}",
            f"prep cache mode: {args.prep_cache_mode}",
            f"output: {_output_destination(args.output)}",
        ]
    if command == "scoring":
        return [
            f"action: scoring {args.scoring_command}",
            f"model: {args.model}",
            f"prepare: {args.prepare}",
            f"external timeout: {_optional_value(args.external_timeout)}",
            f"prep cache mode: {args.prep_cache_mode}",
        ]
    if command == "gromacs":
        output = getattr(args, "output", None) or getattr(args, "json_output", None) or getattr(args, "score_output", None)
        return [
            f"action: gromacs {args.gromacs_command}",
            f"input: {getattr(args, 'input', '(none)')}",
            f"prepare: {args.prepare}",
            f"forcefield: {args.gromacs_forcefield}",
            f"water: {args.gromacs_water}",
            f"solvate: {args.gromacs_solvate}",
            f"run mode: {getattr(args, 'gromacs_run_mode', args.gromacs_command)}",
            f"external timeout: {_optional_value(args.external_timeout)}",
            f"prep cache mode: {args.prep_cache_mode}",
            f"output: {_output_destination(output)}",
        ]
    if command in {"radius-of-gyration", "rg"}:
        return [
            "action: compute radius of gyration",
            f"input: {args.input}",
            f"mode: {args.mode}",
            f"atom set: {args.atom_set}",
            f"output: {_output_destination(args.output)}",
        ]
    if command == "rmsd":
        return [
            "action: compute structure RMSD",
            f"reference: {args.reference}",
            f"target: {args.target}",
            f"atom set: {args.atom_set}",
            f"alignment atom set: {args.alignment_atom_set}",
            f"output: {_output_destination(args.output)}",
        ]
    if command == "examples":
        return _examples_status_lines(args)
    if command == "sources":
        return _sources_status_lines(args)
    if command == "archive":
        return _archive_status_lines(args)
    if command == "geometry":
        return _geometry_status_lines(args)
    if command == "training":
        return _training_status_lines(args)
    if command == "reference":
        return _reference_status_lines(args)
    if command == "molstar":
        status = molstar_asset_status(
            version=getattr(args, "version", DEFAULT_MOLSTAR_VERSION),
            path=getattr(args, "destination", None) or getattr(args, "path", None),
        )
        return [
            f"action: molstar {args.molstar_command}",
            f"asset dir: {status.path}",
            f"asset source: {status.source}",
            f"version: {status.version}",
            f"available: {status.available}",
        ]
    if command == "web":
        return [
            "action: run local web app",
            f"host: {args.host}",
            f"port: {args.port}",
            f"work dir: {args.work_dir}",
            f"Mol* vendor dir: {_optional_value(args.molstar_vendor_dir)}",
            f"random fallback port: {bool(args.random_port_if_taken)}",
        ]
    return ["action: run command"]


def _conversion_status_lines(args: argparse.Namespace) -> list[str]:
    command = str(args.command)
    action_by_command = {
        "pdb-to-structure": "convert PDB to atom-structure JSON",
        "mmcif-to-structure": "convert mmCIF to atom-structure JSON",
        "bcif-to-structure": "convert BinaryCIF to atom-structure JSON",
        "structure-to-pdb": "convert atom-structure JSON to PDB",
        "structure-to-mmcif": "convert atom-structure JSON to mmCIF",
        "structure-to-geometry": "extract residue geometry from atom structure",
        "pdb-to-geometry": "convert PDB to residue geometry",
        "mmcif-to-geometry": "convert mmCIF to residue geometry",
        "bcif-to-geometry": "convert BinaryCIF to residue geometry",
        "geometry-to-structure": "reconstruct atom structure from residue geometry",
        "centroid": "convert heavy atoms to side-chain centroids",
    }
    lines = [
        f"action: {action_by_command[command]}",
        f"input: {args.input}",
        f"output: {args.output}",
    ]
    if hasattr(args, "angle_units") and args.angle_units is not None:
        lines.append(f"angle units: {args.angle_units}")
    if getattr(args, "store_angles", ""):
        lines.append(f"stored angles: {args.store_angles}")
    if getattr(args, "store_lengths", ""):
        lines.append(f"stored lengths: {args.store_lengths}")
    if getattr(args, "hydrogens", None) is not None:
        lines.append(f"hydrogens: {args.hydrogens}")
    if getattr(args, "store_bonds", None) is not None:
        lines.append(f"stored bonds: {args.store_bonds}")
    if getattr(args, "max_chi", None) is not None:
        lines.append(f"max chi: {args.max_chi}")
    if getattr(args, "chain_id_source", None) is not None:
        lines.append(f"chain ID source: {args.chain_id_source}")
    if getattr(args, "model", None) is not None:
        lines.append(f"model: {args.model}")
    if getattr(args, "mode", None) is not None:
        lines.append(f"mode: {args.mode}")
    if getattr(args, "pdb_output", None):
        lines.append(f"PDB output: {args.pdb_output}")
    if getattr(args, "mmcif_output", None):
        lines.append(f"mmCIF output: {args.mmcif_output}")
    if getattr(args, "include_terminal_oxt", False):
        lines.append("terminal OXT: included")
    if getattr(args, "geometry_mode", None) is not None:
        lines.append(f"geometry mode: {args.geometry_mode}")
    if getattr(args, "geometry_table", None):
        lines.append(f"geometry table: {args.geometry_table}")
    if getattr(args, "geometry_profile", None):
        lines.append(f"geometry profile: {args.geometry_profile}")
    if getattr(args, "allow_pdb_chain_truncation", False):
        lines.append("PDB chain truncation: allowed")
    return lines


def _examples_status_lines(args: argparse.Namespace) -> list[str]:
    if args.examples_command == "list":
        return ["action: list example sets"]
    if args.examples_command == "show":
        return ["action: show example manifest", f"example set: {args.name}"]
    if args.examples_command == "fetch":
        return [
            "action: fetch example set",
            f"example set: {args.name}",
            f"destination: {args.destination}",
        ]
    return [
        "action: run example set",
        f"example set: {args.name}",
        f"cache dir: {args.cache_dir}",
        f"model: {args.model}",
        f"fetch missing inputs: {bool(args.fetch)}",
        f"output: {_output_destination(args.output)}",
    ]


def _sources_status_lines(args: argparse.Namespace) -> list[str]:
    if args.sources_command == "list":
        return ["action: list source data records"]
    if args.sources_command == "fetch":
        return [
            "action: fetch source data",
            f"source ID: {args.source_id}",
            f"destination: {args.destination}",
        ]
    return ["action: verify source data", f"cache dir: {args.cache_dir}"]


def _archive_status_lines(args: argparse.Namespace) -> list[str]:
    if args.archive_command == "download":
        return [
            "action: download or plan archive coordinate corpus",
            f"source: {_archive_download_source(args)}",
            f"format: {args.format}",
            f"output root: {args.output_root}",
            f"raw dir: {_archive_raw_dir(args)}",
            f"staging dir: {_optional_value(args.staging_dir)}",
            f"reuse mode: {args.reuse_mode}",
            f"dry run: {bool(args.dry_run)}",
            f"automatic yes: {bool(args.yes)}",
            f"progress redraw: {not bool(args.no_progress_redraw)}",
            f"progress interval: {args.progress_interval}",
            f"progress seconds: {_optional_value(args.progress_seconds)}",
            f"prefetch metadata: {bool(args.prefetch_metadata)}",
            f"metadata source: {args.metadata_source}",
        ]
    if args.archive_command == "snapshots":
        return _archive_snapshot_status_lines(args)
    return ["action: archive command"]


def _archive_snapshot_status_lines(args: argparse.Namespace) -> list[str]:
    if args.snapshots_command == "list":
        return ["action: list archive snapshot presets"]
    snapshot = get_snapshot(args.snapshot_id)
    if args.snapshots_command == "describe":
        return ["action: describe archive snapshot", f"snapshot: {snapshot.id}"]
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    if args.snapshots_command == "verify":
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_root / "manifests"
        return [
            "action: verify archive snapshot",
            f"snapshot: {snapshot.id}",
            f"output root: {output_root}",
            f"manifest dir: {manifest_dir}",
        ]
    if args.snapshots_command == "ids":
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_root / "manifests"
        statuses = args.status or list(SUCCESSFUL_ARCHIVE_STATUSES)
        return [
            "action: write local archive snapshot IDs",
            f"snapshot: {snapshot.id}",
            f"output root: {output_root}",
            f"manifest dir: {manifest_dir}",
            f"statuses: {','.join(statuses)}",
            f"output: {_output_destination(args.output)}",
        ]
    if args.snapshots_command == "metadata":
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_root / "manifests"
        return [
            "action: extract archive snapshot metadata",
            f"snapshot: {snapshot.id}",
            f"output root: {output_root}",
            f"manifest dir: {manifest_dir}",
            f"source: {args.source}",
            f"workers: {args.workers}",
            f"output: {_output_destination(args.output)}",
        ]
    if args.snapshots_command == "relocate":
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_root / "manifests"
        return [
            "action: relocate archive snapshot manifest paths",
            f"snapshot: {snapshot.id}",
            f"output root: {output_root}",
            f"manifest dir: {manifest_dir}",
            f"write: {bool(args.write)}",
            f"relative paths: {not bool(args.absolute_paths)}",
        ]
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_root / "raw"
    return [
        "action: download or plan archive snapshot",
        f"snapshot: {snapshot.id}",
        f"format: {snapshot.file_format}",
        f"output root: {output_root}",
        f"raw dir: {raw_dir}",
        f"staging dir: {_optional_value(args.staging_dir)}",
        f"reuse mode: {args.reuse_mode}",
        f"dry run: {bool(args.dry_run)}",
        f"automatic yes: {bool(args.yes)}",
        f"progress redraw: {not bool(args.no_progress_redraw)}",
        f"progress interval: {args.progress_interval}",
        f"progress seconds: {_optional_value(args.progress_seconds)}",
        f"prefetch metadata: {bool(args.prefetch_metadata)}",
        f"metadata source: {args.metadata_source}",
    ]


def _geometry_status_lines(args: argparse.Namespace) -> list[str]:
    if args.geometry_command != "tables":
        return ["action: geometry command"]
    if args.geometry_tables_command == "build-backbone":
        return [
            "action: build PHEAT backbone geometry tables",
            f"training set: {args.training_set}",
            f"output root: {args.output_root}",
            f"domain: {args.domain}",
            f"max entries: {_optional_value(args.max_entries)}",
            f"table set id: {_optional_value(args.table_set_id)}",
            f"table set version: {args.table_set_version}",
        ]
    if args.geometry_tables_command == "build-cdl":
        return [
            "action: build PHEAT conformation-dependent backbone geometry tables",
            f"training set: {args.training_set}",
            f"output root: {args.output_root}",
            f"domain: {args.domain}",
            f"max entries: {_optional_value(args.max_entries)}",
            f"phi/psi bin size: {args.phi_psi_bin_size}",
            f"min bin count: {args.min_bin_count}",
            f"smoothing: {args.smoothing}",
            f"residue classes: {args.residue_classes}",
            f"table set id: {_optional_value(args.table_set_id)}",
            f"table set version: {args.table_set_version}",
        ]
    if args.geometry_tables_command == "import-cdl":
        return [
            "action: import CDL-like geometry tables",
            f"input: {args.input}",
            f"output root: {args.output_root}",
            f"table set id: {args.table_set_id}",
            f"table set version: {args.table_set_version}",
            f"source license: {_optional_value(args.source_license)}",
        ]
    if args.geometry_tables_command == "build-sidechain-ccd":
        return [
            "action: build PHEAT CCD sidechain geometry tables",
            f"CCD full: {_optional_value(args.ccd_full)}",
            f"CCD dir: {_optional_value(args.ccd_dir)}",
            f"CCD BCIF dir: {_optional_value(args.ccd_bcif_dir)}",
            f"output root: {args.output_root}",
            f"residues: {_optional_value(args.residues)}",
            f"table set id: {args.table_set_id}",
            f"table set version: {args.table_set_version}",
        ]
    if args.geometry_tables_command == "list":
        return ["action: list packaged PHEAT geometry tables"]
    if args.geometry_tables_command == "describe":
        return [
            "action: describe PHEAT geometry tables",
            f"table set: {args.table_set}",
        ]
    return [
        "action: validate PHEAT geometry tables",
        f"table set: {args.table_set}",
    ]


def _training_status_lines(args: argparse.Namespace) -> list[str]:
    if args.training_command == "decoys":
        if args.decoys_command == "list":
            return ["action: list training decoy datasets"]
        if args.decoys_command == "describe":
            return ["action: describe training decoy dataset", f"dataset: {args.dataset_id}"]
        if args.decoys_command == "fetch":
            return [
                "action: fetch training decoy dataset metadata",
                f"dataset: {args.dataset_id}",
                f"output root: {args.output_root}",
                f"automatic yes: {bool(args.yes)}",
            ]
        return ["action: verify training decoy datasets", f"input root: {args.input_root}"]
    if args.training_command == "corpus":
        if args.corpus_command == "inventory":
            return [
                "action: inventory local training snapshot",
                f"snapshot root: {args.snapshot_root}",
                f"domain: {args.domain}",
                f"metadata JSONL: {_optional_value(args.metadata_jsonl)}",
                f"max entries: {_optional_value(args.max_entries)}",
                f"output: {args.output}",
            ]
        if args.corpus_command == "describe":
            return [
                "action: describe training corpus",
                f"training set: {args.training_set}",
            ]
        return [
            "action: select local training corpus",
            f"inventory: {args.inventory}",
            f"output root: {args.output_root}",
            f"sequence identity: {args.sequence_identity}",
            f"sequence clusters: {_optional_value(args.sequence_clusters)}",
            f"corpus id: {_optional_value(args.corpus_id)}",
            f"corpus version: {args.corpus_version}",
            f"include file: {_optional_value(args.include_file)}",
            f"exclude file: {_optional_value(args.exclude_file)}",
            f"holdout file: {_optional_value(args.holdout_file)}",
        ]
    if args.training_command == "tables":
        if args.tables_command == "build":
            return [
                "action: build PHEAT score tables",
                f"training set: {args.training_set}",
                f"output root: {args.output_root}",
                f"models: {args.models}",
                f"domain: {args.domain}",
                f"burial method: {args.burial_method}",
                f"SASA backend: {args.sasa_backend}",
                f"table set id: {_optional_value(args.table_set_id)}",
                f"table set version: {args.table_set_version}",
            ]
        if args.tables_command == "describe":
            return [
                "action: describe PHEAT score tables",
                f"table set: {args.table_set}",
            ]
        return [
            "action: validate PHEAT score tables",
            f"table set: {args.table_set}",
            f"decoy root: {_optional_value(args.decoy_root)}",
        ]
    if args.training_command == "features":
        return [
            "action: extract training features",
            f"training set: {args.training_set}",
            f"models: {args.models}",
            f"domain: {args.domain}",
            f"output: {args.output}",
        ]
    if args.training_command == "ml":
        return [
            "action: train lightweight linear ML scorer",
            f"features: {args.features}",
            f"target: {args.target}",
            f"output: {args.output}",
        ]
    return ["action: training command"]


def _reference_status_lines(args: argparse.Namespace) -> list[str]:
    command = args.reference_command
    if command == "datasets":
        if args.datasets_command == "list":
            return ["action: list reference decoy datasets"]
        return ["action: describe reference decoy dataset", f"dataset: {args.dataset_id}"]
    if command == "validate-spec":
        return ["action: validate corpus spec", f"corpus spec: {args.corpus_spec}"]
    if command == "summarize-spec":
        return ["action: summarize corpus spec", f"corpus spec: {args.corpus_spec}"]
    if command == "init-spec":
        return [
            "action: initialize corpus spec",
            f"template: {args.template}",
            f"output: {args.output}",
        ]
    if command == "build":
        return [
            "action: build local reference corpus",
            f"corpus spec: {args.corpus_spec}",
            f"output root: {args.output_root}",
            f"ccd: {_optional_value(args.ccd)}",
            f"dry run: {bool(args.dry_run)}",
            f"overwrite: {bool(args.overwrite)}",
        ]
    if command == "fetch":
        return [
            "action: fetch or register reference inputs",
            f"reference root: {args.reference_root}",
            f"artifact version: {args.artifact_version}",
            f"snapshot: {args.snapshot_id}",
            f"snapshot root: {_optional_value(args.snapshot_root)}",
            f"datasets: {','.join(args.dataset) if args.dataset else '(none)'}",
            f"include payloads: {bool(args.include_payloads)}",
            f"workers: {args.workers}",
            f"overwrite: {bool(args.overwrite)}",
        ]
    if command == "inventory":
        return [
            "action: build reference inventory",
            f"reference root: {args.reference_root}",
            f"artifact version: {args.artifact_version}",
            f"snapshot root: {_optional_value(args.snapshot_root)}",
            f"domain: {args.domain}",
            f"workers: {args.workers}",
            f"output: {_output_destination(args.output)}",
        ]
    if command == "select":
        return [
            "action: select reference corpus",
            f"inventory: {args.inventory}",
            f"subset: {args.subset}",
            f"method: {args.method}",
            f"sequence identity: {args.sequence_identity}",
            f"max resolution: {_optional_value(args.max_resolution)}",
            f"artifact version: {args.artifact_version}",
        ]
    if command == "build-decoys":
        return [
            "action: build PHEAT reference decoys",
            f"training set: {args.training_set}",
            f"recipes: {args.recipes}",
            f"domain: {args.domain}",
            f"seed: {args.seed}",
            f"attempts per decoy: {args.attempts_per_decoy}",
            f"workers: {args.workers}",
        ]
    if command == "build-scores":
        return [
            "action: build reference score tables",
            f"training set: {args.training_set}",
            f"models: {args.models}",
            f"burial method: {args.burial_method}",
            f"artifact version: {args.artifact_version}",
            f"workers: {args.workers}",
        ]
    if command == "extract-features":
        return [
            "action: extract reference ML features",
            f"training set: {args.training_set}",
            f"decoys: {_optional_value(args.decoys)}",
            f"models: {args.models}",
            f"max entries: {_optional_value(args.max_entries)}",
            f"workers: {args.workers}",
        ]
    if command == "train-ml":
        return [
            "action: train reference ML scorer",
            f"features: {args.features}",
            f"target: {args.target}",
            f"artifact version: {args.artifact_version}",
        ]
    if command == "validate":
        return [
            "action: validate reference features",
            f"features: {args.features}",
            f"artifact version: {args.artifact_version}",
        ]
    if command == "promote":
        return [
            "action: promote reference artifact",
            f"source: {args.source}",
            f"destination: {args.destination}",
        ]
    if command == "package-scoring-assets":
        destination = args.destination_root or str(Path("src/pheat/data/scoring") / args.artifact_version)
        return [
            "action: package reference scoring assets",
            f"reference root: {args.reference_root}",
            f"artifact version: {args.artifact_version}",
            f"destination root: {destination}",
        ]
    if command == "audit-version":
        return [
            "action: audit reference artifact version",
            f"reference root: {args.reference_root}",
            f"artifact version: {args.artifact_version}",
        ]
    if command == "run-unattended":
        return [
            "action: run unattended reference build",
            f"reference root: {args.reference_root}",
            f"artifact version: {args.artifact_version}",
            f"snapshot: {args.snapshot_id}",
            f"snapshot root: {_optional_value(args.snapshot_root)}",
            f"domain: {args.domain}",
            f"method: {args.method}",
            f"sequence identity: {args.sequence_identity}",
            f"max resolution: {_optional_value(args.max_resolution)}",
            f"decoy recipes: {args.decoy_recipes}",
            f"attempts per decoy: {args.attempts_per_decoy}",
            f"workers: {args.workers}",
            f"dry run: {bool(args.dry_run)}",
            f"backup existing: {bool(args.backup_existing)}",
            f"canary entries: {args.canary_entries if not args.skip_canary else 0}",
        ]
    return ["action: reference command"]


def _archive_download_source(args: argparse.Namespace) -> str:
    if args.all_current:
        return "current RCSB/wwPDB holdings"
    if args.ids_file:
        return f"IDs file {args.ids_file}"
    if args.query_json:
        return f"RCSB query JSON {args.query_json}"
    return "unknown"


def _archive_raw_dir(args: argparse.Namespace) -> Path:
    return Path(args.raw_dir) if args.raw_dir else Path(args.output_root) / "raw"


def _output_destination(value: Optional[str]) -> str:
    return value if value else "stdout"


def _optional_value(value: object) -> str:
    return str(value) if value is not None and value != "" else "not set"


def _cmd_pdb_to_heavy(args: argparse.Namespace) -> int:
    structure = load_pdb(args.input, hydrogens=args.hydrogens, store_bonds=args.store_bonds)
    write_heavy_json(structure, args.output)
    return 0


def _cmd_mmcif_to_heavy(args: argparse.Namespace) -> int:
    structure = load_mmcif(
        args.input,
        chain_id_source=args.chain_id_source,
        hydrogens=args.hydrogens,
        store_bonds=args.store_bonds,
    )
    write_heavy_json(structure, args.output)
    return 0


def _cmd_bcif_to_heavy(args: argparse.Namespace) -> int:
    structure = load_bcif(args.input, model=args.model, hydrogens=args.hydrogens, store_bonds=args.store_bonds)
    write_heavy_json(structure, args.output)
    return 0


def _cmd_heavy_to_pdb(args: argparse.Namespace) -> int:
    write_pdb(
        load_heavy_json(args.input),
        args.output,
        allow_chain_truncation=args.allow_pdb_chain_truncation,
    )
    return 0


def _cmd_heavy_to_mmcif(args: argparse.Namespace) -> int:
    write_mmcif(load_heavy_json(args.input), args.output)
    return 0


def _cmd_heavy_to_geometry(args: argparse.Namespace) -> int:
    write_residue_geometry_json(
        structure_to_residue_geometry(
            load_heavy_json(args.input),
            angle_units=args.angle_units,
            stored_angles=args.store_angles,
            stored_lengths=args.store_lengths,
            max_chi=args.max_chi,
        ),
        args.output,
    )
    return 0


def _cmd_pdb_to_geometry(args: argparse.Namespace) -> int:
    write_residue_geometry_json(
        structure_to_residue_geometry(
            load_pdb(args.input, hydrogens=args.hydrogens),
            angle_units=args.angle_units,
            stored_angles=args.store_angles,
            stored_lengths=args.store_lengths,
            max_chi=args.max_chi,
        ),
        args.output,
    )
    return 0


def _cmd_mmcif_to_geometry(args: argparse.Namespace) -> int:
    write_residue_geometry_json(
        structure_to_residue_geometry(
            load_mmcif(args.input, chain_id_source=args.chain_id_source, hydrogens=args.hydrogens),
            angle_units=args.angle_units,
            stored_angles=args.store_angles,
            stored_lengths=args.store_lengths,
            max_chi=args.max_chi,
        ),
        args.output,
    )
    return 0


def _cmd_bcif_to_geometry(args: argparse.Namespace) -> int:
    write_residue_geometry_json(
        structure_to_residue_geometry(
            load_bcif(args.input, model=args.model, hydrogens=args.hydrogens),
            angle_units=args.angle_units,
            stored_angles=args.store_angles,
            stored_lengths=args.store_lengths,
            max_chi=args.max_chi,
        ),
        args.output,
    )
    return 0


def _cmd_geometry_to_heavy(args: argparse.Namespace) -> int:
    structure = structure_from_residue_geometry(
        args.input,
        angle_units=args.angle_units,
        include_terminal_oxt=args.include_terminal_oxt,
        geometry_mode=args.geometry_mode,
        geometry_table=args.geometry_table,
        geometry_profile=args.geometry_profile,
    )
    if args.hydrogens == "generate":
        structure = generate_hydrogens(structure)
    structure = structure_with_bonds(structure, mode=args.store_bonds)
    write_heavy_json(structure, args.output)
    if args.pdb_output:
        write_pdb(
            structure,
            args.pdb_output,
            allow_chain_truncation=args.allow_pdb_chain_truncation,
        )
    if args.mmcif_output:
        write_mmcif(structure, args.mmcif_output)
    return 0


def _cmd_centroid(args: argparse.Namespace) -> int:
    structure = load_heavy_json(args.input)
    centroids = to_centroid_structure(structure, mode=args.mode)
    _write_model_json(args.output, centroids.to_json())
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input, hydrogens=args.hydrogens)
    if args.prepare == "write" and not args.prepared_output:
        raise ValueError("--prepare write requires --prepared-output")
    if args.profiles:
        if not args.table_set:
            raise ValueError("--profiles requires --table-set")
        payload = score_structure_profiles(
            structure,
            model=args.model,
            domain=args.domain,
            table_set=args.table_set,
            profiles=_comma_values(args.profiles),
            burial_method=args.burial_method,
            sasa_backend=args.sasa_backend,
        )
        if args.output:
            _write_json(args.output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = score_structure(
        structure,
        model=args.model,
        domain=args.domain,
        table_set=args.table_set,
        profile=args.profile,
        burial_method=args.burial_method,
        sasa_backend=args.sasa_backend,
        prepare=args.prepare,
        prepared_output=args.prepared_output,
        amber_forcefield=args.amber_forcefield,
        amber_solvent=args.amber_solvent,
        ambertools_work_dir=args.ambertools_work_dir,
        keep_ambertools_files=args.keep_ambertools_files,
        gromacs_forcefield=args.gromacs_forcefield,
        gromacs_water=args.gromacs_water,
        gromacs_solvate=args.gromacs_solvate,
        gromacs_run_mode=args.gromacs_run_mode,
        gromacs_work_dir=args.gromacs_work_dir,
        keep_gromacs_files=args.keep_gromacs_files,
        gromacs_metrics=args.gromacs_metrics,
        gromacs_run_settings=_gromacs_run_settings_from_args(args),
        external_timeout_seconds=args.external_timeout,
        prep_cache_dir=args.prep_cache_dir,
        prep_cache_mode=args.prep_cache_mode,
    )
    if args.output:
        _write_model_json(args.output, result.to_json())
    else:
        print(result.to_json())
    return 0


def _cmd_scoring_validate_options(args: argparse.Namespace) -> int:
    payload = _validate_external_options_payload(args, model=args.model)
    if payload.get("ok"):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2


def _cmd_gromacs_prepare(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input, hydrogens=args.hydrogens)
    payload = prepare_gromacs_structure(
        structure,
        prepare=args.prepare,
        output=args.output,
        topology_output=args.topology,
        gromacs_forcefield=args.gromacs_forcefield,
        gromacs_water=args.gromacs_water,
        gromacs_solvate=args.gromacs_solvate,
        gromacs_work_dir=args.gromacs_work_dir,
        keep_gromacs_files=args.keep_gromacs_files,
        gromacs_run_settings=_gromacs_run_settings_from_args(args),
        external_timeout_seconds=args.external_timeout,
        prep_cache_dir=args.prep_cache_dir,
        prep_cache_mode=args.prep_cache_mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_gromacs_score(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input, hydrogens=args.hydrogens)
    if args.prepare == "write" and not args.prepared_output:
        raise ValueError("--prepare write requires --prepared-output")
    result = _score_with_gromacs_args(structure, args, run_mode=args.gromacs_run_mode)
    if args.output:
        _write_model_json(args.output, result.to_json())
    else:
        print(result.to_json())
    return 0


def _cmd_gromacs_minimize(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input, hydrogens=args.hydrogens)
    result = _score_with_gromacs_args(
        structure,
        args,
        run_mode="minimize",
        prepared_output=args.output,
    )
    if args.score_output:
        _write_model_json(args.score_output, result.to_json())
    else:
        print(result.to_json())
    return 0


def _cmd_gromacs_validate(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input, hydrogens=args.hydrogens)
    result = _score_with_gromacs_args(
        structure,
        args,
        run_mode=args.gromacs_run_mode,
        metrics=True,
    )
    payload = {
        "format": "pheat.gromacs-validation",
        "version": 1,
        "input": args.input,
        "score": json.loads(result.to_json()),
        "radius_of_gyration": radius_of_gyration_result(structure, input_name=args.input, mode="both"),
    }
    if args.json_output:
        _write_json(args.json_output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_gromacs_validate_options(args: argparse.Namespace) -> int:
    payload = _validate_external_options_payload(args, model="gromacs-mdrun")
    if payload.get("ok"):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2


def _score_with_gromacs_args(
    structure: HeavyAtomStructure,
    args: argparse.Namespace,
    *,
    run_mode: str,
    prepared_output: Optional[str] = None,
    metrics: Optional[bool] = None,
):
    return score_structure(
        structure,
        model="gromacs-mdrun",
        domain=args.domain,
        prepare=args.prepare,
        prepared_output=prepared_output if prepared_output is not None else getattr(args, "prepared_output", None),
        gromacs_forcefield=args.gromacs_forcefield,
        gromacs_water=args.gromacs_water,
        gromacs_solvate=args.gromacs_solvate,
        gromacs_run_mode=run_mode,
        gromacs_work_dir=args.gromacs_work_dir,
        keep_gromacs_files=args.keep_gromacs_files,
        gromacs_metrics=args.gromacs_metrics if metrics is None else metrics,
        gromacs_run_settings=_gromacs_run_settings_from_args(args),
        external_timeout_seconds=args.external_timeout,
        prep_cache_dir=args.prep_cache_dir,
        prep_cache_mode=args.prep_cache_mode,
    )


def _gromacs_run_settings_from_args(args: argparse.Namespace) -> GromacsRunSettings:
    return GromacsRunSettings(
        minimize_steps=args.gromacs_minimize_steps,
        emtol=args.gromacs_emtol,
        emstep=args.gromacs_emstep,
        box_distance_nm=args.gromacs_box_distance,
        cutoff_nm=args.gromacs_cutoff,
        coulombtype=args.gromacs_coulombtype,
        vdwtype=args.gromacs_vdwtype,
        nstlist=args.gromacs_nstlist,
        pbc=args.gromacs_pbc,
        comm_mode=args.gromacs_comm_mode,
        mdrun_flags=tuple(args.gromacs_mdrun_flag or ()),
        grompp_maxwarn=args.gromacs_grompp_maxwarn,
    )


def _validate_external_options_payload(args: argparse.Namespace, *, model: str) -> dict[str, Any]:
    return validate_external_scoring_options(
        model=model,
        prepare=args.prepare,
        amber_solvent=getattr(args, "amber_solvent", "vacuum"),
        gromacs_forcefield=args.gromacs_forcefield,
        gromacs_water=args.gromacs_water,
        gromacs_solvate=args.gromacs_solvate,
        gromacs_run_mode=getattr(args, "gromacs_run_mode", "rerun"),
        gromacs_run_settings=_gromacs_run_settings_from_args(args),
        external_timeout_seconds=args.external_timeout,
        prep_cache_dir=args.prep_cache_dir,
        prep_cache_mode=args.prep_cache_mode,
    )


def _comma_values(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _cmd_radius_of_gyration(args: argparse.Namespace) -> int:
    structure = _load_structure(args.input)
    payload = radius_of_gyration_result(
        structure,
        input_name=args.input,
        mode=args.mode,
        atom_set=args.atom_set,
    )
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_rmsd(args: argparse.Namespace) -> int:
    reference = _load_structure(args.reference)
    target = _load_structure(args.target)
    payload = structure_rmsd_result(
        reference,
        target,
        reference_name=args.reference,
        target_name=args.target,
        atom_set=args.atom_set,
        alignment_atom_set=args.alignment_atom_set,
    )
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_examples_list(args: argparse.Namespace) -> int:
    for name in list_example_sets():
        print(name)
    return 0


def _cmd_examples_show(args: argparse.Namespace) -> int:
    print(json.dumps(load_example_manifest(args.name), indent=2, sort_keys=True))
    return 0


def _cmd_examples_fetch(args: argparse.Namespace) -> int:
    paths = fetch_example_set(args.name, args.destination)
    print(json.dumps([str(path) for path in paths], indent=2))
    return 0


def _cmd_examples_run(args: argparse.Namespace) -> int:
    payload = run_example_set(args.name, cache_dir=args.cache_dir, model=args.model, fetch=args.fetch)
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_sources_list(args: argparse.Namespace) -> int:
    print(json.dumps(list_sources(), indent=2, sort_keys=True))
    return 0


def _cmd_sources_fetch(args: argparse.Namespace) -> int:
    path = fetch_source(args.source_id, args.destination)
    if isinstance(path, list):
        print(json.dumps([str(item) for item in path], indent=2, sort_keys=True))
    else:
        print(path)
    return 0


def _cmd_sources_verify(args: argparse.Namespace) -> int:
    print(json.dumps(verify_sources(args.cache_dir), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_build_backbone(args: argparse.Namespace) -> int:
    result = build_backbone_geometry_tables(
        Path(args.training_set),
        output_root=Path(args.output_root),
        domain=args.domain,
        max_entries=args.max_entries,
        table_set_id=args.table_set_id,
        table_set_version=args.table_set_version,
        command_args=getattr(args, "_argv", None),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_build_cdl(args: argparse.Namespace) -> int:
    result = build_cdl_geometry_tables(
        Path(args.training_set),
        output_root=Path(args.output_root),
        domain=args.domain,
        max_entries=args.max_entries,
        table_set_id=args.table_set_id,
        table_set_version=args.table_set_version,
        phi_psi_bin_size=args.phi_psi_bin_size,
        min_bin_count=args.min_bin_count,
        smoothing=args.smoothing,
        residue_classes=args.residue_classes,
        command_args=getattr(args, "_argv", None),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_import_cdl(args: argparse.Namespace) -> int:
    result = import_cdl_geometry_tables(
        Path(args.input),
        output_root=Path(args.output_root),
        table_set_id=args.table_set_id,
        table_set_version=args.table_set_version,
        source_license=args.source_license,
        command_args=getattr(args, "_argv", None),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_build_sidechain_ccd(args: argparse.Namespace) -> int:
    result = build_sidechain_ccd_geometry_tables(
        Path(args.ccd_dir) if args.ccd_dir else None,
        output_root=Path(args.output_root),
        ccd_full=Path(args.ccd_full) if args.ccd_full else None,
        ccd_bcif_dir=Path(args.ccd_bcif_dir) if args.ccd_bcif_dir else None,
        residues=_comma_values(args.residues) if args.residues else None,
        table_set_id=args.table_set_id,
        table_set_version=args.table_set_version,
        command_args=getattr(args, "_argv", None),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_list(args: argparse.Namespace) -> int:
    print(json.dumps(list_packaged_geometry_tables(), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_describe(args: argparse.Namespace) -> int:
    print(json.dumps(describe_geometry_tables(args.table_set), indent=2, sort_keys=True))
    return 0


def _cmd_geometry_tables_validate(args: argparse.Namespace) -> int:
    result = validate_geometry_tables(args.table_set)
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_archive_download(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            run_download_from_args(args, status_stream=_status_stream(args)),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_archive_snapshots_list(args: argparse.Namespace) -> int:
    snapshots = list_snapshots()
    if args.text:
        for snapshot in snapshots:
            print(f"{snapshot['id']}\t{snapshot['format']}\t{snapshot['name']}")
    else:
        print(json.dumps(snapshots, indent=2, sort_keys=True))
    return 0


def _cmd_archive_snapshots_describe(args: argparse.Namespace) -> int:
    print(json.dumps(snapshot_to_dict(get_snapshot(args.snapshot_id)), indent=2, sort_keys=True))
    return 0


def _cmd_archive_snapshots_download(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            run_snapshot_download_from_args(args, status_stream=_status_stream(args)),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_archive_snapshots_verify(args: argparse.Namespace) -> int:
    result = verify_snapshot_from_args(args)
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_archive_snapshots_metadata(args: argparse.Namespace) -> int:
    result = snapshot_metadata_from_args(args, status_stream=_status_stream(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_archive_snapshots_relocate(args: argparse.Namespace) -> int:
    result = relocate_snapshot_manifest_from_args(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["missing"] == 0 else 1


def _cmd_archive_snapshots_ids(args: argparse.Namespace) -> int:
    snapshot = get_snapshot(args.snapshot_id)
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    statuses = args.status or list(SUCCESSFUL_ARCHIVE_STATUSES)
    ids = write_snapshot_ids(
        output_root,
        Path(args.output) if args.output else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        statuses=statuses,
    )
    if args.output:
        print(json.dumps({"snapshot_id": snapshot.id, "count": len(ids), "output": args.output}, indent=2, sort_keys=True))
    else:
        print("".join(f"{pdb_id}\n" for pdb_id in ids), end="")
    return 0


def _cmd_training_decoys_list(args: argparse.Namespace) -> int:
    print(json.dumps(list_decoy_datasets(), indent=2, sort_keys=True))
    return 0


def _cmd_training_decoys_describe(args: argparse.Namespace) -> int:
    print(json.dumps(get_decoy_dataset(args.dataset_id), indent=2, sort_keys=True))
    return 0


def _cmd_training_decoys_fetch(args: argparse.Namespace) -> int:
    result = fetch_decoy_dataset(
        args.dataset_id,
        output_root=Path(args.output_root),
        yes=args.yes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_training_decoys_verify(args: argparse.Namespace) -> int:
    result = verify_decoy_root(Path(args.input_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_training_corpus_inventory(args: argparse.Namespace) -> int:
    result = inventory_snapshot(
        InventoryOptions(
            snapshot_root=Path(args.snapshot_root),
            output=Path(args.output),
            domain=args.domain,
            metadata_jsonl=Path(args.metadata_jsonl) if args.metadata_jsonl else None,
            max_entries=args.max_entries,
            workers=args.workers,
            progress=_progress_callback(args),
        )
    )
    print(json.dumps({"count": len(result), "output": args.output}, indent=2, sort_keys=True))
    return 0


def _cmd_training_corpus_select(args: argparse.Namespace) -> int:
    result = select_corpus(
        SelectionOptions(
            inventory=Path(args.inventory),
            output_root=Path(args.output_root),
            method=args.method,
            max_resolution=args.max_resolution,
            sequence_identity=args.sequence_identity,
            canonical_only=args.canonical_only,
            min_length=args.min_length,
            max_length=args.max_length,
            sequence_clusters=Path(args.sequence_clusters) if args.sequence_clusters else None,
            corpus_id=args.corpus_id,
            corpus_version=args.corpus_version,
            include_file=Path(args.include_file) if args.include_file else None,
            exclude_file=Path(args.exclude_file) if args.exclude_file else None,
            holdout_file=Path(args.holdout_file) if args.holdout_file else None,
            command_args=getattr(args, "_argv", None),
        )
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_training_corpus_describe(args: argparse.Namespace) -> int:
    result = describe_training_corpus(Path(args.training_set))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_training_tables_build(args: argparse.Namespace) -> int:
    result = build_score_tables(
        Path(args.training_set),
        output_root=Path(args.output_root),
        models=normalize_model_list(args.models),
        domain=args.domain,
        burial_method=args.burial_method,
        sasa_backend=args.sasa_backend,
        max_entries=args.max_entries,
        table_set_id=args.table_set_id,
        table_set_version=args.table_set_version,
        command_args=getattr(args, "_argv", None),
        workers=args.workers,
        progress=_progress_callback(args),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_training_tables_describe(args: argparse.Namespace) -> int:
    result = describe_score_tables(Path(args.table_set))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_training_tables_validate(args: argparse.Namespace) -> int:
    result = validate_tables(
        Path(args.table_set),
        decoy_root=Path(args.decoy_root) if args.decoy_root else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_training_features_extract(args: argparse.Namespace) -> int:
    result = extract_features(
        Path(args.training_set),
        output=Path(args.output),
        models=[token.strip() for token in args.models.split(",") if token.strip()],
        domain=args.domain,
        workers=args.workers,
        progress=_progress_callback(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_training_ml_train_linear(args: argparse.Namespace) -> int:
    result = train_linear_model(
        Path(args.features),
        output=Path(args.output),
        target=args.target,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_datasets_list(_args: argparse.Namespace) -> int:
    print(json.dumps(list_reference_decoy_datasets(), indent=2, sort_keys=True))
    return 0


def _cmd_reference_datasets_describe(args: argparse.Namespace) -> int:
    print(json.dumps(get_reference_decoy_dataset(args.dataset_id), indent=2, sort_keys=True))
    return 0


def _cmd_reference_validate_spec(args: argparse.Namespace) -> int:
    print(json.dumps(validate_corpus_spec(Path(args.corpus_spec)), indent=2, sort_keys=True))
    return 0


def _cmd_reference_summarize_spec(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_corpus_spec(Path(args.corpus_spec)), indent=2, sort_keys=True))
    return 0


def _cmd_reference_init_spec(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            init_corpus_spec(args.template, Path(args.output)),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_reference_build(args: argparse.Namespace) -> int:
    result = build_reference_corpus(
        corpus_spec=Path(args.corpus_spec),
        output_root=Path(args.output_root),
        ccd=Path(args.ccd) if args.ccd else None,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        argv=getattr(args, "_argv", ()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_fetch(args: argparse.Namespace) -> int:
    result = fetch_reference_inputs(
        reference_root=Path(args.reference_root),
        artifact_version=args.artifact_version,
        snapshot_id=args.snapshot_id,
        snapshot_root=Path(args.snapshot_root) if args.snapshot_root else None,
        datasets=args.dataset,
        include_payloads=args.include_payloads,
        payload_urls=_dataset_value_mapping(args.payload_url),
        local_files=_dataset_path_mapping(args.local_file),
        local_dirs=_dataset_path_mapping(args.local_dir),
        overwrite=args.overwrite,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_inventory(args: argparse.Namespace) -> int:
    snapshot_root = Path(args.snapshot_root) if args.snapshot_root else Path(".pheat-cache/pdb-archive") / DEFAULT_REFERENCE_SNAPSHOT_ID
    output = Path(args.output) if args.output else Path(args.reference_root) / "inventories" / args.artifact_version / "inventory.jsonl"
    result = inventory_reference(
        snapshot_root=snapshot_root,
        output=output,
        reference_root=Path(args.reference_root),
        artifact_version=args.artifact_version,
        domain=args.domain,
        metadata_jsonl=Path(args.metadata_jsonl) if args.metadata_jsonl else None,
        max_entries=args.max_entries,
        workers=args.workers,
        overwrite=args.overwrite,
        progress=_progress_callback(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_select(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else Path(args.reference_root) / "sets" / (
        f"{args.domain}-30id-xray-{args.subset}-{args.artifact_version}"
    )
    result = select_reference_corpus(
        inventory=Path(args.inventory),
        output_root=output_root,
        subset=args.subset,
        artifact_version=args.artifact_version,
        domain=args.domain,
        method=args.method,
        max_resolution=args.max_resolution,
        sequence_identity=args.sequence_identity,
        sequence_clusters=Path(args.sequence_clusters) if args.sequence_clusters else None,
        include_file=Path(args.include_file) if args.include_file else None,
        exclude_file=Path(args.exclude_file) if args.exclude_file else None,
        holdout_file=Path(args.holdout_file) if args.holdout_file else None,
        min_length=args.min_length,
        max_length=args.max_length,
        canonical_only=args.canonical_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_reference_build_decoys(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else Path(args.reference_root) / "decoys" / args.artifact_version
    result = build_reference_decoys(
        training_set=Path(args.training_set),
        output_root=output_root,
        recipes=_comma_values(args.recipes),
        domain=args.domain,
        artifact_version=args.artifact_version,
        seed=args.seed,
        max_entries=args.max_entries,
        attempts_per_decoy=args.attempts_per_decoy,
        workers=args.workers,
        overwrite=args.overwrite,
        progress=_progress_callback(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_build_scores(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else Path(args.reference_root) / "tables" / args.artifact_version
    result = build_reference_scores(
        training_set=Path(args.training_set),
        output_root=output_root,
        artifact_version=args.artifact_version,
        table_set_id=args.table_set_id,
        models=normalize_model_list(args.models),
        domain=args.domain,
        burial_method=args.burial_method,
        sasa_backend=args.sasa_backend,
        max_entries=args.max_entries,
        workers=args.workers,
        overwrite=args.overwrite,
        progress=_progress_callback(args),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_reference_extract_features(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else Path(args.reference_root) / "features" / args.artifact_version / "features.jsonl"
    result = extract_reference_features(
        training_set=Path(args.training_set),
        output=output,
        decoys=Path(args.decoys) if args.decoys else None,
        models=_comma_values(args.models),
        domain=args.domain,
        artifact_version=args.artifact_version,
        max_entries=args.max_entries,
        workers=args.workers,
        overwrite=args.overwrite,
        progress=_progress_callback(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_train_ml(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else Path(args.reference_root) / "models" / args.artifact_version / "pheat-ml-linear.json"
    result = train_reference_ml(
        features=Path(args.features),
        output=output,
        target=args.target,
        artifact_version=args.artifact_version,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_validate(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else Path(args.reference_root) / "validation" / args.artifact_version / "validation.json"
    result = validate_reference_features(
        features=Path(args.features),
        output=output,
        artifact_version=args.artifact_version,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_promote(args: argparse.Namespace) -> int:
    result = promote_reference_artifact(
        source=Path(args.source),
        destination=Path(args.destination),
        note=args.note,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_package_scoring_assets(args: argparse.Namespace) -> int:
    destination_root = Path(args.destination_root) if args.destination_root else Path("src/pheat/data/scoring") / args.artifact_version
    result = package_reference_scoring_assets(
        reference_root=Path(args.reference_root),
        destination_root=destination_root,
        artifact_version=args.artifact_version,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reference_audit_version(args: argparse.Namespace) -> int:
    result = audit_reference_artifact_version(
        reference_root=Path(args.reference_root),
        artifact_version=args.artifact_version,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["ok"] else 1


def _cmd_reference_run_unattended(args: argparse.Namespace) -> int:
    result = run_reference_unattended(
        reference_root=Path(args.reference_root),
        artifact_version=args.artifact_version,
        snapshot_id=args.snapshot_id,
        snapshot_root=Path(args.snapshot_root) if args.snapshot_root else None,
        domain=args.domain,
        method=args.method,
        max_resolution=args.max_resolution,
        sequence_identity=args.sequence_identity,
        min_length=args.min_length,
        max_length=args.max_length,
        decoy_recipes=_comma_values(args.decoy_recipes),
        attempts_per_decoy=args.attempts_per_decoy,
        models=normalize_model_list(args.models),
        feature_models=_comma_values(args.feature_models),
        sasa_backend=args.sasa_backend,
        metadata_source=args.metadata_source,
        workers=args.workers,
        seed=args.seed,
        overwrite=args.overwrite,
        backup_existing=args.backup_existing,
        run_canary=not args.skip_canary,
        canary_entries=args.canary_entries,
        dry_run=args.dry_run,
        progress=_progress_callback(args),
    )
    print(json.dumps(_omit_large_payloads(result), indent=2, sort_keys=True))
    return 0


def _cmd_molstar_install(args: argparse.Namespace) -> int:
    path = install_molstar_assets(
        version=args.version,
        destination=args.destination,
        timeout=args.timeout,
        force=args.force,
    )
    status = molstar_asset_status(version=args.version, path=args.destination)
    if status.path != path:
        status = molstar_asset_status(version=args.version, path=path)
    print(json.dumps({"status": "ok", **status.to_dict()}, indent=2, sort_keys=True))
    return 0


def _cmd_molstar_status(args: argparse.Namespace) -> int:
    status = molstar_asset_status(version=args.version, path=args.path)
    payload = status.to_dict()
    if not status.available:
        payload["warning"] = molstar_missing_assets_message(status)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status.available else 1


def _cmd_web(args: argparse.Namespace) -> int:
    from pheat.webapp import run_app

    run_app(
        host=args.host,
        port=args.port,
        work_dir=args.work_dir,
        molstar_vendor_dir=args.molstar_vendor_dir,
        random_port_if_taken=args.random_port_if_taken,
    )
    return 0


def _dataset_value_mapping(values: Sequence[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for value in values:
        dataset, payload = _split_dataset_mapping(value)
        output.setdefault(dataset, []).append(payload)
    return output


def _dataset_path_mapping(values: Sequence[str]) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = {}
    for value in values:
        dataset, payload = _split_dataset_mapping(value)
        output.setdefault(dataset, []).append(Path(payload))
    return output


def _split_dataset_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("dataset mappings must use DATASET=VALUE")
    dataset, payload = value.split("=", 1)
    dataset = dataset.strip().lower()
    payload = payload.strip()
    if not dataset or not payload:
        raise ValueError("dataset mappings must use non-empty DATASET=VALUE")
    return dataset, payload


def _load_structure(path: str, *, hydrogens: str = "drop") -> HeavyAtomStructure:
    lower_path = str(path).lower()
    suffix = Path(path).suffix.lower()
    if lower_path.endswith((".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz", ".bcif.gz")):
        return load_training_structure(Path(path))
    if lower_path.endswith(".bcif") or lower_path.endswith(".bcif.gz"):
        return load_bcif(path, hydrogens=hydrogens)
    if suffix == ".pdb" or suffix == ".ent":
        return load_pdb(path, hydrogens=hydrogens)
    if suffix in {".cif", ".mmcif"}:
        return load_mmcif(path, hydrogens=hydrogens)
    return load_heavy_json(path)


def _write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_model_json(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""PHEAT backend API for protein atom-structure conversion and scoring."""

from importlib.metadata import PackageNotFoundError, version

from pheat.bcif import load_bcif
from pheat.centroid import to_centroid_structure
from pheat.domains import SCORING_DOMAINS, filter_structure_for_domain, normalize_domain
from pheat.geometry import (
    apply_kabsch_transform,
    kabsch_align,
    kabsch_rmsd,
    kabsch_transform,
    radius_of_gyration,
)
from pheat.geometry_tables import (
    build_backbone_geometry_tables,
    build_cdl_geometry_tables,
    build_sidechain_ccd_geometry_tables,
    import_cdl_geometry_tables,
    list_packaged_geometry_tables,
    load_geometry_table_set,
    load_packaged_geometry_table,
    validate_geometry_table_set,
)
from pheat.metrics import (
    radius_of_gyration_result,
    structure_radius_of_gyration,
    structure_rmsd_result,
)
from pheat.mmcif import (
    load_mmcif,
    structure_from_mmcif_string,
    structure_to_mmcif_string,
    write_mmcif,
)
from pheat.models import (
    Atom,
    AtomStructure,
    Bond,
    Centroid,
    CentroidStructure,
    DisulfideBond,
    EnergyResult,
    HeavyAtomStructure,
    ResidueGeometry,
    ResidueGeometryStructure,
    ResidueKey,
)
from pheat.pdbio import (
    load_pdb,
    load_structure_json,
    structure_from_pdb_string,
    structure_to_pdb_string,
    write_multimodel_pdb,
    write_pdb,
    write_structure_json,
)
from pheat.residue_geometry import (
    residue_angle_specs,
    structure_from_residue_geometry,
    structure_to_residue_geometry,
)
from pheat.roundtrip import (
    RoundtripCaseSpec,
    combinatorial_roundtrip_case_specs,
    configurable_roundtrip_case_spec,
    run_roundtrip_cases,
    single_roundtrip_case_spec,
)
from pheat.score_contracts import score_input_contract, score_input_contracts
from pheat.score_tables import (
    load_packaged_score_table_set,
    packaged_score_table_ids,
    packaged_score_table_manifest,
)
from pheat.scoring import (
    GromacsRunSettings,
    available_models,
    model_capabilities,
    prepare_gromacs_structure,
    score_model_option_specs,
    score_structure,
    score_structure_profiles,
    supported_models,
    validate_external_scoring_options,
    validate_scoring_options,
)
from pheat.software_provenance import collect_software_provenance
from pheat.nerf import NerfFolder

try:
    __version__ = version("pheat")
except PackageNotFoundError:  # pragma: no cover - only hit from an unpackaged source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "Atom",
    "AtomStructure",
    "Bond",
    "Centroid",
    "CentroidStructure",
    "DisulfideBond",
    "EnergyResult",
    "GromacsRunSettings",
    "HeavyAtomStructure",
    "ResidueKey",
    "ResidueGeometry",
    "ResidueGeometryStructure",
    "RoundtripCaseSpec",
    "normalize_domain",
    "filter_structure_for_domain",
    "SCORING_DOMAINS",
    "available_models",
    "build_backbone_geometry_tables",
    "build_cdl_geometry_tables",
    "build_sidechain_ccd_geometry_tables",
    "model_capabilities",
    "prepare_gromacs_structure",
    "supported_models",
    "collect_software_provenance",
    "combinatorial_roundtrip_case_specs",
    "configurable_roundtrip_case_spec",
    "apply_kabsch_transform",
    "kabsch_align",
    "kabsch_rmsd",
    "kabsch_transform",
    "load_bcif",
    "load_geometry_table_set",
    "load_packaged_geometry_table",
    "import_cdl_geometry_tables",
    "list_packaged_geometry_tables",
    "load_mmcif",
    "load_packaged_score_table_set",
    "load_pdb",
    "load_structure_json",
    "packaged_score_table_ids",
    "packaged_score_table_manifest",
    "radius_of_gyration",
    "radius_of_gyration_result",
    "run_roundtrip_cases",
    "residue_angle_specs",
    "score_model_option_specs",
    "score_structure",
    "score_input_contract",
    "score_input_contracts",
    "score_structure_profiles",
    "single_roundtrip_case_spec",
    "structure_radius_of_gyration",
    "structure_rmsd_result",
    "structure_from_pdb_string",
    "structure_from_mmcif_string",
    "structure_from_residue_geometry",
    "structure_to_mmcif_string",
    "structure_to_pdb_string",
    "structure_to_residue_geometry",
    "NerfFolder",
    "to_centroid_structure",
    "validate_geometry_table_set",
    "validate_external_scoring_options",
    "validate_scoring_options",
    "write_mmcif",
    "write_pdb",
    "write_multimodel_pdb",
    "write_structure_json",
    "__version__",
]

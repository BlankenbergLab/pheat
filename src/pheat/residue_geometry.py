"""Convert between atom structures and residue-geometry definitions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pheat.geometry_tables import geometry_lookup
from pheat.geometry import add
from pheat.geometry import angle_degrees as bond_angle_degrees
from pheat.geometry import cross, dihedral_degrees, distance, dot, norm, normalize, place_atom, scale, sub
from pheat.models import (
    OPTIONAL_RESIDUE_GEOMETRY_ANGLES,
    RESIDUE_GEOMETRY_BACKBONE_LENGTHS,
    Atom,
    HeavyAtomStructure,
    ResidueKey,
    ResidueGeometry,
    ResidueGeometryStructure,
    _normalize_stored_angles,
    _normalize_stored_lengths,
)
from pheat.residues import DEFAULT_CHIS, SIDECHAIN_STEPS, guess_element, one_to_three, three_to_one


# Idealized backbone geometry used for deterministic reconstruction. These rounded
# bond lengths and angles follow standard Engh-Huber-style protein geometry
# conventions (source manifest: engh-huber-1991) but are not a force-field table.
# Optional stored backbone geometry uses these definitions:
# omega = CA(i)-C(i)-N(i+1)-CA(i+1), tau = N(i)-CA(i)-C(i),
# theta = CA(i)-C(i)-N(i+1).
N_CA = 1.458
CA_C = 1.525
C_N = 1.329
C_O = 1.231
CA_CB = 1.530
# Proline closes its side chain onto the backbone nitrogen. These rounded bond
# lengths follow the same ideal protein-geometry convention as the side-chain
# template constants rather than a force-field parameter table.
PRO_CG_CD = 1.500
PRO_N_CD = 1.470
# Rounded wwPDB CCD bond lengths for supported modified residue ring closures.
PCA_CG_CD = 1.520
PCA_N_CD = 1.350
PCA_CD_OE = 1.230
PYL_CD2_CG2 = 1.530
PYL_CA2_CG2 = 1.530
PYL_CG2_CB2 = 1.530

ANGLE_N_CA_C = 111.2
ANGLE_CA_C_N = 116.2
ANGLE_C_N_CA = 121.7
ANGLE_CA_C_O = 120.8
ANGLE_N_CA_CB = 110.5
ANGLE_UNITS = ("radians", "degrees")
DEFAULT_PHI_DEGREES = -60.0
DEFAULT_PSI_DEGREES = -45.0
DEFAULT_OMEGA_DEGREES = 180.0

# Idealized planar aromatic templates in a local frame rooted at CG.  The x axis
# follows the first ring branch (CD1/ND1), and the y axis points toward the
# second branch.  Rebuilding fused and closed rings as an unconstrained chain of
# internal-coordinate steps accumulates closure error; these templates preserve
# chi1/chi2 orientation while enforcing chemically meaningful ring geometry.
_AROMATIC_LOCAL_TEMPLATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "PHE": {
        "CD1": (1.386965, 0.0), "CD2": (-0.663592, 1.217500),
        "CE1": (2.093342, 1.188041), "CE2": (0.039651, 2.406356),
        "CZ": (1.420057, 2.391026),
    },
    "HIS": {
        "ND1": (1.379165, 0.0), "CD2": (-0.376797, 1.300149),
        "CE1": (1.815757, 1.247747), "NE2": (0.770590, 2.054608),
    },
    "TRP": {
        "CD1": (1.362720, 0.0), "CD2": (-0.401066, 1.391571),
        "NE1": (1.835814, 1.288852), "CE2": (0.770321, 2.153414),
        "CE3": (-1.641525, 2.041433), "CZ2": (0.744158, 3.539116),
        "CZ3": (-1.667758, 3.431033), "CH2": (-0.505181, 4.159794),
    },
}


def load_residue_geometry(path: Union[str, Path], *, angle_units: Optional[str] = None) -> ResidueGeometryStructure:
    if angle_units is not None:
        angle_units = _normalize_angle_units(angle_units)
    with open(path, "r", encoding="utf-8") as handle:
        return ResidueGeometryStructure.from_dict(
            json.load(handle),
            angle_units=angle_units,
            strict_format=True,
        )


def write_residue_geometry_json(
    residue_geometry: ResidueGeometryStructure,
    path: Union[str, Path],
    *,
    stored_angles: Optional[Union[str, Iterable[str]]] = None,
    stored_lengths: Optional[Union[str, Iterable[str]]] = None,
    max_chi: Optional[int] = None,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            residue_geometry.to_json(
                stored_angles=stored_angles,
                stored_lengths=stored_lengths,
                max_chi=max_chi,
            )
        )
        handle.write("\n")


def residue_angle_specs(
    sequence: Union[str, Sequence[str]],
    *,
    selective_chi_map: Optional[Mapping[str, Sequence[object]]] = None,
    max_chi: Optional[int] = None,
    stored_angles: Optional[Union[str, Iterable[str]]] = None,
    angle_units: str = "radians",
) -> List[Dict[str, Any]]:
    """Return PHEAT residue-geometry angle fields available for a sequence.

    The returned records are JSON-serializable metadata, not optimizer-specific
    degrees of freedom. A string sequence is interpreted as one-letter amino-acid
    codes. Pass a sequence of tokens, such as ``["MSE", "ALA"]``, when modified
    three-letter component IDs are needed. ``max_chi`` limits chi angles by chi
    index, and ``selective_chi_map`` can restrict named chi angles for specific
    residues before that numeric ceiling is applied.
    """

    normalized_units = _normalize_angle_units(angle_units)
    normalized_max_chi = _normalize_max_chi(max_chi)
    normalized_stored_angles = _normalize_stored_angles(stored_angles)
    residues = _sequence_residue_names(sequence)
    selective = _normalize_selective_chi_map(selective_chi_map or {})
    specs: List[Dict[str, Any]] = []

    for residue_index, resname in enumerate(residues):
        residue_code = _residue_one_letter_code(resname)
        terminal_position = _terminal_position(residue_index, len(residues))
        specs.append(
            _residue_angle_spec(
                residue_index,
                resname,
                residue_code,
                "phi",
                category="backbone-dihedral",
                angle_units=normalized_units,
                applies_to="residue",
                terminal_position=terminal_position,
                coordinate_defined=residue_index > 0,
                required_atoms=["C(i-1)", "N", "CA", "C"],
            )
        )
        specs.append(
            _residue_angle_spec(
                residue_index,
                resname,
                residue_code,
                "psi",
                category="backbone-dihedral",
                angle_units=normalized_units,
                applies_to="residue",
                terminal_position=terminal_position,
                coordinate_defined=residue_index < len(residues) - 1,
                required_atoms=["N", "CA", "C", "N(i+1)"],
            )
        )
        for chi_name in _allowed_chi_names(
            resname,
            residue_code,
            selective_chi_map=selective,
            max_chi=normalized_max_chi,
        ):
            specs.append(
                _residue_angle_spec(
                    residue_index,
                    resname,
                    residue_code,
                    chi_name,
                    category="sidechain-dihedral",
                    angle_units=normalized_units,
                    applies_to="residue",
                    terminal_position=terminal_position,
                    coordinate_defined=True,
                    required_atoms=_chi_required_atoms(resname, chi_name),
                )
            )

    for angle_name in normalized_stored_angles:
        residue_count = len(residues) if angle_name == "tau" else max(0, len(residues) - 1)
        for residue_index in range(residue_count):
            resname = residues[residue_index]
            residue_code = _residue_one_letter_code(resname)
            specs.append(
                _residue_angle_spec(
                    residue_index,
                    resname,
                    residue_code,
                    angle_name,
                    category="backbone-bond-angle" if angle_name in {"tau", "theta"} else "backbone-dihedral",
                    angle_units=normalized_units,
                    applies_to="residue" if angle_name == "tau" else "peptide-link-to-next",
                    terminal_position=_terminal_position(residue_index, len(residues)),
                    peptide_link_to_residue_index=None if angle_name == "tau" else residue_index + 1,
                    coordinate_defined=True,
                    required_atoms=_optional_angle_required_atoms(angle_name),
                    optional=True,
                )
            )
    return specs


def structure_to_residue_geometry(
    structure: HeavyAtomStructure,
    *,
    name: str = "",
    angle_units: str = "radians",
    stored_angles: Optional[Union[str, Iterable[str]]] = None,
    stored_lengths: Optional[Union[str, Iterable[str]]] = None,
    max_chi: Optional[int] = None,
) -> ResidueGeometryStructure:
    """Extract best-effort residue geometry from an atom structure.

    Angles are emitted in radians by default; pass ``angle_units="degrees"`` for degrees.
    Optional stored geometry fields are omitted from JSON by default. Pass
    ``stored_angles`` with any of ``omega``, ``tau``, ``theta``, or ``all`` to store
    peptide dihedral and backbone bond-angle fields in serialized residue-geometry JSON.
    Pass ``max_chi`` to keep only the first N chi angles per residue; the default
    ``None`` stores every extractable chi angle.
    """

    angle_units = _normalize_angle_units(angle_units)
    max_chi = _normalize_max_chi(max_chi)
    normalized_stored_lengths = _normalize_stored_lengths(stored_lengths)
    warnings: List[str] = []
    residues = [
        (key, _resolve_residue_atoms(key, atoms, warnings))
        for key, atoms in structure.atoms_by_residue().items()
    ]
    residue_geometries: List[ResidueGeometry] = []

    for index, (key, atoms) in enumerate(residues):
        chain_id, resseq, _icode, resname, record_name = key
        previous_atoms = _neighbor_backbone(residues, index, -1)
        next_atoms = _neighbor_backbone(residues, index, 1)
        current_backbone = _has_backbone(atoms)

        # Backbone phi/psi/omega need atoms from neighboring residues. Missing
        # neighbors at chain termini, gaps, or malformed residues intentionally
        # become null residue-geometry fields instead of hard failures.
        phi = None
        psi = None
        omega = None
        tau = None
        theta = None
        if current_backbone and previous_atoms:
            phi = _safe_dihedral(
                previous_atoms["C"],
                atoms["N"],
                atoms["CA"],
                atoms["C"],
                warnings,
                f"{chain_id}:{resseq}:{resname}:phi",
            )
        if current_backbone:
            tau = _safe_bond_angle(
                atoms["N"],
                atoms["CA"],
                atoms["C"],
                warnings,
                f"{chain_id}:{resseq}:{resname}:tau",
            )
        if current_backbone and next_atoms:
            psi = _safe_dihedral(
                atoms["N"],
                atoms["CA"],
                atoms["C"],
                next_atoms["N"],
                warnings,
                f"{chain_id}:{resseq}:{resname}:psi",
            )
            omega = _safe_dihedral(
                atoms["CA"],
                atoms["C"],
                next_atoms["N"],
                next_atoms["CA"],
                warnings,
                f"{chain_id}:{resseq}:{resname}:omega",
            )
            theta = _safe_bond_angle(
                atoms["CA"],
                atoms["C"],
                next_atoms["N"],
                warnings,
                f"{chain_id}:{resseq}:{resname}:theta",
            )

        # Chi angles use residue-specific side-chain templates. Unsupported
        # residues stay represented in residue-geometry JSON but cannot provide chis.
        chi = _extract_chis(key, atoms, warnings)
        bond_lengths = _extract_bond_lengths(
            atoms,
            next_atoms,
            normalized_stored_lengths,
        )
        if record_name == "HETATM" and not current_backbone and not chi:
            warnings.append(f"{chain_id}:{resseq}:{resname}: heterogen has no extractable residue geometry")

        residue_geometries.append(
            ResidueGeometry(
                name=resname,
                phi=_degrees_to_units(phi, angle_units),
                psi=_degrees_to_units(psi, angle_units),
                omega=_degrees_to_units(omega, angle_units),
                tau=_degrees_to_units(tau, angle_units),
                theta=_degrees_to_units(theta, angle_units),
                chi=_limit_chis(
                    [_degrees_to_units(value, angle_units) for value in chi],
                    max_chi,
                ),
                bond_lengths=bond_lengths,
                chain_id=chain_id,
                resseq=resseq,
                icode=_icode,
            )
        )

    metadata = {
        "source": "atom_structure",
        "source_name": structure.name,
        "residue_count": len(residue_geometries),
        "warnings": warnings,
    }
    return ResidueGeometryStructure(
        residues=residue_geometries,
        name=name or structure.name,
        angle_units=angle_units,
        metadata=metadata,
        stored_angles=_normalize_stored_angles(stored_angles),
        stored_lengths=normalized_stored_lengths,
        disulfide_bonds=list(structure.disulfide_bonds),
    )


def structure_from_residue_geometry(
    residue_geometry: Union[ResidueGeometryStructure, Mapping[str, object], str, Path],
    *,
    name: str = "",
    angle_units: Optional[str] = None,
    include_terminal_oxt: bool = False,
    geometry_mode: Optional[str] = None,
    geometry_table: Optional[Union[str, Path, Mapping[str, object]]] = None,
    geometry_profile: Optional[str] = None,
) -> HeavyAtomStructure:
    """Reconstruct an approximate atom structure from residue geometry.

    The geometry is suitable for deterministic structural plumbing and scoring tests. It is
    not a substitute for a high-quality internal-coordinate/rotamer modeling package.
    Input phi/psi/omega/chi values use the residue-geometry JSON ``angle_units`` field when
    present, otherwise radians. Pass ``angle_units`` to override the input units.
    Terminal OXT atoms are omitted by default and can be added with
    ``include_terminal_oxt=True``.
    """

    if angle_units is not None:
        angle_units = _normalize_angle_units(angle_units)

    if isinstance(residue_geometry, ResidueGeometryStructure):
        residue_geometry_structure = residue_geometry
    elif isinstance(residue_geometry, Mapping):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(residue_geometry, angle_units=angle_units)
    elif isinstance(residue_geometry, (str, Path)):
        residue_geometry_structure = load_residue_geometry(residue_geometry, angle_units=angle_units)
    else:
        raise TypeError("Unsupported residue-geometry input.")

    # Optional backbone geometry fields are part of the serialized
    # residue-geometry contract. Apply that contract to in-memory objects too, so
    # reconstructing a ResidueGeometryStructure behaves the same as reconstructing
    # the JSON it writes.
    residue_geometry_structure = _filter_unstored_optional_angles(residue_geometry_structure)
    # The public JSON/API can use radians or degrees. Reconstruction geometry is
    # specified in degrees, so normalize once at the boundary.
    residue_geometry_structure = _residue_geometry_to_degrees(residue_geometry_structure, angle_units)
    geometry = geometry_lookup(
        geometry_mode=geometry_mode,
        geometry_table=geometry_table,
        geometry_profile=geometry_profile,
    )
    atoms: List[Atom] = []
    serial = 1
    previous_backbone: Optional[Dict[str, Atom]] = None
    previous_geometry: Optional[ResidueGeometry] = None
    previous_chain_id: Optional[str] = None
    previous_resseq: Optional[int] = None
    warnings = _metadata_warnings(residue_geometry_structure.metadata)

    for index, residue in enumerate(residue_geometry_structure.residues):
        chain_id = residue.chain_id or "A"
        resseq = _effective_resseq(residue, index)
        icode = residue.icode or ""
        try:
            resname = one_to_three(residue.name)
        except ValueError:
            warnings.append(
                f"{chain_id}:{resseq}:{residue.name}: unsupported residue skipped during reconstruction"
            )
            previous_backbone = None
            previous_geometry = None
            previous_chain_id = None
            previous_resseq = None
            continue

        if not _residue_geometry_continues_previous(
            previous_chain_id,
            previous_resseq,
            chain_id,
            resseq,
        ):
            previous_backbone = None
            previous_geometry = None

        if previous_backbone is None or previous_geometry is None:
            # The first residue in each chain defines a deterministic local
            # coordinate frame. Later residues are placed by internal coordinates
            # from the previous peptide unit.
            n_ca = _residue_length(
                residue,
                "N-CA",
                geometry.length("N-CA", N_CA, resname=resname, phi=residue.phi, psi=residue.psi),
            )
            ca_c = _residue_length(
                residue,
                "CA-C",
                geometry.length("CA-C", CA_C, resname=resname, phi=residue.phi, psi=residue.psi),
            )
            tau = residue.tau if residue.tau is not None else geometry.angle(
                "N-CA-C",
                ANGLE_N_CA_C,
                resname=resname,
                phi=residue.phi,
                psi=residue.psi,
            )
            n_coord = (0.0, 0.0, 0.0)
            ca_coord = (n_ca, 0.0, 0.0)
            c_coord = _initial_c_coord(tau, n_ca=n_ca, ca_c=ca_c)
        else:
            psi = previous_geometry.psi if previous_geometry.psi is not None else DEFAULT_PSI_DEGREES
            omega = (
                previous_geometry.omega
                if previous_geometry.omega is not None
                else DEFAULT_OMEGA_DEGREES
            )
            phi = residue.phi if residue.phi is not None else DEFAULT_PHI_DEGREES
            theta = (
                previous_geometry.theta
                if previous_geometry.theta is not None
                else geometry.angle(
                    "CA-C-N",
                    ANGLE_CA_C_N,
                    resname=previous_geometry.name,
                    phi=previous_geometry.phi,
                    psi=previous_geometry.psi,
                )
            )
            tau = residue.tau if residue.tau is not None else geometry.angle(
                "N-CA-C",
                ANGLE_N_CA_C,
                resname=resname,
                phi=residue.phi,
                psi=residue.psi,
            )
            n_coord = place_atom(
                previous_backbone["N"].coord,
                previous_backbone["CA"].coord,
                previous_backbone["C"].coord,
                length=_residue_length(
                    previous_geometry,
                    "C-N",
                    geometry.length(
                        "C-N",
                        C_N,
                        resname=previous_geometry.name,
                        phi=previous_geometry.phi,
                        psi=previous_geometry.psi,
                    ),
                ),
                angle_degrees=theta,
                dihedral_degrees=_placement_dihedral_degrees(psi),
            )
            ca_coord = place_atom(
                previous_backbone["CA"].coord,
                previous_backbone["C"].coord,
                n_coord,
                length=_residue_length(
                    residue,
                    "N-CA",
                    geometry.length("N-CA", N_CA, resname=resname, phi=residue.phi, psi=residue.psi),
                ),
                angle_degrees=geometry.angle(
                    "C-N-CA",
                    ANGLE_C_N_CA,
                    resname=resname,
                    phi=residue.phi,
                    psi=residue.psi,
                ),
                dihedral_degrees=_placement_dihedral_degrees(omega),
            )
            c_coord = place_atom(
                previous_backbone["C"].coord,
                n_coord,
                ca_coord,
                length=_residue_length(
                    residue,
                    "CA-C",
                    geometry.length("CA-C", CA_C, resname=resname, phi=residue.phi, psi=residue.psi),
                ),
                angle_degrees=tau,
                dihedral_degrees=_placement_dihedral_degrees(phi),
            )

        n_atom = _atom("N", "N", n_coord, resname, chain_id, resseq, serial, icode=icode)
        ca_atom = _atom("CA", "C", ca_coord, resname, chain_id, resseq, serial + 1, icode=icode)
        c_atom = _atom("C", "C", c_coord, resname, chain_id, resseq, serial + 2, icode=icode)
        serial += 3
        local_atoms: Dict[str, Atom] = {"N": n_atom, "CA": ca_atom, "C": c_atom}
        atoms.extend([n_atom, ca_atom, c_atom])

        o_coord = place_atom(
            n_atom.coord,
            ca_atom.coord,
            c_atom.coord,
            length=_residue_length(
                residue,
                "C-O",
                geometry.length("C-O", C_O, resname=resname, phi=residue.phi, psi=residue.psi),
            ),
            angle_degrees=geometry.angle(
                "CA-C-O",
                ANGLE_CA_C_O,
                resname=resname,
                phi=residue.phi,
                psi=residue.psi,
            ),
            dihedral_degrees=_placement_dihedral_degrees(
                (residue.psi if residue.psi is not None else DEFAULT_PSI_DEGREES)
                + DEFAULT_OMEGA_DEGREES
            ),
        )
        o_atom = _atom("O", "O", o_coord, resname, chain_id, resseq, serial, icode=icode)
        serial += 1
        local_atoms["O"] = o_atom
        atoms.append(o_atom)

        # PDB terminal OXT is chemically reasonable for a C terminus, but many
        # experimental PDBs omit it. Keep it opt-in so round-trip atom identity
        # can match sources that lacked OXT.
        if include_terminal_oxt and not _next_residue_continues_chain(
            residue_geometry_structure.residues,
            index,
            chain_id,
            resseq,
        ):
            oxt_coord = place_atom(
                n_atom.coord,
                ca_atom.coord,
                c_atom.coord,
                length=_residue_length(
                    residue,
                    "C-O",
                    geometry.length("C-O", C_O, resname=resname, phi=residue.phi, psi=residue.psi),
                ),
                angle_degrees=geometry.angle(
                    "CA-C-O",
                    ANGLE_CA_C_O,
                    resname=resname,
                    phi=residue.phi,
                    psi=residue.psi,
                ),
                dihedral_degrees=_placement_dihedral_degrees(
                    residue.psi if residue.psi is not None else DEFAULT_PSI_DEGREES
                ),
            )
            oxt_atom = _atom("OXT", "O", oxt_coord, resname, chain_id, resseq, serial, icode=icode)
            serial += 1
            local_atoms["OXT"] = oxt_atom
            atoms.append(oxt_atom)

        if resname != "GLY":
            cb_dihedral = -122.5 if resname != "PRO" else -105.0
            cb_coord = place_atom(
                c_atom.coord,
                n_atom.coord,
                ca_atom.coord,
                length=_residue_length(
                    residue,
                    "CA-CB",
                    geometry.length("CA-CB", CA_CB, resname=resname, phi=residue.phi, psi=residue.psi),
                ),
                angle_degrees=geometry.angle(
                    "N-CA-CB",
                    ANGLE_N_CA_CB,
                    resname=resname,
                    phi=residue.phi,
                    psi=residue.psi,
                ),
                dihedral_degrees=_placement_dihedral_degrees(cb_dihedral),
            )
            cb_atom = _atom("CB", "C", cb_coord, resname, chain_id, resseq, serial, icode=icode)
            serial += 1
            local_atoms["CB"] = cb_atom
            atoms.append(cb_atom)

        # Side-chain templates are built as parent/anchor/previous construction
        # steps. The dihedral field maps each new atom to chiN or to a fixed
        # offset from chiN for branched/planar groups.
        for step in geometry.sidechain_steps(resname, SIDECHAIN_STEPS.get(resname, [])):
            if step.parent not in local_atoms or step.anchor not in local_atoms or step.previous not in local_atoms:
                continue
            coord = place_atom(
                local_atoms[step.previous].coord,
                local_atoms[step.anchor].coord,
                local_atoms[step.parent].coord,
                length=_residue_length(residue, f"{step.parent}-{step.atom}", step.length),
                angle_degrees=step.angle,
                dihedral_degrees=_placement_dihedral_degrees(
                    _resolve_dihedral(step.dihedral, residue)
                ),
            )
            atom = _atom(
                step.atom,
                step.element or guess_element(step.atom),
                coord,
                resname,
                chain_id,
                resseq,
                serial,
                icode=icode,
            )
            serial += 1
            local_atoms[step.atom] = atom
            atoms.append(atom)

        if resname in {"PRO", "HYP"}:
            _close_proline_ring(local_atoms, residue)
        if resname == "PCA":
            _close_pca_ring(local_atoms, residue)
        if resname == "PYL":
            _close_pyl_ring(local_atoms, residue)
        if resname in {"PHE", "TYR", "PTR", "HIS", "TRP"}:
            _close_aromatic_ring(local_atoms, resname)

        previous_backbone = {"N": n_atom, "CA": ca_atom, "C": c_atom}
        previous_geometry = residue
        previous_chain_id = chain_id
        previous_resseq = resseq

    metadata = dict(residue_geometry_structure.metadata)
    metadata.setdefault("source", "residue_geometry_reconstruction")
    metadata.setdefault("geometry_note", "idealized approximate heavy-atom geometry")
    if geometry.mode == "table":
        metadata["reconstruction_geometry"] = geometry.metadata()
    if warnings:
        metadata["warnings"] = warnings
    return HeavyAtomStructure(
        atoms=atoms,
        name=name or residue_geometry_structure.name,
        metadata=metadata,
        disulfide_bonds=list(residue_geometry_structure.disulfide_bonds),
    )


def residue_geometry_structure_from_sequence(
    sequence: str,
    *,
    phi: Optional[float] = None,
    psi: Optional[float] = None,
    omega: Optional[float] = None,
    tau: Optional[float] = None,
    theta: Optional[float] = None,
    chain_id: str = "A",
    angle_units: str = "radians",
    stored_angles: Optional[Union[str, Iterable[str]]] = None,
) -> ResidueGeometryStructure:
    angle_units = _normalize_angle_units(angle_units)
    phi = _default_angle(phi, DEFAULT_PHI_DEGREES, angle_units)
    psi = _default_angle(psi, DEFAULT_PSI_DEGREES, angle_units)
    omega = _default_angle(omega, DEFAULT_OMEGA_DEGREES, angle_units)
    tau = _default_angle(tau, ANGLE_N_CA_C, angle_units)
    theta = _default_angle(theta, ANGLE_CA_C_N, angle_units)
    residues = [
        ResidueGeometry(
            name=one_to_three(aa),
            phi=None if index == 0 else phi,
            psi=None if index == len(sequence) - 1 else psi,
            omega=None if index == len(sequence) - 1 else omega,
            tau=tau,
            theta=None if index == len(sequence) - 1 else theta,
            chain_id=chain_id,
            resseq=index + 1,
        )
        for index, aa in enumerate(sequence)
    ]
    return ResidueGeometryStructure(
        residues=residues,
        name=f"sequence:{sequence}",
        angle_units=angle_units,
        stored_angles=_normalize_stored_angles(stored_angles),
    )


def _normalize_angle_units(angle_units: str) -> str:
    normalized = str(angle_units).strip().lower()
    if normalized not in ANGLE_UNITS:
        raise ValueError(f"angle_units must be one of {', '.join(ANGLE_UNITS)}")
    return normalized


def _degrees_to_units(angle: Optional[float], angle_units: str) -> Optional[float]:
    if angle is None:
        return None
    if angle_units == "degrees":
        return float(angle)
    return math.radians(float(angle))


def _units_to_degrees(angle: Optional[float], angle_units: str) -> Optional[float]:
    if angle is None:
        return None
    if angle_units == "degrees":
        return float(angle)
    return math.degrees(float(angle))


def _default_angle(value: Optional[float], default_degrees: float, angle_units: str) -> float:
    if value is not None:
        return float(value)
    converted = _degrees_to_units(default_degrees, angle_units)
    if converted is None:  # pragma: no cover - default_degrees is never None
        raise ValueError("default angle conversion failed")
    return converted


def _residue_geometry_to_degrees(
    residue_geometry: ResidueGeometryStructure,
    angle_units: Optional[str],
) -> ResidueGeometryStructure:
    angle_units = _normalize_angle_units(angle_units or residue_geometry.angle_units)
    residues = [
        ResidueGeometry(
            name=residue.name,
            phi=_units_to_degrees(residue.phi, angle_units),
            psi=_units_to_degrees(residue.psi, angle_units),
            omega=_units_to_degrees(residue.omega, angle_units),
            tau=_units_to_degrees(residue.tau, angle_units),
            theta=_units_to_degrees(residue.theta, angle_units),
            chi=_angles_to_degrees(residue.chi, angle_units),
            bond_lengths=dict(residue.bond_lengths),
            chain_id=residue.chain_id,
            resseq=residue.resseq,
            icode=residue.icode,
        )
        for residue in residue_geometry.residues
    ]
    metadata = dict(residue_geometry.metadata)
    metadata.pop("angle_units", None)
    return ResidueGeometryStructure(
        residues=residues,
        name=residue_geometry.name,
        angle_units="degrees",
        metadata=metadata,
        chi_order=residue_geometry.chi_order,
        stored_angles=residue_geometry.stored_angles,
        stored_lengths=residue_geometry.stored_lengths,
        disulfide_bonds=list(residue_geometry.disulfide_bonds),
    )


def _filter_unstored_optional_angles(residue_geometry: ResidueGeometryStructure) -> ResidueGeometryStructure:
    stored_angles = set(residue_geometry.stored_angles)
    stored_lengths = residue_geometry.stored_lengths
    residues = [
        ResidueGeometry(
            name=residue.name,
            phi=residue.phi,
            psi=residue.psi,
            omega=residue.omega if "omega" in stored_angles else None,
            tau=residue.tau if "tau" in stored_angles else None,
            theta=residue.theta if "theta" in stored_angles else None,
            chi=list(residue.chi),
            bond_lengths=_filter_bond_lengths(residue.bond_lengths, stored_lengths),
            chain_id=residue.chain_id,
            resseq=residue.resseq,
            icode=residue.icode,
        )
        for residue in residue_geometry.residues
    ]
    return ResidueGeometryStructure(
        residues=residues,
        name=residue_geometry.name,
        angle_units=residue_geometry.angle_units,
        metadata=dict(residue_geometry.metadata),
        chi_order=residue_geometry.chi_order,
        stored_angles=tuple(angle for angle in OPTIONAL_RESIDUE_GEOMETRY_ANGLES if angle in stored_angles),
        stored_lengths=stored_lengths,
        disulfide_bonds=list(residue_geometry.disulfide_bonds),
    )


def _filter_bond_lengths(values: Mapping[str, float], stored_lengths: Sequence[str]) -> Dict[str, float]:
    if not stored_lengths:
        return {}
    if "all" in stored_lengths:
        return dict(values)
    return {key: value for key, value in values.items() if _length_requested(key, stored_lengths)}


def _angles_to_degrees(angles: Sequence[float], angle_units: str) -> List[float]:
    converted = [_units_to_degrees(angle, angle_units) for angle in angles]
    return [float(angle) for angle in converted if angle is not None]


def _normalize_max_chi(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_chi must be a non-negative integer or None")
    if value < 0:
        raise ValueError("max_chi must be a non-negative integer or None")
    return value


def _limit_chis(values: Sequence[Optional[float]], max_chi: Optional[int]) -> List[float]:
    if max_chi is None:
        return [float(value) for value in values if value is not None]
    return [float(value) for value in values[:max_chi] if value is not None]


def _sequence_residue_names(sequence: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(sequence, str):
        tokens = [item for item in sequence.strip().upper()]
    else:
        tokens = [str(item).strip().upper() for item in sequence]
    if not tokens:
        raise ValueError("sequence must contain at least one residue")
    return [one_to_three(token) for token in tokens]


def _normalize_selective_chi_map(values: Mapping[str, Sequence[object]]) -> Dict[str, Tuple[str, ...]]:
    normalized: Dict[str, Tuple[str, ...]] = {}
    for key, raw_items in values.items():
        resname = one_to_three(str(key))
        items = raw_items.split(",") if isinstance(raw_items, str) else raw_items
        chi_names = tuple(dict.fromkeys(_normalize_chi_name(item) for item in items))
        normalized[resname] = chi_names
        code = _residue_one_letter_code(resname)
        if code is not None:
            normalized[code] = chi_names
    return normalized


def _normalize_chi_name(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("chi indexes must be positive")
        return f"chi{value}"
    token = str(value).strip().lower()
    if token.isdigit():
        numeric = int(token)
        if numeric <= 0:
            raise ValueError("chi indexes must be positive")
        return f"chi{numeric}"
    if token.startswith("chi") and token[3:].isdigit() and int(token[3:]) > 0:
        return f"chi{int(token[3:])}"
    raise ValueError(f"invalid chi angle name: {value!r}")


def _residue_one_letter_code(resname: str) -> Optional[str]:
    try:
        return three_to_one(resname)
    except ValueError:
        return None


def _terminal_position(residue_index: int, residue_count: int) -> str:
    if residue_count == 1:
        return "single-residue"
    if residue_index == 0:
        return "n-terminal"
    if residue_index == residue_count - 1:
        return "c-terminal"
    return "internal"


def _available_chi_names(resname: str) -> Tuple[str, ...]:
    names = set()
    for step in SIDECHAIN_STEPS.get(resname, []):
        index = _chi_index(step.dihedral)
        if index is not None:
            names.add(f"chi{index + 1}")
    return tuple(sorted(names, key=lambda name: int(name[3:])))


def _allowed_chi_names(
    resname: str,
    residue_code: Optional[str],
    *,
    selective_chi_map: Mapping[str, Tuple[str, ...]],
    max_chi: Optional[int],
) -> Tuple[str, ...]:
    available = _available_chi_names(resname)
    configured = selective_chi_map.get(resname)
    if configured is None and residue_code is not None:
        configured = selective_chi_map.get(residue_code)
    allowed = tuple(name for name in available if configured is None or name in set(configured))
    if max_chi is not None:
        allowed = tuple(name for name in allowed if int(name[3:]) <= max_chi)
    return allowed


def _chi_required_atoms(resname: str, chi_name: str) -> List[str]:
    target_index = int(chi_name[3:]) - 1
    for step in SIDECHAIN_STEPS.get(resname, []):
        if _chi_index(step.dihedral) == target_index:
            return [step.previous, step.anchor, step.parent, step.atom]
    return []


def _optional_angle_required_atoms(angle_name: str) -> List[str]:
    if angle_name == "omega":
        return ["CA", "C", "N(i+1)", "CA(i+1)"]
    if angle_name == "tau":
        return ["N", "CA", "C"]
    if angle_name == "theta":
        return ["CA", "C", "N(i+1)"]
    return []


def _residue_angle_spec(
    residue_index: int,
    resname: str,
    residue_code: Optional[str],
    angle_name: str,
    *,
    category: str,
    angle_units: str,
    applies_to: str,
    terminal_position: str,
    coordinate_defined: bool,
    required_atoms: Sequence[str],
    peptide_link_to_residue_index: Optional[int] = None,
    optional: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "key": f"{residue_index}_{angle_name}",
        "residue_index": residue_index,
        "residue_number": residue_index + 1,
        "residue_name": resname,
        "angle_name": angle_name,
        "category": category,
        "angle_units": angle_units,
        "applies_to": applies_to,
        "terminal_position": terminal_position,
        "coordinate_defined": bool(coordinate_defined),
        "required_atoms": list(required_atoms),
        "optional_residue_geometry_field": bool(optional),
        "source": "pheat.residue_geometry",
    }
    if residue_code is not None:
        payload["residue_code"] = residue_code
    if peptide_link_to_residue_index is not None:
        payload["peptide_link_to_residue_index"] = peptide_link_to_residue_index
        payload["peptide_link_to_residue_number"] = peptide_link_to_residue_index + 1
    return payload


def _initial_c_coord(
    tau_degrees: float,
    *,
    n_ca: float = N_CA,
    ca_c: float = CA_C,
) -> Tuple[float, float, float]:
    theta = math.radians(tau_degrees)
    return (
        n_ca - ca_c * math.cos(theta),
        ca_c * math.sin(theta),
        0.0,
    )


def _normalize_degrees(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def _placement_dihedral_degrees(conventional_degrees: float) -> float:
    """Convert a public signed dihedral into the internal-coordinate placer frame."""

    return _normalize_degrees(float(conventional_degrees) - 180.0)


def _atom(
    name: str,
    element: str,
    coord: Sequence[float],
    resname: str,
    chain_id: str,
    resseq: int,
    serial: int,
    icode: str = "",
) -> Atom:
    return Atom(
        name=name,
        element=element,
        x=float(coord[0]),
        y=float(coord[1]),
        z=float(coord[2]),
        resname=resname,
        chain_id=chain_id,
        resseq=resseq,
        icode=icode,
        record_name="ATOM",
        serial=serial,
        occupancy=1.0,
        bfactor=0.0,
    )


def _residue_length(residue: ResidueGeometry, key: str, fallback: float) -> float:
    return float(residue.bond_lengths.get(key, fallback))


def _close_aromatic_ring(local_atoms: Dict[str, Atom], resname: str) -> None:
    """Project a reconstructed aromatic ring onto a rigid planar template."""

    template_name = "PHE" if resname in {"PHE", "TYR", "PTR"} else resname
    template = _AROMATIC_LOCAL_TEMPLATES[template_name]
    first_name = "ND1" if template_name == "HIS" else "CD1"
    second_name = "CD2"
    required = {"CG", first_name, second_name, *template}
    if not required.issubset(local_atoms):
        return

    origin = local_atoms["CG"].coord
    x_axis = normalize(sub(local_atoms[first_name].coord, origin))
    second_vector = sub(local_atoms[second_name].coord, origin)
    normal = normalize(cross(x_axis, second_vector))
    y_axis = normalize(cross(normal, x_axis))

    old_oh = local_atoms["OH"].coord if "OH" in local_atoms else None
    for atom_name, (x_value, y_value) in template.items():
        local_atoms[atom_name].coord = add(
            origin,
            add(scale(x_axis, x_value), scale(y_axis, y_value)),
        )

    if resname in {"TYR", "PTR"} and "OH" in local_atoms:
        cz = local_atoms["CZ"].coord
        ring_center = scale(
            add(add(local_atoms["CG"].coord, local_atoms["CE1"].coord), local_atoms["CE2"].coord),
            1.0 / 3.0,
        )
        local_atoms["OH"].coord = add(cz, scale(normalize(sub(cz, ring_center)), 1.36))
        if resname == "PTR" and old_oh is not None:
            displacement = sub(local_atoms["OH"].coord, old_oh)
            for atom_name in ("P", "O1P", "O2P", "O3P"):
                if atom_name in local_atoms:
                    local_atoms[atom_name].coord = add(local_atoms[atom_name].coord, displacement)


def _close_proline_ring(local_atoms: Dict[str, Atom], residue: ResidueGeometry) -> None:
    if not all(name in local_atoms for name in ("N", "CG", "CD")):
        return
    cd_atom = local_atoms["CD"]
    cd_atom.coord = _place_atom_from_two_bond_lengths(
        parent=local_atoms["CG"].coord,
        closing_parent=local_atoms["N"].coord,
        parent_length=_residue_length(residue, "CG-CD", PRO_CG_CD),
        closing_length=_residue_length(residue, "N-CD", PRO_N_CD),
        preferred=cd_atom.coord,
        fallback_reference=local_atoms.get("CB", local_atoms["N"]).coord,
    )


def _close_pca_ring(local_atoms: Dict[str, Atom], residue: ResidueGeometry) -> None:
    """Close the pyroglutamate lactam ring and keep its carbonyl oxygen bonded."""

    if not all(name in local_atoms for name in ("N", "CG", "CD")):
        return
    cd_atom = local_atoms["CD"]
    cd_atom.coord = _place_atom_from_two_bond_lengths(
        parent=local_atoms["CG"].coord,
        closing_parent=local_atoms["N"].coord,
        parent_length=_residue_length(residue, "CG-CD", PCA_CG_CD),
        closing_length=_residue_length(residue, "N-CD", PCA_N_CD),
        preferred=cd_atom.coord,
        fallback_reference=local_atoms.get("CB", local_atoms["N"]).coord,
    )
    if "OE" in local_atoms and "CB" in local_atoms:
        local_atoms["OE"].coord = place_atom(
            local_atoms["CB"].coord,
            local_atoms["CG"].coord,
            local_atoms["CD"].coord,
            length=_residue_length(residue, "CD-OE", PCA_CD_OE),
            angle_degrees=120.0,
            dihedral_degrees=_placement_dihedral_degrees(0.0),
        )


def _close_pyl_ring(local_atoms: Dict[str, Atom], residue: ResidueGeometry) -> None:
    """Close pyrrolysine's pyrroline ring after template placement."""

    if not all(name in local_atoms for name in ("CA2", "CD2", "CG2")):
        return
    cg2_atom = local_atoms["CG2"]
    cg2_atom.coord = _place_atom_from_two_bond_lengths(
        parent=local_atoms["CD2"].coord,
        closing_parent=local_atoms["CA2"].coord,
        parent_length=_residue_length(residue, "CD2-CG2", PYL_CD2_CG2),
        closing_length=_residue_length(residue, "CA2-CG2", PYL_CA2_CG2),
        preferred=cg2_atom.coord,
        fallback_reference=local_atoms.get("CB2", local_atoms["CA2"]).coord,
    )
    if "CB2" in local_atoms:
        local_atoms["CB2"].coord = place_atom(
            local_atoms["CA2"].coord,
            local_atoms["CD2"].coord,
            local_atoms["CG2"].coord,
            length=_residue_length(residue, "CG2-CB2", PYL_CG2_CB2),
            angle_degrees=113.0,
            dihedral_degrees=_placement_dihedral_degrees(120.0),
        )


def _place_atom_from_two_bond_lengths(
    *,
    parent: Sequence[float],
    closing_parent: Sequence[float],
    parent_length: float,
    closing_length: float,
    preferred: Sequence[float],
    fallback_reference: Sequence[float],
) -> Tuple[float, float, float]:
    """Move a terminal atom onto the circle that satisfies two bond lengths."""

    axis_raw = sub(closing_parent, parent)
    axis_length = norm(axis_raw)
    if axis_length < 1e-12:
        return (float(preferred[0]), float(preferred[1]), float(preferred[2]))
    if axis_length > parent_length + closing_length or axis_length < abs(parent_length - closing_length):
        return (float(preferred[0]), float(preferred[1]), float(preferred[2]))

    axis = scale(axis_raw, 1.0 / axis_length)
    along_axis = (
        parent_length * parent_length
        - closing_length * closing_length
        + axis_length * axis_length
    ) / (2.0 * axis_length)
    circle_center = add(parent, scale(axis, along_axis))
    circle_radius = math.sqrt(max(parent_length * parent_length - along_axis * along_axis, 0.0))

    direction = _perpendicular_component(preferred, circle_center, axis)
    if norm(direction) < 1e-12:
        direction = _perpendicular_component(fallback_reference, circle_center, axis)
    if norm(direction) < 1e-12:
        direction = cross(axis, (1.0, 0.0, 0.0))
    if norm(direction) < 1e-12:
        direction = cross(axis, (0.0, 1.0, 0.0))
    return add(circle_center, scale(normalize(direction), circle_radius))


def _perpendicular_component(
    point: Sequence[float],
    origin: Sequence[float],
    axis: Sequence[float],
) -> Tuple[float, float, float]:
    relative = sub(point, origin)
    return sub(relative, scale(axis, dot(relative, axis)))


def _resolve_dihedral(spec: object, residue: ResidueGeometry) -> float:
    if isinstance(spec, (float, int)):
        return float(spec)
    if isinstance(spec, tuple):
        base, offset = spec
        return _resolve_dihedral(base, residue) + float(offset)
    if isinstance(spec, str) and spec.startswith("chi"):
        index = int(spec[3:]) - 1
        if index < len(residue.chi):
            return float(residue.chi[index])
        return DEFAULT_CHIS[index] if index < len(DEFAULT_CHIS) else -60.0
    raise ValueError(f"Unsupported dihedral spec: {spec!r}")


def _resolve_residue_atoms(
    key: ResidueKey,
    atoms: Sequence[Atom],
    warnings: List[str],
) -> Dict[str, Atom]:
    resolved: Dict[str, Atom] = {}
    for atom in atoms:
        name = atom.name.strip().upper()
        if name not in resolved:
            resolved[name] = atom
            continue
        existing = resolved[name]
        if _occupancy(atom) > _occupancy(existing):
            resolved[name] = atom
        chain_id, resseq, _icode, resname, _record_name = key
        warnings.append(
            f"{chain_id}:{resseq}:{resname}: duplicate atom {name}; selected highest occupancy"
        )
    return resolved


def _occupancy(atom: Atom) -> float:
    return -1.0 if atom.occupancy is None else float(atom.occupancy)


def _has_backbone(atoms: Mapping[str, Atom]) -> bool:
    return all(name in atoms for name in ("N", "CA", "C"))


def _effective_resseq(residue: ResidueGeometry, index: int) -> int:
    return residue.resseq if residue.resseq is not None else index + 1


def _residue_geometry_continues_previous(
    previous_chain_id: Optional[str],
    previous_resseq: Optional[int],
    chain_id: str,
    resseq: int,
) -> bool:
    if previous_chain_id is None or previous_resseq is None:
        return False
    if previous_chain_id != chain_id:
        return False
    return _residue_geometry_resseq_continues(previous_resseq, resseq)


def _residue_geometry_resseq_continues(current_resseq: int, next_resseq: int) -> bool:
    resseq_delta = next_resseq - current_resseq
    # Consecutive residue numbers are the normal polymer path. Equal numbers are
    # allowed because residue-geometry JSON currently does not retain PDB
    # insertion codes, so insertion-code neighbors collapse to the same resseq.
    return resseq_delta in {0, 1}


def _neighbor_backbone(
    residues: Sequence[Tuple[ResidueKey, Dict[str, Atom]]],
    index: int,
    direction: int,
) -> Optional[Dict[str, Atom]]:
    current_key = residues[index][0]
    cursor = index + direction
    if cursor < 0 or cursor >= len(residues):
        return None
    key, atoms = residues[cursor]
    if not _is_adjacent_polymer_residue(current_key, key, direction):
        return None
    if not _has_backbone(atoms):
        return None
    return atoms


def _is_adjacent_polymer_residue(current: ResidueKey, neighbor: ResidueKey, direction: int) -> bool:
    if current[0] != neighbor[0]:
        return False
    if current[4] != "ATOM" or neighbor[4] != "ATOM":
        return False
    resseq_delta = neighbor[1] - current[1]
    if resseq_delta == direction:
        return True
    # PDB insertion-code residues can share the same numeric resseq while still
    # being adjacent in file order.
    return resseq_delta == 0 and bool(current[2] or neighbor[2]) and current[2] != neighbor[2]


def _safe_dihedral(
    atom_a: Atom,
    atom_b: Atom,
    atom_c: Atom,
    atom_d: Atom,
    warnings: List[str],
    label: str,
) -> Optional[float]:
    try:
        return dihedral_degrees(atom_a.coord, atom_b.coord, atom_c.coord, atom_d.coord)
    except ValueError as exc:
        warnings.append(f"{label}: {exc}")
        return None


def _safe_bond_angle(
    atom_a: Atom,
    atom_b: Atom,
    atom_c: Atom,
    warnings: List[str],
    label: str,
) -> Optional[float]:
    try:
        return bond_angle_degrees(atom_a.coord, atom_b.coord, atom_c.coord)
    except ValueError as exc:
        warnings.append(f"{label}: {exc}")
        return None


def _extract_bond_lengths(
    atoms: Mapping[str, Atom],
    next_atoms: Optional[Mapping[str, Atom]],
    stored_lengths: Sequence[str],
) -> Dict[str, float]:
    if not stored_lengths:
        return {}
    resname = next(iter(atoms.values())).resname.strip().upper() if atoms else ""
    lengths: Dict[str, float] = {}
    _store_length(lengths, stored_lengths, "N-CA", atoms.get("N"), atoms.get("CA"))
    _store_length(lengths, stored_lengths, "CA-C", atoms.get("CA"), atoms.get("C"))
    _store_length(lengths, stored_lengths, "C-O", atoms.get("C"), atoms.get("O"))
    if next_atoms is not None:
        _store_length(lengths, stored_lengths, "C-N", atoms.get("C"), next_atoms.get("N"))
    if resname != "GLY":
        _store_length(lengths, stored_lengths, "CA-CB", atoms.get("CA"), atoms.get("CB"))
    for step in SIDECHAIN_STEPS.get(resname, []):
        _store_length(
            lengths,
            stored_lengths,
            f"{step.parent}-{step.atom}",
            atoms.get(step.parent),
            atoms.get(step.atom),
        )
    for key in _ring_closure_length_keys(resname):
        first, second = key.split("-", 1)
        _store_length(lengths, stored_lengths, key, atoms.get(first), atoms.get(second))
    return lengths


def _store_length(
    lengths: Dict[str, float],
    stored_lengths: Sequence[str],
    key: str,
    atom_a: Optional[Atom],
    atom_b: Optional[Atom],
) -> None:
    if atom_a is None or atom_b is None or not _length_requested(key, stored_lengths):
        return
    lengths[key] = distance(atom_a.coord, atom_b.coord)


def _length_requested(key: str, stored_lengths: Sequence[str]) -> bool:
    if "all" in stored_lengths:
        return True
    if key in stored_lengths:
        return True
    if key in RESIDUE_GEOMETRY_BACKBONE_LENGTHS:
        return "backbone" in stored_lengths
    return "sidechain" in stored_lengths


def _ring_closure_length_keys(resname: str) -> Tuple[str, ...]:
    if resname in {"PRO", "HYP"}:
        return ("N-CD",)
    if resname == "PCA":
        return ("N-CD",)
    if resname == "PYL":
        return ("CA2-CG2",)
    if resname in {"PHE", "TYR", "PTR"}:
        return ("CE2-CZ",)
    if resname == "HIS":
        return ("CE1-NE2",)
    if resname == "TRP":
        return ("NE1-CE2", "CZ3-CH2")
    return ()


def _extract_chis(key: ResidueKey, atoms: Mapping[str, Atom], warnings: List[str]) -> List[float]:
    chain_id, resseq, _icode, resname, record_name = key
    steps = SIDECHAIN_STEPS.get(resname, [])
    if not steps:
        if record_name == "ATOM" and resname not in {"ALA", "GLY"}:
            warnings.append(f"{chain_id}:{resseq}:{resname}: no side-chain chi template")
        return []

    chis: Dict[int, float] = {}
    for step in steps:
        chi_index = _chi_index(step.dihedral)
        if chi_index is None or chi_index in chis:
            continue
        names = [step.previous, step.anchor, step.parent, step.atom]
        if not all(name in atoms for name in names):
            warnings.append(
                f"{chain_id}:{resseq}:{resname}: missing atoms for chi{chi_index + 1}"
            )
            continue
        angle = _safe_dihedral(
            atoms[step.previous],
            atoms[step.anchor],
            atoms[step.parent],
            atoms[step.atom],
            warnings,
            f"{chain_id}:{resseq}:{resname}:chi{chi_index + 1}",
        )
        if angle is not None:
            chis[chi_index] = angle

    if not chis:
        return []
    return [chis[index] for index in sorted(chis)]


def _chi_index(spec: object) -> Optional[int]:
    if isinstance(spec, str) and spec.startswith("chi"):
        return int(spec[3:]) - 1
    return None


def _metadata_warnings(metadata: Mapping[str, object]) -> List[str]:
    warnings = metadata.get("warnings", [])
    if isinstance(warnings, list):
        return [str(item) for item in warnings]
    return [str(warnings)]


def _next_residue_continues_chain(
    residues: Sequence[ResidueGeometry],
    index: int,
    chain_id: str,
    resseq: int,
) -> bool:
    if index + 1 >= len(residues):
        return False
    next_residue = residues[index + 1]
    if (next_residue.chain_id or "A") != chain_id:
        return False
    try:
        one_to_three(next_residue.name)
    except ValueError:
        return False
    next_resseq = _effective_resseq(next_residue, index + 1)
    return _residue_geometry_resseq_continues(resseq, next_resseq)

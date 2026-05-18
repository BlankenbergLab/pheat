"""Core serializable data models.

The models are intentionally implemented with dataclasses instead of a validation
framework so the backend remains usable before optional scientific dependencies are
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


Coordinate = Tuple[float, float, float]
ResidueKey = Tuple[str, int, str, str, str]
ResidueReference = Tuple[str, int, str]
ATOM_STRUCTURE_FORMAT = "pheat.atom-structure"
ATOM_STRUCTURE_VERSION = 1
HEAVY_ATOM_STRUCTURE_FORMAT = ATOM_STRUCTURE_FORMAT
HEAVY_ATOM_STRUCTURE_VERSION = ATOM_STRUCTURE_VERSION
RESIDUE_GEOMETRY_STRUCTURE_FORMAT = "pheat.residue-geometry-structure"
RESIDUE_GEOMETRY_STRUCTURE_VERSION = 1
RESIDUE_GEOMETRY_CHI_ORDER = "chi1_to_chiN"
OPTIONAL_RESIDUE_GEOMETRY_ANGLES = ("omega", "tau", "theta")
RESIDUE_GEOMETRY_LENGTH_SELECTORS = ("all", "backbone", "sidechain")
RESIDUE_GEOMETRY_BACKBONE_LENGTHS = frozenset({"N-CA", "CA-C", "C-O", "C-N"})
CANONICAL_FLOAT_DECIMALS = 12


@dataclass(frozen=True)
class DisulfideBond:
    """An explicit inter-residue CYS SG-SG connectivity annotation."""

    chain_id_1: str
    resseq_1: int
    chain_id_2: str
    resseq_2: int
    icode_1: str = ""
    icode_2: str = ""
    source: str = "unknown"

    @property
    def residue_ref_1(self) -> ResidueReference:
        return (self.chain_id_1 or "", int(self.resseq_1), self.icode_1 or "")

    @property
    def residue_ref_2(self) -> ResidueReference:
        return (self.chain_id_2 or "", int(self.resseq_2), self.icode_2 or "")

    @property
    def canonical_key(self) -> Tuple[ResidueReference, ResidueReference]:
        refs = sorted((self.residue_ref_1, self.residue_ref_2))
        return (refs[0], refs[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id_1": self.chain_id_1,
            "resseq_1": self.resseq_1,
            "icode_1": self.icode_1,
            "chain_id_2": self.chain_id_2,
            "resseq_2": self.resseq_2,
            "icode_2": self.icode_2,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DisulfideBond":
        return cls(
            chain_id_1=str(data.get("chain_id_1") or ""),
            resseq_1=int(data["resseq_1"]),
            icode_1=str(data.get("icode_1") or ""),
            chain_id_2=str(data.get("chain_id_2") or ""),
            resseq_2=int(data["resseq_2"]),
            icode_2=str(data.get("icode_2") or ""),
            source=str(data.get("source") or "unknown"),
        )


@dataclass
class Atom:
    """A single atom parsed from or written to PDB-like records."""

    name: str
    element: str
    x: float
    y: float
    z: float
    resname: str
    chain_id: str = "A"
    resseq: int = 1
    icode: str = ""
    record_name: str = "ATOM"
    serial: Optional[int] = None
    altloc: str = ""
    occupancy: Optional[float] = None
    bfactor: Optional[float] = None
    charge: str = ""
    model: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def coord(self) -> Coordinate:
        return (self.x, self.y, self.z)

    @coord.setter
    def coord(self, value: Sequence[float]) -> None:
        self.x, self.y, self.z = float(value[0]), float(value[1]), float(value[2])

    @property
    def residue_key(self) -> ResidueKey:
        return (
            self.chain_id or "",
            int(self.resseq),
            self.icode or "",
            self.resname.strip().upper(),
            "HETATM" if self.record_name.upper().startswith("HET") else "ATOM",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "element": self.element,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "resname": self.resname,
            "chain_id": self.chain_id,
            "resseq": self.resseq,
            "icode": self.icode,
            "record_name": self.record_name,
            "serial": self.serial,
            "altloc": self.altloc,
            "occupancy": self.occupancy,
            "bfactor": self.bfactor,
            "charge": self.charge,
            "model": self.model,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Atom":
        return cls(
            name=str(data["name"]),
            element=str(data.get("element") or "").upper(),
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            resname=str(data["resname"]),
            chain_id=str(data.get("chain_id") or ""),
            resseq=int(data.get("resseq", 1)),
            icode=str(data.get("icode") or ""),
            record_name=str(data.get("record_name") or "ATOM"),
            serial=data.get("serial"),
            altloc=str(data.get("altloc") or ""),
            occupancy=data.get("occupancy"),
            bfactor=data.get("bfactor"),
            charge=str(data.get("charge") or ""),
            model=data.get("model"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Bond:
    """A declared or template covalent bond between two atoms in ``atoms`` order."""

    atom_index_1: int
    atom_index_2: int
    source: str = "unknown"
    length: Optional[float] = None
    order: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        first = int(self.atom_index_1)
        second = int(self.atom_index_2)
        if first == second:
            raise ValueError("bond atom indices must reference two distinct atoms")
        if second < first:
            first, second = second, first
        self.atom_index_1 = first
        self.atom_index_2 = second
        self.source = str(self.source or "unknown")
        self.metadata = dict(self.metadata or {})
        if self.length is not None:
            self.length = float(self.length)
        if self.order is not None:
            self.order = str(self.order)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.atom_index_1, self.atom_index_2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_index_1": self.atom_index_1,
            "atom_index_2": self.atom_index_2,
            "source": self.source,
            "length": self.length,
            "order": self.order,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Bond":
        return cls(
            atom_index_1=int(data["atom_index_1"]),
            atom_index_2=int(data["atom_index_2"]),
            source=str(data.get("source") or "unknown"),
            length=_optional_float(data.get("length")),
            order=None if data.get("order") is None else str(data.get("order")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AtomStructure:
    """A PDB-derived or generated atom structure.

    PHEAT defaults to heavy atoms, but the structure model can also store source
    or generated hydrogens when callers opt into all-atom workflows.
    """

    atoms: List[Atom]
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    disulfide_bonds: List[DisulfideBond] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    atom_scope: str = ""

    def __post_init__(self) -> None:
        self.disulfide_bonds = _normalize_disulfide_bonds(self.disulfide_bonds)
        self.bonds = _normalize_bonds(self.bonds)
        self.atom_scope = _normalize_atom_scope(self.atom_scope, self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    def __len__(self) -> int:
        return len(self.atoms)

    def residue_keys(self) -> List[ResidueKey]:
        keys: List[ResidueKey] = []
        seen = set()
        for atom in self.atoms:
            key = atom.residue_key
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    def atoms_for_residue(self, key: ResidueKey) -> List[Atom]:
        return [atom for atom in self.atoms if atom.residue_key == key]

    def atoms_by_residue(self) -> Dict[ResidueKey, List[Atom]]:
        residues: Dict[ResidueKey, List[Atom]] = {}
        for atom in self.atoms:
            residues.setdefault(atom.residue_key, []).append(atom)
        return residues

    def atom_lookup(self) -> Dict[Tuple[ResidueKey, str], Atom]:
        return {(atom.residue_key, atom.name.strip().upper()): atom for atom in self.atoms}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "format": ATOM_STRUCTURE_FORMAT,
            "version": ATOM_STRUCTURE_VERSION,
            "atom_scope": self.atom_scope,
            "name": self.name,
            "metadata": self.metadata,
            "disulfide_bonds": [bond.to_dict() for bond in self.disulfide_bonds],
            "atoms": [atom.to_dict() for atom in self.atoms],
        }
        if self.bonds:
            payload["bonds"] = [bond.to_dict() for bond in self.bonds]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AtomStructure":
        _validate_canonical_header(
            data,
            expected_format=ATOM_STRUCTURE_FORMAT,
            expected_version=ATOM_STRUCTURE_VERSION,
            label="atom structure",
        )
        atoms = [Atom.from_dict(item) for item in data.get("atoms", [])]
        disulfide_bonds = [
            DisulfideBond.from_dict(item)
            for item in data.get("disulfide_bonds", [])
        ]
        bonds = [Bond.from_dict(item) for item in data.get("bonds", [])]
        return cls(
            atoms=atoms,
            name=str(data.get("name") or ""),
            metadata=dict(data.get("metadata") or {}),
            disulfide_bonds=disulfide_bonds,
            bonds=bonds,
            atom_scope=str(data.get("atom_scope") or ""),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return _canonical_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AtomStructure":
        return cls.from_dict(json.loads(text))


HeavyAtomStructure = AtomStructure


@dataclass
class ResidueGeometry:
    """Residue-level internal geometry input.

    Optional backbone geometry fields use PHEAT's documented definitions:
    omega = CA(i)-C(i)-N(i+1)-CA(i+1), tau = N(i)-CA(i)-C(i),
    and theta = CA(i)-C(i)-N(i+1).
    """

    name: str
    phi: Optional[float] = None
    psi: Optional[float] = None
    omega: Optional[float] = None
    tau: Optional[float] = None
    theta: Optional[float] = None
    chi: List[float] = field(default_factory=list)
    bond_lengths: Dict[str, float] = field(default_factory=dict)
    chain_id: str = "A"
    resseq: Optional[int] = None
    icode: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int = 1, chain_id: str = "A") -> "ResidueGeometry":
        chi_value = data.get("chi", data.get("chis", []))
        if chi_value is None:
            chi: List[float] = []
        elif isinstance(chi_value, Mapping):
            chi = [float(chi_value[key]) for key in sorted(chi_value)]
        else:
            chi = [float(value) for value in chi_value]
        resseq_value = data.get("resseq", index)
        return cls(
            name=str(data.get("name") or data.get("resname") or data.get("aa")),
            phi=_optional_float(data.get("phi")),
            psi=_optional_float(data.get("psi")),
            omega=_optional_float(data.get("omega")),
            tau=_optional_float(data.get("tau")),
            theta=_optional_float(data.get("theta")),
            chi=chi,
            bond_lengths=_normalize_bond_lengths(data.get("bond_lengths") or {}),
            chain_id=str(data.get("chain_id") or chain_id),
            resseq=index if resseq_value is None else int(resseq_value),
            icode=str(data.get("icode") or ""),
        )

    def to_dict(
        self,
        *,
        stored_angles: Optional[Union[str, Iterable[str]]] = None,
        stored_lengths: Optional[Union[str, Iterable[str]]] = None,
        max_chi: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "phi": self.phi,
            "psi": self.psi,
            "chi": _limit_chi_values(self.chi, max_chi),
            "chain_id": self.chain_id,
            "resseq": self.resseq,
        }
        if self.icode:
            payload["icode"] = self.icode
        for angle_name in _normalize_stored_angles(stored_angles):
            payload[angle_name] = getattr(self, angle_name)
        selected_lengths = _selected_bond_lengths(self.bond_lengths, stored_lengths)
        if selected_lengths:
            payload["bond_lengths"] = selected_lengths
        return payload


@dataclass
class ResidueGeometryStructure:
    """A protein chain or set of chains defined by residue-level geometry."""

    residues: List[ResidueGeometry]
    name: str = ""
    angle_units: str = "radians"
    metadata: Dict[str, Any] = field(default_factory=dict)
    chi_order: str = RESIDUE_GEOMETRY_CHI_ORDER
    stored_angles: Tuple[str, ...] = field(default_factory=tuple)
    stored_lengths: Tuple[str, ...] = field(default_factory=tuple)
    disulfide_bonds: List[DisulfideBond] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.angle_units = _normalize_angle_units(self.angle_units)
        self.metadata = dict(self.metadata or {})
        self.metadata.pop("angle_units", None)
        if self.chi_order != RESIDUE_GEOMETRY_CHI_ORDER:
            raise ValueError(f"chi_order must be {RESIDUE_GEOMETRY_CHI_ORDER!r}")
        self.stored_angles = _normalize_stored_angles(self.stored_angles)
        self.stored_lengths = _normalize_stored_lengths(self.stored_lengths)
        self.disulfide_bonds = _normalize_disulfide_bonds(self.disulfide_bonds)

    def to_dict(
        self,
        *,
        stored_angles: Optional[Union[str, Iterable[str]]] = None,
        stored_lengths: Optional[Union[str, Iterable[str]]] = None,
        max_chi: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_stored_angles = (
            self.stored_angles if stored_angles is None else _normalize_stored_angles(stored_angles)
        )
        normalized_stored_lengths = (
            self.stored_lengths if stored_lengths is None else _normalize_stored_lengths(stored_lengths)
        )
        normalized_max_chi = _normalize_max_chi(max_chi)
        return {
            "format": RESIDUE_GEOMETRY_STRUCTURE_FORMAT,
            "version": RESIDUE_GEOMETRY_STRUCTURE_VERSION,
            "angle_units": self.angle_units,
            "chi_order": self.chi_order,
            "name": self.name,
            "metadata": self.metadata,
            "disulfide_bonds": [bond.to_dict() for bond in self.disulfide_bonds],
            "residues": [
                residue.to_dict(
                    stored_angles=normalized_stored_angles,
                    stored_lengths=normalized_stored_lengths,
                    max_chi=normalized_max_chi,
                )
                for residue in self.residues
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        angle_units: Optional[str] = None,
        strict_format: bool = False,
    ) -> "ResidueGeometryStructure":
        if strict_format or "format" in data or "version" in data:
            _validate_canonical_header(
                data,
                expected_format=RESIDUE_GEOMETRY_STRUCTURE_FORMAT,
                expected_version=RESIDUE_GEOMETRY_STRUCTURE_VERSION,
                label="residue-geometry structure",
            )
            if "angle_units" not in data:
                raise ValueError("Canonical residue-geometry JSON requires 'angle_units'.")
            if "residues" not in data:
                raise ValueError("Canonical residue-geometry JSON requires 'residues'.")
        if "residues" in data:
            residue_items = list(data.get("residues", []))
            residues = [
                ResidueGeometry.from_dict(item, index=index + 1)
                for index, item in enumerate(residue_items)
            ]
            stored_angles = _stored_angles_from_residue_items(residue_items)
            stored_lengths = _stored_lengths_from_residue_items(residue_items)
        elif "sequence" in data:
            from pheat.residues import one_to_three

            sequence = str(data["sequence"]).strip()
            phi_values = _list_or_fill(data.get("phi"), len(sequence), None)
            psi_values = _list_or_fill(data.get("psi"), len(sequence), None)
            sequence_angle_units = _angle_units_from_data(data, angle_units)
            omega_default = None if "omega" in data else _default_omega_for_units(sequence_angle_units)
            omega_values = _list_or_fill(
                data.get("omega"),
                len(sequence),
                omega_default,
            )
            tau_values = _list_or_fill(data.get("tau"), len(sequence), None)
            theta_values = _list_or_fill(data.get("theta"), len(sequence), None)
            chi_values = _list_or_fill(data.get("chi", data.get("chis")), len(sequence), [])
            chain_id = str(data.get("chain_id") or "A")
            residues = []
            provided_stored_angles = data.get("stored_angles")
            if provided_stored_angles is None:
                stored_angles = tuple(
                    angle_name
                    for angle_name in OPTIONAL_RESIDUE_GEOMETRY_ANGLES
                    if angle_name in data
                )
            else:
                stored_angles = _normalize_stored_angles(provided_stored_angles)
            stored_lengths = ()
            for index, aa in enumerate(sequence, start=1):
                is_first = index == 1
                is_last = index == len(sequence)
                residues.append(
                    ResidueGeometry(
                        name=one_to_three(aa),
                        phi=None if is_first else _optional_float(phi_values[index - 1]),
                        psi=None if is_last else _optional_float(psi_values[index - 1]),
                        omega=None if is_last else _optional_float(omega_values[index - 1]),
                        tau=_optional_float(tau_values[index - 1]),
                        theta=None if is_last else _optional_float(theta_values[index - 1]),
                        chi=[float(value) for value in (chi_values[index - 1] or [])],
                        bond_lengths={},
                        chain_id=chain_id,
                        resseq=index,
                    )
                )
        else:
            raise ValueError("Residue-geometry JSON must contain either 'residues' or 'sequence'.")
        metadata = dict(data.get("metadata") or {})
        metadata.pop("angle_units", None)
        disulfide_bonds = [
            DisulfideBond.from_dict(item)
            for item in data.get("disulfide_bonds", [])
        ]
        return cls(
            residues=residues,
            name=str(data.get("name") or ""),
            angle_units=_angle_units_from_data(data, angle_units),
            metadata=metadata,
            chi_order=str(data.get("chi_order") or RESIDUE_GEOMETRY_CHI_ORDER),
            stored_angles=stored_angles,
            stored_lengths=stored_lengths,
            disulfide_bonds=disulfide_bonds,
        )

    def to_json(
        self,
        *,
        indent: Optional[int] = 2,
        stored_angles: Optional[Union[str, Iterable[str]]] = None,
        stored_lengths: Optional[Union[str, Iterable[str]]] = None,
        max_chi: Optional[int] = None,
    ) -> str:
        return _canonical_json_dumps(
            self.to_dict(stored_angles=stored_angles, stored_lengths=stored_lengths, max_chi=max_chi),
            indent=indent,
        )

    @classmethod
    def from_json(cls, text: str, *, angle_units: Optional[str] = None) -> "ResidueGeometryStructure":
        return cls.from_dict(json.loads(text), angle_units=angle_units, strict_format=True)


@dataclass
class Centroid:
    name: str
    x: float
    y: float
    z: float
    resname: str
    chain_id: str
    resseq: int
    atom_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def coord(self) -> Coordinate:
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "resname": self.resname,
            "chain_id": self.chain_id,
            "resseq": self.resseq,
            "atom_names": self.atom_names,
            "metadata": self.metadata,
        }


@dataclass
class CentroidStructure:
    centroids: List[Centroid]
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "pheat.centroid-structure",
            "version": 1,
            "name": self.name,
            "metadata": self.metadata,
            "centroids": [centroid.to_dict() for centroid in self.centroids],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return _canonical_json_dumps(self.to_dict(), indent=indent)


@dataclass
class EnergyResult:
    model: str
    total: float
    units: str
    terms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total": self.total,
            "units": self.units,
            "terms": self.terms,
            "warnings": self.warnings,
            "citations": self.citations,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return _canonical_json_dumps(self.to_dict(), indent=indent)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _normalize_disulfide_bonds(values: Iterable[Union[DisulfideBond, Mapping[str, Any]]]) -> List[DisulfideBond]:
    bonds: Dict[Tuple[ResidueReference, ResidueReference], DisulfideBond] = {}
    for value in values:
        bond = value if isinstance(value, DisulfideBond) else DisulfideBond.from_dict(value)
        key = bond.canonical_key
        if key in bonds:
            continue
        bonds[key] = bond
    return [bonds[key] for key in sorted(bonds)]


def _normalize_bonds(values: Iterable[Union[Bond, Mapping[str, Any]]]) -> List[Bond]:
    bonds: Dict[Tuple[int, int], Bond] = {}
    for value in values:
        bond = value if isinstance(value, Bond) else Bond.from_dict(value)
        bonds.setdefault(bond.key, bond)
    return [bonds[key] for key in sorted(bonds)]


def _normalize_atom_scope(value: str, atoms: Sequence[Atom]) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = "all" if any(_is_hydrogen_atom(atom) for atom in atoms) else "heavy"
    if normalized not in {"heavy", "all"}:
        raise ValueError("atom_scope must be one of heavy, all")
    if normalized == "heavy" and any(_is_hydrogen_atom(atom) for atom in atoms):
        raise ValueError("atom_scope='heavy' cannot contain hydrogen atoms")
    return normalized


def _is_hydrogen_atom(atom: Atom) -> bool:
    return atom.element.strip().upper() in {"H", "D", "T"}


def _canonical_json_dumps(payload: Mapping[str, Any], *, indent: Optional[int]) -> str:
    return json.dumps(_canonical_json_value(payload), indent=indent, sort_keys=True)


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, CANONICAL_FLOAT_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, Mapping):
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def _list_or_fill(value: Any, length: int, default: Any) -> List[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != length:
            raise ValueError(f"Expected {length} values, found {len(values)}.")
        return values
    return [value for _ in range(length)]


def _angle_units_from_data(data: Mapping[str, Any], explicit: Optional[str]) -> str:
    if explicit is not None:
        return _normalize_angle_units(explicit)
    if data.get("angle_units") is not None:
        return _normalize_angle_units(data["angle_units"])
    return "radians"


def _validate_canonical_header(
    data: Mapping[str, Any],
    *,
    expected_format: str,
    expected_version: int,
    label: str,
) -> None:
    found_format = data.get("format")
    if found_format != expected_format:
        raise ValueError(f"Canonical {label} JSON requires format {expected_format!r}.")
    found_version = data.get("version")
    if found_version != expected_version:
        raise ValueError(f"Canonical {label} JSON requires version {expected_version!r}.")


def _normalize_angle_units(angle_units: object) -> str:
    normalized = str(angle_units).strip().lower()
    if normalized not in {"radians", "degrees"}:
        raise ValueError("angle_units must be one of radians, degrees")
    return normalized


def _default_omega_for_units(angle_units: str) -> float:
    if angle_units == "degrees":
        return 180.0
    return math.pi


def _normalize_stored_angles(value: Optional[Union[str, Iterable[str]]]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in value if str(token).strip()]
    if "all" in tokens:
        tokens = list(OPTIONAL_RESIDUE_GEOMETRY_ANGLES)
    unknown = sorted(set(tokens) - set(OPTIONAL_RESIDUE_GEOMETRY_ANGLES))
    if unknown:
        raise ValueError(
            "stored optional residue-geometry angles must be chosen from "
            f"{', '.join(OPTIONAL_RESIDUE_GEOMETRY_ANGLES)}"
        )
    return tuple(angle for angle in OPTIONAL_RESIDUE_GEOMETRY_ANGLES if angle in tokens)


def _normalize_stored_lengths(value: Optional[Union[str, Iterable[str]]]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in value if str(token).strip()]
    normalized = []
    for token in tokens:
        lower = token.lower()
        if lower in RESIDUE_GEOMETRY_LENGTH_SELECTORS:
            normalized.append(lower)
            continue
        normalized.append(_normalize_bond_length_key(token))
    return tuple(dict.fromkeys(normalized))


def _normalize_bond_lengths(value: Mapping[str, Any]) -> Dict[str, float]:
    return {
        _normalize_bond_length_key(key): float(length)
        for key, length in value.items()
    }


def _normalize_bond_length_key(value: object) -> str:
    token = str(value).strip().upper().replace(" ", "")
    if token.count("-") != 1:
        raise ValueError("bond length keys must use ATOM-ATOM form")
    first, second = token.split("-", 1)
    if not first or not second:
        raise ValueError("bond length keys must use ATOM-ATOM form")
    return f"{first}-{second}"


def _selected_bond_lengths(
    values: Mapping[str, float],
    stored_lengths: Optional[Union[str, Iterable[str]]],
) -> Dict[str, float]:
    normalized = _normalize_stored_lengths(stored_lengths)
    if not normalized or not values:
        return {}
    if "all" in normalized:
        return dict(sorted(values.items()))
    allowed = {token for token in normalized if token not in RESIDUE_GEOMETRY_LENGTH_SELECTORS}
    if "backbone" in normalized:
        allowed.update(key for key in values if key in RESIDUE_GEOMETRY_BACKBONE_LENGTHS)
    if "sidechain" in normalized:
        allowed.update(key for key in values if key not in RESIDUE_GEOMETRY_BACKBONE_LENGTHS)
    return {key: values[key] for key in sorted(values) if key in allowed}


def _normalize_max_chi(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_chi must be a non-negative integer or None")
    if value < 0:
        raise ValueError("max_chi must be a non-negative integer or None")
    return value


def _limit_chi_values(values: Sequence[float], max_chi: Optional[int]) -> List[float]:
    normalized = _normalize_max_chi(max_chi)
    if normalized is None:
        return list(values)
    return list(values[:normalized])


def _stored_angles_from_residue_items(items: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    present = []
    for angle_name in OPTIONAL_RESIDUE_GEOMETRY_ANGLES:
        if any(angle_name in item for item in items):
            present.append(angle_name)
    return tuple(present)


def _stored_lengths_from_residue_items(items: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    keys = []
    for item in items:
        for key in (item.get("bond_lengths") or {}):
            normalized = _normalize_bond_length_key(key)
            if normalized not in keys:
                keys.append(normalized)
    return tuple(keys)

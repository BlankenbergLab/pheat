"""PDB parsing and writing for PHEAT atom structures."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from urllib.request import urlretrieve

from pheat.bonds import declared_pairs_from_serials, structure_with_bonds
from pheat.domains import filter_structure_for_domain
from pheat.hydrogens import generate_hydrogens, is_hydrogen_element, normalize_hydrogen_policy
from pheat.models import Atom, DisulfideBond, HeavyAtomStructure, ResidueReference
from pheat.residues import guess_element

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"
_PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")
_FORMAT_EXTENSIONS = {"pdb": "pdb", "cif": "cif", "mmcif": "cif"}
_default_cache_dir: Optional[Path] = None


def _normalize_format(format: str) -> str:
    fmt = format.lower()
    if fmt not in _FORMAT_EXTENSIONS:
        raise ValueError(
            f"Unknown format '{format}'. Expected one of: pdb, cif, mmcif."
        )
    return "cif" if fmt == "mmcif" else fmt


def _resolve_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    global _default_cache_dir
    if cache_dir is None:
        if _default_cache_dir is None:
            _default_cache_dir = Path(tempfile.mkdtemp(prefix="pheat-pdb-"))
        return _default_cache_dir
    return Path(cache_dir)


def fetch_pdb(
    pdb_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    *,
    format: str = "pdb",
    overwrite: bool = False,
    downloader: Optional[Callable[[str, str], Any]] = None,
) -> Path:
    """Download a PDB or mmCIF file from RCSB into a cache directory.

    Parameters
    ----------
    pdb_id:
        Four-character RCSB identifier (case-insensitive).
    cache_dir:
        Destination directory. If ``None``, a process-wide temp directory is
        lazily created on first call and reused on subsequent calls.
    format:
        ``"pdb"`` (default), ``"cif"``, or ``"mmcif"`` (alias for ``cif``).
    overwrite:
        Re-download even when a cached copy exists.
    downloader:
        Optional callable ``(url, dest)`` replacing :func:`urllib.request.urlretrieve`.
        Useful for tests.

    Returns
    -------
    Path to the cached file.
    """
    if not isinstance(pdb_id, str) or not _PDB_ID_RE.match(pdb_id):
        raise ValueError(
            f"Invalid PDB id {pdb_id!r}: expected a 4-character alphanumeric string."
        )
    fmt = _normalize_format(format)
    ext = _FORMAT_EXTENSIONS[fmt]
    target_dir = _resolve_cache_dir(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pdb_id_upper = pdb_id.upper()
    output = target_dir / f"{pdb_id_upper}.{ext}"
    if output.exists() and not overwrite:
        return output
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id_upper, ext=ext)
    fetcher = downloader if downloader is not None else urlretrieve
    fetcher(url, str(output))
    return output


def load_pdb_by_id(
    pdb_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    *,
    format: str = "pdb",
    overwrite: bool = False,
    downloader: Optional[Callable[[str, str], Any]] = None,
    **load_kwargs: Any,
) -> HeavyAtomStructure:
    """Fetch (if needed) and parse a PDB/mmCIF entry by RCSB id.

    Extra keyword arguments are forwarded to :func:`load_pdb` or
    :func:`pheat.mmcif.load_mmcif` depending on ``format``.
    """
    fmt = _normalize_format(format)
    path = fetch_pdb(
        pdb_id,
        cache_dir,
        format=fmt,
        overwrite=overwrite,
        downloader=downloader,
    )
    if fmt == "cif":
        from pheat.mmcif import load_mmcif
        return load_mmcif(path, **load_kwargs)
    return load_pdb(path, **load_kwargs)


def load_pdb_ca_by_id(
    pdb_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    *,
    format: str = "pdb",
    overwrite: bool = False,
    downloader: Optional[Callable[[str, str], Any]] = None,
):
    """Return a ``(N, 3)`` numpy array of Cα coordinates for an RCSB entry."""
    import numpy as np

    structure = load_pdb_by_id(
        pdb_id,
        cache_dir,
        format=format,
        overwrite=overwrite,
        downloader=downloader,
    )
    return np.array(
        [(a.x, a.y, a.z) for a in structure.atoms if a.name.strip() == "CA"],
        dtype=float,
    )


def load_pdb(
    path: Union[str, Path],
    *,
    model: Optional[int] = 1,
    name: str = "",
    hydrogens: str = "drop",
    store_bonds: str = "none",
) -> HeavyAtomStructure:
    with open(path, "r", encoding="utf-8") as handle:
        return structure_from_pdb_string(
            handle.read(),
            model=model,
            name=name or Path(path).name,
            hydrogens=hydrogens,
            store_bonds=store_bonds,
        )


def write_pdb(
    structure: HeavyAtomStructure,
    path: Union[str, Path],
    *,
    allow_chain_truncation: bool = False,
    remarks: Optional[Iterable[str]] = None,
    domain: Optional[str] = None,
) -> None:
    text = structure_to_pdb_string(
        structure,
        allow_chain_truncation=allow_chain_truncation,
        remarks=remarks,
        domain=domain,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_multimodel_pdb(
    structures: Iterable[HeavyAtomStructure],
    path: Union[str, Path],
    *,
    remarks_per_model: Optional[Iterable[Optional[Iterable[str]]]] = None,
    allow_chain_truncation: bool = False,
    domain: Optional[str] = None,
) -> None:
    """Write structures as a multi-model PDB with optional per-model remarks.

    Each input structure becomes one ``MODEL``/``ENDMDL`` block. Per-model
    remarks, when provided, are written inside that model block before atom
    records. Existing single-structure ``END`` records are omitted so viewers
    can treat the file as a trajectory-like ensemble.
    """

    structure_list = list(structures)
    if remarks_per_model is None:
        remarks_list: List[Optional[Iterable[str]]] = [None] * len(structure_list)
    else:
        remarks_list = list(remarks_per_model)
        if len(remarks_list) != len(structure_list):
            raise ValueError("remarks_per_model length must match structures length.")

    lines: List[str] = []
    for model_index, (structure, remarks) in enumerate(zip(structure_list, remarks_list), start=1):
        lines.append(f"MODEL     {model_index:>4}")
        pdb_text = structure_to_pdb_string(
            structure,
            allow_chain_truncation=allow_chain_truncation,
            remarks=remarks,
            domain=domain,
        )
        for line in pdb_text.splitlines():
            record = line[:6].strip().upper()
            if record in {"MODEL", "ENDMDL", "END"}:
                continue
            lines.append(line)
        lines.append("ENDMDL")
    lines.append("END")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_heavy_json(path: Union[str, Path]) -> HeavyAtomStructure:
    with open(path, "r", encoding="utf-8") as handle:
        return HeavyAtomStructure.from_dict(json.load(handle))


load_structure_json = load_heavy_json


def write_heavy_json(structure: HeavyAtomStructure, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(structure.to_json())
        handle.write("\n")


write_structure_json = write_heavy_json


def structure_from_pdb_string(
    text: str,
    *,
    model: Optional[int] = 1,
    name: str = "",
    hydrogens: str = "drop",
    store_bonds: str = "none",
) -> HeavyAtomStructure:
    hydrogen_policy = normalize_hydrogen_policy(hydrogens)
    atoms: List[Atom] = []
    current_model: Optional[int] = None
    saw_model = False
    warnings: List[str] = []
    ssbond_bonds: List[DisulfideBond] = []
    conect_pairs: List[Tuple[int, int]] = []
    dropped_hydrogen_count = 0
    preserved_hydrogen_count = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        record = line[0:6].strip().upper()
        if record == "SSBOND":
            try:
                ssbond_bonds.append(_parse_ssbond_line(line))
            except ValueError as exc:
                warnings.append(f"line {line_number}: {exc}")
            continue
        if record == "CONECT":
            try:
                conect_pairs.extend(_parse_conect_line(line))
            except ValueError as exc:
                warnings.append(f"line {line_number}: {exc}")
            continue
        if record == "MODEL":
            # Multi-model PDBs are common for NMR structures. Keep one model by
            # default so downstream conversions have deterministic atom sets.
            saw_model = True
            try:
                current_model = int(line[10:14].strip() or len({atom.model for atom in atoms}) + 1)
            except ValueError:
                current_model = 1
            continue
        if record == "ENDMDL":
            current_model = None
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        atom_model = current_model if saw_model else None
        if model is not None and saw_model and atom_model != model:
            continue
        try:
            atom = _parse_atom_line(line, record, atom_model)
        except ValueError as exc:
            warnings.append(f"line {line_number}: {exc}")
            continue
        # PHEAT defaults to compact heavy-atom structures. The count is preserved
        # in metadata so all-atom inputs remain auditable. Callers can opt into
        # preserving source hydrogens or generating hydrogens later.
        if is_hydrogen_element(atom.element) and hydrogen_policy != "preserve":
            dropped_hydrogen_count += 1
            continue
        if is_hydrogen_element(atom.element):
            preserved_hydrogen_count += 1
        atoms.append(atom)

    if dropped_hydrogen_count:
        warnings.append(f"dropped {dropped_hydrogen_count} hydrogen atoms from all-atom PDB input")
    metadata = {
        "source": "pdb",
        "warnings": warnings,
        "dropped_hydrogen_count": dropped_hydrogen_count,
        "preserved_hydrogen_count": preserved_hydrogen_count,
        "hydrogen_policy": _metadata_hydrogen_policy(
            requested=hydrogen_policy,
            dropped_count=dropped_hydrogen_count,
            preserved_count=preserved_hydrogen_count,
        ),
    }
    if saw_model:
        metadata["selected_model"] = model
    disulfide_bonds = _merge_disulfide_bonds(
        ssbond_bonds,
        _disulfide_bonds_from_conect(atoms, conect_pairs),
    )
    structure = HeavyAtomStructure(
        atoms=atoms,
        name=name,
        metadata=metadata,
        disulfide_bonds=disulfide_bonds,
    )
    if hydrogen_policy == "generate":
        structure = generate_hydrogens(structure)
        if conect_pairs:
            structure.metadata.setdefault("warnings", [])
            structure.metadata["warnings"] = list(structure.metadata["warnings"]) + [
                "source CONECT bond storage was skipped because hydrogens were regenerated"
            ]
        return structure_with_bonds(structure, mode=store_bonds)
    declared_pairs = declared_pairs_from_serials(structure.atoms, conect_pairs)
    return structure_with_bonds(structure, mode=store_bonds, declared_pairs=declared_pairs)


def structure_to_pdb_string(
    structure: HeavyAtomStructure,
    *,
    allow_chain_truncation: bool = False,
    remarks: Optional[Iterable[str]] = None,
    domain: Optional[str] = None,
) -> str:
    if domain is not None:
        structure, _coverage = filter_structure_for_domain(structure, domain=domain)
    lines = []
    if remarks is not None:
        for remark in remarks:
            lines.extend(_format_remark_lines(remark))
    writable_disulfide_bonds = _disulfide_bonds_with_sg_atoms(structure)
    for index, bond in enumerate(writable_disulfide_bonds, start=1):
        lines.append(_format_ssbond_line(bond, index, allow_chain_truncation=allow_chain_truncation))
    distinct_models: List[Optional[int]] = sorted(
        set(atom.model for atom in structure.atoms),
        key=lambda m: (m is None, m if m is not None else 0),
    )
    multi_model = len(distinct_models) > 1
    serial_by_atom_id: Dict[int, int] = {}
    for model_num in distinct_models:
        model_atoms = (
            [a for a in structure.atoms if a.model == model_num]
            if multi_model
            else list(structure.atoms)
        )
        if multi_model:
            display_model = model_num if model_num is not None else 1
            lines.append(f"MODEL     {display_model:>4}")
        previous_chain = None
        serial = 1
        for atom in model_atoms:
            if previous_chain is not None and atom.chain_id != previous_chain:
                lines.append("TER")
            lines.append(
                _format_atom_line(
                    atom,
                    serial,
                    allow_chain_truncation=allow_chain_truncation,
                )
            )
            serial_by_atom_id[id(atom)] = serial
            previous_chain = atom.chain_id
            serial += 1
        if previous_chain is not None:
            lines.append("TER")
        if multi_model:
            lines.append("ENDMDL")
    for serial_a, serial_b in _disulfide_conect_serial_pairs(
        writable_disulfide_bonds,
        structure,
        serial_by_atom_id,
    ):
        lines.append(_format_conect_line(serial_a, serial_b))
    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse_atom_line(line: str, record: str, model: Optional[int]) -> Atom:
    padded = line.rstrip("\n").ljust(80)
    try:
        serial = int(padded[6:11].strip() or "0")
        name = padded[12:16].strip()
        altloc = padded[16].strip()
        resname = padded[17:20].strip().upper()
        chain_id = padded[21].strip() or "A"
        resseq = int(padded[22:26].strip())
        icode = padded[26].strip()
        x = float(padded[30:38].strip())
        y = float(padded[38:46].strip())
        z = float(padded[46:54].strip())
    except ValueError as exc:
        raise ValueError(f"invalid ATOM/HETATM field ({exc})") from exc

    occupancy = _parse_optional_float(padded[54:60])
    bfactor = _parse_optional_float(padded[60:66])
    element = padded[76:78].strip().upper() or guess_element(name)
    charge = padded[78:80].strip()
    return Atom(
        name=name,
        element=element,
        x=x,
        y=y,
        z=z,
        resname=resname,
        chain_id=chain_id,
        resseq=resseq,
        icode=icode,
        record_name=record,
        serial=serial,
        altloc=altloc,
        occupancy=occupancy,
        bfactor=bfactor,
        charge=charge,
        model=model,
    )


def _format_atom_line(atom: Atom, serial: int, *, allow_chain_truncation: bool) -> str:
    record = "HETATM" if atom.record_name.upper().startswith("HET") else "ATOM  "
    name = _format_atom_name(atom.name, atom.element)
    altloc = (atom.altloc or " ")[:1]
    resname = atom.resname[:3].rjust(3)
    chain_id = _pdb_chain_id(atom.chain_id, allow_chain_truncation=allow_chain_truncation)
    icode = (atom.icode or " ")[:1]
    occupancy = 1.0 if atom.occupancy is None else float(atom.occupancy)
    bfactor = 0.0 if atom.bfactor is None else float(atom.bfactor)
    element = (atom.element or guess_element(atom.name)).upper()[:2].rjust(2)
    charge = (atom.charge or "")[:2].rjust(2)
    line = (
        f"{record}{serial:5d} {name}{altloc}{resname} {chain_id}"
        f"{atom.resseq:4d}{icode}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element}{charge}"
    )
    return line.rstrip()


def _format_atom_name(name: str, element: str) -> str:
    # PDB atom names are column-sensitive: one-letter elements are right-shifted
    # inside the 4-character atom-name field, while two-letter elements are not.
    token = name.strip()[:4]
    if len(token) == 4:
        return token
    if len((element or "").strip()) == 1:
        return f" {token:<3}"
    return f"{token:<4}"


def _format_remark_lines(remark: str, *, number: int = 1) -> List[str]:
    # PDB REMARK records use a 3-digit type code right-justified in columns 8-10
    # with the remark text starting at column 12. We default to "REMARK   1" for
    # caller-defined free-form annotations, matching the convention used by most
    # downstream tools that scan for free-form REMARK metadata.
    rendered = []
    text = "" if remark is None else str(remark)
    for raw_line in text.splitlines() or [""]:
        rendered.append(f"REMARK {number:>3d} {raw_line}".rstrip())
    return rendered


def _parse_optional_float(text: str) -> Optional[float]:
    stripped = text.strip()
    return None if not stripped else float(stripped)


def _parse_ssbond_line(line: str) -> DisulfideBond:
    padded = line.rstrip("\n").ljust(80)
    try:
        return DisulfideBond(
            chain_id_1=padded[15].strip() or "A",
            resseq_1=int(padded[17:21].strip()),
            icode_1=padded[21].strip(),
            chain_id_2=padded[29].strip() or "A",
            resseq_2=int(padded[31:35].strip()),
            icode_2=padded[35].strip(),
            source="ssbond",
        )
    except ValueError:
        tokens = line.split()
        if len(tokens) < 8 or tokens[0].upper() != "SSBOND":
            raise ValueError("invalid SSBOND record")
        try:
            resseq_1, icode_1 = _parse_resseq_token(tokens[4])
            resseq_2, icode_2 = _parse_resseq_token(tokens[7])
        except ValueError as exc:
            raise ValueError(f"invalid SSBOND residue number ({exc})") from exc
        return DisulfideBond(
            chain_id_1=tokens[3],
            resseq_1=resseq_1,
            icode_1=icode_1,
            chain_id_2=tokens[6],
            resseq_2=resseq_2,
            icode_2=icode_2,
            source="ssbond",
        )


def _parse_resseq_token(token: str) -> Tuple[int, str]:
    digits = ""
    icode = ""
    for char in token.strip():
        if char in "+-" and not digits:
            digits += char
        elif char.isdigit():
            digits += char
        else:
            icode += char
    if not digits:
        raise ValueError(token)
    return int(digits), icode[:1]


def _parse_conect_line(line: str) -> List[Tuple[int, int]]:
    serial_fields = [
        line[6:11],
        line[11:16],
        line[16:21],
        line[21:26],
        line[26:31],
    ]
    serials = [int(field.strip()) for field in serial_fields if field.strip()]
    if not serials:
        raise ValueError("invalid CONECT record")
    source = serials[0]
    return [(source, target) for target in serials[1:]]


def _metadata_hydrogen_policy(
    *,
    requested: str,
    dropped_count: int,
    preserved_count: int,
) -> str:
    if requested == "generate":
        return "generated"
    if requested == "preserve":
        return "preserved" if preserved_count else "none-present"
    if dropped_count:
        return "dropped"
    return "none-present"


def _merge_disulfide_bonds(
    ssbond_bonds: Iterable[DisulfideBond],
    conect_bonds: Iterable[DisulfideBond],
) -> List[DisulfideBond]:
    by_key: Dict[Tuple[ResidueReference, ResidueReference], DisulfideBond] = {}
    for bond in ssbond_bonds:
        by_key[bond.canonical_key] = bond
    for bond in conect_bonds:
        by_key.setdefault(bond.canonical_key, bond)
    return [by_key[key] for key in sorted(by_key)]


def _disulfide_bonds_from_conect(
    atoms: Iterable[Atom],
    conect_pairs: Iterable[Tuple[int, int]],
) -> List[DisulfideBond]:
    atoms_by_serial = {
        int(atom.serial): atom
        for atom in atoms
        if atom.serial is not None
    }
    bonds: Dict[Tuple[ResidueReference, ResidueReference], DisulfideBond] = {}
    for serial_a, serial_b in conect_pairs:
        atom_a = atoms_by_serial.get(serial_a)
        atom_b = atoms_by_serial.get(serial_b)
        if atom_a is None or atom_b is None:
            continue
        if not (_is_cys_sg(atom_a) and _is_cys_sg(atom_b)):
            continue
        bond = DisulfideBond(
            chain_id_1=atom_a.chain_id,
            resseq_1=atom_a.resseq,
            icode_1=atom_a.icode,
            chain_id_2=atom_b.chain_id,
            resseq_2=atom_b.resseq,
            icode_2=atom_b.icode,
            source="conect",
        )
        bonds.setdefault(bond.canonical_key, bond)
    return [bonds[key] for key in sorted(bonds)]


def _is_cys_sg(atom: Atom) -> bool:
    return atom.resname.strip().upper() == "CYS" and atom.name.strip().upper() == "SG"


def _format_ssbond_line(
    bond: DisulfideBond,
    index: int,
    *,
    allow_chain_truncation: bool,
) -> str:
    chain_id_1 = _pdb_chain_id(
        bond.chain_id_1,
        allow_chain_truncation=allow_chain_truncation,
    )
    chain_id_2 = _pdb_chain_id(
        bond.chain_id_2,
        allow_chain_truncation=allow_chain_truncation,
    )
    icode_1 = (bond.icode_1 or " ")[:1]
    icode_2 = (bond.icode_2 or " ")[:1]
    return (
        f"SSBOND {index:3d} CYS {chain_id_1}{bond.resseq_1:4d}{icode_1}"
        f"   CYS {chain_id_2}{bond.resseq_2:4d}{icode_2}"
        "                          2.03"
    ).rstrip()


def _format_conect_line(serial_a: int, serial_b: int) -> str:
    return f"CONECT{serial_a:5d}{serial_b:5d}"


def _pdb_chain_id(chain_id: str, *, allow_chain_truncation: bool) -> str:
    normalized = (chain_id or "A").strip() or "A"
    if len(normalized) <= 1:
        return normalized
    if allow_chain_truncation:
        return normalized[:1]
    raise ValueError(
        f"PDB format supports one-character chain IDs; found {normalized!r}. "
        "Use mmCIF output or pass allow_chain_truncation=True."
    )


def _disulfide_bonds_with_sg_atoms(structure: HeavyAtomStructure) -> List[DisulfideBond]:
    writable = []
    for bond in structure.disulfide_bonds:
        atom_a = _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_1)
        atom_b = _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_2)
        if atom_a is not None and atom_b is not None:
            writable.append(bond)
    return writable


def _disulfide_conect_serial_pairs(
    disulfide_bonds: Iterable[DisulfideBond],
    structure: HeavyAtomStructure,
    serial_by_atom_id: Dict[int, int],
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for bond in disulfide_bonds:
        atom_a = _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_1)
        atom_b = _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_2)
        if atom_a is None or atom_b is None:
            continue
        serial_a = serial_by_atom_id[id(atom_a)]
        serial_b = serial_by_atom_id[id(atom_b)]
        pairs.append((min(serial_a, serial_b), max(serial_a, serial_b)))
    return sorted(set(pairs))


def _find_disulfide_sg_atom(
    atoms: Iterable[Atom],
    residue_ref: ResidueReference,
) -> Optional[Atom]:
    chain_id, resseq, icode = residue_ref
    for atom in atoms:
        if not _is_cys_sg(atom):
            continue
        if (atom.chain_id or "") != (chain_id or ""):
            continue
        if int(atom.resseq) != int(resseq):
            continue
        if (atom.icode or "") != (icode or ""):
            continue
        return atom
    return None

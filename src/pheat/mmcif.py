"""mmCIF parsing and writing for PHEAT atom structures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from pheat.bonds import structure_with_bonds
from pheat.hydrogens import generate_hydrogens, is_hydrogen_element, normalize_hydrogen_policy
from pheat.models import Atom, DisulfideBond, HeavyAtomStructure, ResidueReference
from pheat.residues import guess_element

MMCIF_CHAIN_ID_SOURCES = ("auth", "label")


def load_mmcif(
    path: Union[str, Path],
    *,
    model: Optional[int] = 1,
    name: str = "",
    chain_id_source: str = "auth",
    hydrogens: str = "drop",
    store_bonds: str = "none",
) -> HeavyAtomStructure:
    with open(path, "r", encoding="utf-8") as handle:
        return structure_from_mmcif_string(
            handle.read(),
            model=model,
            name=name or Path(path).name,
            chain_id_source=chain_id_source,
            hydrogens=hydrogens,
            store_bonds=store_bonds,
        )


def write_mmcif(structure: HeavyAtomStructure, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(structure_to_mmcif_string(structure))


def structure_from_mmcif_string(
    text: str,
    *,
    model: Optional[int] = 1,
    name: str = "",
    chain_id_source: str = "auth",
    hydrogens: str = "drop",
    store_bonds: str = "none",
) -> HeavyAtomStructure:
    """Parse coordinate-bearing mmCIF text into a PHEAT atom structure."""

    source = _normalize_chain_id_source(chain_id_source)
    hydrogen_policy = normalize_hydrogen_policy(hydrogens)
    data = _mmcif_dict_from_text(text)
    atom_count = _category_length(data, "_atom_site.id")
    warnings: list[str] = []
    atoms: list[Atom] = []
    dropped_hydrogen_count = 0
    preserved_hydrogen_count = 0
    saw_model = "_atom_site.pdbx_PDB_model_num" in data

    for index in range(atom_count):
        atom_model = _optional_int(_value(data, "_atom_site.pdbx_PDB_model_num", index))
        if atom_model is None and saw_model:
            atom_model = 1
        if model is not None and saw_model and atom_model != model:
            continue

        record_name = (_value(data, "_atom_site.group_PDB", index) or "ATOM").upper()
        atom_name = _atom_site_value(
            data,
            index,
            "auth" if source == "auth" else "label",
            "atom_id",
        )
        resname = _atom_site_value(
            data,
            index,
            "auth" if source == "auth" else "label",
            "comp_id",
        ).upper()
        chain_id = _atom_site_value(
            data,
            index,
            "auth" if source == "auth" else "label",
            "asym_id",
        ) or "A"
        resseq_text = _atom_site_value(
            data,
            index,
            "auth" if source == "auth" else "label",
            "seq_id",
        )
        resseq = _optional_int(resseq_text)
        if resseq is None:
            resseq = index + 1
            warnings.append(
                f"atom_site row {index + 1}: missing residue number; used atom row index"
            )

        element = (_value(data, "_atom_site.type_symbol", index) or guess_element(atom_name)).upper()
        if is_hydrogen_element(element) and hydrogen_policy != "preserve":
            dropped_hydrogen_count += 1
            continue
        if is_hydrogen_element(element):
            preserved_hydrogen_count += 1

        try:
            atom = Atom(
                name=atom_name,
                element=element,
                x=float(_value(data, "_atom_site.Cartn_x", index)),
                y=float(_value(data, "_atom_site.Cartn_y", index)),
                z=float(_value(data, "_atom_site.Cartn_z", index)),
                resname=resname,
                chain_id=chain_id,
                resseq=resseq,
                icode=_clean_optional(_value(data, "_atom_site.pdbx_PDB_ins_code", index)),
                record_name="HETATM" if record_name.startswith("HET") else "ATOM",
                serial=_optional_int(_value(data, "_atom_site.id", index)),
                altloc=_clean_optional(_value(data, "_atom_site.label_alt_id", index)),
                occupancy=_optional_float(_value(data, "_atom_site.occupancy", index)),
                bfactor=_optional_float(_value(data, "_atom_site.B_iso_or_equiv", index)),
                model=atom_model,
                metadata={
                    "mmcif_chain_id_source": source,
                    "label_asym_id": _clean_optional(_value(data, "_atom_site.label_asym_id", index)),
                    "auth_asym_id": _clean_optional(_value(data, "_atom_site.auth_asym_id", index)),
                    "label_seq_id": _clean_optional(_value(data, "_atom_site.label_seq_id", index)),
                    "auth_seq_id": _clean_optional(_value(data, "_atom_site.auth_seq_id", index)),
                },
            )
        except ValueError as exc:
            warnings.append(f"atom_site row {index + 1}: invalid coordinate field ({exc})")
            continue
        atoms.append(atom)

    if dropped_hydrogen_count:
        warnings.append(f"dropped {dropped_hydrogen_count} hydrogen atoms from all-atom mmCIF input")
    metadata = {
        "source": "mmcif",
        "warnings": warnings,
        "dropped_hydrogen_count": dropped_hydrogen_count,
        "preserved_hydrogen_count": preserved_hydrogen_count,
        "hydrogen_policy": _metadata_hydrogen_policy(
            requested=hydrogen_policy,
            dropped_count=dropped_hydrogen_count,
            preserved_count=preserved_hydrogen_count,
        ),
        "chain_id_source": source,
    }
    if saw_model:
        metadata["selected_model"] = model
    structure = HeavyAtomStructure(
        atoms=atoms,
        name=name or _entry_id(data),
        metadata=metadata,
        disulfide_bonds=_disulfide_bonds_from_struct_conn(data, chain_id_source=source),
    )
    if hydrogen_policy == "generate":
        return structure_with_bonds(generate_hydrogens(structure), mode=store_bonds)
    return structure_with_bonds(
        structure,
        mode=store_bonds,
        declared_pairs=_declared_pairs_from_struct_conn(data, structure.atoms, chain_id_source=source),
    )


def structure_to_mmcif_string(structure: HeavyAtomStructure) -> str:
    """Write a compact coordinate mmCIF for PHEAT analysis/visualization exchange."""

    entry_id = _entry_token(structure.name or "PHEAT")
    lines = [
        f"data_{entry_id}",
        "#",
        f"_entry.id {_quote(entry_id)}",
        "#",
    ]
    disulfides = _disulfide_bonds_with_sg_atoms(structure)
    if disulfides:
        lines.extend(_struct_conn_lines(disulfides))
    chain_ids = sorted({atom.chain_id or "A" for atom in structure.atoms})
    chain_to_entity: dict[str, str] = {chain: str(i + 1) for i, chain in enumerate(chain_ids)}
    lines.extend(
        [
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    for index, atom in enumerate(structure.atoms, start=1):
        record = "HETATM" if atom.record_name.upper().startswith("HET") else "ATOM"
        serial = atom.serial if atom.serial is not None else index
        model = atom.model if atom.model is not None else 1
        occupancy = 1.0 if atom.occupancy is None else float(atom.occupancy)
        bfactor = 0.0 if atom.bfactor is None else float(atom.bfactor)
        element = (atom.element or guess_element(atom.name)).upper()
        entity_id = chain_to_entity[atom.chain_id or "A"]
        values = [
            record,
            serial,
            element,
            atom.name.strip(),
            atom.altloc or ".",
            atom.resname.strip().upper(),
            atom.chain_id or "A",
            entity_id,
            atom.resseq,
            atom.icode or "?",
            f"{atom.x:.3f}",
            f"{atom.y:.3f}",
            f"{atom.z:.3f}",
            f"{occupancy:.2f}",
            f"{bfactor:.2f}",
            atom.resseq,
            atom.resname.strip().upper(),
            atom.chain_id or "A",
            atom.name.strip(),
            model,
        ]
        lines.append(" ".join(_quote(value) for value in values))
    lines.append("#")
    return "\n".join(lines) + "\n"


def _mmcif_dict_from_text(text: str) -> Mapping[str, Any]:
    """Return a small PDBx/mmCIF dictionary for coordinate categories.

    This parser intentionally covers the core CIF token rules PHEAT needs for
    structure exchange: whitespace-delimited tokens, quoted values, semicolon
    multiline values, scalar data items, and looped categories.  Keeping this
    local avoids routing mmCIF input through an external structure-parser object model.
    """

    tokens = _mmcif_tokens(text)
    data: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if lower.startswith("data_") or lower.startswith("save_") or lower in {"global_", "stop_"}:
            index += 1
            continue
        if lower == "loop_":
            index = _parse_loop(tokens, index + 1, data)
            continue
        if token.startswith("_"):
            if index + 1 < len(tokens) and not _is_mmcif_control_token(tokens[index + 1]):
                data[token] = tokens[index + 1]
                index += 2
            else:
                data[token] = ""
                index += 1
            continue
        index += 1
    return data


def _parse_loop(tokens: Sequence[str], index: int, data: dict[str, Any]) -> int:
    tags = []
    while index < len(tokens) and tokens[index].startswith("_"):
        tags.append(tokens[index])
        index += 1
    if not tags:
        return index

    values = []
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("_") or _is_mmcif_control_token(token):
            break
        values.append(token)
        index += 1

    width = len(tags)
    for tag in tags:
        data[tag] = []
    remainder = len(values) % width
    if remainder != 0:
        raise ValueError(
            f"mmCIF loop for {tags[0]!r} has {len(values)} values but column count is {width};"
            f" last row is incomplete ({remainder} of {width} values present)"
        )
    for offset in range(0, len(values), width):
        row = values[offset : offset + width]
        for tag, value in zip(tags, row):
            data[tag].append(value)
    return index


def _is_mmcif_control_token(token: str) -> bool:
    lower = token.lower()
    return lower == "loop_" or lower.startswith("data_") or lower.startswith("save_") or lower in {"global_", "stop_"}


def _mmcif_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if line.startswith(";"):
            multiline = [line[1:]]
            line_index += 1
            while line_index < len(lines):
                if lines[line_index].startswith(";"):
                    break
                multiline.append(lines[line_index])
                line_index += 1
            tokens.append("\n".join(multiline))
            line_index += 1
            continue
        tokens.extend(_mmcif_line_tokens(line))
        line_index += 1
    return tokens


def _mmcif_line_tokens(line: str) -> list[str]:
    tokens = []
    index = 0
    while index < len(line):
        char = line[index]
        if char.isspace():
            index += 1
            continue
        if char == "#":
            break
        if char in {"'", '"'}:
            token, index = _read_quoted_mmcif_token(line, index)
            tokens.append(token)
            continue
        start = index
        while index < len(line) and not line[index].isspace():
            if line[index] == "#":
                break
            index += 1
        if start != index:
            tokens.append(line[start:index])
        if index < len(line) and line[index] == "#":
            break
    return tokens


def _read_quoted_mmcif_token(line: str, index: int) -> tuple[str, int]:
    quote = line[index]
    index += 1
    output: list[str] = []
    while index < len(line):
        char = line[index]
        if char == quote:
            next_index = index + 1
            if next_index == len(line) or line[next_index].isspace() or line[next_index] == "#":
                return "".join(output), next_index
        output.append(char)
        index += 1
    return "".join(output), index


def _normalize_chain_id_source(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in MMCIF_CHAIN_ID_SOURCES:
        raise ValueError(f"chain_id_source must be one of {', '.join(MMCIF_CHAIN_ID_SOURCES)}")
    return normalized


def _category_length(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key, [])
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


def _value(data: Mapping[str, Any], key: str, index: int) -> str:
    value = data.get(key, "")
    if isinstance(value, list):
        if index >= len(value):
            return ""
        return str(value[index])
    return str(value)


def _atom_site_value(data: Mapping[str, Any], index: int, source: str, field: str) -> str:
    primary = _clean_optional(_value(data, f"_atom_site.{source}_{field}", index))
    if primary:
        return primary
    fallback_source = "label" if source == "auth" else "auth"
    fallback = _clean_optional(_value(data, f"_atom_site.{fallback_source}_{field}", index))
    if fallback:
        return fallback
    if field == "atom_id":
        return _clean_optional(_value(data, "_atom_site.label_atom_id", index))
    if field == "comp_id":
        return _clean_optional(_value(data, "_atom_site.label_comp_id", index))
    if field == "asym_id":
        return _clean_optional(_value(data, "_atom_site.label_asym_id", index))
    return ""


def _entry_id(data: Mapping[str, Any]) -> str:
    return _clean_optional(str(data.get("_entry.id") or "")) or "mmcif"


def _clean_optional(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in {"", ".", "?"} else text


def _optional_int(value: Any) -> Optional[int]:
    text = _clean_optional(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _optional_float(value: Any) -> Optional[float]:
    text = _clean_optional(value)
    if not text:
        return None
    return float(text)


def _disulfide_bonds_from_struct_conn(
    data: Mapping[str, Any],
    *,
    chain_id_source: str,
) -> list[DisulfideBond]:
    row_count = _category_length(data, "_struct_conn.id")
    bonds: dict[tuple[ResidueReference, ResidueReference], DisulfideBond] = {}
    source_prefix = "auth" if chain_id_source == "auth" else "label"
    for index in range(row_count):
        conn_type = _value(data, "_struct_conn.conn_type_id", index).lower()
        atom_1 = _struct_conn_value(data, index, 1, source_prefix, "atom_id").upper()
        atom_2 = _struct_conn_value(data, index, 2, source_prefix, "atom_id").upper()
        comp_1 = _struct_conn_value(data, index, 1, source_prefix, "comp_id").upper()
        comp_2 = _struct_conn_value(data, index, 2, source_prefix, "comp_id").upper()
        if "disulf" not in conn_type and (atom_1, atom_2, comp_1, comp_2) != (
            "SG",
            "SG",
            "CYS",
            "CYS",
        ):
            continue
        chain_1 = _struct_conn_value(data, index, 1, source_prefix, "asym_id")
        chain_2 = _struct_conn_value(data, index, 2, source_prefix, "asym_id")
        resseq_1 = _optional_int(_struct_conn_value(data, index, 1, source_prefix, "seq_id"))
        resseq_2 = _optional_int(_struct_conn_value(data, index, 2, source_prefix, "seq_id"))
        if resseq_1 is None or resseq_2 is None:
            continue
        bond = DisulfideBond(
            chain_id_1=chain_1,
            resseq_1=resseq_1,
            icode_1=_clean_optional(_value(data, "_struct_conn.pdbx_ptnr1_PDB_ins_code", index)),
            chain_id_2=chain_2,
            resseq_2=resseq_2,
            icode_2=_clean_optional(_value(data, "_struct_conn.pdbx_ptnr2_PDB_ins_code", index)),
            source="struct_conn",
        )
        bonds.setdefault(bond.canonical_key, bond)
    return [bonds[key] for key in sorted(bonds)]


def _declared_pairs_from_struct_conn(
    data: Mapping[str, Any],
    atoms: Sequence[Atom],
    *,
    chain_id_source: str,
) -> list[tuple[int, int]]:
    row_count = _category_length(data, "_struct_conn.id")
    source_prefix = "auth" if chain_id_source == "auth" else "label"
    pairs = []
    for index in range(row_count):
        atom_1 = _atom_from_struct_conn(data, atoms, index, 1, source_prefix)
        atom_2 = _atom_from_struct_conn(data, atoms, index, 2, source_prefix)
        if atom_1 is None or atom_2 is None:
            continue
        pairs.append((atoms.index(atom_1), atoms.index(atom_2)))
    return pairs


def _atom_from_struct_conn(
    data: Mapping[str, Any],
    atoms: Sequence[Atom],
    index: int,
    partner: int,
    source_prefix: str,
) -> Optional[Atom]:
    chain_id = _struct_conn_value(data, index, partner, source_prefix, "asym_id")
    atom_name = _struct_conn_value(data, index, partner, source_prefix, "atom_id").upper()
    resseq = _optional_int(_struct_conn_value(data, index, partner, source_prefix, "seq_id"))
    comp_id = _struct_conn_value(data, index, partner, source_prefix, "comp_id").upper()
    icode = _clean_optional(_value(data, f"_struct_conn.pdbx_ptnr{partner}_PDB_ins_code", index))
    if resseq is None:
        return None
    for atom in atoms:
        if (atom.chain_id or "") != (chain_id or ""):
            continue
        if int(atom.resseq) != int(resseq):
            continue
        if atom.resname.strip().upper() != comp_id:
            continue
        if atom.name.strip().upper() != atom_name:
            continue
        if (atom.icode or "") != (icode or ""):
            continue
        return atom
    return None


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


def _struct_conn_value(
    data: Mapping[str, Any],
    index: int,
    partner: int,
    source_prefix: str,
    field: str,
) -> str:
    value = _clean_optional(_value(data, f"_struct_conn.ptnr{partner}_{source_prefix}_{field}", index))
    if value:
        return value
    fallback_prefix = "label" if source_prefix == "auth" else "auth"
    return _clean_optional(_value(data, f"_struct_conn.ptnr{partner}_{fallback_prefix}_{field}", index))


def _struct_conn_lines(disulfides: Sequence[DisulfideBond]) -> list[str]:
    lines = [
        "loop_",
        "_struct_conn.id",
        "_struct_conn.conn_type_id",
        "_struct_conn.ptnr1_label_asym_id",
        "_struct_conn.ptnr1_label_comp_id",
        "_struct_conn.ptnr1_label_seq_id",
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.pdbx_ptnr1_PDB_ins_code",
        "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_auth_comp_id",
        "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.ptnr1_auth_atom_id",
        "_struct_conn.ptnr2_label_asym_id",
        "_struct_conn.ptnr2_label_comp_id",
        "_struct_conn.ptnr2_label_seq_id",
        "_struct_conn.ptnr2_label_atom_id",
        "_struct_conn.pdbx_ptnr2_PDB_ins_code",
        "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_auth_comp_id",
        "_struct_conn.ptnr2_auth_seq_id",
        "_struct_conn.ptnr2_auth_atom_id",
    ]
    for index, bond in enumerate(disulfides, start=1):
        values = [
            f"disulf{index}",
            "disulf",
            bond.chain_id_1 or "A",
            "CYS",
            bond.resseq_1,
            "SG",
            bond.icode_1 or "?",
            bond.chain_id_1 or "A",
            "CYS",
            bond.resseq_1,
            "SG",
            bond.chain_id_2 or "A",
            "CYS",
            bond.resseq_2,
            "SG",
            bond.icode_2 or "?",
            bond.chain_id_2 or "A",
            "CYS",
            bond.resseq_2,
            "SG",
        ]
        lines.append(" ".join(_quote(value) for value in values))
    lines.append("#")
    return lines


def _disulfide_bonds_with_sg_atoms(structure: HeavyAtomStructure) -> list[DisulfideBond]:
    writable = []
    for bond in structure.disulfide_bonds:
        if _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_1) is not None:
            if _find_disulfide_sg_atom(structure.atoms, bond.residue_ref_2) is not None:
                writable.append(bond)
    return writable


def _find_disulfide_sg_atom(
    atoms: Iterable[Atom],
    residue_ref: ResidueReference,
) -> Optional[Atom]:
    chain_id, resseq, icode = residue_ref
    for atom in atoms:
        if atom.resname.strip().upper() != "CYS" or atom.name.strip().upper() != "SG":
            continue
        if (atom.chain_id or "") != (chain_id or ""):
            continue
        if int(atom.resseq) != int(resseq):
            continue
        if (atom.icode or "") != (icode or ""):
            continue
        return atom
    return None


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+/-]+$")


def _quote(value: object) -> str:
    text = str(value)
    if text == "":
        return "?"
    if text in {".", "?"}:
        return text
    if _SAFE_TOKEN_RE.match(text) and not text.lower().startswith(("data_", "loop_", "save_")):
        return text
    if '"' not in text:
        return f'"{text}"'
    if "'" not in text:
        return f"'{text}'"
    return '"' + text.replace('"', '\\"') + '"'


def _entry_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:64].strip("._")
    return token or "PHEAT"

"""BinaryCIF input support for PHEAT atom structures."""

from __future__ import annotations

import copy
import gzip
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from pheat.bonds import structure_with_bonds
from pheat.hydrogens import generate_hydrogens, is_hydrogen_element, normalize_hydrogen_policy
from pheat.models import Atom, HeavyAtomStructure


def load_bcif(
    path: Union[str, Path],
    *,
    model: Optional[int] = 1,
    name: str = "",
    hydrogens: str = "drop",
    store_bonds: str = "none",
) -> HeavyAtomStructure:
    """Load a BinaryCIF file into PHEAT atom-structure JSON form.

    PHEAT decodes the coordinate-bearing ``_atom_site`` category directly
    instead of routing through an external structure-parser object model.  That
    keeps BCIF archive workflows usable for RCSB model-server files whose
    optional masked string columns need direct BinaryCIF handling.
    """

    source = Path(path)
    structure = _load_bcif_native(
        source,
        model=model,
        name=name or source.name,
        hydrogens=hydrogens,
        store_bonds=store_bonds,
    )
    return structure


def decode_bcif_column(column: Mapping[str, Any], *, row_count: Any = None) -> list[Any]:
    """Decode one BinaryCIF column, applying nonzero mask values as ``None``."""

    row_count_int = int(row_count or 0)
    mask_values = None
    if isinstance(column.get("mask"), Mapping):
        try:
            mask_values = _array_to_list(_decode_column_payload(column["mask"]))
        except Exception:
            mask_values = None

    try:
        values = _array_to_list(_decode_column_payload(column["data"]))
    except Exception:
        if mask_values is not None:
            return [None for _ in range(len(mask_values))]
        return [None for _ in range(row_count_int)]

    if mask_values is None:
        return values

    length = row_count_int or max(len(values), len(mask_values))
    output = []
    for index in range(length):
        masked = index < len(mask_values) and int(mask_values[index] or 0) != 0
        output.append(None if masked else _at(values, index))
    return output


def _load_bcif_native(
    path: Path,
    *,
    model: Optional[int],
    name: str,
    hydrogens: str,
    store_bonds: str,
) -> HeavyAtomStructure:
    """Decode the coordinate-bearing atom_site table from a BinaryCIF file."""

    payload = _read_bcif_payload(path)
    block = _first_data_block(payload)
    categories = {str(category["name"]): category for category in block.get("categories", [])}
    atom_site = categories.get("_atom_site")
    if atom_site is None:
        raise ValueError("BinaryCIF file does not contain an _atom_site category")
    columns = {str(column["name"]): column for column in atom_site.get("columns", [])}
    row_count = _category_row_count(atom_site, columns)

    group_pdb = _column_values(columns, "group_PDB", row_count=row_count)
    atom_names = _first_column_values(columns, ("auth_atom_id", "label_atom_id"), row_count=row_count)
    resnames = _first_column_values(columns, ("auth_comp_id", "label_comp_id"), row_count=row_count)
    chain_ids = _first_column_values(columns, ("label_asym_id", "auth_asym_id"), row_count=row_count)
    auth_chain_ids = _optional_column_values(columns, "auth_asym_id", row_count=row_count)
    label_chain_ids = _optional_column_values(columns, "label_asym_id", row_count=row_count)
    resseq_values = _first_column_values(columns, ("auth_seq_id", "label_seq_id"), row_count=row_count)
    label_seq_ids = _optional_column_values(columns, "label_seq_id", row_count=row_count)
    auth_seq_ids = _optional_column_values(columns, "auth_seq_id", row_count=row_count)
    insertion_codes = _optional_column_values(columns, "pdbx_PDB_ins_code", row_count=row_count)
    altlocs = _optional_column_values(columns, "label_alt_id", row_count=row_count)
    elements = _optional_column_values(columns, "type_symbol", row_count=row_count)
    serials = _optional_column_values(columns, "id", row_count=row_count)
    occupancies = _optional_column_values(columns, "occupancy", row_count=row_count)
    bfactors = _optional_column_values(columns, "B_iso_or_equiv", row_count=row_count)
    model_numbers = _optional_column_values(columns, "pdbx_PDB_model_num", row_count=row_count)
    xs = _column_values(columns, "Cartn_x", row_count=row_count)
    ys = _column_values(columns, "Cartn_y", row_count=row_count)
    zs = _column_values(columns, "Cartn_z", row_count=row_count)

    hydrogen_policy = normalize_hydrogen_policy(hydrogens)
    atoms: list[Atom] = []
    warnings: list[str] = []
    dropped_hydrogen_count = 0
    preserved_hydrogen_count = 0
    saw_model = "pdbx_PDB_model_num" in columns

    for index in range(row_count):
        atom_model = _optional_int(_at(model_numbers, index)) if model_numbers is not None else 1
        if atom_model is None:
            atom_model = 1
        if model is not None and atom_model != model:
            continue

        element = str(_at(elements, index) or "").strip().upper()
        if is_hydrogen_element(element) and hydrogen_policy != "preserve":
            dropped_hydrogen_count += 1
            continue
        if is_hydrogen_element(element):
            preserved_hydrogen_count += 1

        resseq = _optional_int(_at(resseq_values, index))
        if resseq is None:
            resseq = index + 1
            warnings.append(f"atom_site row {index + 1}: missing residue number; used atom row index")
        record_name = str(_at(group_pdb, index) or "ATOM").upper()
        atoms.append(
            Atom(
                name=str(_at(atom_names, index) or "").strip(),
                element=element,
                x=float(_at(xs, index)),
                y=float(_at(ys, index)),
                z=float(_at(zs, index)),
                resname=str(_at(resnames, index) or "").strip().upper(),
                chain_id=str(_at(chain_ids, index) or "A").strip() or "A",
                resseq=resseq,
                icode=_clean_optional(_at(insertion_codes, index)),
                record_name="HETATM" if record_name.startswith("HET") else "ATOM",
                serial=_optional_int(_at(serials, index)),
                altloc=_clean_optional(_at(altlocs, index)),
                occupancy=_optional_float(_at(occupancies, index)),
                bfactor=_optional_float(_at(bfactors, index)),
                model=atom_model,
                metadata={
                    "bcif_parser": "pheat-native-atom-site",
                    "bcif_chain_id_source": "label_asym_id",
                    "label_asym_id": _clean_optional(_at(label_chain_ids, index)),
                    "auth_asym_id": _clean_optional(_at(auth_chain_ids, index)),
                    "label_seq_id": _clean_optional(_at(label_seq_ids, index)),
                    "auth_seq_id": _clean_optional(_at(auth_seq_ids, index)),
                },
            )
        )

    atoms = _deduplicate_atoms_by_occupancy(atoms, warnings)
    if dropped_hydrogen_count:
        warnings.append(f"dropped {dropped_hydrogen_count} hydrogen atoms from all-atom BCIF input")
    metadata = {
        "source": "bcif",
        "warnings": warnings,
        "dropped_hydrogen_count": dropped_hydrogen_count,
        "preserved_hydrogen_count": preserved_hydrogen_count,
        "hydrogen_policy": _metadata_hydrogen_policy(
            requested=hydrogen_policy,
            dropped_count=dropped_hydrogen_count,
            preserved_count=preserved_hydrogen_count,
        ),
        "chain_id_source": "label",
        "parser": "pheat-native-atom-site",
    }
    if saw_model:
        metadata["selected_model"] = model
    structure = HeavyAtomStructure(atoms=atoms, name=name or _entry_id_from_payload(payload, path), metadata=metadata)
    if hydrogen_policy == "generate":
        structure = generate_hydrogens(structure)
    return structure_with_bonds(structure, mode=store_bonds)


def _deduplicate_atoms_by_occupancy(atoms: Sequence[Atom], warnings: list[str]) -> list[Atom]:
    selected: dict[tuple[str, int, str, str, str, str], Atom] = {}
    duplicate_count = 0
    for atom in atoms:
        key = (
            atom.chain_id or "",
            int(atom.resseq),
            atom.icode or "",
            atom.resname.strip().upper(),
            atom.record_name.strip().upper(),
            atom.name.strip().upper(),
        )
        existing = selected.get(key)
        if existing is None:
            selected[key] = atom
            continue
        duplicate_count += 1
        if _atom_altloc_rank(atom) > _atom_altloc_rank(existing):
            selected[key] = atom
    if duplicate_count:
        warnings.append(
            f"selected highest-occupancy atom for {duplicate_count} duplicate BinaryCIF atom-site rows"
        )
    return list(selected.values())


def _atom_altloc_rank(atom: Atom) -> tuple[float, int]:
    occupancy = -1.0 if atom.occupancy is None else float(atom.occupancy)
    altloc = (atom.altloc or "").strip().upper()
    altloc_rank = 2 if not altloc else 1 if altloc == "A" else 0
    return (occupancy, altloc_rank)


def _read_bcif_payload(path: Path) -> Mapping[str, Any]:
    msgpack = _msgpack_module()
    with (gzip.open(path, "rb") if str(path).lower().endswith(".gz") else path.open("rb")) as handle:
        payload = msgpack.unpack(handle, raw=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"BinaryCIF payload is not a mapping: {path}")
    return payload


def _category_row_count(category: Mapping[str, Any], columns: Mapping[str, Mapping[str, Any]]) -> int:
    row_count = int(category.get("rowCount") or 0)
    if row_count:
        return row_count
    for column in columns.values():
        values = decode_bcif_column(column, row_count=0)
        if values:
            return len(values)
    return 0


def _msgpack_module() -> Any:
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError(
            "BinaryCIF support requires optional dependencies. Install PHEAT with "
            "`.[scientific]` or `.[all]` so msgpack and numpy are available."
        ) from exc
    _numpy_module()
    return msgpack


def _numpy_module() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError(
            "BinaryCIF support requires optional dependencies. Install PHEAT with "
            "`.[scientific]` or `.[all]` so msgpack and numpy are available."
        ) from exc
    return numpy


def _column_values(columns: Mapping[str, Mapping[str, Any]], name: str, *, row_count: int) -> list[Any]:
    if name not in columns:
        raise KeyError(f"_atom_site.{name}")
    return decode_bcif_column(columns[name], row_count=row_count)


def _first_column_values(
    columns: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    row_count: int,
) -> list[Any]:
    for name in names:
        if name in columns:
            values = decode_bcif_column(columns[name], row_count=row_count)
            if values:
                return values
    raise KeyError(f"none of {', '.join('_atom_site.' + name for name in names)} are present")


def _optional_column_values(
    columns: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    row_count: int,
) -> Optional[list[Any]]:
    if name not in columns:
        return None
    return decode_bcif_column(columns[name], row_count=row_count)


def _decode_column_payload(payload: Mapping[str, Any]) -> Any:
    current = copy.deepcopy(payload.get("data"))
    encodings = list(payload.get("encoding") or [])
    while encodings:
        encoding = encodings.pop()
        kind = str(encoding.get("kind") or "")
        if kind == "ByteArray":
            current = _decode_byte_array(current, int(encoding["type"]))
        elif kind == "FixedPoint":
            current = _decode_fixed_point(current, encoding)
        elif kind == "IntervalQuantization":
            current = _decode_interval_quantization(current, encoding)
        elif kind == "RunLength":
            current = _decode_run_length(current, encoding)
        elif kind == "Delta":
            current = _decode_delta(current, encoding)
        elif kind == "IntegerPacking":
            current = _decode_integer_packing(current, encoding)
        elif kind == "StringArray":
            current = _decode_string_array(current, encoding)
        else:
            raise ValueError(f"unsupported BinaryCIF encoding kind: {kind}")
    return current


def _decode_byte_array(data: Any, type_code: int) -> Any:
    numpy = _numpy_module()
    dtype_by_code = {
        1: numpy.dtype("<i1"),
        2: numpy.dtype("<i2"),
        3: numpy.dtype("<i4"),
        4: numpy.dtype("<u1"),
        5: numpy.dtype("<u2"),
        6: numpy.dtype("<u4"),
        32: numpy.dtype("<f4"),
        33: numpy.dtype("<f8"),
    }
    if type_code not in dtype_by_code:
        raise ValueError(f"unsupported BinaryCIF ByteArray type: {type_code}")
    return numpy.frombuffer(data, dtype_by_code[type_code])


def _decode_fixed_point(data: Any, encoding: Mapping[str, Any]) -> Any:
    numpy = _numpy_module()
    dtype = _numpy_dtype(int(encoding["srcType"]))
    return numpy.divide(data, float(encoding["factor"]), dtype=dtype)


def _decode_interval_quantization(data: Any, encoding: Mapping[str, Any]) -> Any:
    numpy = _numpy_module()
    min_value = float(encoding["min"])
    max_value = float(encoding["max"])
    steps = int(encoding["num_steps"])
    delta = (max_value - min_value) / float(steps - 1)
    return numpy.add(min_value, numpy.multiply(data, delta, dtype=_numpy_dtype(int(encoding["srcType"]))))


def _decode_run_length(data: Any, encoding: Mapping[str, Any]) -> Any:
    numpy = _numpy_module()
    values = data[::2].astype(_numpy_dtype(int(encoding["srcType"])))
    counts = data[1::2]
    decoded = numpy.repeat(values, counts)
    expected = int(encoding["srcSize"])
    if len(decoded) != expected:
        raise ValueError(f"BinaryCIF RunLength decoded {len(decoded)} values, expected {expected}")
    return decoded


def _decode_delta(data: Any, encoding: Mapping[str, Any]) -> Any:
    decoded = data.astype(_numpy_dtype(int(encoding["srcType"])), copy=True)
    if len(decoded):
        decoded[0] += int(encoding["origin"])
        decoded.cumsum(out=decoded)
    return decoded


def _decode_integer_packing(data: Any, encoding: Mapping[str, Any]) -> Any:
    numpy = _numpy_module()
    src_size = int(encoding["srcSize"])
    byte_count = int(encoding["byteCount"])
    is_unsigned = bool(encoding["isUnsigned"])
    if is_unsigned:
        limit = {1: 0xFF, 2: 0xFFFF}.get(byte_count)
        dtype = numpy.dtype("<u4")
        lower_limit = None
    else:
        limit = {1: 0x7F, 2: 0x7FFF}.get(byte_count)
        lower_limit = {1: -0x80, 2: -0x8000}.get(byte_count)
        dtype = numpy.dtype("<i4")
    if limit is None:
        raise ValueError(f"unsupported BinaryCIF IntegerPacking byte count: {byte_count}")

    output = []
    accumulator = 0
    for raw_value in data:
        value = int(raw_value)
        accumulator += value
        if value == limit or (lower_limit is not None and value == lower_limit):
            continue
        output.append(accumulator)
        accumulator = 0
    if len(output) != src_size:
        raise ValueError(f"BinaryCIF IntegerPacking decoded {len(output)} values, expected {src_size}")
    return numpy.asarray(output, dtype=dtype)


def _decode_string_array(data: Any, encoding: Mapping[str, Any]) -> list[Optional[str]]:
    offsets = _array_to_list(
        _decode_column_payload(
            {
                "data": encoding["offsets"],
                "encoding": encoding.get("offsetEncoding") or [],
            }
        )
    )
    lookup_values = _array_to_list(
        _decode_column_payload(
            {
                "data": data,
                "encoding": encoding.get("dataEncoding") or [],
            }
        )
    )
    string_data = str(encoding.get("stringData") or "")
    unique_strings = [
        string_data[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(max(0, len(offsets) - 1))
    ]
    output: list[Optional[str]] = []
    for lookup in lookup_values:
        lookup_int = int(lookup)
        output.append(None if lookup_int < 0 else unique_strings[lookup_int])
    return output


def _numpy_dtype(type_code: int) -> Any:
    numpy = _numpy_module()
    dtype_by_code = {
        1: numpy.dtype("<i1"),
        2: numpy.dtype("<i2"),
        3: numpy.dtype("<i4"),
        4: numpy.dtype("<u1"),
        5: numpy.dtype("<u2"),
        6: numpy.dtype("<u4"),
        32: numpy.dtype("<f4"),
        33: numpy.dtype("<f8"),
    }
    if type_code not in dtype_by_code:
        raise ValueError(f"unsupported BinaryCIF numeric type: {type_code}")
    return dtype_by_code[type_code]


def _array_to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(values)


def _first_data_block(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    blocks = payload.get("dataBlocks") or []
    if not blocks:
        raise ValueError("BinaryCIF file does not contain any data blocks")
    block = blocks[0]
    if not isinstance(block, Mapping):
        raise ValueError("BinaryCIF data block is not a mapping")
    return block


def _entry_id_from_payload(payload: Mapping[str, Any], path: Path) -> str:
    try:
        block = _first_data_block(payload)
        categories = {str(category["name"]): category for category in block.get("categories", [])}
        entry = categories.get("_entry")
        if entry is not None:
            columns = {str(column["name"]): column for column in entry.get("columns", [])}
            values = _optional_column_values(columns, "id", row_count=int(entry.get("rowCount") or 1))
            if values and values[0]:
                return str(values[0])
    except Exception:
        pass
    return _entry_id_from_path(path)


def _entry_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".bcif.gz", ".bcif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _at(values: Optional[Sequence[Any]], index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def _clean_optional(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {".", "?"} else text


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


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

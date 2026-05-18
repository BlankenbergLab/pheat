"""Small CCD-like component annotation helpers for corpus manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from pheat.models import HeavyAtomStructure
from pheat.residues import CANONICAL_RESIDUES, MODIFIED_RESIDUE_PARENTS


_ION_IDS = {
    "AL",
    "BA",
    "BR",
    "CA",
    "CD",
    "CL",
    "CO",
    "CU",
    "FE",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NI",
    "SR",
    "ZN",
}


@dataclass(frozen=True)
class ComponentRecord:
    """Minimal CCD-like metadata used by manifest generation."""

    component_id: str
    component_type: Optional[str] = None
    name: Optional[str] = None
    formula: Optional[str] = None
    atom_count: Optional[int] = None
    classification: Optional[str] = None
    metadata_source: str = "user"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source: str) -> "ComponentRecord":
        component_id = str(
            payload.get("component_id")
            or payload.get("id")
            or payload.get("ccd_id")
            or payload.get("chem_comp_id")
            or ""
        ).strip().upper()
        if not component_id:
            raise ValueError("CCD component metadata row is missing component_id")
        atom_count = payload.get("atom_count")
        return cls(
            component_id=component_id,
            component_type=_optional_string(payload.get("component_type") or payload.get("type")),
            name=_optional_string(payload.get("name")),
            formula=_optional_string(payload.get("formula")),
            atom_count=_optional_int(atom_count),
            classification=_normalize_classification(payload.get("classification")),
            metadata_source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "name": self.name,
            "formula": self.formula,
            "atom_count": self.atom_count,
            "classification": self.classification,
            "metadata_source": self.metadata_source,
        }


def load_component_table(path: Optional[str | Path] = None) -> dict[str, ComponentRecord]:
    """Load optional user CCD-like metadata and merge it over demo fallback rows.

    Supported user formats are JSONL and CSV. The accepted fields are
    component_id, component_type, name, formula, atom_count, and classification.
    The built-in rows are intentionally small demo/test metadata, not a vendored
    copy of the wwPDB Chemical Component Dictionary.
    """

    table = _demo_component_table()
    if path is None:
        return table
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            record = ComponentRecord.from_mapping(payload, source=str(source))
            table[record.component_id] = record
        return table
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record = ComponentRecord.from_mapping(row, source=str(source))
                table[record.component_id] = record
        return table
    raise ValueError(f"Unsupported CCD metadata format for {source}; use CSV or JSONL")


def annotate_structure_components(
    structure: HeavyAtomStructure,
    *,
    component_table: Optional[Mapping[str, ComponentRecord]] = None,
) -> dict[str, Any]:
    """Return heterogen, ligand, modified-residue, water, ion, and unknown summaries."""

    table = dict(component_table or load_component_table())
    grouped: dict[tuple[str, str, int, str, str], list[Any]] = {}
    for atom in structure.atoms:
        key = (
            atom.resname.strip().upper(),
            atom.chain_id or "",
            int(atom.resseq),
            atom.icode or "",
            "HETATM" if atom.record_name.upper().startswith("HET") else "ATOM",
        )
        grouped.setdefault(key, []).append(atom)

    component_summaries: dict[str, dict[str, Any]] = {}
    unknown_ids: set[str] = set()
    for (component_id, _chain_id, _resseq, _icode, record_name), atoms in grouped.items():
        classification = classify_component(
            component_id,
            record_name=record_name,
            component_table=table,
        )
        if classification == "standard_residue":
            continue
        record = table.get(component_id)
        if classification == "unknown":
            unknown_ids.add(component_id)
        summary = component_summaries.setdefault(
            component_id,
            {
                "component_id": component_id,
                "component_type": record.component_type if record else None,
                "name": record.name if record else None,
                "formula": record.formula if record else None,
                "ccd_atom_count": record.atom_count if record else None,
                "classification": classification,
                "metadata_source": record.metadata_source if record else None,
                "residue_count": 0,
                "atom_count": 0,
            },
        )
        summary["residue_count"] += 1
        summary["atom_count"] += len(atoms)

    summaries = sorted(component_summaries.values(), key=lambda item: item["component_id"])
    heterogens = [
        item
        for item in summaries
        if item["classification"] in {"water", "ion", "ligand", "unknown"}
    ]
    modified = [item for item in summaries if item["classification"] == "modified_residue"]
    ligands = [item for item in summaries if item["classification"] == "ligand"]
    return {
        "heterogen_summary": heterogens,
        "modified_residue_summary": modified,
        "ligand_summary": ligands,
        "water_count": sum(int(item["residue_count"]) for item in summaries if item["classification"] == "water"),
        "ion_count": sum(int(item["residue_count"]) for item in summaries if item["classification"] == "ion"),
        "unknown_component_ids": sorted(unknown_ids),
    }


def classify_component(
    component_id: str,
    *,
    record_name: str = "",
    component_table: Optional[Mapping[str, ComponentRecord]] = None,
) -> str:
    """Classify a component ID for manifest summaries."""

    component = component_id.strip().upper()
    table = component_table or load_component_table()
    record = table.get(component)
    if record and record.classification:
        return record.classification
    if component in CANONICAL_RESIDUES:
        return "standard_residue"
    if component in MODIFIED_RESIDUE_PARENTS:
        return "modified_residue"
    if component in {"HOH", "WAT", "DOD"}:
        return "water"
    if component in _ION_IDS:
        return "ion"
    if record and (record.component_type or "").lower().find("non-polymer") >= 0:
        return "ligand"
    if str(record_name or "").upper().startswith("HET"):
        return "unknown"
    return "unknown"


def _demo_component_table() -> dict[str, ComponentRecord]:
    rows: list[Mapping[str, Any]] = [
        {
            "component_id": "HOH",
            "component_type": "non-polymer",
            "name": "water",
            "formula": "H2 O",
            "atom_count": 1,
            "classification": "water",
        },
        {
            "component_id": "WAT",
            "component_type": "non-polymer",
            "name": "water",
            "formula": "H2 O",
            "atom_count": 1,
            "classification": "water",
        },
        {
            "component_id": "NA",
            "component_type": "non-polymer",
            "name": "sodium ion",
            "formula": "Na",
            "atom_count": 1,
            "classification": "ion",
        },
        {
            "component_id": "CL",
            "component_type": "non-polymer",
            "name": "chloride ion",
            "formula": "Cl",
            "atom_count": 1,
            "classification": "ion",
        },
        {
            "component_id": "MSE",
            "component_type": "L-peptide linking",
            "name": "selenomethionine",
            "formula": "C5 H11 N O2 Se",
            "classification": "modified_residue",
        },
        {
            "component_id": "HEM",
            "component_type": "non-polymer",
            "name": "heme",
            "formula": "C34 H32 Fe N4 O4",
            "classification": "ligand",
        },
        {
            "component_id": "ATP",
            "component_type": "non-polymer",
            "name": "adenosine triphosphate",
            "formula": "C10 H16 N5 O13 P3",
            "classification": "ligand",
        },
    ]
    return {
        record.component_id: record
        for record in (
            ComponentRecord.from_mapping(row, source="pheat-demo-fallback") for row in rows
        )
    }


def _normalize_classification(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "modified": "modified_residue",
        "modified_residue": "modified_residue",
        "standard": "standard_residue",
        "standard_residue": "standard_residue",
        "water": "water",
        "ion": "ion",
        "ligand": "ligand",
        "unknown": "unknown",
    }
    return aliases.get(normalized, normalized)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)

"""Packaged JSON Schema definitions for PHEAT exchange formats."""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


_SCHEMA_FILES = {
    "atom-structure": "atom-structure.schema.json",
    "residue-geometry-structure": "residue-geometry-structure.schema.json",
    "centroid-structure": "centroid-structure.schema.json",
    "energy-result": "energy-result.schema.json",
    "residue-angle-specs": "residue-angle-specs.schema.json",
    "radius-of-gyration-result": "radius-of-gyration-result.schema.json",
    "score-model-option-specs": "score-model-option-specs.schema.json",
    "scoring-options-validation": "scoring-options-validation.schema.json",
    "score-table-set": "score-table-set.schema.json",
    "geometry-table-set": "geometry-table-set.schema.json",
    "training-corpus": "training-corpus.schema.json",
    "corpus-spec": "corpus-spec.schema.json",
    "reference-dataset-manifest": "reference-dataset-manifest.schema.json",
    "benchmark-result": "benchmark-result.schema.json",
    "decoy-set-manifest": "decoy-set-manifest.schema.json",
    "software-comparison": "software-comparison.schema.json",
}


def available_schemas() -> Tuple[str, ...]:
    """Return the canonical schema names bundled with the package."""

    return tuple(_SCHEMA_FILES)


def schema_filename(name: str) -> str:
    """Return the packaged JSON filename for a schema name or filename."""

    if name in _SCHEMA_FILES:
        return _SCHEMA_FILES[name]
    if name in _SCHEMA_FILES.values():
        return name
    raise ValueError(f"Unknown PHEAT schema: {name}")


def load_schema(name: str) -> Dict[str, Any]:
    """Load a packaged JSON Schema by canonical name or filename."""

    filename = schema_filename(name)
    text = resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")
    return json.loads(text)


def validate_json_object(payload: Mapping[str, Any], schema_name: str) -> None:
    """Validate a JSON-like object against a packaged schema.

    The core package does not import jsonschema at module import time. Validation
    commands and tests run in the development environment where jsonschema
    is available; users of the validation CLI get a direct dependency hint if it
    is missing.
    """

    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError(
            "Schema validation requires jsonschema. Install pheat[training] or pheat[all]."
        ) from exc

    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    error = next(validator.iter_errors(payload), None)
    if error is None:
        return
    path = ".".join(str(part) for part in error.absolute_path)
    location = f" at {path}" if path else ""
    raise ValueError(f"{schema_name} validation failed{location}: {error.message}") from ValidationError(
        error.message
    )


def validate_json_file(path: str | Path, schema_name: str) -> None:
    """Validate a JSON file against a packaged schema."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    validate_json_object(payload, schema_name)

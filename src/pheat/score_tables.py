"""Score-table JSON helpers for trained PHEAT scoring models."""

from __future__ import annotations

import hashlib
import json
import lzma
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Union

SCORE_TABLE_SET_FORMAT = "pheat.score-table-set"
SCORE_TABLE_SET_VERSION = 1
DEFAULT_PROFILE = "protein-heavy-contacts"
PACKAGED_SCORE_TABLE_PREFIX = "packaged:"
DEFAULT_PACKAGED_SCORE_TABLE_VERSION = "v0"
SCORE_TABLE_XZ_PRESET = 6


def load_score_table_set(value: Optional[Union[str, Path, Mapping[str, Any]]]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        text_value = str(value)
        if text_value.startswith(PACKAGED_SCORE_TABLE_PREFIX):
            return load_packaged_score_table_set(text_value)
        path = Path(value)
        payload = _load_score_table_payload_bytes(path.read_bytes(), compressed=_is_xz_path(path))
        if not isinstance(payload, dict):
            raise ValueError("score table set JSON must contain an object")
    validate_score_table_set(payload)
    return payload


def packaged_score_table_manifest(
    version: str = DEFAULT_PACKAGED_SCORE_TABLE_VERSION,
) -> dict[str, Any]:
    """Load the manifest for packaged PHEAT score-table assets."""

    manifest_resource = _packaged_score_resource(version, "manifest.json")
    try:
        payload = json.loads(manifest_resource.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown packaged score-table artifact version '{version}'") from exc
    if not isinstance(payload, dict):
        raise ValueError("packaged score-table manifest must contain an object")
    if payload.get("format") != "pheat.packaged-score-assets":
        raise ValueError("packaged score-table manifest has an unsupported format")
    return payload


def packaged_score_table_ids(
    version: str = DEFAULT_PACKAGED_SCORE_TABLE_VERSION,
) -> list[str]:
    """Return packaged score-table asset IDs for a packaged artifact version."""

    manifest = packaged_score_table_manifest(version)
    assets = manifest.get("assets", {})
    if not isinstance(assets, Mapping):
        return []
    return sorted(str(asset_id) for asset_id in assets)


def load_packaged_score_table_set(table_id: str) -> dict[str, Any]:
    """Load a packaged score-table set by exact ID or ``packaged:<id>`` reference."""

    normalized = str(table_id)
    if normalized.startswith(PACKAGED_SCORE_TABLE_PREFIX):
        normalized = normalized[len(PACKAGED_SCORE_TABLE_PREFIX) :]
    manifest = packaged_score_table_manifest(DEFAULT_PACKAGED_SCORE_TABLE_VERSION)
    assets = manifest.get("assets", {})
    if not isinstance(assets, Mapping) or normalized not in assets:
        known = ", ".join(packaged_score_table_ids())
        raise ValueError(f"Unknown packaged score table '{normalized}'. Known packaged tables: {known}")
    asset = assets[normalized]
    if not isinstance(asset, Mapping):
        raise ValueError(f"Packaged score table '{normalized}' has invalid manifest metadata")
    path = asset.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Packaged score table '{normalized}' has no resource path")
    table_resource = _packaged_score_resource(
        DEFAULT_PACKAGED_SCORE_TABLE_VERSION,
        *path.split("/"),
    )
    raw = table_resource.read_bytes()
    expected_sha256 = asset.get("sha256")
    if isinstance(expected_sha256, str) and _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"Packaged score table '{normalized}' checksum does not match manifest")
    compressed = asset.get("compression") == "xz" or _is_xz_path(Path(path))
    payload = _load_score_table_payload_bytes(raw, compressed=compressed)
    if not isinstance(payload, dict):
        raise ValueError(f"Packaged score table '{normalized}' must contain an object")
    validate_score_table_set(payload)
    expected_uncompressed_sha256 = asset.get("uncompressed_sha256")
    if (
        isinstance(expected_uncompressed_sha256, str)
        and _sha256_bytes(score_table_set_json_bytes(payload)) != expected_uncompressed_sha256
    ):
        raise ValueError(f"Packaged score table '{normalized}' uncompressed checksum does not match manifest")
    return payload


def _packaged_score_resource(version: str, *parts: str):
    resource = resources.files("pheat").joinpath("data").joinpath("scoring").joinpath(version)
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def validate_score_table_set(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != SCORE_TABLE_SET_FORMAT:
        raise ValueError(f"score table set requires format {SCORE_TABLE_SET_FORMAT!r}")
    if payload.get("version") != SCORE_TABLE_SET_VERSION:
        raise ValueError(f"score table set requires version {SCORE_TABLE_SET_VERSION!r}")
    if not isinstance(payload.get("profiles"), Mapping):
        raise ValueError("score table set requires a profiles object")
    if not payload.get("default_profile"):
        raise ValueError("score table set requires default_profile")


def write_score_table_set(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    output = Path(path)
    raw = score_table_set_json_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _is_xz_path(output):
        output.write_bytes(compress_score_table_set_bytes(raw))
    else:
        output.write_bytes(raw)


def score_table_set_json_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_score_table_set(payload)
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def compress_score_table_set_bytes(raw: bytes) -> bytes:
    return lzma.compress(raw, preset=SCORE_TABLE_XZ_PRESET)


def _load_score_table_payload_bytes(raw: bytes, *, compressed: bool) -> Any:
    if compressed:
        raw = lzma.decompress(raw)
    return json.loads(raw)


def _is_xz_path(path: Path) -> bool:
    return path.suffix == ".xz" or path.name.endswith(".json.xz")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def table_for_model(
    table_set: Optional[Mapping[str, Any]],
    model: str,
    *,
    profile: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    if table_set is None:
        return None
    profile_payload = profile_payload_for(table_set, profile)
    models = profile_payload.get("models")
    if not isinstance(models, Mapping):
        return None
    table = models.get(model)
    return table if isinstance(table, Mapping) else None


def profile_payload_for(
    table_set: Mapping[str, Any],
    profile: Optional[str] = None,
) -> Mapping[str, Any]:
    profiles = table_set.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("score table set requires a profiles object")
    selected = profile or str(table_set.get("default_profile") or "")
    if not selected:
        raise ValueError("score table set requires a default profile")
    payload = profiles.get(selected)
    if not isinstance(payload, Mapping):
        known = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Unknown score table profile '{selected}'. Known profiles: {known}")
    return payload


def profile_ids(table_set: Mapping[str, Any]) -> list[str]:
    profiles = table_set.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("score table set requires a profiles object")
    return sorted(str(key) for key in profiles)


def default_profile(table_set: Mapping[str, Any]) -> str:
    return str(table_set.get("default_profile") or DEFAULT_PROFILE)

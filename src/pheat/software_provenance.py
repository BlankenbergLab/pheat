"""Selected-component software provenance for PHEAT reports."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

from pheat.scoring import model_capabilities


BASE_PACKAGE_COMPONENTS = (
    {"name": "platformdirs", "role": "PHEAT cache and platform path handling", "required": True},
)
FEATURE_PACKAGE_COMPONENTS = {
    "web": (
        {"name": "fastapi", "role": "PHEAT local web application", "required": True},
        {"name": "python-multipart", "role": "web file-upload form parsing", "required": True},
    ),
    "training": (
        {"name": "numpy", "role": "training and numeric table generation", "required": True},
        {"name": "scipy", "role": "training and numeric analysis", "required": True},
        {"name": "PyYAML", "role": "training/reference configuration", "required": True},
        {"name": "jsonschema", "role": "schema validation", "required": True},
    ),
}
PACKAGE_REQUIREMENT_ROLES = {
    "openmm": "selected PHEAT scoring dependency",
    "pdbfixer": "selected PHEAT structure-preparation dependency",
    "freesasa": "selected PHEAT solvent-accessible surface dependency",
    "numpy": "selected PHEAT numeric dependency",
    "scipy": "selected PHEAT scientific-computing dependency",
    "PyYAML": "selected PHEAT YAML dependency",
    "jsonschema": "selected PHEAT schema-validation dependency",
}
EXTERNAL_TOOL_REQUIREMENT_ROLES = {
    "gmx": "selected GROMACS scorer executable",
    "tleap": "selected AmberTools topology-preparation executable",
    "sander": "selected AmberTools energy/minimization executable",
}
EXTERNAL_TOOL_VERSION_COMMANDS = {
    "gmx": ("gmx", "--version"),
    "tleap": ("tleap", "-h"),
    "sander": ("sander", "-h"),
}


def distribution_version(name: str) -> Optional[str]:
    """Return an installed distribution version, or None when unavailable."""

    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def installed_distributions() -> list[dict[str, str]]:
    """Return all installed distributions for optional forensic sidecars."""

    rows = []
    for dist in importlib_metadata.distributions():
        metadata = dist.metadata
        name = metadata.get("Name") or getattr(dist, "_normalized_name", "")
        version = metadata.get("Version") or ""
        if name:
            rows.append({"name": str(name), "version": str(version)})
    return sorted(rows, key=lambda item: item["name"].lower())


def _package_component(name: str, role: str, *, required: bool) -> dict[str, Any]:
    version = distribution_version(name)
    return {
        "name": name,
        "version": version,
        "role": role,
        "required": bool(required),
        "selected": True,
        "status": "available" if version else "missing",
    }


def _merge_package_component(rows: dict[str, dict[str, Any]], component: Mapping[str, Any]) -> None:
    name = str(component.get("name") or "")
    if not name:
        return
    existing = rows.get(name)
    if existing is None:
        rows[name] = dict(component)
        return
    existing["required"] = bool(existing.get("required")) or bool(component.get("required"))
    roles = [str(part) for part in str(existing.get("role") or "").split("; ") if part]
    new_role = str(component.get("role") or "")
    if new_role and new_role not in roles:
        roles.append(new_role)
    existing["role"] = "; ".join(roles)
    if component.get("status") == "available":
        existing["status"] = "available"
        existing["version"] = component.get("version")


def _first_nonempty_line(text: str) -> Optional[str]:
    for line in str(text or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return None


def external_tool_component(name: str, role: str, *, required: bool) -> dict[str, Any]:
    """Return selected external executable provenance."""

    executable_path = shutil.which(name)
    payload = {
        "name": name,
        "path": executable_path,
        "version": None,
        "role": role,
        "required": bool(required),
        "selected": True,
        "status": "available" if executable_path else "missing",
        "details": None,
    }
    if not executable_path:
        payload["details"] = "executable was not found on PATH"
        return payload
    command = EXTERNAL_TOOL_VERSION_COMMANDS.get(name, (name, "--version"))
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # pragma: no cover - platform-specific executable failures
        payload["details"] = f"version probe failed: {exc}"
        return payload
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    payload["version"] = _first_nonempty_line(output)
    if completed.returncode != 0 and not payload["version"]:
        payload["details"] = f"version probe exited with code {completed.returncode}"
    return payload


def _capabilities_by_model() -> dict[str, dict[str, Any]]:
    return {str(item.get("model") or ""): dict(item) for item in model_capabilities()}


def _selected_model_requirements(
    selected_score_models: Sequence[str],
    *,
    selected_score_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    include_optional_requirements: bool = False,
) -> list[dict[str, Any]]:
    capabilities = _capabilities_by_model()
    selected_score_options = selected_score_options or {}
    requirements: dict[str, dict[str, Any]] = {}

    def add_requirement(requirement: object, *, role: str, required: bool) -> None:
        name = str(requirement or "").strip()
        if not name:
            return
        existing = requirements.get(name)
        if existing is None:
            requirements[name] = {"name": name, "role": role, "required": bool(required)}
            return
        existing["required"] = bool(existing.get("required")) or bool(required)
        roles = [str(part) for part in str(existing.get("role") or "").split("; ") if part]
        if role and role not in roles:
            roles.append(role)
        existing["role"] = "; ".join(roles)

    for model in selected_score_models:
        model_id = str(model)
        capability = capabilities.get(model_id) or {}
        role = f"selected score model: {model_id}"
        for requirement in capability.get("requires") or []:
            add_requirement(requirement, role=role, required=True)
        options = selected_score_options.get(model_id) or {}
        include_optional = include_optional_requirements or bool(options)
        if include_optional:
            for requirement in capability.get("optional_requires") or []:
                add_requirement(requirement, role=f"optional dependency for {model_id}", required=False)
    return list(requirements.values())


def collect_software_provenance(
    *,
    selected_score_models: Optional[Sequence[str]] = None,
    selected_features: Optional[Sequence[str]] = None,
    selected_score_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    include_optional_requirements: bool = False,
    include_installed_distributions: bool = False,
) -> dict[str, Any]:
    """Collect selected-component provenance for PHEAT reports.

    The main component lists intentionally describe the selected run path rather
    than every package that happens to be installed in the environment. Set
    ``include_installed_distributions`` for an exhaustive debugging inventory.
    """

    selected_score_models = list(dict.fromkeys(str(model) for model in (selected_score_models or []) if model))
    selected_features = list(dict.fromkeys(str(feature) for feature in (selected_features or []) if feature))

    package_rows_by_name: dict[str, dict[str, Any]] = {}
    for spec in BASE_PACKAGE_COMPONENTS:
        _merge_package_component(
            package_rows_by_name,
            _package_component(str(spec["name"]), str(spec["role"]), required=bool(spec["required"])),
        )
    for feature in selected_features:
        for spec in FEATURE_PACKAGE_COMPONENTS.get(feature, ()):  # unknown features may be caller-owned
            _merge_package_component(
                package_rows_by_name,
                _package_component(str(spec["name"]), str(spec["role"]), required=bool(spec["required"])),
            )

    external_tool_rows: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for spec in _selected_model_requirements(
        selected_score_models,
        selected_score_options=selected_score_options,
        include_optional_requirements=include_optional_requirements,
    ):
        requirement = str(spec.get("name") or "")
        role = str(spec.get("role") or "selected scorer dependency")
        required = bool(spec.get("required"))
        if requirement.startswith("executable:"):
            tool_name = requirement.removeprefix("executable:").strip()
            if tool_name and tool_name not in seen_tools:
                seen_tools.add(tool_name)
                external_tool_rows.append(
                    external_tool_component(
                        tool_name,
                        EXTERNAL_TOOL_REQUIREMENT_ROLES.get(tool_name, role),
                        required=required,
                    )
                )
            continue
        _merge_package_component(
            package_rows_by_name,
            _package_component(
                requirement,
                PACKAGE_REQUIREMENT_ROLES.get(requirement, role),
                required=required,
            ),
        )

    payload = {
        "format": "pheat.software-provenance",
        "version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "pheat": {"version": distribution_version("pheat")},
        "selected_score_models": selected_score_models,
        "selected_features": selected_features,
        "package_components": sorted(
            package_rows_by_name.values(),
            key=lambda item: str(item.get("name") or "").lower(),
        ),
        "external_tools": sorted(
            external_tool_rows,
            key=lambda item: str(item.get("name") or "").lower(),
        ),
    }
    if include_installed_distributions:
        payload["installed_distributions"] = installed_distributions()
    return payload

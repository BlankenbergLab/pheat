"""Runtime management for self-hosted Mol* browser assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from platformdirs import user_cache_dir

DEFAULT_MOLSTAR_VERSION = "5.9.0"
MOLSTAR_ENV_VAR = "PHEAT_MOLSTAR_DIR"
REQUIRED_MOLSTAR_FILES = ("molstar.js", "molstar.css", "LICENSE")
MOLSTAR_INSTALL_COMMAND = "pheat molstar install"


@dataclass(frozen=True)
class MolstarAssetStatus:
    """Current Mol* asset availability at a resolved location."""

    path: Path
    version: str
    source: str
    missing: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "path": str(self.path),
            "version": self.version,
            "source": self.source,
            "missing": list(self.missing),
            "install_command": MOLSTAR_INSTALL_COMMAND,
        }


def default_molstar_asset_dir(version: str = DEFAULT_MOLSTAR_VERSION) -> Path:
    """Return PHEAT's platform-aware per-user Mol* cache directory."""

    return Path(user_cache_dir("pheat")).expanduser() / "vendor" / "molstar" / str(version)


def molstar_asset_dir(
    *,
    version: str = DEFAULT_MOLSTAR_VERSION,
    path: str | Path | None = None,
) -> tuple[Path, str]:
    """Return the active Mol* asset directory and its configuration source."""

    if path is not None:
        return Path(path).expanduser(), "argument"
    env_path = os.environ.get(MOLSTAR_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser(), MOLSTAR_ENV_VAR
    return default_molstar_asset_dir(version), "platform-cache"


def molstar_asset_status(
    *,
    version: str = DEFAULT_MOLSTAR_VERSION,
    path: str | Path | None = None,
) -> MolstarAssetStatus:
    """Return availability metadata for the active Mol* asset directory."""

    asset_dir, source = molstar_asset_dir(version=version, path=path)
    missing = tuple(filename for filename in REQUIRED_MOLSTAR_FILES if not (asset_dir / filename).exists())
    return MolstarAssetStatus(path=asset_dir, version=str(version), source=source, missing=missing)


def molstar_missing_assets_message(status: MolstarAssetStatus) -> str:
    missing = ", ".join(status.missing) if status.missing else "none"
    return (
        f"Mol* assets are unavailable in {status.path}: missing {missing}. "
        f"Run `{MOLSTAR_INSTALL_COMMAND}`."
    )


def warn_missing_molstar_assets(
    status: MolstarAssetStatus,
    *,
    stacklevel: int = 2,
    warning_func: Callable[[str], None] | None = None,
) -> str:
    """Warn that Mol* assets are unavailable and return the warning text."""

    message = molstar_missing_assets_message(status)
    if warning_func is not None:
        warning_func(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
    return message


def resolve_molstar_assets(
    *,
    version: str = DEFAULT_MOLSTAR_VERSION,
    path: str | Path | None = None,
    warn: bool = True,
) -> Optional[Path]:
    """Return the active Mol* asset directory, or ``None`` if required files are missing."""

    status = molstar_asset_status(version=version, path=path)
    if status.available:
        return status.path
    if warn:
        warn_missing_molstar_assets(status, stacklevel=3)
    return None


def copy_molstar_assets(
    destination: str | Path,
    *,
    version: str = DEFAULT_MOLSTAR_VERSION,
    source: str | Path | None = None,
    warn: bool = True,
) -> tuple[Optional[Path], Optional[str]]:
    """Copy resolved Mol* assets into ``destination``.

    Returns ``(destination_path, None)`` when assets are available. If the source
    assets are missing, returns ``(None, message)`` and optionally emits a
    warning. The destination is an exact directory, not a parent report folder.
    """

    status = molstar_asset_status(version=version, path=source)
    if not status.available:
        message = molstar_missing_assets_message(status)
        if warn:
            warn_missing_molstar_assets(status, stacklevel=3)
        return None, message

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_MOLSTAR_FILES:
        shutil.copy2(status.path / filename, destination_path / filename)
    manifest = status.path / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, destination_path / "manifest.json")
    return destination_path, None


def install_molstar_assets(
    *,
    version: str = DEFAULT_MOLSTAR_VERSION,
    destination: str | Path | None = None,
    timeout: float = 60.0,
    force: bool = False,
    npm: str = "npm",
) -> Path:
    """Install pinned Mol* browser assets into the PHEAT runtime cache."""

    target, _source = molstar_asset_dir(version=version, path=destination)
    status = molstar_asset_status(version=version, path=target)
    if status.available and not force:
        return target

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir_text:
        tmpdir = Path(tmpdir_text)
        pack = subprocess.run(
            [npm, "pack", "--silent", f"molstar@{version}", "--pack-destination", str(tmpdir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
        if pack.returncode != 0:
            stderr = pack.stderr.strip()
            stdout = pack.stdout.strip()
            detail = stderr or stdout or f"exit code {pack.returncode}"
            raise RuntimeError(f"npm failed to download molstar@{version}: {detail}")

        package_path = _npm_pack_output_path(tmpdir, pack.stdout)
        extract_dir = tmpdir / "package"
        extract_dir.mkdir()
        _safe_extract_tarball(package_path, extract_dir)

        viewer_dir = extract_dir / "package" / "build" / "viewer"
        license_path = extract_dir / "package" / "LICENSE"
        sources = {
            "molstar.js": viewer_dir / "molstar.js",
            "molstar.css": viewer_dir / "molstar.css",
            "LICENSE": license_path,
        }
        missing = [name for name, path in sources.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"molstar@{version} package did not contain required assets: {', '.join(missing)}"
            )

        for filename, source_path in sources.items():
            shutil.copy2(source_path, target / filename)
        strip_molstar_source_maps(target)
        _write_manifest(target, version=version)
    return target


def strip_molstar_source_maps(vendor_dir: str | Path) -> list[Path]:
    """Remove trailing source-map comments from generated Mol* assets."""

    vendor_path = Path(vendor_dir)
    changed: list[Path] = []
    patterns = {
        "molstar.js": "//# sourceMappingURL=molstar.js.map",
        "molstar.css": "/*# sourceMappingURL=molstar.css.map */",
    }
    for filename, marker in patterns.items():
        path = vendor_path / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[-1].strip() == marker:
            path.write_text("\n".join(lines[:-1]).rstrip() + "\n", encoding="utf-8")
            changed.append(path)
    return changed


def _npm_pack_output_path(tmpdir: Path, stdout: str) -> Path:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if lines:
        candidate = Path(lines[-1])
        if not candidate.is_absolute():
            candidate = tmpdir / candidate.name
        if candidate.exists():
            return candidate
    matches = sorted(tmpdir.glob("molstar-*.tgz"))
    if matches:
        return matches[-1]
    raise FileNotFoundError("npm pack did not produce a molstar tarball")


def _safe_extract_tarball(package_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(package_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not _is_relative_to(member_path, destination_resolved):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        archive.extractall(destination)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_manifest(target: Path, *, version: str) -> None:
    files = {}
    for filename in REQUIRED_MOLSTAR_FILES:
        path = target / filename
        payload = path.read_bytes()
        files[filename] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "format": "pheat.molstar-assets",
        "version": 1,
        "molstar_version": str(version),
        "source_package": f"molstar@{version}",
        "source_url": f"https://www.npmjs.com/package/molstar/v/{version}",
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MOLSTAR_VERSION",
    "MOLSTAR_ENV_VAR",
    "MOLSTAR_INSTALL_COMMAND",
    "MolstarAssetStatus",
    "REQUIRED_MOLSTAR_FILES",
    "copy_molstar_assets",
    "default_molstar_asset_dir",
    "install_molstar_assets",
    "molstar_asset_dir",
    "molstar_asset_status",
    "molstar_missing_assets_message",
    "resolve_molstar_assets",
    "strip_molstar_source_maps",
    "warn_missing_molstar_assets",
]

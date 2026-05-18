"""MkDocs hook for publishing package schemas as static documentation assets."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any


def copy_schema_assets(*, repo_root: Path, site_dir: Path) -> list[Path]:
    """Copy the single source-of-truth package schemas into the built site."""

    source_dir = repo_root / "src" / "pheat" / "schemas"
    destination_dir = site_dir / "schemas"
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for source_path in sorted(source_dir.glob("*.schema.json")):
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    return copied


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Publish schema URLs after MkDocs has generated the static site."""

    repo_root = Path(config.config_file_path).resolve().parent
    copy_schema_assets(repo_root=repo_root, site_dir=Path(config["site_dir"]))

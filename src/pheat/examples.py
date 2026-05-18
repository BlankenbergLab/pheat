"""Example-set manifests and optional PDB fetching."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast
from urllib.request import urlretrieve

Downloader = Callable[[str, str], object]


def list_example_sets() -> List[str]:
    base = resources.files("pheat").joinpath("data/examples")
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def load_example_manifest(name: str) -> Dict[str, object]:
    path = resources.files("pheat").joinpath("data/examples").joinpath(name).joinpath("manifest.json")
    with path.open("r", encoding="utf-8") as handle:
        return cast(Dict[str, object], json.load(handle))


def fetch_example_set(
    name: str,
    destination: Union[str, Path] = ".pheat-cache/examples",
    *,
    downloader: Optional[Downloader] = None,
) -> List[Path]:
    manifest = load_example_manifest(name)
    output_dir = Path(destination) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    fetched: List[Path] = []
    for item in _manifest_examples(manifest):
        pdb_id = item.get("pdb_id")
        if not pdb_id:
            continue
        output = output_dir / f"{str(pdb_id).lower()}.pdb"
        if not output.exists():
            _download_to_path(_pdb_download_url(str(pdb_id)), output, downloader=downloader)
        fetched.append(output)
    return fetched


def run_example_set(
    name: str,
    *,
    cache_dir: Union[str, Path] = ".pheat-cache/examples",
    model: str = "generic",
    fetch: bool = False,
) -> Dict[str, object]:
    """Run smoke conversions/scoring for an example set.

    Network fetching is opt-in. Without `fetch=True`, PDB-backed examples are reported as
    missing until the user runs `pheat examples fetch`.
    """

    from pheat.centroid import to_centroid_structure
    from pheat.pdbio import load_pdb
    from pheat.scoring import score_structure
    from pheat.residue_geometry import structure_from_residue_geometry

    if fetch:
        fetch_example_set(name, cache_dir)
    manifest = load_example_manifest(name)
    root = Path(cache_dir) / name
    results: List[Dict[str, Any]] = []
    for item in _manifest_examples(manifest):
        item_id = str(item["id"])
        pdb_id = item.get("pdb_id")
        sequence = item.get("sequence")
        if item.get("requires_user_coordinates"):
            results.append({"id": item_id, "status": "skipped", "reason": "requires user coordinates"})
            continue
        if pdb_id:
            path = root / f"{str(pdb_id).lower()}.pdb"
            if not path.exists():
                results.append({"id": item_id, "status": "missing", "path": str(path)})
                continue
            structure = load_pdb(path)
        elif sequence:
            structure = structure_from_residue_geometry({"sequence": sequence, "name": item_id})
        else:
            results.append({"id": item_id, "status": "skipped", "reason": "no pdb_id or sequence"})
            continue

        score = score_structure(structure, model=model)
        centroids = to_centroid_structure(structure)
        results.append(
            {
                "id": item_id,
                "status": "ok",
                "atom_count": len(structure.atoms),
                "residue_count": len(structure.residue_keys()),
                "centroid_count": len(centroids.centroids),
                "score": score.to_dict(),
            }
        )
    return {"name": name, "model": model, "results": results}


def _manifest_examples(manifest: Mapping[str, object]) -> List[Mapping[str, Any]]:
    examples = manifest.get("examples", [])
    if not isinstance(examples, list):
        return []
    return [cast(Mapping[str, Any], item) for item in examples]


def _pdb_download_url(pdb_id: str) -> str:
    return f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"


def _download_to_path(
    url: str,
    output: Path,
    *,
    downloader: Optional[Downloader] = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    fetch = downloader or urlretrieve
    try:
        fetch(url, str(temp_path))
        temp_path.replace(output)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} to {output}: {exc}") from exc

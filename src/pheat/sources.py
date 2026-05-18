"""Source manifest loading and verification."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from importlib import resources
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast
from urllib.request import urlretrieve

Downloader = Callable[[str, str], object]
SourceFetchResult = Union[Path, List[Path]]
PROVENANCE_FILENAME = "pheat-source-provenance.json"


def load_source_manifest() -> Dict[str, Any]:
    with resources.files("pheat").joinpath("data/sources/manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return cast(Dict[str, Any], json.load(handle))


def list_sources() -> List[Mapping[str, object]]:
    sources = load_source_manifest().get("sources", [])
    if not isinstance(sources, list):
        return []
    return [cast(Mapping[str, object], source) for source in sources]


def fetch_source(
    source_id: str,
    destination: Union[str, Path],
    *,
    downloader: Optional[Downloader] = None,
) -> SourceFetchResult:
    source = _find_source(source_id)
    if source.get("downloadable") is False:
        raise ValueError(
            f"Source '{source_id}' is reference-only and is not fetched or vendored by PHEAT."
        )
    files = _source_files(source)
    if not files:
        raise ValueError(f"Source '{source_id}' does not define downloadable URLs.")
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination_path.resolve()
    outputs = []
    records = []
    for file_payload in files:
        filename = str(file_payload["filename"])
        output = destination_path / filename
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(
                f"Source '{source_id}' has an unsafe filename {filename!r}; "
                "filenames must be relative and may not contain parent-directory references."
            )
        if not output.resolve().is_relative_to(destination_resolved):
            raise ValueError(
                f"Source '{source_id}' filename {filename!r} would escape the destination directory."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        url = str(file_payload["url"])
        temp_output, headers = _download_to_temporary_path(url, output, downloader=downloader)
        expected = _expected_sha256(source, file_payload)
        try:
            actual = sha256_file(temp_output)
            if expected and actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {output}: expected {expected}, found {actual}"
                )
            byte_count = temp_output.stat().st_size
            temp_output.replace(output)
            outputs.append(output)
            records.append(
                {
                    "url": url,
                    "filename": str(file_payload["filename"]),
                    "relative_path": str(file_payload["filename"]),
                    "path": str(output),
                    "bytes": byte_count,
                    "sha256": actual,
                    "expected_sha256": str(expected) if expected else None,
                    "downloaded_at": _utc_now(),
                    "last_modified": _header_value(headers, "last-modified"),
                    "etag": _header_value(headers, "etag"),
                }
            )
        except Exception:
            temp_output.unlink(missing_ok=True)
            raise
    _write_source_provenance(destination_path, source, records)
    return outputs[0] if len(outputs) == 1 else outputs


def verify_sources(cache_dir: Union[str, Path]) -> Dict[str, object]:
    cache = Path(cache_dir)
    results: Dict[str, object] = {}
    provenance = _provenance_by_source(cache)
    for source in list_sources():
        source_id = str(source["id"])
        if source.get("downloadable") is False:
            results[source_id] = {"status": "reference-only", "downloadable": False}
            continue
        if source_id in provenance:
            provenance_path, source_record = provenance[source_id]
            results[source_id] = _verify_provenance_record(source, provenance_path, source_record)
            continue
        results[source_id] = _verify_source_files_without_provenance(source, cache)
    return results


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_source(source_id: str) -> Mapping[str, object]:
    for source in list_sources():
        if source.get("id") == source_id:
            return source
    raise KeyError(f"Unknown source '{source_id}'.")


def _source_files(source: Mapping[str, object]) -> List[Mapping[str, object]]:
    files = source.get("files")
    if isinstance(files, list):
        normalized = []
        for item in files:
            if not isinstance(item, Mapping):
                continue
            if item.get("url") and item.get("filename"):
                normalized.append(cast(Mapping[str, object], item))
        return normalized
    urls = source.get("urls", [])
    if not isinstance(urls, list) or not urls:
        return []
    return [
        {
            "url": str(urls[0]),
            "filename": str(source.get("filename") or f"{source.get('id', 'source')}.dat"),
            "sha256": source.get("sha256"),
        }
    ]


def _expected_sha256(source: Mapping[str, object], file_payload: Mapping[str, object]) -> Optional[str]:
    value = file_payload.get("sha256")
    if value:
        return str(value)
    files = _source_files(source)
    if len(files) == 1 and source.get("sha256"):
        return str(source["sha256"])
    return None


def _download_to_temporary_path(
    url: str,
    output: Path,
    *,
    downloader: Optional[Downloader] = None,
) -> tuple[Path, Optional[object]]:
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
        result = fetch(url, str(temp_path))
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} to {output}: {exc}") from exc
    return temp_path, _download_headers(result)


def _download_headers(result: object) -> Optional[object]:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return None


def _header_value(headers: Optional[object], name: str) -> Optional[str]:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        return str(value) if value is not None else None
    return None


def _write_source_provenance(
    destination: Path,
    source: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    path = destination / PROVENANCE_FILENAME
    payload: Dict[str, object]
    if path.exists():
        try:
            payload = cast(Dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            payload = {}
    else:
        payload = {}
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    source_id = str(source["id"])
    source_record = {
        "id": source_id,
        "name": source.get("name"),
        "type": source.get("type"),
        "license": source.get("license"),
        "license_url": source.get("license_url"),
        "license_note": source.get("license_note"),
        "citation": source.get("citation"),
        "files": [dict(record) for record in records],
    }
    sources = [
        item
        for item in sources
        if not isinstance(item, Mapping) or str(item.get("id")) != source_id
    ]
    sources.append(source_record)
    payload = {
        "format": "pheat.source-cache-provenance",
        "version": 1,
        "generated_at": _utc_now(),
        "pheat_version": _pheat_version(),
        "sources": sources,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provenance_by_source(cache: Path) -> Dict[str, tuple[Path, Mapping[str, object]]]:
    records: Dict[str, tuple[Path, Mapping[str, object]]] = {}
    if not cache.exists():
        return records
    for path in sorted(cache.rglob(PROVENANCE_FILENAME)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sources = payload.get("sources", []) if isinstance(payload, Mapping) else []
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, Mapping) and source.get("id"):
                records[str(source["id"])] = (path, cast(Mapping[str, object], source))
    return records


def _verify_provenance_record(
    source: Mapping[str, object],
    provenance_path: Path,
    source_record: Mapping[str, object],
) -> Dict[str, object]:
    base = provenance_path.parent
    file_records = source_record.get("files", [])
    files: List[Dict[str, object]] = []
    status = "ok"
    if not isinstance(file_records, list) or not file_records:
        return {"status": "missing", "provenance": str(provenance_path), "files": []}
    for record in file_records:
        if not isinstance(record, Mapping):
            continue
        relative_path = str(record.get("relative_path") or record.get("filename") or "")
        path = base / relative_path
        expected = str(record.get("sha256") or _expected_sha_for_filename(source, relative_path) or "")
        if not path.exists():
            status = "missing"
            files.append({"status": "missing", "path": str(path), "filename": relative_path})
            continue
        actual = sha256_file(path)
        file_status = "ok" if not expected or actual == expected else "checksum-mismatch"
        if file_status != "ok" and status == "ok":
            status = file_status
        files.append(
            {
                "status": file_status,
                "path": str(path),
                "filename": relative_path,
                "sha256": actual,
                "expected_sha256": expected or None,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "status": status,
        "downloadable": True,
        "provenance": str(provenance_path),
        "files": files,
    }


def _verify_source_files_without_provenance(source: Mapping[str, object], cache: Path) -> Dict[str, object]:
    files: List[Dict[str, object]] = []
    status = "ok"
    for file_payload in _source_files(source):
        filename = str(file_payload["filename"])
        path = cache / filename
        expected = _expected_sha256(source, file_payload)
        if not path.exists():
            status = "missing"
            files.append({"status": "missing", "path": str(path), "filename": filename})
            continue
        actual = sha256_file(path)
        file_status = "ok" if not expected or actual == expected else "checksum-mismatch"
        if file_status != "ok" and status == "ok":
            status = file_status
        files.append(
            {
                "status": file_status,
                "path": str(path),
                "filename": filename,
                "sha256": actual,
                "expected_sha256": expected,
                "bytes": path.stat().st_size,
            }
        )
    if len(files) == 1:
        result: Dict[str, object] = dict(files[0])
        result["downloadable"] = True
        return result
    return {"status": status, "downloadable": True, "files": files}


def _expected_sha_for_filename(source: Mapping[str, object], filename: str) -> Optional[str]:
    for file_payload in _source_files(source):
        if str(file_payload.get("filename")) == filename:
            return _expected_sha256(source, file_payload)
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pheat_version() -> str:
    try:
        return version("pheat")
    except PackageNotFoundError:
        return "0.0.0+unknown"

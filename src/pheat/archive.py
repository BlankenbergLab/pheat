"""PDB archive corpus download and provenance helpers.

The routines in this module intentionally keep the transport and provenance
logic explicit. RCSB Search/Data API schemas can be inspected for version and
checksum provenance, but schema bodies are not written into PHEAT outputs unless
a future caller adds that behavior explicitly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pheat.bcif import decode_bcif_column

DEFAULT_OUTPUT_ROOT = Path(".pheat-cache/pdb-archive")
SEARCH_QUERY_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CURRENT_ENTRY_IDS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
DATA_GRAPHQL_URL = "https://data.rcsb.org/graphql"
SNAPSHOT_METADATA_SOURCES = ("auto", "rcsb-api", "bcif-local")


@dataclass(frozen=True)
class SchemaSpec:
    service: str
    kind: str
    url: str
    used_for: str


@dataclass(frozen=True)
class HttpPayload:
    body: bytes
    headers: Mapping[str, str]
    status: int = 200
    url: str = ""


@dataclass(frozen=True)
class ArchivePaths:
    output_root: Path
    raw_dir: Path
    manifest_dir: Path
    processed_dir: Path
    analysis_dir: Path
    failed_dir: Path


@dataclass(frozen=True)
class DownloadRecord:
    pdb_id: str
    url: str
    path: Path
    status: str
    source_path: Optional[Path] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class SnapshotSpec:
    id: str
    name: str
    description: str
    file_format: str
    source_type: str
    coordinate_url_template: str
    metadata_urls: Sequence[str]
    license_note: str
    default_output_root: Path
    checksum_policy: str


Fetcher = Callable[[str, Optional[Mapping[str, Any]]], HttpPayload]
Downloader = Callable[[str, Path], None]
Clock = Callable[[], float]
ReuseMode = str
StatusStream = Any


SCHEMA_SPECS = (
    SchemaSpec(
        service="search",
        kind="openapi",
        url="https://search.rcsb.org/openapi.json",
        used_for="Search API endpoint, version, and license provenance",
    ),
    SchemaSpec(
        service="search-metadata",
        kind="json-schema",
        url="https://search.rcsb.org/rcsbsearch/v2/metadata/schema",
        used_for="Search attribute discovery, descriptions, units, and ranges",
    ),
)

SNAPSHOTS = (
    SnapshotSpec(
        id="wwpdb-current-mmcif",
        name="wwPDB current holdings, gzipped mmCIF",
        description="Current RCSB/wwPDB entry IDs downloaded as gzipped mmCIF coordinate files.",
        file_format="cif",
        source_type="current-holdings",
        coordinate_url_template="https://files.rcsb.org/download/{pdb_id}.cif.gz",
        metadata_urls=(CURRENT_ENTRY_IDS_URL,),
        license_note=(
            "PDB archive data files and RCSB PDB programmatic data are distributed under "
            "the RCSB usage policy and its CC0 statement plus external-resource caveats."
        ),
        default_output_root=DEFAULT_OUTPUT_ROOT / "wwpdb-current-mmcif",
        checksum_policy="pheat-sha256-manifest",
    ),
    SnapshotSpec(
        id="wwpdb-current-pdb",
        name="wwPDB current holdings, gzipped legacy PDB",
        description="Current RCSB/wwPDB entry IDs downloaded as gzipped legacy PDB coordinate files.",
        file_format="pdb",
        source_type="current-holdings",
        coordinate_url_template="https://files.rcsb.org/download/{pdb_id}.pdb.gz",
        metadata_urls=(CURRENT_ENTRY_IDS_URL,),
        license_note=(
            "PDB archive data files and RCSB PDB programmatic data are distributed under "
            "the RCSB usage policy and its CC0 statement plus external-resource caveats."
        ),
        default_output_root=DEFAULT_OUTPUT_ROOT / "wwpdb-current-pdb",
        checksum_policy="pheat-sha256-manifest",
    ),
    SnapshotSpec(
        id="rcsb-current-bcif",
        name="RCSB current holdings, gzipped BinaryCIF",
        description="Current RCSB/wwPDB entry IDs downloaded as gzipped RCSB BinaryCIF files.",
        file_format="bcif",
        source_type="current-holdings",
        coordinate_url_template="https://models.rcsb.org/{pdb_id}.bcif.gz",
        metadata_urls=(CURRENT_ENTRY_IDS_URL,),
        license_note=(
            "PDB archive data files and RCSB PDB programmatic data are distributed under "
            "the RCSB usage policy and its CC0 statement plus external-resource caveats."
        ),
        default_output_root=DEFAULT_OUTPUT_ROOT / "rcsb-current-bcif",
        checksum_policy="pheat-sha256-manifest",
    ),
)

ENTRY_METADATA_QUERY = """
query PheatEntryMetadata($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    entry { id }
    struct { title pdbx_CASP_flag }
    struct_keywords { text pdbx_keywords }
    exptl { method }
    refine {
      B_iso_mean
      ls_R_factor_R_free
      ls_R_factor_R_work
      ls_R_factor_obs
      ls_d_res_high
      ls_d_res_low
      ls_number_reflns_obs
      ls_percent_reflns_obs
      pdbx_refine_id
    }
    pdbx_database_status {
      status_code
      pdb_format_compatible
      recvd_initial_deposition_date
    }
    rcsb_accession_info {
      deposit_date
      initial_release_date
      revision_date
      major_revision
      minor_revision
      status_code
    }
    rcsb_entry_container_identifiers {
      assembly_ids
      entity_ids
      model_ids
      non_polymer_entity_ids
      polymer_entity_ids
    }
    rcsb_entry_info {
      assembly_count
      branched_entity_count
      cis_peptide_count
      deposited_atom_count
      deposited_hydrogen_atom_count
      deposited_model_count
      deposited_polymer_entity_instance_count
      deposited_polymer_monomer_count
      deposited_solvent_atom_count
      deposited_unmodeled_polymer_monomer_count
      disulfide_bond_count
      entity_count
      experimental_method
      experimental_method_count
      inter_mol_covalent_bond_count
      inter_mol_metalic_bond_count
      molecular_weight
      nonpolymer_entity_count
      polymer_composition
      polymer_entity_count
      polymer_entity_count_protein
      resolution_combined
      selected_polymer_entity_types
      solvent_entity_count
      structure_determination_methodology
    }
    pdbx_vrpt_summary_geometry {
      angles_RMSZ
      bonds_RMSZ
      clashscore
      percent_ramachandran_outliers
      percent_rotamer_outliers
    }
    pdbx_vrpt_summary_diffraction {
      DCC_R
      DCC_Rfree
      EDS_R
      EDS_res_high
      Fo_Fc_correlation
      data_completeness
      percent_RSRZ_outliers
    }
    exptl_crystal_grow {
      method
      pH
      temp
      pdbx_details
    }
  }
}
"""

POLYMER_ENTITY_METADATA_QUERY = """
query PheatPolymerEntityMetadata($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    rcsb_polymer_entity_container_identifiers {
      asym_ids
      auth_asym_ids
      entry_id
      entity_id
    }
    entity_poly {
      nstd_linkage
      nstd_monomer
      pdbx_seq_one_letter_code_can
      pdbx_strand_id
      rcsb_entity_polymer_type
      rcsb_sample_sequence_length
      type
    }
    rcsb_polymer_entity {
      formula_weight
      pdbx_description
    }
    rcsb_entity_source_organism {
      ncbi_scientific_name
      ncbi_taxonomy_id
      source_type
    }
    rcsb_polymer_entity_group_membership {
      aggregation_method
      group_id
      similarity_cutoff
    }
  }
}
"""


def default_archive_paths(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    raw_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    analysis_dir: Optional[Path] = None,
    failed_dir: Optional[Path] = None,
) -> ArchivePaths:
    root = Path(output_root)
    return ArchivePaths(
        output_root=root,
        raw_dir=Path(raw_dir) if raw_dir is not None else root / "raw",
        manifest_dir=Path(manifest_dir) if manifest_dir is not None else root / "manifests",
        processed_dir=Path(processed_dir) if processed_dir is not None else root / "processed",
        analysis_dir=Path(analysis_dir) if analysis_dir is not None else root / "analysis",
        failed_dir=Path(failed_dir) if failed_dir is not None else root / "failed",
    )


def ensure_archive_paths(paths: ArchivePaths) -> None:
    for path in [
        paths.output_root,
        paths.raw_dir,
        paths.manifest_dir,
        paths.processed_dir,
        paths.analysis_dir,
        paths.failed_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_http_json_bytes(url: str, json_payload: Optional[Mapping[str, Any]] = None) -> HttpPayload:
    data = None
    headers = {"Accept": "application/json"}
    if json_payload is not None:
        data = json.dumps(json_payload, sort_keys=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=60) as response:  # nosec B310 - user-facing archival utility.
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return HttpPayload(
                body=response.read(),
                headers=response_headers,
                status=int(getattr(response, "status", 200)),
                url=url,
            )
    except HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"HTTP {exc.code} fetching {url}: {body[:200]!r}") from exc


def schema_provenance_records(
    *,
    fetcher: Fetcher = fetch_http_json_bytes,
    retrieved_at: Optional[datetime] = None,
    specs: Sequence[SchemaSpec] = SCHEMA_SPECS,
) -> list[dict[str, Any]]:
    """Fetch known RCSB schema documents and return provenance without content."""

    timestamp = (retrieved_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    records: list[dict[str, Any]] = []
    for spec in specs:
        payload = fetcher(spec.url, None)
        body = payload.body
        document = _decode_json_or_none(body)
        records.append(
            {
                "service": spec.service,
                "kind": spec.kind,
                "url": spec.url,
                "retrieved_at": timestamp.isoformat().replace("+00:00", "Z"),
                "content_sha256": hashlib.sha256(body).hexdigest(),
                "content_length": len(body),
                "etag": _header(payload.headers, "etag"),
                "last_modified": _header(payload.headers, "last-modified"),
                "declared_license": _schema_license(document),
                "version": _schema_version(document),
                "schema_id": document.get("$id") if isinstance(document, dict) else None,
                "schema_uri": document.get("$schema") if isinstance(document, dict) else None,
                "stored": False,
                "used_for": spec.used_for,
            }
        )
    return records


def write_schema_provenance(
    manifest_dir: Path,
    *,
    fetcher: Fetcher = fetch_http_json_bytes,
) -> Path:
    records = schema_provenance_records(fetcher=fetcher)
    output = Path(manifest_dir) / "api-schemas.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schemas": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def ids_from_file(path: Path) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(normalize_pdb_id(value))
    return sorted(set(ids))


def ids_from_current_holdings(*, fetcher: Fetcher = fetch_http_json_bytes) -> list[str]:
    payload = fetcher(CURRENT_ENTRY_IDS_URL, None)
    decoded = json.loads(payload.body.decode("utf-8"))
    if not isinstance(decoded, list):
        raise ValueError("current holdings endpoint did not return a JSON list")
    return sorted({normalize_pdb_id(str(item)) for item in decoded})


def ids_from_search_query(
    query: Mapping[str, Any],
    *,
    fetcher: Fetcher = fetch_http_json_bytes,
) -> list[str]:
    payload = fetcher(SEARCH_QUERY_URL, query)
    decoded = json.loads(payload.body.decode("utf-8"))
    result_set = decoded.get("result_set", []) if isinstance(decoded, dict) else []
    ids = []
    for item in result_set:
        if isinstance(item, Mapping) and "identifier" in item:
            ids.append(normalize_pdb_id(str(item["identifier"])))
    return sorted(set(ids))


def coordinate_url(pdb_id: str, file_format: str) -> str:
    normalized = normalize_pdb_id(pdb_id)
    if file_format == "cif":
        return f"https://files.rcsb.org/download/{normalized}.cif.gz"
    if file_format == "pdb":
        return f"https://files.rcsb.org/download/{normalized}.pdb.gz"
    if file_format == "bcif":
        return f"https://models.rcsb.org/{normalized}.bcif.gz"
    raise ValueError(f"Unsupported coordinate format: {file_format}")


def coordinate_filename(pdb_id: str, file_format: str) -> str:
    normalized = normalize_pdb_id(pdb_id).lower()
    if file_format == "cif":
        return f"{normalized}.cif.gz"
    if file_format == "pdb":
        return f"{normalized}.pdb.gz"
    if file_format == "bcif":
        return f"{normalized}.bcif.gz"
    raise ValueError(f"Unsupported coordinate format: {file_format}")


def snapshot_ids() -> list[str]:
    return [snapshot.id for snapshot in SNAPSHOTS]


def get_snapshot(snapshot_id: str) -> SnapshotSpec:
    for snapshot in SNAPSHOTS:
        if snapshot.id == snapshot_id:
            return snapshot
    known = ", ".join(snapshot_ids())
    raise ValueError(f"Unknown archive snapshot '{snapshot_id}'. Known snapshots: {known}")


def snapshot_to_dict(snapshot: SnapshotSpec) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "name": snapshot.name,
        "description": snapshot.description,
        "format": snapshot.file_format,
        "source_type": snapshot.source_type,
        "coordinate_url_template": snapshot.coordinate_url_template,
        "metadata_urls": list(snapshot.metadata_urls),
        "license_note": snapshot.license_note,
        "default_output_root": str(snapshot.default_output_root),
        "checksum_policy": snapshot.checksum_policy,
    }


def list_snapshots() -> list[dict[str, Any]]:
    return [snapshot_to_dict(snapshot) for snapshot in SNAPSHOTS]


def plan_download_records(
    ids: Iterable[str],
    paths: ArchivePaths,
    *,
    file_format: str = "cif",
    reuse_raw_dirs: Sequence[Path] = (),
) -> list[DownloadRecord]:
    records = []
    expected_sha256_by_filename = _manifest_sha256_by_filename(paths.manifest_dir)
    for pdb_id in sorted({normalize_pdb_id(item) for item in ids}):
        filename = coordinate_filename(pdb_id, file_format)
        target = paths.raw_dir / filename
        url = coordinate_url(pdb_id, file_format)
        if target.exists():
            expected_sha256 = expected_sha256_by_filename.get(filename.lower())
            if expected_sha256 is not None and sha256_file(target) != expected_sha256:
                records.append(DownloadRecord(pdb_id=pdb_id, url=url, path=target, status="pending"))
                continue
            records.append(_record_for_existing(pdb_id, url, target, "present"))
            continue
        reused = _find_reused_file(filename, reuse_raw_dirs)
        if reused is not None:
            records.append(_record_for_existing(pdb_id, url, target, "reused", source_path=reused))
            continue
        records.append(DownloadRecord(pdb_id=pdb_id, url=url, path=target, status="pending"))
    return records


def download_pending_records(
    records: Sequence[DownloadRecord],
    *,
    downloader: Downloader,
    staging_dir: Optional[Path] = None,
    cleanup_staging: bool = False,
    status_stream: Optional[StatusStream] = None,
    progress_interval: int = 100,
    progress_seconds: Optional[float] = None,
    progress_redraw: bool = True,
    clock: Clock = monotonic,
) -> list[DownloadRecord]:
    completed = []
    pending_total = sum(1 for record in records if record.status == "pending")
    progress = _DownloadProgressReporter(
        status_stream,
        total=pending_total,
        interval=progress_interval,
        seconds=progress_seconds,
        redraw=progress_redraw,
        clock=clock,
    )
    progress.start()
    for record in records:
        if record.status != "pending":
            completed.append(record)
            continue
        try:
            downloaded = _download_record(
                record,
                downloader=downloader,
                staging_dir=staging_dir,
                cleanup_staging=cleanup_staging,
            )
            completed.append(downloaded)
            progress.record_success(downloaded)
        except Exception as exc:
            _status_print(
                status_stream,
                f"failed download for {record.pdb_id}: {exc}",
            )
            progress.record_failure()
            completed.append(
                DownloadRecord(
                    pdb_id=record.pdb_id,
                    url=record.url,
                    path=record.path,
                    status="failed",
                    error=str(exc),
                )
            )
    progress.finish()
    return completed


def materialize_reused_records(records: Sequence[DownloadRecord], *, reuse_mode: ReuseMode) -> list[DownloadRecord]:
    """Copy reused raw files into the snapshot unless reference mode was requested."""

    if reuse_mode == "reference":
        return list(records)
    if reuse_mode != "copy":
        raise ValueError(f"Unsupported reuse mode: {reuse_mode}")

    materialized = []
    for record in records:
        if record.status != "reused":
            materialized.append(record)
            continue
        if record.source_path is None:
            materialized.append(
                DownloadRecord(
                    pdb_id=record.pdb_id,
                    url=record.url,
                    path=record.path,
                    status="failed",
                    error="reused record has no source_path",
                )
            )
            continue
        try:
            source_sha256 = sha256_file(record.source_path)
            copied = copy_verified_file(
                record.source_path,
                record.path,
                expected_sha256=source_sha256,
            )
            materialized.append(
                DownloadRecord(
                    pdb_id=record.pdb_id,
                    url=record.url,
                    path=record.path,
                    status="reused",
                    source_path=record.source_path,
                    bytes=copied.stat().st_size,
                    sha256=sha256_file(copied),
                )
            )
        except Exception as exc:
            materialized.append(
                DownloadRecord(
                    pdb_id=record.pdb_id,
                    url=record.url,
                    path=record.path,
                    status="failed",
                    source_path=record.source_path,
                    error=str(exc),
                )
            )
    return materialized


def _download_record(
    record: DownloadRecord,
    *,
    downloader: Downloader,
    staging_dir: Optional[Path],
    cleanup_staging: bool,
) -> DownloadRecord:
    if staging_dir is None:
        downloaded = atomic_download(record.url, record.path, downloader=downloader)
        return DownloadRecord(
            pdb_id=record.pdb_id,
            url=record.url,
            path=record.path,
            status="downloaded",
            bytes=downloaded.stat().st_size,
            sha256=sha256_file(downloaded),
        )

    staged_path = Path(staging_dir) / record.path.name
    if staged_path.resolve(strict=False) == record.path.resolve(strict=False):
        downloaded = atomic_download(record.url, record.path, downloader=downloader)
        return DownloadRecord(
            pdb_id=record.pdb_id,
            url=record.url,
            path=record.path,
            status="downloaded",
            bytes=downloaded.stat().st_size,
            sha256=sha256_file(downloaded),
        )

    staged = atomic_download(record.url, staged_path, downloader=downloader)
    staged_sha256 = sha256_file(staged)
    promoted = copy_verified_file(staged, record.path, expected_sha256=staged_sha256)
    if cleanup_staging:
        staged.unlink(missing_ok=True)
    return DownloadRecord(
        pdb_id=record.pdb_id,
        url=record.url,
        path=record.path,
        status="downloaded",
        bytes=promoted.stat().st_size,
        sha256=sha256_file(promoted),
    )


def atomic_download(url: str, output: Path, *, downloader: Downloader) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".part",
        dir=output.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        downloader(url, temp_path)
        temp_path.replace(output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return output


def copy_verified_file(source: Path, target: Path, *, expected_sha256: Optional[str] = None) -> Path:
    """Copy a file through a same-directory temp file and verify its checksum before publish."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copy2(source, temp_path)
        actual_sha256 = sha256_file(temp_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                f"sha256 mismatch while copying {source} to {target}: "
                f"expected {expected_sha256}, observed {actual_sha256}"
            )
        temp_path.replace(target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def urlretrieve_downloader(url: str, output: Path) -> None:
    from urllib.request import urlretrieve

    urlretrieve(url, str(output))  # nosec B310 - user-facing archival utility.


def write_files_manifest(records: Sequence[DownloadRecord], manifest_dir: Path) -> Path:
    output = Path(manifest_dir) / "files.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_download_record_to_dict(record), sort_keys=True) + "\n")
    return output


def write_ids(ids: Sequence[str], manifest_dir: Path) -> Path:
    output = Path(manifest_dir) / "ids.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{pdb_id}\n" for pdb_id in ids), encoding="utf-8")
    return output


def write_filters_manifest(payload: Mapping[str, Any], manifest_dir: Path) -> Path:
    output = Path(manifest_dir) / "filters.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_snapshot_metadata(
    snapshot_id: str,
    *,
    paths: ArchivePaths,
    source: str = "auto",
    output: Optional[Path] = None,
    manifest_output: Optional[Path] = None,
    failures_output: Optional[Path] = None,
    max_entries: Optional[int] = None,
    batch_size: int = 100,
    workers: Any = "auto",
    overwrite: bool = False,
    fetcher: Fetcher = fetch_http_json_bytes,
    status_stream: Optional[StatusStream] = sys.stderr,
) -> dict[str, Any]:
    """Build compact, normalized entry metadata for a local coordinate snapshot."""

    snapshot = get_snapshot(snapshot_id)
    if source not in SNAPSHOT_METADATA_SOURCES:
        known = ", ".join(SNAPSHOT_METADATA_SOURCES)
        raise ValueError(f"Unsupported metadata source '{source}'. Known sources: {known}")
    manifest_path = paths.manifest_dir / "files.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Snapshot manifest not found: {manifest_path}")
    metadata_path = output or paths.manifest_dir / "metadata.jsonl"
    metadata_manifest_path = manifest_output or paths.manifest_dir / "metadata-manifest.json"
    metadata_failures_path = failures_output or paths.manifest_dir / "metadata-failures.jsonl"
    _ensure_can_write(metadata_path, overwrite=overwrite)
    _ensure_can_write(metadata_manifest_path, overwrite=overwrite)
    _ensure_can_write(metadata_failures_path, overwrite=overwrite)

    rows = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row.get("status", "")).strip().lower() in {"downloaded", "present", "reused", "verified"}
    ]
    if max_entries is not None:
        rows = rows[:max_entries]
    requested_ids = [str(row.get("pdb_id") or "").upper() for row in rows if row.get("pdb_id")]
    _status_print(
        status_stream,
        f"building {snapshot.id} metadata for {len(requested_ids)} entries using source {source}",
    )
    source_used = source
    api_error = None
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_failures_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temp_path = _temp_output_path(metadata_path)
    failures_temp_path = _temp_output_path(metadata_failures_path)
    metadata_temp_path.unlink(missing_ok=True)
    failures_temp_path.unlink(missing_ok=True)
    if source == "bcif-local":
        metadata_rows, failures = _metadata_from_local_snapshot(
            rows,
            snapshot=snapshot,
            output_root=paths.output_root,
            workers=workers,
        )
        _write_metadata_rows(metadata_temp_path, metadata_rows)
        _write_metadata_rows(failures_temp_path, failures)
        entry_count = len(metadata_rows)
        failure_count = len(failures)
    else:
        try:
            counts = _write_metadata_from_rcsb_api(
                requested_ids,
                snapshot=snapshot,
                fetcher=fetcher,
                batch_size=batch_size,
                metadata_path=metadata_temp_path,
                failures_path=failures_temp_path,
                status_stream=status_stream,
            )
            source_used = "rcsb-api"
            entry_count = counts["entry_count"]
            failure_count = counts["failure_count"]
        except Exception as exc:
            metadata_temp_path.unlink(missing_ok=True)
            failures_temp_path.unlink(missing_ok=True)
            if source == "rcsb-api":
                raise
            api_error = str(exc)
            _status_print(status_stream, f"RCSB API metadata failed; falling back to local BCIF: {exc}")
            metadata_rows, failures = _metadata_from_local_snapshot(
                rows,
                snapshot=snapshot,
                output_root=paths.output_root,
                workers=workers,
            )
            source_used = "bcif-local"
            _write_metadata_rows(metadata_temp_path, metadata_rows)
            _write_metadata_rows(failures_temp_path, failures)
            entry_count = len(metadata_rows)
            failure_count = len(failures)
    metadata_temp_path.replace(metadata_path)
    failures_temp_path.replace(metadata_failures_path)
    summary = {
        "format": "pheat.archive-snapshot-metadata",
        "version": 1,
        "snapshot_id": snapshot.id,
        "snapshot_format": snapshot.file_format,
        "created_at": _utc_now(),
        "source_requested": source,
        "source_used": source_used,
        "api_error": api_error,
        "metadata_source_urls": [DATA_GRAPHQL_URL] if source_used == "rcsb-api" else [],
        "output_root": str(paths.output_root),
        "files_manifest": str(manifest_path),
        "files_manifest_sha256": sha256_file(manifest_path),
        "metadata_jsonl": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "failures_jsonl": str(metadata_failures_path),
        "failure_count": failure_count,
        "entry_count": entry_count,
        "requested_count": len(requested_ids),
        "batch_size": int(batch_size),
        "workers": _normalize_worker_count(workers),
        "kept_fields": [
            "method",
            "resolution",
            "dates",
            "quality",
            "composition",
            "polymer_entities",
            "sequence_clusters",
            "environment",
        ],
        "omitted_by_default": [
            "atom_site_coordinates",
            "raw_api_payloads",
            "schema_bodies",
            "full_citations",
            "per_residue_validation",
            "full_crystallization_text",
        ],
    }
    metadata_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def relocate_snapshot_manifest(
    snapshot_id: str,
    *,
    paths: ArchivePaths,
    write: bool = False,
    relative_paths: bool = True,
    backup: bool = False,
) -> dict[str, Any]:
    """Rewrite stale archive file paths so a moved snapshot is usable in-place."""

    snapshot = get_snapshot(snapshot_id)
    manifest_path = paths.manifest_dir / "files.jsonl"
    filters_path = paths.manifest_dir / "filters.json"
    if not manifest_path.exists():
        raise ValueError(f"Snapshot manifest not found: {manifest_path}")
    rows = _read_jsonl(manifest_path)
    rewritten = []
    unchanged = 0
    missing = 0
    changed = 0
    for row in rows:
        updated = dict(row)
        resolved = resolve_manifest_row_path(row, output_root=paths.output_root, raw_dir=paths.raw_dir)
        if resolved is None or not resolved.exists():
            missing += 1
            rewritten.append(updated)
            continue
        new_path = _manifest_path_text(resolved, output_root=paths.output_root, relative=relative_paths)
        if updated.get("path") != new_path:
            updated["path"] = new_path
            changed += 1
        else:
            unchanged += 1
        source_path = updated.get("source_path")
        if source_path:
            source = resolve_path_text(str(source_path), output_root=paths.output_root)
            if source is not None and source.exists():
                updated["source_path"] = _manifest_path_text(source, output_root=paths.output_root, relative=relative_paths)
        rewritten.append(updated)
    filters = _relocated_filters(filters_path, paths=paths, relative_paths=relative_paths) if filters_path.exists() else None
    if write:
        if backup:
            backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
            shutil.copy2(manifest_path, backup_path)
            if filters_path.exists():
                shutil.copy2(filters_path, filters_path.with_suffix(filters_path.suffix + ".bak"))
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rewritten:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if filters is not None:
            filters_path.write_text(json.dumps(filters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "snapshot_id": snapshot.id,
        "manifest_path": str(manifest_path),
        "output_root": str(paths.output_root),
        "write": write,
        "relative_paths": relative_paths,
        "total": len(rows),
        "changed": changed,
        "unchanged": unchanged,
        "missing": missing,
        "filters_rewritten": bool(write and filters is not None),
    }


def _prefetch_snapshot_metadata_after_download(
    *,
    paths: ArchivePaths,
    filter_metadata: Optional[Mapping[str, Any]],
    metadata_source: str,
    metadata_batch_size: int,
    metadata_workers: Any,
    fetcher: Fetcher,
    status_stream: Optional[StatusStream],
) -> dict[str, Any]:
    snapshot_id = str((filter_metadata or {}).get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise ValueError("--prefetch-metadata is only supported for named snapshot downloads")
    _status_print(status_stream, f"prefetching {snapshot_id} metadata into {paths.manifest_dir / 'metadata.jsonl'}...")
    return write_snapshot_metadata(
        snapshot_id,
        paths=paths,
        source=metadata_source,
        batch_size=metadata_batch_size,
        workers=metadata_workers,
        overwrite=True,
        fetcher=fetcher,
        status_stream=status_stream,
    )


def resolve_path_text(path_text: str, *, output_root: Path) -> Optional[Path]:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(output_root) / path


def resolve_manifest_row_path(
    row: Mapping[str, Any],
    *,
    output_root: Path,
    raw_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve absolute, relative, and stale moved-snapshot manifest paths."""

    root = Path(output_root)
    raw = Path(raw_dir) if raw_dir is not None else root / "raw"
    for key in ("path", "source_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        candidates = [path if path.is_absolute() else root / path]
        if path.name:
            candidates.append(raw / path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
    value = row.get("pdb_id")
    if value:
        for extension in ("bcif.gz", "cif.gz", "pdb.gz", "pdb", "cif", "bcif"):
            candidate = raw / f"{str(value).lower()}.{extension}"
            if candidate.exists():
                return candidate
    path_value = row.get("path")
    if path_value:
        path = Path(str(path_value))
        return path if path.is_absolute() else root / path
    return None


def _metadata_from_rcsb_api(
    ids: Sequence[str],
    *,
    snapshot: SnapshotSpec,
    fetcher: Fetcher,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ids = [normalize_pdb_id(pdb_id) for pdb_id in ids]
    if batch_size <= 0:
        raise ValueError("metadata batch size must be positive")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch in _batched(normalized_ids, batch_size):
        batch_rows, batch_failures = _metadata_from_rcsb_api_batch(
            batch,
            snapshot=snapshot,
            fetcher=fetcher,
            batch_size=batch_size,
        )
        rows.extend(batch_rows)
        failures.extend(batch_failures)
    return rows, failures


def _write_metadata_from_rcsb_api(
    ids: Sequence[str],
    *,
    snapshot: SnapshotSpec,
    fetcher: Fetcher,
    batch_size: int,
    metadata_path: Path,
    failures_path: Path,
    status_stream: Optional[StatusStream],
) -> dict[str, int]:
    normalized_ids = [normalize_pdb_id(pdb_id) for pdb_id in ids]
    if batch_size <= 0:
        raise ValueError("metadata batch size must be positive")
    total = len(normalized_ids)
    processed = 0
    entry_count = 0
    failure_count = 0
    batch_count = math.ceil(total / batch_size) if total else 0
    started_at = monotonic()
    last_progress_at = started_at
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    _metadata_progress(
        status_stream,
        processed=processed,
        total=total,
        batches_done=0,
        batch_count=batch_count,
        entry_count=entry_count,
        failure_count=failure_count,
        started_at=started_at,
    )
    with metadata_path.open("w", encoding="utf-8") as metadata_handle, failures_path.open("w", encoding="utf-8") as failures_handle:
        for batch_index, batch in enumerate(_batched(normalized_ids, batch_size), start=1):
            batch_rows, batch_failures = _metadata_from_rcsb_api_batch(
                batch,
                snapshot=snapshot,
                fetcher=fetcher,
                batch_size=batch_size,
            )
            for row in batch_rows:
                metadata_handle.write(json.dumps(row, sort_keys=True) + "\n")
            for failure in batch_failures:
                failures_handle.write(json.dumps(failure, sort_keys=True) + "\n")
            metadata_handle.flush()
            failures_handle.flush()
            processed += len(batch)
            entry_count += len(batch_rows)
            failure_count += len(batch_failures)
            now = monotonic()
            if batch_index == 1 or batch_index == batch_count or batch_index % 10 == 0 or now - last_progress_at >= 30.0:
                _metadata_progress(
                    status_stream,
                    processed=processed,
                    total=total,
                    batches_done=batch_index,
                    batch_count=batch_count,
                    entry_count=entry_count,
                    failure_count=failure_count,
                    started_at=started_at,
                )
                last_progress_at = now
    return {"entry_count": entry_count, "failure_count": failure_count}


def _metadata_from_rcsb_api_batch(
    ids: Sequence[str],
    *,
    snapshot: SnapshotSpec,
    fetcher: Fetcher,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ids = [normalize_pdb_id(pdb_id) for pdb_id in ids]
    entries_by_id: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    payload = _graphql_fetch(fetcher, ENTRY_METADATA_QUERY, {"ids": normalized_ids})
    for error in payload.get("errors", []) if isinstance(payload, Mapping) else []:
        failures.append({"source": "rcsb-api", "scope": "entry", "error": str(error), "pdb_ids": list(normalized_ids)})
    data = payload.get("data") if isinstance(payload, Mapping) else None
    for entry in (data or {}).get("entries") or []:
        if isinstance(entry, Mapping):
            pdb_id = str(entry.get("rcsb_id") or _nested(entry, "entry", "id") or "").upper()
            if pdb_id:
                entries_by_id[pdb_id] = entry
    polymer_entity_ids = sorted(
        {
            f"{pdb_id}_{entity_id}"
            for pdb_id, entry in entries_by_id.items()
            for entity_id in _as_list(_nested(entry, "rcsb_entry_container_identifiers", "polymer_entity_ids"))
            if entity_id is not None
        }
    )
    polymer_by_entry: dict[str, list[Mapping[str, Any]]] = {}
    for entity_batch in _batched(polymer_entity_ids, max(1, batch_size * 2)):
        payload = _graphql_fetch(fetcher, POLYMER_ENTITY_METADATA_QUERY, {"ids": entity_batch})
        for error in payload.get("errors", []) if isinstance(payload, Mapping) else []:
            failures.append({"source": "rcsb-api", "scope": "polymer_entity", "error": str(error), "entity_ids": list(entity_batch)})
        data = payload.get("data") if isinstance(payload, Mapping) else None
        for entity in (data or {}).get("polymer_entities") or []:
            if not isinstance(entity, Mapping):
                continue
            entry_id = str(_nested(entity, "rcsb_polymer_entity_container_identifiers", "entry_id") or "").upper()
            if entry_id:
                polymer_by_entry.setdefault(entry_id, []).append(entity)
    rows = []
    for pdb_id in normalized_ids:
        entry = entries_by_id.get(pdb_id)
        if not entry:
            failures.append({"source": "rcsb-api", "scope": "entry", "pdb_id": pdb_id, "error": "entry metadata missing"})
            continue
        rows.append(_metadata_row_from_api_entry(entry, polymer_by_entry.get(pdb_id, []), snapshot=snapshot))
    return rows, failures


def _metadata_from_local_snapshot(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot: SnapshotSpec,
    output_root: Path,
    workers: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    worker_count = _normalize_worker_count(workers)
    raw_dir = Path(output_root) / "raw"
    tasks = [(dict(row), snapshot.file_format, str(output_root), str(raw_dir), snapshot.id) for row in rows]
    if worker_count == 1:
        results = [_metadata_from_local_manifest_row(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(_metadata_from_local_manifest_row, tasks))
    metadata_rows = []
    failures = []
    for result in results:
        if result.get("ok"):
            metadata_rows.append(result["metadata"])
        else:
            failures.append(result["failure"])
    return metadata_rows, failures


def _metadata_from_local_manifest_row(task: tuple[dict[str, Any], str, str, str, str]) -> dict[str, Any]:
    row, file_format, output_root_text, raw_dir_text, snapshot_id = task
    pdb_id = str(row.get("pdb_id") or "").upper()
    path = resolve_manifest_row_path(row, output_root=Path(output_root_text), raw_dir=Path(raw_dir_text))
    if path is None or not path.exists():
        return {
            "ok": False,
            "failure": {"source": "bcif-local", "pdb_id": pdb_id, "error": "coordinate file missing"},
        }
    if file_format != "bcif":
        return {
            "ok": False,
            "failure": {
                "source": "bcif-local",
                "pdb_id": pdb_id,
                "path": str(path),
                "error": f"local metadata extraction currently supports bcif snapshots, not {file_format}",
            },
        }
    try:
        return {"ok": True, "metadata": _metadata_row_from_bcif(path, pdb_id=pdb_id, snapshot_id=snapshot_id)}
    except Exception as exc:
        return {
            "ok": False,
            "failure": {"source": "bcif-local", "pdb_id": pdb_id, "path": str(path), "error": str(exc)},
        }


def _metadata_row_from_api_entry(
    entry: Mapping[str, Any],
    polymer_entities: Sequence[Mapping[str, Any]],
    *,
    snapshot: SnapshotSpec,
) -> dict[str, Any]:
    pdb_id = str(entry.get("rcsb_id") or _nested(entry, "entry", "id") or "").upper()
    entry_info = _mapping(entry.get("rcsb_entry_info"))
    accession = _mapping(entry.get("rcsb_accession_info"))
    status = _mapping(entry.get("pdbx_database_status"))
    methods = [
        str(item.get("method"))
        for item in _as_list(entry.get("exptl"))
        if isinstance(item, Mapping) and item.get("method")
    ]
    experimental_method = entry_info.get("experimental_method") or _first(methods)
    resolution = _first_float(entry_info.get("resolution_combined"))
    if resolution is None:
        resolution = _first_float([item.get("ls_d_res_high") for item in _as_mapping_list(entry.get("refine"))])
    quality = _quality_from_api(entry)
    polymer_rows = [_polymer_entity_from_api(item) for item in polymer_entities if isinstance(item, Mapping)]
    sequence_clusters = _sequence_clusters_from_polymer_entities(polymer_entities)
    text_sources = _api_text_sources(entry, polymer_rows)
    environment = _environment_from_text(
        text_sources,
        methodology=str(entry_info.get("structure_determination_methodology") or ""),
        method=str(experimental_method or ""),
    )
    return _clean_metadata_row(
        {
            "format": "pheat.archive-entry-metadata",
            "version": 1,
            "pdb_id": pdb_id,
            "metadata_source": "rcsb-api",
            "metadata_retrieved_at": _utc_now(),
            "source_snapshot_id": snapshot.id,
            "method": _normalize_method(experimental_method or _first(methods)),
            "experimental_method": experimental_method,
            "experimental_methods": methods,
            "resolution": resolution,
            "resolution_angstrom": resolution,
            "dates": {
                "deposit": _date_only(accession.get("deposit_date") or status.get("recvd_initial_deposition_date")),
                "initial_release": _date_only(accession.get("initial_release_date")),
                "revision": _date_only(accession.get("revision_date")),
            },
            "status": {
                "code": accession.get("status_code") or status.get("status_code"),
                "major_revision": accession.get("major_revision"),
                "minor_revision": accession.get("minor_revision"),
                "pdb_format_compatible": status.get("pdb_format_compatible"),
            },
            "quality": quality,
            "composition": _composition_from_api(entry),
            "polymer_entities": polymer_rows,
            "sequence_clusters": sequence_clusters,
            "environment": environment,
            "title": _nested(entry, "struct", "title"),
            "keywords": {
                "text": _nested(entry, "struct_keywords", "text"),
                "category": _nested(entry, "struct_keywords", "pdbx_keywords"),
            },
        }
    )


def _metadata_row_from_bcif(path: Path, *, pdb_id: str, snapshot_id: str) -> dict[str, Any]:
    categories = _bcif_categories(path)
    entry_id = _cat_first(categories, "_entry", "id") or pdb_id or path.stem.split(".", 1)[0].upper()
    exptl_methods = [str(value) for value in _cat_values(categories, "_exptl", "method") if value]
    method = _normalize_method(_first(exptl_methods))
    resolution = _first_float(
        _cat_values(categories, "_refine", "ls_d_res_high")
        or _cat_values(categories, "_refine_hist", "d_res_high")
    )
    polymer_entities = _polymer_entities_from_bcif(categories)
    text_sources = _bcif_text_sources(categories, polymer_entities)
    environment = _environment_from_text(text_sources, methodology="", method=method or "")
    return _clean_metadata_row(
        {
            "format": "pheat.archive-entry-metadata",
            "version": 1,
            "pdb_id": str(entry_id).upper(),
            "metadata_source": "bcif-local",
            "metadata_retrieved_at": _utc_now(),
            "source_snapshot_id": snapshot_id,
            "method": method,
            "experimental_method": _first(exptl_methods),
            "experimental_methods": exptl_methods,
            "resolution": resolution,
            "resolution_angstrom": resolution,
            "dates": {
                "deposit": _date_only(_cat_first(categories, "_pdbx_database_status", "recvd_initial_deposition_date")),
                "initial_release": _date_only(_last(_cat_values(categories, "_pdbx_audit_revision_history", "revision_date"))),
                "revision": _date_only(_last(_cat_values(categories, "_pdbx_audit_revision_history", "revision_date"))),
            },
            "status": {
                "code": _cat_first(categories, "_pdbx_database_status", "status_code"),
                "pdb_format_compatible": _cat_first(categories, "_pdbx_database_status", "pdb_format_compatible"),
            },
            "quality": _quality_from_bcif(categories),
            "composition": _composition_from_bcif(categories),
            "polymer_entities": polymer_entities,
            "sequence_clusters": {},
            "environment": environment,
            "title": _cat_first(categories, "_struct", "title"),
            "keywords": {
                "text": _cat_first(categories, "_struct_keywords", "text"),
                "category": _cat_first(categories, "_struct_keywords", "pdbx_keywords"),
            },
        }
    )


def _graphql_fetch(fetcher: Fetcher, query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = fetcher(DATA_GRAPHQL_URL, {"query": query, "variables": dict(variables)})
    decoded = json.loads(payload.body.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("RCSB GraphQL endpoint did not return a JSON object")
    return decoded


def _quality_from_api(entry: Mapping[str, Any]) -> dict[str, Any]:
    refine = _first_mapping(entry.get("refine"))
    geometry = _first_mapping(entry.get("pdbx_vrpt_summary_geometry"))
    diffraction = _first_mapping(entry.get("pdbx_vrpt_summary_diffraction"))
    return _clean_dict(
        {
            "r_work": _optional_float(refine.get("ls_R_factor_R_work")),
            "r_free": _optional_float(refine.get("ls_R_factor_R_free")),
            "r_obs": _optional_float(refine.get("ls_R_factor_obs")),
            "b_iso_mean": _optional_float(refine.get("B_iso_mean")),
            "refine_resolution_high": _optional_float(refine.get("ls_d_res_high")),
            "refine_resolution_low": _optional_float(refine.get("ls_d_res_low")),
            "clashscore": _optional_float(geometry.get("clashscore")),
            "ramachandran_outliers_percent": _optional_float(geometry.get("percent_ramachandran_outliers")),
            "rotamer_outliers_percent": _optional_float(geometry.get("percent_rotamer_outliers")),
            "bond_rmsz": _optional_float(geometry.get("bonds_RMSZ")),
            "angle_rmsz": _optional_float(geometry.get("angles_RMSZ")),
            "data_completeness_percent": _optional_float(diffraction.get("data_completeness")),
            "rsrz_outliers_percent": _optional_float(diffraction.get("percent_RSRZ_outliers")),
            "density_r": _optional_float(diffraction.get("EDS_R")),
            "dcc_r": _optional_float(diffraction.get("DCC_R")),
            "dcc_rfree": _optional_float(diffraction.get("DCC_Rfree")),
        }
    )


def _quality_from_bcif(categories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return _clean_dict(
        {
            "r_work": _first_float(_cat_values(categories, "_refine", "ls_R_factor_R_work")),
            "r_free": _first_float(_cat_values(categories, "_refine", "ls_R_factor_R_free")),
            "r_obs": _first_float(_cat_values(categories, "_refine", "ls_R_factor_obs")),
            "b_iso_mean": _first_float(_cat_values(categories, "_refine", "B_iso_mean")),
            "refine_resolution_high": _first_float(_cat_values(categories, "_refine", "ls_d_res_high")),
            "refine_resolution_low": _first_float(_cat_values(categories, "_refine", "ls_d_res_low")),
            "clashscore": _first_float(_cat_values(categories, "_pdbx_vrpt_summary_geometry", "clashscore")),
            "ramachandran_outliers_percent": _first_float(
                _cat_values(categories, "_pdbx_vrpt_summary_geometry", "percent_ramachandran_outliers")
            ),
            "rotamer_outliers_percent": _first_float(
                _cat_values(categories, "_pdbx_vrpt_summary_geometry", "percent_rotamer_outliers")
            ),
            "bond_rmsz": _first_float(_cat_values(categories, "_pdbx_vrpt_summary_geometry", "bonds_RMSZ")),
            "angle_rmsz": _first_float(_cat_values(categories, "_pdbx_vrpt_summary_geometry", "angles_RMSZ")),
            "data_completeness_percent": _first_float(
                _cat_values(categories, "_pdbx_vrpt_summary_diffraction", "data_completeness")
            ),
            "rsrz_outliers_percent": _first_float(
                _cat_values(categories, "_pdbx_vrpt_summary_diffraction", "percent_RSRZ_outliers")
            ),
        }
    )


def _composition_from_api(entry: Mapping[str, Any]) -> dict[str, Any]:
    info = _mapping(entry.get("rcsb_entry_info"))
    identifiers = _mapping(entry.get("rcsb_entry_container_identifiers"))
    return _clean_dict(
        {
            "selected_polymer_entity_types": info.get("selected_polymer_entity_types"),
            "polymer_composition": info.get("polymer_composition"),
            "structure_determination_methodology": info.get("structure_determination_methodology"),
            "atom_count": _optional_int(info.get("deposited_atom_count")),
            "hydrogen_atom_count": _optional_int(info.get("deposited_hydrogen_atom_count")),
            "model_count": _optional_int(info.get("deposited_model_count")),
            "polymer_entity_count": _optional_int(info.get("polymer_entity_count")),
            "protein_entity_count": _optional_int(info.get("polymer_entity_count_protein")),
            "nonpolymer_entity_count": _optional_int(info.get("nonpolymer_entity_count")),
            "solvent_entity_count": _optional_int(info.get("solvent_entity_count")),
            "solvent_atom_count": _optional_int(info.get("deposited_solvent_atom_count")),
            "assembly_count": _optional_int(info.get("assembly_count")),
            "cis_peptide_count": _optional_int(info.get("cis_peptide_count")),
            "disulfide_bond_count": _optional_int(info.get("disulfide_bond_count")),
            "metal_bond_count": _optional_int(info.get("inter_mol_metalic_bond_count")),
            "covalent_bond_count": _optional_int(info.get("inter_mol_covalent_bond_count")),
            "polymer_entity_ids": [str(item) for item in _as_list(identifiers.get("polymer_entity_ids"))],
            "nonpolymer_entity_ids": [str(item) for item in _as_list(identifiers.get("non_polymer_entity_ids"))],
            "assembly_ids": [str(item) for item in _as_list(identifiers.get("assembly_ids"))],
        }
    )


def _composition_from_bcif(categories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    entity_types = [str(item).lower() for item in _cat_values(categories, "_entity", "type") if item]
    entity_ids = [str(item) for item in _cat_values(categories, "_entity", "id") if item]
    polymer_entities = _polymer_entities_from_bcif(categories)
    model_numbers = set(_cat_values(categories, "_atom_site", "pdbx_PDB_model_num"))
    return _clean_dict(
        {
            "selected_polymer_entity_types": "Protein (only)"
            if polymer_entities and all(str(item.get("type", "")).lower().startswith("polypeptide") for item in polymer_entities)
            else None,
            "atom_count": len(_cat_values(categories, "_atom_site", "id")),
            "model_count": len([item for item in model_numbers if item is not None]) or None,
            "polymer_entity_count": entity_types.count("polymer"),
            "protein_entity_count": sum(
                1 for item in polymer_entities if str(item.get("polymer_type", "")).lower() == "protein"
            ),
            "nonpolymer_entity_count": entity_types.count("non-polymer"),
            "solvent_entity_count": entity_types.count("water"),
            "polymer_entity_ids": [str(item.get("entity_id")) for item in polymer_entities if item.get("entity_id")],
            "entity_ids": entity_ids,
        }
    )


def _polymer_entity_from_api(entity: Mapping[str, Any]) -> dict[str, Any]:
    identifiers = _mapping(entity.get("rcsb_polymer_entity_container_identifiers"))
    entity_poly = _mapping(entity.get("entity_poly"))
    description = _mapping(entity.get("rcsb_polymer_entity"))
    organisms = [
        _clean_dict(
            {
                "scientific_name": item.get("ncbi_scientific_name"),
                "taxonomy_id": _optional_int(item.get("ncbi_taxonomy_id")),
                "source_type": item.get("source_type"),
            }
        )
        for item in _as_mapping_list(entity.get("rcsb_entity_source_organism"))
    ]
    return _clean_dict(
        {
            "entity_id": identifiers.get("entity_id"),
            "rcsb_id": entity.get("rcsb_id"),
            "asym_ids": [str(item) for item in _as_list(identifiers.get("asym_ids"))],
            "auth_asym_ids": [str(item) for item in _as_list(identifiers.get("auth_asym_ids"))],
            "type": entity_poly.get("type"),
            "polymer_type": entity_poly.get("rcsb_entity_polymer_type"),
            "sequence": _compact_sequence(entity_poly.get("pdbx_seq_one_letter_code_can")),
            "sequence_length": _optional_int(entity_poly.get("rcsb_sample_sequence_length")),
            "nonstandard_monomer": _yes_no_bool(entity_poly.get("nstd_monomer")),
            "nonstandard_linkage": _yes_no_bool(entity_poly.get("nstd_linkage")),
            "description": description.get("pdbx_description"),
            "organisms": [item for item in organisms if item],
        }
    )


def _polymer_entities_from_bcif(categories: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    entity_ids = [str(item) for item in _cat_values(categories, "_entity_poly", "entity_id") if item]
    types = _cat_values(categories, "_entity_poly", "type")
    strands = _cat_values(categories, "_entity_poly", "pdbx_strand_id")
    sequences = _cat_values(categories, "_entity_poly", "pdbx_seq_one_letter_code_can")
    nstd_monomers = _cat_values(categories, "_entity_poly", "nstd_monomer")
    nstd_linkages = _cat_values(categories, "_entity_poly", "nstd_linkage")
    descriptions_by_entity = {
        str(entity_id): description
        for entity_id, description in zip(
            _cat_values(categories, "_entity", "id"),
            _cat_values(categories, "_entity", "pdbx_description"),
        )
        if entity_id
    }
    rows = []
    for index, entity_id in enumerate(entity_ids):
        sequence = _compact_sequence(_indexed(sequences, index))
        rows.append(
            _clean_dict(
                {
                    "entity_id": entity_id,
                    "asym_ids": [item.strip() for item in str(_indexed(strands, index) or "").split(",") if item.strip()],
                    "type": _indexed(types, index),
                    "polymer_type": "Protein"
                    if str(_indexed(types, index) or "").lower().startswith("polypeptide")
                    else None,
                    "sequence": sequence,
                    "sequence_length": len(sequence) if sequence else None,
                    "nonstandard_monomer": _yes_no_bool(_indexed(nstd_monomers, index)),
                    "nonstandard_linkage": _yes_no_bool(_indexed(nstd_linkages, index)),
                    "description": descriptions_by_entity.get(entity_id),
                }
            )
        )
    return rows


def _sequence_clusters_from_polymer_entities(entities: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for entity in entities:
        for membership in _as_mapping_list(entity.get("rcsb_polymer_entity_group_membership")):
            if str(membership.get("aggregation_method") or "") != "sequence_identity":
                continue
            cutoff = _optional_float(membership.get("similarity_cutoff"))
            group_id = membership.get("group_id")
            if cutoff is None or group_id is None:
                continue
            key = _percent_key(cutoff)
            clusters.setdefault(key, [])
            if str(group_id) not in clusters[key]:
                clusters[key].append(str(group_id))
    return {key: sorted(values) for key, values in sorted(clusters.items())}


def _api_text_sources(entry: Mapping[str, Any], polymer_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    texts = [
        _nested(entry, "struct", "title"),
        _nested(entry, "struct_keywords", "text"),
        _nested(entry, "struct_keywords", "pdbx_keywords"),
    ]
    for grow in _as_mapping_list(entry.get("exptl_crystal_grow")):
        texts.extend([grow.get("method"), grow.get("pdbx_details")])
    for polymer in polymer_rows:
        texts.extend([polymer.get("description"), polymer.get("polymer_type"), polymer.get("type")])
    return [str(text) for text in texts if text]


def _bcif_text_sources(
    categories: Mapping[str, Mapping[str, Any]],
    polymer_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    texts = [
        _cat_first(categories, "_struct", "title"),
        _cat_first(categories, "_struct_keywords", "text"),
        _cat_first(categories, "_struct_keywords", "pdbx_keywords"),
    ]
    texts.extend(_cat_values(categories, "_exptl_crystal_grow", "method"))
    texts.extend(_cat_values(categories, "_exptl_crystal_grow", "pdbx_details"))
    for polymer in polymer_rows:
        texts.extend([polymer.get("description"), polymer.get("polymer_type"), polymer.get("type")])
    return [str(text) for text in texts if text]


def _environment_from_text(texts: Sequence[str], *, methodology: str, method: str) -> dict[str, Any]:
    joined = " ".join(texts).lower()
    membrane_terms = [
        "membrane",
        "transmembrane",
        "lipid",
        "detergent",
        "micelle",
        "nanodisc",
        "amphipol",
    ]
    aqueous_terms = [
        "aqueous",
        "water",
        "buffer",
        "tris",
        "hepes",
        "mes",
        "mops",
        "phosphate",
        "nacl",
        "sodium chloride",
        "potassium chloride",
        "crystal grow",
    ]
    membrane_hits = sorted({term for term in membrane_terms if term in joined})
    aqueous_hits = sorted({term for term in aqueous_terms if term in joined})
    methodology_text = str(methodology or "").lower()
    method_text = str(method or "").lower()
    return _clean_dict(
        {
            "membrane": bool(membrane_hits),
            "aqueous_like": bool(aqueous_hits) and not bool(membrane_hits),
            "computed_model": "computational" in methodology_text or "computed" in method_text,
            "integrative_model": "integrative" in methodology_text or "integrative" in method_text,
            "evidence": {
                "membrane_terms": membrane_hits,
                "aqueous_terms": aqueous_hits,
            },
        }
    )


def _bcif_categories(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError(
            "Local BinaryCIF metadata extraction requires msgpack. Install PHEAT with `.[scientific]` or `.[all]`."
        ) from exc
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        payload = msgpack.unpackb(handle.read(), raw=False)
    blocks = payload.get("dataBlocks") if isinstance(payload, Mapping) else None
    if not blocks:
        raise ValueError(f"BinaryCIF file has no data blocks: {path}")
    return {category["name"]: category for category in blocks[0].get("categories", [])}


def _decode_bcif_category(category: Mapping[str, Any]) -> dict[str, list[Any]]:
    row_count = category.get("rowCount")
    output = {}
    for column in category.get("columns", []):
        name = str(column.get("name") or "")
        if not name:
            continue
        output[name] = _decode_bcif_column(column, row_count=row_count)
    return output


def _decode_bcif_column(column: Mapping[str, Any], *, row_count: Any = None) -> list[Any]:
    return decode_bcif_column(column, row_count=row_count)


def _cat_values(categories: Mapping[str, Mapping[str, Any]], category_name: str, column_name: str) -> list[Any]:
    category = categories.get(category_name)
    if category is None:
        return []
    decoded = _decode_bcif_category(category)
    return [value for value in decoded.get(column_name, []) if value is not None]


def _cat_first(categories: Mapping[str, Mapping[str, Any]], category_name: str, column_name: str) -> Any:
    return _first(_cat_values(categories, category_name, column_name))


def _relocated_filters(filters_path: Path, *, paths: ArchivePaths, relative_paths: bool) -> dict[str, Any]:
    filters = json.loads(filters_path.read_text(encoding="utf-8"))
    if not isinstance(filters, dict):
        return {}
    output = dict(filters)
    output["output_root"] = "." if relative_paths else str(paths.output_root)
    output["raw_dir"] = "raw" if relative_paths else str(paths.raw_dir)
    output["manifest_dir"] = "manifests" if relative_paths else str(paths.manifest_dir)
    if output.get("staging_dir"):
        staging = Path(str(output["staging_dir"]))
        if not staging.is_absolute():
            output["staging_dir"] = str(staging)
    return output


def _manifest_path_text(path: Path, *, output_root: Path, relative: bool) -> str:
    if relative:
        try:
            return str(path.resolve().relative_to(Path(output_root).resolve()))
        except ValueError:
            return str(path)
    return str(path)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise ValueError(f"Output already exists: {path}. Use --overwrite to replace it.")


def _normalize_worker_count(value: Any) -> int:
    if str(value).strip().lower() == "auto":
        try:
            import os

            return max(1, os.cpu_count() or 1)
        except Exception:
            return 1
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be positive or 'auto'")
    return workers


def _temp_output_path(path: Path) -> Path:
    return Path(path).with_name(f".{Path(path).name}.part")


def _write_metadata_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _metadata_progress(
    stream: Optional[StatusStream],
    *,
    processed: int,
    total: int,
    batches_done: int,
    batch_count: int,
    entry_count: int,
    failure_count: int,
    started_at: float,
) -> None:
    elapsed = max(0.0, monotonic() - started_at)
    rate = processed / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, total - processed)
    eta = (remaining / rate) if rate > 0.0 else None
    percent = (processed / total * 100.0) if total else 100.0
    _status_print(
        stream,
        "metadata progress: "
        f"{processed}/{total} entries ({percent:.1f}%), "
        f"batches {batches_done}/{batch_count}, "
        f"written {entry_count}, failed {failure_count}, "
        f"{rate:.2f} entries/s, elapsed {_format_duration(elapsed)}, ETA {_format_eta(eta)}",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
    except Exception:
        pass
    return [value]


def _as_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, Mapping)]


def _first(value: Any) -> Any:
    values = _as_list(value)
    return values[0] if values else None


def _last(value: Any) -> Any:
    values = _as_list(value)
    return values[-1] if values else None


def _first_mapping(value: Any) -> Mapping[str, Any]:
    for item in _as_list(value):
        if isinstance(item, Mapping):
            return item
    return {}


def _first_float(value: Any) -> Optional[float]:
    for item in _as_list(value):
        parsed = _optional_float(item)
        if parsed is not None:
            return parsed
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _indexed(values: Sequence[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _compact_sequence(value: Any) -> Optional[str]:
    if value is None:
        return None
    sequence = "".join(str(value).split())
    return sequence or None


def _yes_no_bool(value: Any) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None


def _percent_key(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _normalize_method(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if text else None


def _date_only(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _clean_metadata_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return _clean_dict(dict(row))


def _clean_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            nested = _clean_dict(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            nested_list: list[Any] = [
                _clean_dict(item) if isinstance(item, Mapping) else item
                for item in value
                if item not in (None, "")
            ]
            nested_list = [item for item in nested_list if item not in ({}, [])]
            if nested_list:
                cleaned[key] = nested_list
        elif value not in (None, ""):
            cleaned[key] = value
    return cleaned


def run_download(
    ids: Sequence[str],
    *,
    paths: ArchivePaths,
    file_format: str = "cif",
    reuse_raw_dirs: Sequence[Path] = (),
    reuse_mode: ReuseMode = "copy",
    staging_dir: Optional[Path] = None,
    cleanup_staging: bool = False,
    prefetch_metadata: bool = False,
    metadata_source: str = "auto",
    metadata_batch_size: int = 100,
    metadata_workers: Any = "auto",
    yes: bool = False,
    dry_run: bool = False,
    record_schema_provenance: bool = True,
    filter_metadata: Optional[Mapping[str, Any]] = None,
    fetcher: Fetcher = fetch_http_json_bytes,
    downloader: Downloader = urlretrieve_downloader,
    prompt_input: Callable[[str], str] = input,
    status_stream: Optional[StatusStream] = sys.stderr,
    progress_interval: int = 100,
    progress_seconds: Optional[float] = None,
    progress_redraw: bool = True,
    clock: Clock = monotonic,
) -> dict[str, Any]:
    ensure_archive_paths(paths)
    normalized_ids = sorted({normalize_pdb_id(item) for item in ids})
    records = plan_download_records(
        normalized_ids,
        paths,
        file_format=file_format,
        reuse_raw_dirs=reuse_raw_dirs,
    )
    pending_count = sum(1 for record in records if record.status == "pending")
    present_count = sum(1 for record in records if record.status == "present")
    reused_count = sum(1 for record in records if record.status == "reused")

    filter_payload = {
        "file_format": file_format,
        "output_root": str(paths.output_root),
        "raw_dir": str(paths.raw_dir),
        "manifest_dir": str(paths.manifest_dir),
        "reuse_raw_dirs": [str(path) for path in reuse_raw_dirs],
        "reuse_mode": reuse_mode,
        "staging_dir": str(staging_dir) if staging_dir is not None else None,
        "cleanup_staging": cleanup_staging,
        "prefetch_metadata": bool(prefetch_metadata),
        "metadata_source": metadata_source if prefetch_metadata else None,
        "metadata_batch_size": int(metadata_batch_size),
        "schema_provenance_recorded": bool(record_schema_provenance),
    }
    if filter_metadata:
        filter_payload.update(dict(filter_metadata))

    write_ids(normalized_ids, paths.manifest_dir)
    write_filters_manifest(filter_payload, paths.manifest_dir)
    if record_schema_provenance:
        _status_print(status_stream, "recording RCSB API schema provenance...")
        write_schema_provenance(paths.manifest_dir, fetcher=fetcher)

    _status_print(
        status_stream,
        f"Selected {len(normalized_ids)} PDB IDs: {present_count} present, "
        f"{reused_count} reused, {pending_count} pending downloads.",
    )
    _status_print(status_stream, f"Output root: {paths.output_root}")

    if dry_run:
        write_files_manifest(records, paths.manifest_dir)
        return _summary(records, dry_run=True)
    if pending_count and not yes:
        download_destination = staging_dir if staging_dir is not None else paths.raw_dir
        prompt = f"Download {pending_count} files into {download_destination}? [y/N] "
        if status_stream is None:
            answer = prompt_input(prompt).strip().lower()
        else:
            print(prompt, file=status_stream, end="", flush=True)
            answer = prompt_input("").strip().lower()
        if answer not in {"y", "yes"}:
            write_files_manifest(records, paths.manifest_dir)
            return _summary(records, cancelled=True)

    materialized_records = materialize_reused_records(records, reuse_mode=reuse_mode)
    final_records = download_pending_records(
        materialized_records,
        downloader=downloader,
        staging_dir=staging_dir,
        cleanup_staging=cleanup_staging,
        status_stream=status_stream,
        progress_interval=progress_interval,
        progress_seconds=progress_seconds,
        progress_redraw=progress_redraw,
        clock=clock,
    )
    write_files_manifest(final_records, paths.manifest_dir)
    summary = _summary(final_records)
    if prefetch_metadata:
        summary["metadata"] = _prefetch_snapshot_metadata_after_download(
            paths=paths,
            filter_metadata=filter_metadata,
            metadata_source=metadata_source,
            metadata_batch_size=metadata_batch_size,
            metadata_workers=metadata_workers,
            fetcher=fetcher,
            status_stream=status_stream,
        )
    return summary


def normalize_pdb_id(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("PDB ID cannot be blank")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_reused_file(source: Path, target: Path) -> Path:
    return copy_verified_file(source, target, expected_sha256=sha256_file(source))


def add_download_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach PDB archive download options to an argparse parser."""

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--all-current", action="store_true", help="Use all current RCSB PDB IDs.")
    source.add_argument("--ids-file", help="Read PDB IDs from a text file.")
    source.add_argument("--query-json", help="Run an RCSB Search API JSON query from this file.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--processed-dir")
    parser.add_argument("--analysis-dir")
    parser.add_argument("--failed-dir")
    parser.add_argument("--reuse-raw-dir", action="append", default=[])
    parser.add_argument(
        "--reuse-mode",
        choices=["copy", "reference"],
        default="copy",
        help="Copy reused files into this archive or keep manifest references to the source files.",
    )
    parser.add_argument("--staging-dir", help="Download pending files here before verified promotion to raw-dir.")
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Remove successfully promoted staged files.",
    )
    parser.add_argument("--format", choices=["cif", "pdb", "bcif"], default="cif")
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not prompt before downloads.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Report download progress every N processed pending files. Default: 100.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        help=(
            "Also report progress after this many seconds. Default: 5 for terminal "
            "redraw, 30 for line mode. Use 0 to disable time-based progress."
        ),
    )
    parser.add_argument(
        "--no-progress-redraw",
        action="store_true",
        help="Disable dynamic terminal progress redraw and emit line-oriented progress.",
    )
    parser.add_argument(
        "--skip-schema-provenance",
        action="store_true",
        help="Do not fetch remote API schema documents for provenance.",
    )
    parser.add_argument(
        "--prefetch-metadata",
        action="store_true",
        help="After a successful download, write compact snapshot metadata under manifests/metadata.jsonl.",
    )
    parser.add_argument("--metadata-source", choices=SNAPSHOT_METADATA_SOURCES, default="auto")
    parser.add_argument("--metadata-batch-size", type=int, default=100)
    parser.add_argument("--metadata-workers", default="auto")
    return parser


def add_snapshot_download_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach snapshot download options to an argparse parser."""

    parser.add_argument("--output-root")
    parser.add_argument("--raw-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--processed-dir")
    parser.add_argument("--analysis-dir")
    parser.add_argument("--failed-dir")
    parser.add_argument("--reuse-raw-dir", action="append", default=[])
    parser.add_argument(
        "--reuse-mode",
        choices=["copy", "reference"],
        default="copy",
        help="Copy reused files into this snapshot or keep manifest references to the source files.",
    )
    parser.add_argument("--staging-dir", help="Download pending files here before verified promotion to raw-dir.")
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Remove successfully promoted staged files.",
    )
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not prompt before downloads.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Report download progress every N processed pending files. Default: 100.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        help=(
            "Also report progress after this many seconds. Default: 5 for terminal "
            "redraw, 30 for line mode. Use 0 to disable time-based progress."
        ),
    )
    parser.add_argument(
        "--no-progress-redraw",
        action="store_true",
        help="Disable dynamic terminal progress redraw and emit line-oriented progress.",
    )
    parser.add_argument(
        "--skip-schema-provenance",
        action="store_true",
        help="Do not fetch remote API schema documents for provenance.",
    )
    parser.add_argument(
        "--prefetch-metadata",
        action="store_true",
        help="After a successful download, write compact snapshot metadata under manifests/metadata.jsonl.",
    )
    parser.add_argument("--metadata-source", choices=SNAPSHOT_METADATA_SOURCES, default="auto")
    parser.add_argument("--metadata-batch-size", type=int, default=100)
    parser.add_argument("--metadata-workers", default="auto")
    return parser


def add_snapshot_verify_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach snapshot verification options to an argparse parser."""

    parser.add_argument("--output-root")
    parser.add_argument("--manifest-dir")
    return parser


def add_snapshot_metadata_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach snapshot metadata extraction options to an argparse parser."""

    parser.add_argument("--output-root")
    parser.add_argument("--raw-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--source", choices=SNAPSHOT_METADATA_SOURCES, default="auto")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("-o", "--output")
    parser.add_argument("--manifest-output")
    parser.add_argument("--failures-output")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def add_snapshot_relocate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach snapshot manifest relocation options to an argparse parser."""

    parser.add_argument("--output-root")
    parser.add_argument("--raw-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--write", action="store_true", help="Rewrite files.jsonl and filters.json in place.")
    parser.add_argument("--backup", action="store_true", help="Write .bak copies before rewriting.")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of relative paths under output-root.",
    )
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or plan PDB archive coordinate corpora with provenance manifests.",
    )
    return add_download_arguments(parser)


def run_download_from_args(
    args: argparse.Namespace,
    *,
    fetcher: Optional[Fetcher] = None,
    downloader: Optional[Downloader] = None,
    prompt_input: Callable[[str], str] = input,
    status_stream: Optional[StatusStream] = sys.stderr,
    progress_interval: int = 100,
    progress_seconds: Optional[float] = None,
) -> dict[str, Any]:
    fetcher = fetcher or fetch_http_json_bytes
    downloader = downloader or urlretrieve_downloader
    paths = default_archive_paths(
        Path(args.output_root),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        failed_dir=Path(args.failed_dir) if args.failed_dir else None,
    )
    _report_download_request(args, paths, status_stream=status_stream)
    progress_interval = _progress_interval_from_args(args, progress_interval)
    progress_seconds = _progress_seconds_from_args(args, progress_seconds)
    ids = _ids_from_args(args, fetcher=fetcher)
    if args.max_entries is not None:
        ids = ids[: args.max_entries]
    return run_download(
        ids,
        paths=paths,
        file_format=args.format,
        reuse_raw_dirs=[Path(path) for path in args.reuse_raw_dir],
        reuse_mode=args.reuse_mode,
        staging_dir=Path(args.staging_dir) if args.staging_dir else None,
        cleanup_staging=args.cleanup_staging,
        prefetch_metadata=args.prefetch_metadata,
        metadata_source=args.metadata_source,
        metadata_batch_size=args.metadata_batch_size,
        metadata_workers=args.metadata_workers,
        yes=args.yes,
        dry_run=args.dry_run,
        record_schema_provenance=not args.skip_schema_provenance,
        fetcher=fetcher,
        downloader=downloader,
        prompt_input=prompt_input,
        status_stream=status_stream,
        progress_interval=progress_interval,
        progress_seconds=progress_seconds,
        progress_redraw=not bool(args.no_progress_redraw),
    )


def run_snapshot_download_from_args(
    args: argparse.Namespace,
    *,
    fetcher: Optional[Fetcher] = None,
    downloader: Optional[Downloader] = None,
    prompt_input: Callable[[str], str] = input,
    status_stream: Optional[StatusStream] = sys.stderr,
    progress_interval: int = 100,
    progress_seconds: Optional[float] = None,
) -> dict[str, Any]:
    fetcher = fetcher or fetch_http_json_bytes
    downloader = downloader or urlretrieve_downloader
    snapshot = get_snapshot(args.snapshot_id)
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    paths = default_archive_paths(
        output_root,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        failed_dir=Path(args.failed_dir) if args.failed_dir else None,
    )
    _report_snapshot_download_request(args, snapshot, paths, status_stream=status_stream)
    progress_interval = _progress_interval_from_args(args, progress_interval)
    progress_seconds = _progress_seconds_from_args(args, progress_seconds)
    ids = ids_from_current_holdings(fetcher=fetcher)
    if args.max_entries is not None:
        ids = ids[: args.max_entries]
    summary = run_download(
        ids,
        paths=paths,
        file_format=snapshot.file_format,
        reuse_raw_dirs=[Path(path) for path in args.reuse_raw_dir],
        reuse_mode=args.reuse_mode,
        staging_dir=Path(args.staging_dir) if args.staging_dir else None,
        cleanup_staging=args.cleanup_staging,
        prefetch_metadata=args.prefetch_metadata,
        metadata_source=args.metadata_source,
        metadata_batch_size=args.metadata_batch_size,
        metadata_workers=args.metadata_workers,
        yes=args.yes,
        dry_run=args.dry_run,
        record_schema_provenance=not args.skip_schema_provenance,
        filter_metadata={
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "snapshot_source_type": snapshot.source_type,
            "snapshot_checksum_policy": snapshot.checksum_policy,
        },
        fetcher=fetcher,
        downloader=downloader,
        prompt_input=prompt_input,
        status_stream=status_stream,
        progress_interval=progress_interval,
        progress_seconds=progress_seconds,
        progress_redraw=not bool(args.no_progress_redraw),
    )
    return {"snapshot_id": snapshot.id, **summary}


def verify_snapshot_from_args(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = get_snapshot(args.snapshot_id)
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    paths = default_archive_paths(
        output_root,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
    )
    return verify_snapshot(snapshot.id, paths=paths)


def snapshot_metadata_from_args(
    args: argparse.Namespace,
    *,
    fetcher: Optional[Fetcher] = None,
    status_stream: Optional[StatusStream] = sys.stderr,
) -> dict[str, Any]:
    fetcher = fetcher or fetch_http_json_bytes
    snapshot = get_snapshot(args.snapshot_id)
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    paths = default_archive_paths(
        output_root,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
    )
    return write_snapshot_metadata(
        snapshot.id,
        paths=paths,
        source=args.source,
        output=Path(args.output) if args.output else None,
        manifest_output=Path(args.manifest_output) if args.manifest_output else None,
        failures_output=Path(args.failures_output) if args.failures_output else None,
        max_entries=args.max_entries,
        batch_size=args.batch_size,
        workers=args.workers,
        overwrite=args.overwrite,
        fetcher=fetcher,
        status_stream=status_stream,
    )


def relocate_snapshot_manifest_from_args(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = get_snapshot(args.snapshot_id)
    output_root = Path(args.output_root) if args.output_root else snapshot.default_output_root
    paths = default_archive_paths(
        output_root,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
    )
    return relocate_snapshot_manifest(
        snapshot.id,
        paths=paths,
        write=args.write,
        relative_paths=not args.absolute_paths,
        backup=args.backup,
    )


def verify_snapshot(snapshot_id: str, *, paths: ArchivePaths) -> dict[str, Any]:
    snapshot = get_snapshot(snapshot_id)
    manifest_path = paths.manifest_dir / "files.jsonl"
    filters_path = paths.manifest_dir / "filters.json"
    if not manifest_path.exists():
        raise ValueError(f"Snapshot manifest not found: {manifest_path}")

    warnings = []
    if filters_path.exists():
        filters = json.loads(filters_path.read_text(encoding="utf-8"))
        recorded_snapshot = filters.get("snapshot_id") if isinstance(filters, dict) else None
        if recorded_snapshot and recorded_snapshot != snapshot.id:
            raise ValueError(
                f"Manifest was created for snapshot '{recorded_snapshot}', not '{snapshot.id}'"
            )
        if isinstance(filters, dict) and filters.get("file_format") != snapshot.file_format:
            warnings.append(
                f"Manifest file_format is {filters.get('file_format')!r}; "
                f"snapshot expects {snapshot.file_format!r}"
            )
    else:
        warnings.append(f"Snapshot filters manifest not found: {filters_path}")

    records = []
    counts = {
        "verified": 0,
        "missing": 0,
        "checksum_mismatch": 0,
        "failed": 0,
        "unverified": 0,
    }
    for row in _read_jsonl(manifest_path):
        record = _verify_manifest_row(row, output_root=paths.output_root, raw_dir=paths.raw_dir)
        records.append(record)
        status = record["status"]
        if status in counts:
            counts[status] += 1
    ok = not (counts["missing"] or counts["checksum_mismatch"] or counts["failed"])
    return {
        "snapshot_id": snapshot.id,
        "ok": ok,
        "manifest_path": str(manifest_path),
        "output_root": str(paths.output_root),
        "total": len(records),
        **counts,
        "warnings": warnings,
        "records": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_download_from_args(args)
    except Exception as exc:
        print(f"pheat archive download: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


class _DownloadProgressReporter:
    def __init__(
        self,
        stream: Optional[StatusStream],
        *,
        total: int,
        interval: int,
        seconds: Optional[float],
        redraw: bool,
        clock: Clock,
    ) -> None:
        if interval < 1:
            raise ValueError("--progress-interval must be at least 1")
        if seconds is not None and seconds < 0:
            raise ValueError("--progress-seconds cannot be negative")
        self.stream = stream
        self.total = total
        self.interval = interval
        self.redraw = redraw and _stream_supports_redraw(stream)
        self.seconds = _default_progress_seconds(seconds, redraw=self.redraw)
        self.clock = clock
        self.start_time = clock()
        self.last_report_time = self.start_time
        self.last_report_processed = 0
        self.processed = 0
        self.downloaded = 0
        self.failed = 0
        self.bytes_downloaded = 0
        self.reported = False

    def start(self) -> None:
        if self.total:
            self._report(force=True, state="starting")

    def record_success(self, record: DownloadRecord) -> None:
        self.processed += 1
        self.downloaded += 1
        self.bytes_downloaded += int(record.bytes or 0)
        self._report_if_due(state="downloading")

    def record_failure(self) -> None:
        self.processed += 1
        self.failed += 1
        if self.processed < self.total:
            self._report(force=True, state="downloading")

    def finish(self) -> None:
        if self.total:
            self._report(force=True, state="complete", final=True)

    def _report_if_due(self, *, state: str) -> None:
        if self.processed >= self.total:
            return
        now = self.clock()
        by_count = self.processed == self.total or (
            self.processed - self.last_report_processed >= self.interval
        )
        by_time = self.seconds > 0 and (now - self.last_report_time) >= self.seconds
        if by_count or by_time:
            self._report(force=True, state=state, now=now)

    def _report(
        self,
        *,
        force: bool,
        state: str,
        now: Optional[float] = None,
        final: bool = False,
    ) -> None:
        if not force or self.total == 0:
            return
        observed_now = self.clock() if now is None else now
        elapsed = max(0.0, observed_now - self.start_time)
        message = _format_download_progress(
            state=state,
            processed=self.processed,
            total=self.total,
            downloaded=self.downloaded,
            failed=self.failed,
            bytes_downloaded=self.bytes_downloaded,
            elapsed=elapsed,
        )
        _progress_print(self.stream, message, redraw=self.redraw, final=final)
        self.reported = True
        self.last_report_time = observed_now
        self.last_report_processed = self.processed


def _progress_interval_from_args(args: argparse.Namespace, default: int) -> int:
    value = int(getattr(args, "progress_interval", default))
    if value < 1:
        raise ValueError("--progress-interval must be at least 1")
    return value


def _progress_seconds_from_args(
    args: argparse.Namespace,
    default: Optional[float],
) -> Optional[float]:
    value = getattr(args, "progress_seconds", default)
    if value is not None and float(value) < 0:
        raise ValueError("--progress-seconds cannot be negative")
    return None if value is None else float(value)


def _status_print(stream: Optional[StatusStream], message: str) -> None:
    if stream is not None:
        if hasattr(stream, "status"):
            stream.status(message)
        else:
            print(message, file=stream, flush=True)


def _progress_print(
    stream: Optional[StatusStream],
    message: str,
    *,
    redraw: bool,
    final: bool = False,
) -> None:
    if stream is None:
        return
    if hasattr(stream, "progress"):
        stream.progress(message, redraw=redraw, final=final)
    elif redraw and _stream_supports_redraw(stream):
        print(f"\r{message}", file=stream, end="\n" if final else "", flush=True)
    else:
        print(message, file=stream, flush=True)


def _stream_supports_redraw(stream: Optional[StatusStream]) -> bool:
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _default_progress_seconds(value: Optional[float], *, redraw: bool) -> float:
    if value is not None:
        return float(value)
    return 5.0 if redraw else 30.0


def _format_download_progress(
    *,
    state: str,
    processed: int,
    total: int,
    downloaded: int,
    failed: int,
    bytes_downloaded: int,
    elapsed: float,
) -> str:
    percent = 100.0 if total == 0 else (processed / total) * 100.0
    files_per_second = processed / elapsed if elapsed > 0 else 0.0
    bytes_per_second = bytes_downloaded / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta = remaining / files_per_second if files_per_second > 0 else None
    return (
        f"download progress [{state}]: {processed}/{total} pending "
        f"({percent:.1f}%) | downloaded {downloaded} | failed {failed} | "
        f"elapsed {_format_duration(elapsed)} | {files_per_second:.2f} files/s | "
        f"{_format_byte_rate(bytes_per_second)} | ETA {_format_eta(eta)}"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    return _format_duration(seconds)


def _format_byte_rate(bytes_per_second: float) -> str:
    units = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"]
    value = max(0.0, bytes_per_second)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{value:.0f} {units[unit_index]}"
    return f"{value:.2f} {units[unit_index]}"


def _report_download_request(
    args: argparse.Namespace,
    paths: ArchivePaths,
    *,
    status_stream: Optional[StatusStream],
) -> None:
    _status_print(status_stream, f"archive output root: {paths.output_root}")
    _status_print(status_stream, f"archive raw dir: {paths.raw_dir}")
    if args.staging_dir:
        _status_print(status_stream, f"archive staging dir: {args.staging_dir}")
    _status_print(status_stream, f"archive format: {args.format}")
    _status_print(status_stream, f"archive reuse mode: {args.reuse_mode}")
    _status_print(status_stream, f"archive dry run: {bool(args.dry_run)}")
    _status_print(status_stream, f"archive automatic yes: {bool(args.yes)}")
    if args.all_current:
        _status_print(status_stream, "fetching current RCSB/wwPDB holdings...")
    elif args.query_json:
        _status_print(status_stream, f"fetching RCSB query results from {args.query_json}...")
    elif args.ids_file:
        _status_print(status_stream, f"reading PDB IDs from {args.ids_file}...")


def _report_snapshot_download_request(
    args: argparse.Namespace,
    snapshot: SnapshotSpec,
    paths: ArchivePaths,
    *,
    status_stream: Optional[StatusStream],
) -> None:
    _status_print(status_stream, f"archive snapshot: {snapshot.id}")
    _status_print(status_stream, f"archive snapshot format: {snapshot.file_format}")
    _status_print(status_stream, f"archive output root: {paths.output_root}")
    _status_print(status_stream, f"archive raw dir: {paths.raw_dir}")
    if args.staging_dir:
        _status_print(status_stream, f"archive staging dir: {args.staging_dir}")
    _status_print(status_stream, f"archive reuse mode: {args.reuse_mode}")
    _status_print(status_stream, f"archive dry run: {bool(args.dry_run)}")
    _status_print(status_stream, f"archive automatic yes: {bool(args.yes)}")
    _status_print(status_stream, "fetching current RCSB/wwPDB holdings...")


def _ids_from_args(
    args: argparse.Namespace,
    *,
    fetcher: Fetcher = fetch_http_json_bytes,
) -> list[str]:
    if args.all_current:
        return ids_from_current_holdings(fetcher=fetcher)
    if args.ids_file:
        return ids_from_file(Path(args.ids_file))
    if args.query_json:
        query = json.loads(Path(args.query_json).read_text(encoding="utf-8"))
        if not isinstance(query, Mapping):
            raise ValueError("--query-json must contain a JSON object")
        return ids_from_search_query(query, fetcher=fetcher)
    raise ValueError("one input mode is required")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"Manifest row in {path} is not a JSON object")
            rows.append(decoded)
    return rows


def _manifest_sha256_by_filename(manifest_dir: Path) -> dict[str, str]:
    manifest_path = Path(manifest_dir) / "files.jsonl"
    if not manifest_path.exists():
        return {}
    expected = {}
    for row in _read_jsonl(manifest_path):
        path_text = row.get("path")
        checksum_payload = row.get("checksums")
        if not path_text or not isinstance(checksum_payload, dict):
            continue
        sha256 = checksum_payload.get("sha256")
        if isinstance(sha256, str):
            expected[Path(str(path_text)).name.lower()] = sha256
    return expected


def _verify_manifest_row(
    row: Mapping[str, Any],
    *,
    output_root: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> dict[str, Any]:
    path_text = row.get("path")
    source_path_text = row.get("source_path")
    path = Path(str(path_text)) if path_text else None
    source_path = Path(str(source_path_text)) if source_path_text else None
    checksum_payload = row.get("checksums")
    checksums: Mapping[str, Any] = checksum_payload if isinstance(checksum_payload, dict) else {}
    expected_sha256 = checksums.get("sha256")
    result = {
        "pdb_id": row.get("pdb_id"),
        "path": str(path) if path is not None else None,
        "source_path": str(source_path) if source_path is not None else None,
        "expected_sha256": expected_sha256,
        "actual_sha256": None,
        "status": "unverified",
        "error": None,
    }

    if row.get("status") == "failed":
        result["status"] = "failed"
        result["error"] = row.get("error") or "manifest row status is failed"
        return result
    if path is None:
        result["error"] = "manifest row has no path"
        return result

    if output_root is not None:
        verify_path = resolve_manifest_row_path(row, output_root=output_root, raw_dir=raw_dir)
    else:
        verify_path = path if path.exists() else source_path
    if verify_path is None or not verify_path.exists():
        result["status"] = "missing"
        result["error"] = f"file not found: {path}"
        return result
    if not expected_sha256:
        result["path"] = str(verify_path)
        result["error"] = "manifest row has no sha256 checksum"
        return result

    actual_sha256 = sha256_file(verify_path)
    result["path"] = str(verify_path)
    result["actual_sha256"] = actual_sha256
    if actual_sha256 == expected_sha256:
        result["status"] = "verified"
    else:
        result["status"] = "checksum_mismatch"
        result["error"] = "sha256 checksum mismatch"
    return result


def _decode_json_or_none(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _schema_license(document: Any) -> Any:
    if isinstance(document, dict):
        info = document.get("info")
        if isinstance(info, dict) and isinstance(info.get("license"), dict):
            return dict(info["license"])
    return None


def _schema_version(document: Any) -> Optional[str]:
    if isinstance(document, dict):
        info = document.get("info")
        if isinstance(info, dict) and info.get("version") is not None:
            return str(info["version"])
        comment = document.get("$comment")
        if isinstance(comment, str) and "Schema version:" in comment:
            return comment.split("Schema version:", 1)[1].strip()
    return None


def _header(headers: Mapping[str, str], key: str) -> Optional[str]:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.title())


def _find_reused_file(filename: str, reuse_raw_dirs: Sequence[Path]) -> Optional[Path]:
    lower_name = filename.lower()
    for directory in reuse_raw_dirs:
        root = Path(directory)
        direct = root / filename
        if direct.exists():
            return direct
        if root.exists():
            for candidate in root.iterdir():
                if candidate.name.lower() == lower_name and candidate.is_file():
                    return candidate
    return None


def _record_for_existing(
    pdb_id: str,
    url: str,
    path: Path,
    status: str,
    *,
    source_path: Optional[Path] = None,
) -> DownloadRecord:
    real_path = source_path or path
    return DownloadRecord(
        pdb_id=pdb_id,
        url=url,
        path=path,
        status=status,
        source_path=source_path,
        bytes=real_path.stat().st_size,
        sha256=sha256_file(real_path),
    )


def _download_record_to_dict(record: DownloadRecord) -> dict[str, Any]:
    return {
        "pdb_id": record.pdb_id,
        "url": record.url,
        "path": str(record.path),
        "status": record.status,
        "source_path": str(record.source_path) if record.source_path else None,
        "bytes": record.bytes,
        "checksums": {"sha256": record.sha256} if record.sha256 else {},
        "error": record.error,
    }


def _summary(
    records: Sequence[DownloadRecord],
    *,
    dry_run: bool = False,
    cancelled: bool = False,
) -> dict[str, Any]:
    return {
        "total": len(records),
        "present": sum(1 for record in records if record.status == "present"),
        "reused": sum(1 for record in records if record.status == "reused"),
        "pending": sum(1 for record in records if record.status == "pending"),
        "downloaded": sum(1 for record in records if record.status == "downloaded"),
        "failed": sum(1 for record in records if record.status == "failed"),
        "dry_run": dry_run,
        "cancelled": cancelled,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

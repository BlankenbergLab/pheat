import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pheat.archive import (
    DATA_GRAPHQL_URL,
    HttpPayload,
    default_archive_paths,
    relocate_snapshot_manifest,
    run_download,
    schema_provenance_records,
    write_snapshot_metadata,
)
from pheat.cli import main as cli_main


OPENAPI_PAYLOAD = {
    "openapi": "3.0.1",
    "info": {
        "title": "RCSB Search API",
        "license": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        "version": "2.6.0",
    },
}

METADATA_SCHEMA_PAYLOAD = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "$id": "https://example.invalid/rcsb-search-metadata.json",
    "$comment": "Schema version: 1.56.0",
    "type": "object",
}

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def fake_schema_fetcher(url, json_payload=None):
    if json_payload is not None:
        raise AssertionError("schema provenance should not POST query payloads")
    if url.endswith("/openapi.json"):
        body = json.dumps(OPENAPI_PAYLOAD, sort_keys=True).encode("utf-8")
        return HttpPayload(
            body=body,
            headers={"etag": '"openapi-etag"', "last-modified": "Wed, 20 May 2026 00:00:00 GMT"},
            url=url,
        )
    if url.endswith("/metadata/schema"):
        body = json.dumps(METADATA_SCHEMA_PAYLOAD, sort_keys=True).encode("utf-8")
        return HttpPayload(body=body, headers={}, url=url)
    raise AssertionError(f"unexpected URL {url}")


def fake_current_holdings_fetcher(url, json_payload=None):
    if json_payload is not None:
        raise AssertionError("current holdings should not POST query payloads")
    if url.endswith("/holdings/current/entry_ids"):
        body = json.dumps(["2DEF", "1ABC"]).encode("utf-8")
        return HttpPayload(body=body, headers={}, url=url)
    raise AssertionError(f"unexpected URL {url}")


def fake_graphql_metadata_fetcher(url, json_payload=None):
    if url != DATA_GRAPHQL_URL:
        raise AssertionError(f"unexpected URL {url}")
    if not isinstance(json_payload, dict):
        raise AssertionError("metadata fetch should POST a GraphQL payload")
    query = json_payload.get("query", "")
    if "entries(entry_ids" in query:
        body = {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1ABC",
                        "entry": {"id": "1ABC"},
                        "struct": {"title": "Aqueous buffer test protein", "pdbx_CASP_flag": "N"},
                        "struct_keywords": {"text": "HYDROLASE, water soluble", "pdbx_keywords": "HYDROLASE"},
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        "refine": [
                            {
                                "ls_d_res_high": 1.8,
                                "ls_R_factor_R_work": 0.18,
                                "ls_R_factor_R_free": 0.22,
                                "B_iso_mean": 20.0,
                            }
                        ],
                        "pdbx_database_status": {
                            "status_code": "REL",
                            "pdb_format_compatible": "Y",
                            "recvd_initial_deposition_date": "2020-01-02T00:00:00.000+00:00",
                        },
                        "rcsb_accession_info": {
                            "deposit_date": "2020-01-02T00:00:00.000+00:00",
                            "initial_release_date": "2020-03-04T00:00:00.000+00:00",
                            "revision_date": "2021-05-06T00:00:00.000+00:00",
                            "status_code": "REL",
                        },
                        "rcsb_entry_container_identifiers": {
                            "polymer_entity_ids": ["1"],
                            "non_polymer_entity_ids": ["2"],
                            "assembly_ids": ["1"],
                            "model_ids": [1],
                        },
                        "rcsb_entry_info": {
                            "experimental_method": "X-ray",
                            "structure_determination_methodology": "experimental",
                            "resolution_combined": [1.8],
                            "selected_polymer_entity_types": "Protein (only)",
                            "polymer_composition": "homomeric protein",
                            "deposited_atom_count": 100,
                            "deposited_model_count": 1,
                            "polymer_entity_count": 1,
                            "polymer_entity_count_protein": 1,
                            "nonpolymer_entity_count": 1,
                            "solvent_entity_count": 1,
                        },
                        "pdbx_vrpt_summary_geometry": [
                            {
                                "clashscore": 2.0,
                                "percent_ramachandran_outliers": 0.1,
                                "percent_rotamer_outliers": 0.2,
                            }
                        ],
                        "pdbx_vrpt_summary_diffraction": [{"data_completeness": 99.0}],
                        "exptl_crystal_grow": [{"method": "VAPOR DIFFUSION", "pdbx_details": "Tris buffer"}],
                    }
                ]
            }
        }
        return HttpPayload(body=json.dumps(body).encode("utf-8"), headers={}, url=url)
    if "polymer_entities(entity_ids" in query:
        body = {
            "data": {
                "polymer_entities": [
                    {
                        "rcsb_id": "1ABC_1",
                        "rcsb_polymer_entity_container_identifiers": {
                            "entry_id": "1ABC",
                            "entity_id": "1",
                            "asym_ids": ["A"],
                            "auth_asym_ids": ["A"],
                        },
                        "entity_poly": {
                            "type": "polypeptide(L)",
                            "rcsb_entity_polymer_type": "Protein",
                            "pdbx_seq_one_letter_code_can": "AG",
                            "rcsb_sample_sequence_length": 2,
                            "nstd_monomer": "no",
                            "nstd_linkage": "no",
                        },
                        "rcsb_polymer_entity": {"pdbx_description": "test protein"},
                        "rcsb_entity_source_organism": [
                            {"ncbi_scientific_name": "Escherichia coli", "ncbi_taxonomy_id": 562}
                        ],
                        "rcsb_polymer_entity_group_membership": [
                            {
                                "aggregation_method": "sequence_identity",
                                "group_id": "cluster_30",
                                "similarity_cutoff": 30.0,
                            }
                        ],
                    }
                ]
            }
        }
        return HttpPayload(body=json.dumps(body).encode("utf-8"), headers={}, url=url)
    raise AssertionError(f"unexpected GraphQL query {query}")


def _capture_cli(args):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status = cli_main(args)
    return status, stdout.getvalue()


def _trailing_json_object(text):
    return json.loads(text[text.find("{") :])


class _TtyBuffer(io.StringIO):
    def isatty(self):
        return True


class ArchiveTests(unittest.TestCase):
    def test_schema_provenance_records_versions_checksums_without_content(self):
        records = schema_provenance_records(fetcher=fake_schema_fetcher)

        search = records[0]
        self.assertEqual(search["service"], "search")
        self.assertEqual(search["kind"], "openapi")
        self.assertEqual(search["declared_license"]["name"], "Apache 2.0")
        self.assertEqual(search["version"], "2.6.0")
        self.assertEqual(search["etag"], '"openapi-etag"')
        self.assertFalse(search["stored"])
        self.assertNotIn("content", search)

        expected_hash = hashlib.sha256(
            json.dumps(OPENAPI_PAYLOAD, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(search["content_sha256"], expected_hash)

        metadata = records[1]
        self.assertEqual(metadata["service"], "search-metadata")
        self.assertEqual(metadata["schema_uri"], "http://json-schema.org/draft-04/schema#")
        self.assertEqual(metadata["schema_id"], "https://example.invalid/rcsb-search-metadata.json")
        self.assertEqual(metadata["version"], "1.56.0")
        self.assertIsNone(metadata["declared_license"])
        self.assertFalse(metadata["stored"])

    def test_run_download_dry_run_records_manifests_and_schema_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir))

            summary = run_download(
                ["1abc", "2def", "1ABC"],
                paths=paths,
                dry_run=True,
                fetcher=fake_schema_fetcher,
            )

            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["pending"], 2)
            self.assertTrue((paths.manifest_dir / "ids.txt").exists())
            self.assertEqual(
                (paths.manifest_dir / "ids.txt").read_text(encoding="utf-8"),
                "1ABC\n2DEF\n",
            )
            api_schemas = json.loads(
                (paths.manifest_dir / "api-schemas.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(api_schemas["schemas"]), 2)
            self.assertFalse(api_schemas["schemas"][0]["stored"])
            self.assertNotIn('"openapi": "3.0.1"', json.dumps(api_schemas))
            self.assertNotIn('"title": "RCSB Search API"', json.dumps(api_schemas))

            file_rows = [
                json.loads(line)
                for line in (paths.manifest_dir / "files.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in file_rows], ["pending", "pending"])

    def test_cli_archive_download_dry_run_records_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ids_file = root / "ids.txt"
            ids_file.write_text("2def\n1abc\n1ABC\n", encoding="utf-8")
            output_root = root / "archive"

            self.assertEqual(
                cli_main(
                    [
                        "archive",
                        "download",
                        "--ids-file",
                        str(ids_file),
                        "--output-root",
                        str(output_root),
                        "--dry-run",
                        "--skip-schema-provenance",
                    ]
                ),
                0,
            )

            manifest_dir = output_root / "manifests"
            self.assertEqual(
                (manifest_dir / "ids.txt").read_text(encoding="utf-8"),
                "1ABC\n2DEF\n",
            )
            self.assertFalse((manifest_dir / "api-schemas.json").exists())
            file_rows = [
                json.loads(line)
                for line in (manifest_dir / "files.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["pdb_id"] for row in file_rows], ["1ABC", "2DEF"])
            self.assertEqual([row["status"] for row in file_rows], ["pending", "pending"])

    def test_cli_archive_snapshots_list_and_describe(self):
        status, output = _capture_cli(["archive", "snapshots", "list"])
        self.assertEqual(status, 0)
        snapshots = json.loads(output)
        snapshot_ids = {snapshot["id"] for snapshot in snapshots}
        self.assertIn("wwpdb-current-mmcif", snapshot_ids)
        self.assertIn("wwpdb-current-pdb", snapshot_ids)
        self.assertIn("rcsb-current-bcif", snapshot_ids)

        status, output = _capture_cli(
            ["archive", "snapshots", "describe", "wwpdb-current-mmcif"]
        )
        self.assertEqual(status, 0)
        snapshot = json.loads(output)
        self.assertEqual(snapshot["id"], "wwpdb-current-mmcif")
        self.assertEqual(snapshot["format"], "cif")
        self.assertEqual(snapshot["checksum_policy"], "pheat-sha256-manifest")

    def test_cli_reference_select_help_renders_percent_text(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as captured, contextlib.redirect_stdout(stdout):
            cli_main(["reference", "select", "--help"])

        self.assertEqual(captured.exception.code, 0)
        self.assertIn("30 percent ID", stdout.getvalue())

    def test_cli_archive_snapshot_download_dry_run_records_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "archive"

            with mock.patch(
                "pheat.archive.fetch_http_json_bytes",
                side_effect=fake_current_holdings_fetcher,
            ):
                status, output = _capture_cli(
                    [
                        "archive",
                        "snapshots",
                        "download",
                        "wwpdb-current-mmcif",
                        "--output-root",
                        str(output_root),
                        "--max-entries",
                        "1",
                        "--dry-run",
                        "--skip-schema-provenance",
                    ]
                )

            self.assertEqual(status, 0)
            summary = _trailing_json_object(output)
            self.assertEqual(summary["snapshot_id"], "wwpdb-current-mmcif")
            self.assertEqual(summary["total"], 1)
            manifest_dir = output_root / "manifests"
            filters = json.loads((manifest_dir / "filters.json").read_text(encoding="utf-8"))
            self.assertEqual(filters["snapshot_id"], "wwpdb-current-mmcif")
            self.assertEqual(filters["file_format"], "cif")
            self.assertEqual(
                (manifest_dir / "ids.txt").read_text(encoding="utf-8"),
                "1ABC\n",
            )

    def test_cli_archive_snapshot_download_stages_promotes_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "archive"
            staging_dir = root / "staging"
            payload = b"binary cif payload\n"

            def downloader(url, output):
                self.assertEqual(url, "https://models.rcsb.org/1ABC.bcif.gz")
                output.write_bytes(payload)

            with (
                mock.patch(
                    "pheat.archive.fetch_http_json_bytes",
                    side_effect=fake_current_holdings_fetcher,
                ),
                mock.patch("pheat.archive.urlretrieve_downloader", side_effect=downloader),
            ):
                status, output = _capture_cli(
                    [
                        "archive",
                        "snapshots",
                        "download",
                        "rcsb-current-bcif",
                        "--output-root",
                        str(output_root),
                        "--staging-dir",
                        str(staging_dir),
                        "--cleanup-staging",
                        "--max-entries",
                        "1",
                        "--skip-schema-provenance",
                        "--yes",
                    ]
                )

            self.assertEqual(status, 0)
            summary = _trailing_json_object(output)
            self.assertEqual(summary["snapshot_id"], "rcsb-current-bcif")
            self.assertEqual(summary["downloaded"], 1)
            final_file = output_root / "raw" / "1abc.bcif.gz"
            self.assertEqual(final_file.read_bytes(), payload)
            self.assertFalse((staging_dir / "1abc.bcif.gz").exists())
            file_rows = [
                json.loads(line)
                for line in (output_root / "manifests" / "files.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(file_rows[0]["path"], str(final_file))
            self.assertEqual(file_rows[0]["checksums"]["sha256"], hashlib.sha256(payload).hexdigest())
            filters = json.loads((output_root / "manifests" / "filters.json").read_text(encoding="utf-8"))
            self.assertEqual(filters["staging_dir"], str(staging_dir))
            self.assertTrue(filters["cleanup_staging"])

    def test_run_download_reports_redraw_speed_and_eta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir))
            stream = _TtyBuffer()
            now = [100.0]

            def clock():
                return now[0]

            def downloader(_url, output):
                now[0] += 2.0
                output.write_bytes(b"x" * 2048)

            summary = run_download(
                ["1ABC", "2DEF"],
                paths=paths,
                downloader=downloader,
                yes=True,
                record_schema_provenance=False,
                status_stream=stream,
                progress_interval=10,
                progress_seconds=0,
                progress_redraw=True,
                clock=clock,
            )

            self.assertEqual(summary["downloaded"], 2)
            progress = stream.getvalue()
            self.assertIn("\rdownload progress [starting]", progress)
            self.assertIn("\rdownload progress [complete]", progress)
            self.assertIn("2/2 pending (100.0%)", progress)
            self.assertIn("files/s", progress)
            self.assertIn("KiB/s", progress)
            self.assertIn("ETA 00:00:00", progress)

    def test_run_download_can_disable_redraw_for_line_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir))
            stream = _TtyBuffer()

            def downloader(_url, output):
                output.write_bytes(b"payload\n")

            summary = run_download(
                ["1ABC"],
                paths=paths,
                downloader=downloader,
                yes=True,
                record_schema_provenance=False,
                status_stream=stream,
                progress_interval=1,
                progress_seconds=0,
                progress_redraw=False,
            )

            self.assertEqual(summary["downloaded"], 1)
            progress = stream.getvalue()
            self.assertNotIn("\rdownload progress", progress)
            self.assertIn("download progress [complete]: 1/1 pending (100.0%)", progress)

    def test_run_download_failures_advance_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir))
            stream = io.StringIO()

            def downloader(url, output):
                if url.endswith("1ABC.cif.gz"):
                    raise RuntimeError("simulated failure")
                output.write_bytes(b"payload\n")

            summary = run_download(
                ["1ABC", "2DEF"],
                paths=paths,
                downloader=downloader,
                yes=True,
                record_schema_provenance=False,
                status_stream=stream,
                progress_interval=1,
                progress_seconds=0,
                progress_redraw=False,
            )

            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual(summary["failed"], 1)
            progress = stream.getvalue()
            self.assertIn("failed download for 1ABC: simulated failure", progress)
            self.assertIn("2/2 pending (100.0%)", progress)
            self.assertIn("failed 1", progress)

    def test_cli_archive_snapshot_verify_accepts_matching_checksums(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "archive"
            manifest_dir = root / "manifests"
            raw_dir = root / "raw"
            manifest_dir.mkdir(parents=True)
            raw_dir.mkdir()
            coordinate_file = raw_dir / "1abc.cif.gz"
            coordinate_file.write_bytes(b"coordinates\n")
            digest = hashlib.sha256(b"coordinates\n").hexdigest()
            (manifest_dir / "filters.json").write_text(
                json.dumps({"snapshot_id": "wwpdb-current-mmcif", "file_format": "cif"}) + "\n",
                encoding="utf-8",
            )
            (manifest_dir / "files.jsonl").write_text(
                json.dumps(
                    {
                        "pdb_id": "1ABC",
                        "path": str(coordinate_file),
                        "status": "downloaded",
                        "checksums": {"sha256": digest},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            status, output = _capture_cli(
                ["archive", "snapshots", "verify", "wwpdb-current-mmcif", "--output-root", str(root)]
            )

            self.assertEqual(status, 0)
            result = json.loads(output)
            self.assertTrue(result["ok"])
            self.assertEqual(result["verified"], 1)
            self.assertEqual(result["missing"], 0)
            self.assertEqual(result["checksum_mismatch"], 0)

    def test_snapshot_verify_accepts_relative_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "archive"
            manifest_dir = root / "manifests"
            raw_dir = root / "raw"
            manifest_dir.mkdir(parents=True)
            raw_dir.mkdir()
            coordinate_file = raw_dir / "1abc.bcif.gz"
            coordinate_file.write_bytes(b"coordinates\n")
            digest = hashlib.sha256(b"coordinates\n").hexdigest()
            (manifest_dir / "filters.json").write_text(
                json.dumps({"snapshot_id": "rcsb-current-bcif", "file_format": "bcif"}) + "\n",
                encoding="utf-8",
            )
            (manifest_dir / "files.jsonl").write_text(
                json.dumps(
                    {
                        "pdb_id": "1ABC",
                        "path": "raw/1abc.bcif.gz",
                        "status": "downloaded",
                        "checksums": {"sha256": digest},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            status, output = _capture_cli(
                ["archive", "snapshots", "verify", "rcsb-current-bcif", "--output-root", str(root)]
            )

            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output)["ok"])

    def test_relocate_snapshot_manifest_rewrites_stale_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "archive"
            paths = default_archive_paths(root)
            paths.raw_dir.mkdir(parents=True)
            paths.manifest_dir.mkdir(parents=True)
            coordinate_file = paths.raw_dir / "1abc.bcif.gz"
            coordinate_file.write_bytes(b"coordinates\n")
            digest = hashlib.sha256(b"coordinates\n").hexdigest()
            (paths.manifest_dir / "filters.json").write_text(
                json.dumps(
                    {
                        "snapshot_id": "rcsb-current-bcif",
                        "file_format": "bcif",
                        "output_root": "/old/archive",
                        "raw_dir": "/old/archive/raw",
                        "manifest_dir": "/old/archive/manifests",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (paths.manifest_dir / "files.jsonl").write_text(
                json.dumps(
                    {
                        "pdb_id": "1ABC",
                        "path": "/old/archive/raw/1abc.bcif.gz",
                        "status": "downloaded",
                        "checksums": {"sha256": digest},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = relocate_snapshot_manifest("rcsb-current-bcif", paths=paths, write=True)

            self.assertEqual(result["changed"], 1)
            row = json.loads((paths.manifest_dir / "files.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["path"], "raw/1abc.bcif.gz")
            filters = json.loads((paths.manifest_dir / "filters.json").read_text(encoding="utf-8"))
            self.assertEqual(filters["raw_dir"], "raw")

    def test_cli_archive_snapshot_verify_reports_missing_and_mismatched_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "archive"
            manifest_dir = root / "manifests"
            raw_dir = root / "raw"
            manifest_dir.mkdir(parents=True)
            raw_dir.mkdir()
            mismatched_file = raw_dir / "1abc.cif.gz"
            missing_file = raw_dir / "2def.cif.gz"
            mismatched_file.write_bytes(b"changed\n")
            (manifest_dir / "filters.json").write_text(
                json.dumps({"snapshot_id": "wwpdb-current-mmcif", "file_format": "cif"}) + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "pdb_id": "1ABC",
                    "path": str(mismatched_file),
                    "status": "downloaded",
                    "checksums": {"sha256": hashlib.sha256(b"original\n").hexdigest()},
                },
                {
                    "pdb_id": "2DEF",
                    "path": str(missing_file),
                    "status": "downloaded",
                    "checksums": {"sha256": hashlib.sha256(b"missing\n").hexdigest()},
                },
            ]
            (manifest_dir / "files.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            status, output = _capture_cli(
                ["archive", "snapshots", "verify", "wwpdb-current-mmcif", "--output-root", str(root)]
            )

            self.assertEqual(status, 1)
            result = json.loads(output)
            self.assertFalse(result["ok"])
            self.assertEqual(result["checksum_mismatch"], 1)
            self.assertEqual(result["missing"], 1)

    def test_run_download_copies_reused_files_by_default_and_downloads_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reuse_dir = root / "external-archive"
            reuse_dir.mkdir()
            reused_file = reuse_dir / "1abc.cif.gz"
            reused_file.write_bytes(b"already downloaded\n")
            paths = default_archive_paths(root / "pheat-archive")
            downloaded = []

            def downloader(url, output):
                downloaded.append(url)
                output.write_bytes(f"downloaded {url}\n".encode("utf-8"))

            summary = run_download(
                ["1abc", "2def"],
                paths=paths,
                reuse_raw_dirs=[reuse_dir],
                yes=True,
                record_schema_provenance=False,
                downloader=downloader,
            )

            self.assertEqual(summary["reused"], 1)
            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual(len(downloaded), 1)
            self.assertTrue((paths.raw_dir / "2def.cif.gz").exists())
            self.assertEqual((paths.raw_dir / "1abc.cif.gz").read_bytes(), b"already downloaded\n")

            file_rows = [
                json.loads(line)
                for line in (paths.manifest_dir / "files.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            statuses = {row["pdb_id"]: row["status"] for row in file_rows}
            self.assertEqual(statuses, {"1ABC": "reused", "2DEF": "downloaded"})
            self.assertEqual(set(file_rows[0]["checksums"]), {"sha256"})
            self.assertIn("external-archive/1abc.cif.gz", file_rows[0]["source_path"])

    def test_run_download_can_reference_reused_files_without_copying(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reuse_dir = root / "external-archive"
            reuse_dir.mkdir()
            reused_file = reuse_dir / "1abc.cif.gz"
            reused_file.write_bytes(b"already downloaded\n")
            paths = default_archive_paths(root / "pheat-archive")

            summary = run_download(
                ["1abc"],
                paths=paths,
                reuse_raw_dirs=[reuse_dir],
                reuse_mode="reference",
                yes=True,
                record_schema_provenance=False,
                downloader=lambda _url, _output: self.fail("should not download"),
            )

            self.assertEqual(summary["reused"], 1)
            self.assertFalse((paths.raw_dir / "1abc.cif.gz").exists())
            file_row = json.loads((paths.manifest_dir / "files.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(file_row["status"], "reused")
            self.assertEqual(file_row["source_path"], str(reused_file))

            result = _capture_cli(
                ["archive", "snapshots", "verify", "wwpdb-current-mmcif", "--output-root", str(paths.output_root)]
            )
            self.assertEqual(result[0], 0)
            verify_payload = json.loads(result[1])
            self.assertTrue(verify_payload["ok"])
            self.assertEqual(verify_payload["verified"], 1)

    def test_run_download_redownloads_present_files_that_mismatch_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = default_archive_paths(root / "pheat-archive")
            paths.raw_dir.mkdir(parents=True)
            paths.manifest_dir.mkdir(parents=True)
            good_file = paths.raw_dir / "1abc.cif.gz"
            stale_file = paths.raw_dir / "2def.cif.gz"
            good_file.write_bytes(b"good\n")
            stale_file.write_bytes(b"stale\n")
            redownloaded_payload = b"redownloaded\n"
            rows = [
                {
                    "pdb_id": "1ABC",
                    "path": str(good_file),
                    "status": "downloaded",
                    "checksums": {"sha256": hashlib.sha256(b"good\n").hexdigest()},
                },
                {
                    "pdb_id": "2DEF",
                    "path": str(stale_file),
                    "status": "downloaded",
                    "checksums": {"sha256": hashlib.sha256(b"original\n").hexdigest()},
                },
            ]
            (paths.manifest_dir / "files.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            downloaded = []

            def downloader(url, output):
                downloaded.append(url)
                output.write_bytes(redownloaded_payload)

            summary = run_download(
                ["1abc", "2def"],
                paths=paths,
                yes=True,
                record_schema_provenance=False,
                downloader=downloader,
            )

            self.assertEqual(summary["present"], 1)
            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual(downloaded, ["https://files.rcsb.org/download/2DEF.cif.gz"])
            self.assertEqual(good_file.read_bytes(), b"good\n")
            self.assertEqual(stale_file.read_bytes(), redownloaded_payload)

    def test_write_snapshot_metadata_uses_rcsb_api_and_keeps_compact_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir) / "archive")
            paths.manifest_dir.mkdir(parents=True)
            paths.raw_dir.mkdir(parents=True)
            (paths.manifest_dir / "files.jsonl").write_text(
                json.dumps({"pdb_id": "1ABC", "path": "raw/1abc.bcif.gz", "status": "downloaded"}) + "\n",
                encoding="utf-8",
            )

            summary = write_snapshot_metadata(
                "rcsb-current-bcif",
                paths=paths,
                source="rcsb-api",
                fetcher=fake_graphql_metadata_fetcher,
                status_stream=None,
            )

            self.assertEqual(summary["entry_count"], 1)
            metadata_row = json.loads((paths.manifest_dir / "metadata.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(metadata_row["pdb_id"], "1ABC")
            self.assertEqual(metadata_row["method"], "x-ray")
            self.assertEqual(metadata_row["resolution"], 1.8)
            self.assertEqual(metadata_row["quality"]["r_free"], 0.22)
            self.assertEqual(metadata_row["sequence_clusters"]["30"], ["cluster_30"])
            self.assertTrue(metadata_row["environment"]["aqueous_like"])
            self.assertNotIn("raw_api_payload", metadata_row)

    def test_write_snapshot_metadata_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir) / "archive")
            paths.manifest_dir.mkdir(parents=True)
            (paths.manifest_dir / "files.jsonl").write_text(
                json.dumps({"pdb_id": "1ABC", "path": "raw/1abc.bcif.gz", "status": "downloaded"}) + "\n",
                encoding="utf-8",
            )
            stream = io.StringIO()

            summary = write_snapshot_metadata(
                "rcsb-current-bcif",
                paths=paths,
                source="rcsb-api",
                fetcher=fake_graphql_metadata_fetcher,
                status_stream=stream,
            )

            self.assertEqual(summary["entry_count"], 1)
            self.assertIn("metadata progress: 0/1 entries", stream.getvalue())
            self.assertIn("metadata progress: 1/1 entries", stream.getvalue())

    def test_snapshot_download_can_prefetch_metadata_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "archive"
            payload = b"binary cif payload\n"

            def downloader(_url, output):
                output.write_bytes(payload)

            with (
                mock.patch("pheat.archive.fetch_http_json_bytes", side_effect=fake_current_holdings_fetcher),
                mock.patch("pheat.archive.urlretrieve_downloader", side_effect=downloader),
                mock.patch("pheat.archive.write_snapshot_metadata") as metadata_mock,
            ):
                metadata_mock.return_value = {"entry_count": 1, "failure_count": 0}
                status, output = _capture_cli(
                    [
                        "archive",
                        "snapshots",
                        "download",
                        "rcsb-current-bcif",
                        "--output-root",
                        str(output_root),
                        "--max-entries",
                        "1",
                        "--skip-schema-provenance",
                        "--prefetch-metadata",
                        "--metadata-source",
                        "rcsb-api",
                        "--yes",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(_trailing_json_object(output)["metadata"]["entry_count"], 1)
            metadata_mock.assert_called_once()
            filters = json.loads((output_root / "manifests" / "filters.json").read_text(encoding="utf-8"))
            self.assertTrue(filters["prefetch_metadata"])
            self.assertEqual(filters["metadata_source"], "rcsb-api")

    def test_write_snapshot_metadata_can_extract_minimal_local_bcif_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir) / "archive")
            paths.manifest_dir.mkdir(parents=True)
            paths.raw_dir.mkdir(parents=True)
            target = paths.raw_dir / "tiny.bcif.gz"
            target.write_bytes((FIXTURE_DIR / "tiny.bcif.gz").read_bytes())
            (paths.manifest_dir / "files.jsonl").write_text(
                json.dumps({"pdb_id": "TINY", "path": "raw/tiny.bcif.gz", "status": "downloaded"}) + "\n",
                encoding="utf-8",
            )

            summary = write_snapshot_metadata(
                "rcsb-current-bcif",
                paths=paths,
                source="bcif-local",
                workers=1,
                status_stream=None,
            )

            self.assertEqual(summary["source_used"], "bcif-local")
            row = json.loads((paths.manifest_dir / "metadata.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["pdb_id"], "TINY_MMCIF")
            self.assertEqual(row["metadata_source"], "bcif-local")
            self.assertEqual(row["composition"]["atom_count"], 10)

    def test_run_download_prompts_and_can_cancel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = default_archive_paths(Path(tmpdir))
            attempted = []

            def downloader(url, output):
                attempted.append((url, output))

            summary = run_download(
                ["1abc"],
                paths=paths,
                yes=False,
                record_schema_provenance=False,
                downloader=downloader,
                prompt_input=lambda _prompt: "",
            )

            self.assertTrue(summary["cancelled"])
            self.assertEqual(summary["pending"], 1)
            self.assertEqual(attempted, [])
            self.assertFalse((paths.raw_dir / "1abc.cif.gz").exists())

    def test_rcsb_api_is_not_a_dependency(self):
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        environment = (repo_root / "environment.yml").read_text(encoding="utf-8")

        self.assertNotIn("rcsb-api", pyproject)
        self.assertNotIn("rcsbapi", pyproject)
        self.assertNotIn("rcsb-api", environment)
        self.assertNotIn("rcsbapi", environment)

    def test_readme_documents_archive_snapshots_list(self):
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")

        self.assertIn("pheat archive snapshots list", readme)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pheat
from pheat.archive import HttpPayload
from pheat.cli import _build_parser, _emit_startup_status, main as cli_main


def _capture_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli_main(args)
    return status, stdout.getvalue(), stderr.getvalue()


class CliStatusTests(unittest.TestCase):
    def test_startup_status_builder_covers_all_cli_commands(self):
        parser = _build_parser()
        cases = [
            ["pdb-to-structure", "input.pdb", "-o", "heavy.json"],
            ["mmcif-to-structure", "input.cif", "-o", "heavy.json"],
            ["bcif-to-structure", "input.bcif.gz", "-o", "heavy.json"],
            ["structure-to-pdb", "heavy.json", "-o", "out.pdb"],
            ["structure-to-mmcif", "heavy.json", "-o", "out.cif"],
            ["structure-to-geometry", "heavy.json", "-o", "geometry.json"],
            ["pdb-to-geometry", "input.pdb", "-o", "geometry.json"],
            ["mmcif-to-geometry", "input.cif", "-o", "geometry.json"],
            ["bcif-to-geometry", "input.bcif.gz", "-o", "geometry.json"],
            ["geometry-to-structure", "geometry.json", "-o", "heavy.json"],
            ["centroid", "heavy.json", "-o", "centroid.json"],
            ["score", "input.pdb", "--model", "generic"],
            ["radius-of-gyration", "input.pdb"],
            ["rg", "heavy.json", "--mode", "unweighted"],
            ["rmsd", "reference.pdb", "target.pdb"],
            ["examples", "list"],
            ["examples", "show", "diversity"],
            ["examples", "fetch", "diversity"],
            ["examples", "run", "diversity"],
            ["sources", "list"],
            ["sources", "fetch", "demo-source"],
            ["sources", "verify"],
            [
                "geometry",
                "tables",
                "build-backbone",
                "--training-set",
                "selected",
                "--output-root",
                "geometry",
            ],
            [
                "geometry",
                "tables",
                "build-cdl",
                "--training-set",
                "selected",
                "--output-root",
                "geometry",
            ],
            [
                "geometry",
                "tables",
                "import-cdl",
                "--input",
                "cdl.json",
                "--output-root",
                "geometry",
            ],
            [
                "geometry",
                "tables",
                "build-sidechain-ccd",
                "--ccd-dir",
                "ccd",
                "--output-root",
                "geometry",
            ],
            ["geometry", "tables", "list"],
            ["geometry", "tables", "describe", "--table-set", "geometry/geometry-tables.json"],
            ["geometry", "tables", "validate", "--table-set", "geometry/geometry-tables.json"],
            ["scoring", "validate-options", "--model", "gromacs-mdrun"],
            ["gromacs", "prepare", "input.pdb", "-o", "prepared.gro", "--topology", "topol.top"],
            ["gromacs", "score", "input.pdb"],
            ["gromacs", "minimize", "input.pdb", "-o", "minimized.gro"],
            ["gromacs", "validate", "input.pdb"],
            ["gromacs", "validate-options"],
            ["archive", "download", "--ids-file", "ids.txt"],
            ["archive", "snapshots", "list"],
            ["archive", "snapshots", "describe", "rcsb-current-bcif"],
            ["archive", "snapshots", "download", "rcsb-current-bcif"],
            ["archive", "snapshots", "verify", "rcsb-current-bcif"],
            ["archive", "snapshots", "metadata", "rcsb-current-bcif"],
            ["archive", "snapshots", "relocate", "rcsb-current-bcif"],
            ["archive", "snapshots", "ids", "rcsb-current-bcif"],
            ["training", "decoys", "list"],
            ["training", "decoys", "describe", "3drobot"],
            ["training", "decoys", "fetch", "3drobot"],
            ["training", "decoys", "verify"],
            ["training", "corpus", "inventory", "--snapshot-root", "snapshot", "-o", "inventory.jsonl"],
            [
                "training",
                "corpus",
                "select",
                "--inventory",
                "inventory.jsonl",
                "--output-root",
                "selected",
            ],
            ["training", "corpus", "describe", "--training-set", "selected"],
            [
                "training",
                "tables",
                "build",
                "--training-set",
                "selected",
                "--output-root",
                "tables",
            ],
            ["training", "tables", "describe", "--table-set", "tables/score-tables.json"],
            ["training", "tables", "validate", "--table-set", "tables/score-tables.json"],
            ["training", "features", "extract", "--training-set", "selected", "-o", "features.jsonl"],
            ["training", "ml", "train-linear", "--features", "features.jsonl", "-o", "model.json"],
            ["reference", "datasets", "list"],
            ["reference", "datasets", "describe", "3drobot"],
            ["reference", "fetch", "--dataset", "3drobot"],
            ["reference", "inventory", "--snapshot-root", "snapshot", "-o", "inventory.jsonl"],
            ["reference", "select", "--inventory", "inventory.jsonl"],
            ["reference", "build-decoys", "--training-set", "selected"],
            ["reference", "build-scores", "--training-set", "selected"],
            ["reference", "extract-features", "--training-set", "selected", "--decoys", "decoys.jsonl"],
            ["reference", "train-ml", "--features", "features.jsonl"],
            ["reference", "validate", "--features", "features.jsonl"],
            ["reference", "promote", "--source", "artifact-v0", "--destination", "artifact-v1", "--note", "reviewed"],
            ["web"],
        ]

        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    _emit_startup_status(parser.parse_args(argv))
                status = stderr.getvalue()
                self.assertIn(f"pheat {pheat.__version__}", status)
                self.assertIn("command:", status)
                self.assertIn("action:", status)

    def test_score_parser_accepts_supported_optional_models(self):
        parser = _build_parser()
        args = parser.parse_args(["score", "input.pdb", "--model", "openmm-prepared"])
        self.assertEqual(args.model, "openmm-prepared")
        args = parser.parse_args(["score", "input.pdb", "--model", "ambertools-sander", "--prepare", "auto"])
        self.assertEqual(args.model, "ambertools-sander")
        self.assertEqual(args.prepare, "auto")
        args = parser.parse_args(
            [
                "score",
                "input.pdb",
                "--model",
                "gromacs-mdrun",
                "--gromacs-forcefield",
                "amber19sb",
                "--gromacs-run-mode",
                "rerun",
                "--gromacs-preflight",
                "warn",
                "--external-timeout",
                "30",
                "--prep-cache-dir",
                "cache",
                "--prep-cache-mode",
                "readwrite",
                "--gromacs-cutoff",
                "1.2",
                "--gromacs-mdrun-flag=-ntomp",
                "--gromacs-mdrun-flag",
                "1",
            ]
        )
        self.assertEqual(args.model, "gromacs-mdrun")
        self.assertEqual(args.gromacs_forcefield, "amber19sb")
        self.assertEqual(args.gromacs_preflight, "warn")
        self.assertEqual(args.external_timeout, 30.0)
        self.assertEqual(args.prep_cache_mode, "readwrite")
        self.assertEqual(args.gromacs_cutoff, 1.2)
        self.assertEqual(args.gromacs_mdrun_flag, ["-ntomp", "1"])

    def test_json_stdout_remains_parseable_with_status_on_stderr(self):
        status, stdout, stderr = _capture_cli(["archive", "snapshots", "list"])

        self.assertEqual(status, 0)
        self.assertGreater(len(json.loads(stdout)), 0)
        self.assertIn(f"pheat {pheat.__version__}", stderr)
        self.assertIn("command: archive snapshots list", stderr)

    def test_quiet_suppresses_startup_status(self):
        status, stdout, stderr = _capture_cli(["--quiet", "archive", "snapshots", "list"])

        self.assertEqual(status, 0)
        self.assertGreater(len(json.loads(stdout)), 0)
        self.assertEqual(stderr, "")

    def test_quiet_can_still_write_log_file(self):
        def fetcher(url, json_payload=None):
            self.assertIsNone(json_payload)
            self.assertTrue(url.endswith("/holdings/current/entry_ids"))
            return HttpPayload(body=b'["1ABC"]', headers={}, url=url)

        def downloader(_url, output):
            output.write_bytes(b"payload\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "pheat.log"
            with (
                mock.patch("pheat.archive.fetch_http_json_bytes", side_effect=fetcher),
                mock.patch("pheat.archive.urlretrieve_downloader", side_effect=downloader),
            ):
                status, stdout, stderr = _capture_cli(
                    [
                        "--quiet",
                        "--log",
                        str(log_path),
                        "archive",
                        "snapshots",
                        "download",
                        "rcsb-current-bcif",
                        "--output-root",
                        str(root / "archive"),
                        "--max-entries",
                        "1",
                        "--skip-schema-provenance",
                        "--yes",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(json.loads(stdout)["downloaded"], 1)
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn(f"pheat {pheat.__version__}", log_text)
            self.assertIn("command: archive snapshots download rcsb-current-bcif", log_text)
            self.assertIn("download progress [complete]", log_text)

    def test_verbose_twice_prints_traceback_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.json"
            status, _stdout, stderr = _capture_cli(
                ["-vv", "pdb-to-structure", str(Path(tmpdir) / "missing.pdb"), "-o", str(output)]
            )

        self.assertEqual(status, 2)
        self.assertIn("Traceback", stderr)
        self.assertIn("missing.pdb", stderr)

    def test_version_prints_package_version(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli_main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"pheat {pheat.__version__}")

    def test_archive_snapshot_download_reports_before_fetching_holdings(self):
        def fetcher(url, json_payload=None):
            self.assertIsNone(json_payload)
            self.assertTrue(url.endswith("/holdings/current/entry_ids"))
            return HttpPayload(body=b'["1ABC"]', headers={}, url=url)

        with mock.patch("pheat.archive.fetch_http_json_bytes", side_effect=fetcher):
            status, stdout, stderr = _capture_cli(
                [
                    "archive",
                    "snapshots",
                    "download",
                    "rcsb-current-bcif",
                    "--output-root",
                    "/tmp/pheat-bcif",
                    "--staging-dir",
                    "./pheat-bcif-staging",
                    "--max-entries",
                    "1",
                    "--dry-run",
                    "--skip-schema-provenance",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout)["snapshot_id"], "rcsb-current-bcif")
        self.assertIn(f"pheat {pheat.__version__}", stderr)
        self.assertIn("command: archive snapshots download rcsb-current-bcif", stderr)
        self.assertIn("archive snapshot: rcsb-current-bcif", stderr)
        self.assertIn("archive staging dir: ./pheat-bcif-staging", stderr)
        self.assertIn("fetching current RCSB/wwPDB holdings", stderr)


if __name__ == "__main__":
    unittest.main()

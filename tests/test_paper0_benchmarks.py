import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator

from pheat.cli import main as cli_main
from pheat.schemas import load_schema


ROOT = Path(__file__).resolve().parents[1]


def _capture_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli_main(args)
    return status, stdout.getvalue(), stderr.getvalue()


class Paper0BenchmarkTests(unittest.TestCase):
    def test_io_benchmark_script_runs_and_reports_missing_comparator(self):
        spec = ROOT / "benchmarks" / "paper-0" / "corpus_specs" / "user_defined_ids_demo.yml"
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, stderr = _capture_cli(
                [
                    "--quiet",
                    "reference",
                    "build",
                    "--corpus-spec",
                    str(spec),
                    "--output-root",
                    str(Path(tmpdir) / "corpus"),
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0, stderr)
            manifest = json.loads(stdout)["manifest"]
            output = Path(tmpdir) / "io.jsonl"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "paper-0" / "scripts" / "02_compare_io_tools.py"),
                    "--manifest",
                    manifest,
                    "--output",
                    str(output),
                    "--comparators",
                    "pheat,gemmi",
                ],
                cwd=ROOT,
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["comparator"] == "pheat" and row["parse_success"] for row in rows))
            self.assertTrue(any(row["comparator"] == "gemmi" for row in rows))
            for row in rows:
                Draft202012Validator(load_schema("benchmark-result")).validate(row)


if __name__ == "__main__":
    unittest.main()

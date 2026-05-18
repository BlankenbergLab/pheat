import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from pheat.ccd import annotate_structure_components, load_component_table
from pheat.cli import main as cli_main
from pheat.pdbio import load_pdb
from pheat.schemas import load_schema, validate_json_object


ROOT = Path(__file__).resolve().parents[1]


def _capture_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli_main(args)
    return status, stdout.getvalue(), stderr.getvalue()


class ReferenceCorpusTests(unittest.TestCase):
    def test_example_corpus_spec_validates_and_invalid_spec_fails(self):
        spec = ROOT / "examples" / "corpora" / "user_defined_ids_demo.yml"
        status, stdout, stderr = _capture_cli(["--quiet", "reference", "validate-spec", str(spec)])

        self.assertEqual(status, 0, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["corpus_id"], "pheat-user-defined-ids-demo")

        with self.assertRaises(ValueError):
            validate_json_object({"version": "v1", "source": {"type": "id_list", "format": "pdb"}}, "corpus-spec")

    def test_reference_build_writes_manifest_checksums_and_valid_rows(self):
        spec = ROOT / "examples" / "corpora" / "user_defined_ids_demo.yml"
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, stderr = _capture_cli(
                [
                    "--quiet",
                    "reference",
                    "build",
                    "--corpus-spec",
                    str(spec),
                    "--output-root",
                    tmpdir,
                    "--overwrite",
                ]
            )

            self.assertEqual(status, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["selected_count"], 2)
            self.assertEqual(payload["excluded_count"], 0)
            manifest = Path(payload["manifest"])
            checksums = Path(payload["checksums"])
            self.assertTrue(manifest.exists())
            self.assertTrue(checksums.exists())

            schema = load_schema("reference-dataset-manifest")
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                Draft202012Validator(schema).validate(row)
                self.assertTrue(row["selected"])
                self.assertIn("atom_structure_json", row["generated_artifacts"])

    def test_dry_run_rcsb_style_spec_writes_plan_without_network(self):
        spec = ROOT / "examples" / "corpora" / "xray_aqueous_30id_v1.yml"
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, stderr = _capture_cli(
                [
                    "--quiet",
                    "reference",
                    "build",
                    "--corpus-spec",
                    str(spec),
                    "--output-root",
                    tmpdir,
                    "--dry-run",
                    "--overwrite",
                ]
            )

            self.assertEqual(status, 0, stderr)
            payload = json.loads(stdout)
            self.assertTrue(payload["dry_run"])
            self.assertTrue((Path(tmpdir) / "plan.json").exists())

    def test_ccd_annotation_unknown_and_local_override(self):
        structure = load_pdb(ROOT / "tests" / "fixtures" / "tiny.pdb")
        fallback = annotate_structure_components(structure)
        ion = next(item for item in fallback["heterogen_summary"] if item["component_id"] == "ZN")
        self.assertEqual(ion["classification"], "ion")
        self.assertIsNone(ion["name"])

        with tempfile.TemporaryDirectory() as tmpdir:
            table = Path(tmpdir) / "ccd.csv"
            table.write_text(
                "component_id,component_type,name,formula,atom_count,classification\n"
                "ZN,non-polymer,zinc ion,Zn,1,ligand\n",
                encoding="utf-8",
            )
            annotated = annotate_structure_components(
                structure,
                component_table=load_component_table(table),
            )

        ligand = next(item for item in annotated["ligand_summary"] if item["component_id"] == "ZN")
        self.assertEqual(ligand["name"], "zinc ion")
        self.assertEqual(ligand["classification"], "ligand")


if __name__ == "__main__":
    unittest.main()

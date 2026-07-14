import contextlib
import copy
import gzip
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from pheat.cli import main as cli_main
from pheat.geometry import distance
from pheat.geometry_tables import GEOMETRY_TABLE_SET_FORMAT
from pheat.models import Atom, HeavyAtomStructure
from pheat.pdbio import load_heavy_json, load_pdb
from pheat.residue_geometry import residue_angle_specs, residue_geometry_structure_from_sequence, structure_from_residue_geometry
from pheat.schemas import load_schema
from pheat.score_tables import (
    load_score_table_set,
    load_packaged_score_table_set,
    packaged_score_table_ids,
    packaged_score_table_manifest,
    write_score_table_set,
)
from pheat.scoring import (
    GromacsRunSettings,
    available_models,
    model_capabilities,
    parse_gromacs_xvg_terms,
    parse_sander_energy_terms,
    score_model_option_specs,
    score_structure,
    supported_models,
    validate_external_scoring_options,
    validate_scoring_options,
)
from pheat.sasa import SASA_BACKENDS, residue_burial
from pheat.training import (
    _accumulate_distance_bins,
    _accumulate_hbond,
    _accumulate_mj,
    build_score_tables,
    list_decoy_datasets,
    normalize_workers,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
TINY_PDB = FIXTURE_DIR / "tiny.pdb"
TWO_MU7_PDB = FIXTURE_DIR / "2mu7.pdb"
AG_HEAVY = FIXTURE_DIR / "expected" / "ag.heavy.json"
CCD_DIR = FIXTURE_DIR / "ccd"
REFERENCE_ARCHIVE_V0 = FIXTURE_DIR / "reference_archive_v0"

MULTICHAIN_PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       1.300   2.420   0.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       1.900  -0.700  -1.200  1.00 10.00           C
TER
ATOM      6  N   GLY B   1      10.000   0.000   0.000  1.00 10.00           N
ATOM      7  CA  GLY B   1      11.458   0.000   0.000  1.00 10.00           C
ATOM      8  C   GLY B   1      12.000   1.420   0.000  1.00 10.00           C
ATOM      9  O   GLY B   1      11.300   2.420   0.000  1.00 10.00           O
TER
END
"""


def _capture_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli_main(args)
    return status, stdout.getvalue(), stderr.getvalue()


def _write_snapshot(root, rows):
    manifest_dir = root / "manifests"
    raw_dir = root / "raw"
    manifest_dir.mkdir(parents=True)
    raw_dir.mkdir(exist_ok=True)
    (manifest_dir / "files.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class TrainingCommandTests(unittest.TestCase):
    def test_training_contact_accumulators_use_spatial_cutoffs(self):
        residue_points = {
            ("A", 1, "", "ALA", "ATOM"): (0.0, 0.0, 0.0),
            ("A", 4, "", "GLY", "ATOM"): (6.0, 0.0, 0.0),
            ("B", 1, "", "VAL", "ATOM"): (20.0, 0.0, 0.0),
            ("C", 1, "", "LEU", "ATOM"): (21.0, 0.0, 0.0),
        }
        accumulators = {
            "pheat-dfire": {"counts": {}, "totals": {}},
            "pheat-goap": {"counts": {}, "totals": {}},
        }
        _accumulate_distance_bins(accumulators, residue_points)
        self.assertEqual(accumulators["pheat-dfire"]["counts"]["ALA-GLY"]["06"], 1)
        self.assertEqual(accumulators["pheat-dfire"]["counts"]["ALA-VAL"]["20"], 1)
        self.assertEqual(accumulators["pheat-dfire"]["counts"]["GLY-VAL"]["14"], 1)
        self.assertEqual(accumulators["pheat-dfire"]["counts"]["GLY-LEU"]["15"], 1)
        self.assertEqual(accumulators["pheat-dfire"]["counts"]["LEU-VAL"]["01"], 1)
        self.assertNotIn("ALA-LEU", accumulators["pheat-dfire"]["counts"])
        self.assertEqual(accumulators["pheat-goap"]["counts"], accumulators["pheat-dfire"]["counts"])

        mj_accumulator = {"counts": {}, "totals": {}}
        _accumulate_mj(mj_accumulator, residue_points)
        self.assertEqual(mj_accumulator["counts"], {"ALA-GLY": 1, "LEU-VAL": 1})

        hbond_accumulator = {"counts": {}, "totals": {}}
        structure = HeavyAtomStructure(
            atoms=[
                Atom("N", "N", 0.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1),
                Atom("O", "O", 3.0, 0.0, 0.0, "GLY", chain_id="A", resseq=2),
                Atom("O", "O", 3.0, 0.0, 0.0, "SER", chain_id="A", resseq=4),
                Atom("S", "S", 3.5, 0.0, 0.0, "CYS", chain_id="B", resseq=1),
                Atom("C", "C", 3.0, 0.0, 0.0, "ALA", chain_id="C", resseq=1),
            ],
        )
        _accumulate_hbond(hbond_accumulator, structure)
        self.assertEqual(hbond_accumulator["counts"], {"N-O": 1, "N-S": 1})

    def test_sasa_backend_surface_uses_freesasa_not_biopython(self):
        self.assertEqual(SASA_BACKENDS, ("auto", "freesasa"))
        self.assertNotIn("biopython", SASA_BACKENDS)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli_main(
                    [
                        "training",
                        "tables",
                        "build",
                        "--training-set",
                        str(FIXTURE_DIR),
                        "--output-root",
                        str(FIXTURE_DIR),
                        "--burial-method",
                        "sasa",
                        "--sasa-backend",
                        "biopython",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_sasa_missing_freesasa_error_is_actionable(self):
        structure = load_pdb(TINY_PDB)
        with mock.patch.dict("sys.modules", {"freesasa": None}):
            with self.assertRaisesRegex(RuntimeError, "FreeSASA is required"):
                residue_burial(structure, method="sasa", backend="auto")

    def test_score_table_build_uses_selected_chain_not_full_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_path = root / "two-chain.heavy.json"
            structure = HeavyAtomStructure(
                atoms=[
                    Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1),
                    Atom("CB", "C", 0.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1),
                    Atom("CA", "C", 6.0, 0.0, 0.0, "GLY", chain_id="A", resseq=4),
                    Atom("CA", "C", 1.0, 0.0, 0.0, "VAL", chain_id="B", resseq=1),
                    Atom("CB", "C", 1.0, 0.0, 0.0, "VAL", chain_id="B", resseq=1),
                    Atom("CA", "C", 2.0, 0.0, 0.0, "LEU", chain_id="B", resseq=4),
                    Atom("CB", "C", 2.0, 0.0, 0.0, "LEU", chain_id="B", resseq=4),
                ],
                name="two-chain",
            )
            structure_path.write_text(structure.to_json(), encoding="utf-8")
            training_set = root / "selected"
            training_set.mkdir()
            entry = {
                "pdb_id": "TEST",
                "chain_id": "A",
                "path": str(structure_path),
                "sequence": "AG",
                "length": 2,
                "status": "ok",
            }
            (training_set / "selected.jsonl").write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")

            payload = build_score_tables(
                training_set,
                output_root=root / "tables",
                models=("pheat-dfire",),
                max_entries=1,
                workers=1,
            )

            counts = payload["profiles"]["protein-heavy-contacts"]["models"]["pheat-dfire"]["counts"]
            self.assertEqual(counts, {"ALA-GLY": {"06": 1}})

    def test_archive_snapshot_ids_exports_successful_local_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "snapshot"
            raw_dir = root / "raw"
            good = raw_dir / "1abc.pdb"
            failed = raw_dir / "2def.pdb"
            raw_dir.mkdir(parents=True)
            good.write_text(TINY_PDB.read_text(encoding="utf-8"), encoding="utf-8")
            rows = [
                {
                    "pdb_id": "1ABC",
                    "path": str(good),
                    "status": "downloaded",
                    "checksums": {"sha256": hashlib.sha256(good.read_bytes()).hexdigest()},
                },
                {"pdb_id": "2DEF", "path": str(failed), "status": "failed"},
                {"pdb_id": "3GHI", "path": str(good), "status": "present"},
            ]
            _write_snapshot(root, rows)

            status, stdout, stderr = _capture_cli(
                ["archive", "snapshots", "ids", "rcsb-current-bcif", "--output-root", str(root)]
            )

            self.assertEqual(status, 0)
            self.assertEqual(stdout.splitlines(), ["1ABC", "3GHI"])
            self.assertIn("action: write local archive snapshot IDs", stderr)

    def test_training_inventory_select_and_contact_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshot"
            raw = snapshot / "raw"
            raw.mkdir(parents=True)
            pdb_path = raw / "1abc.pdb"
            pdb_path.write_text(TINY_PDB.read_text(encoding="utf-8"), encoding="utf-8")
            _write_snapshot(
                snapshot,
                [
                    {
                        "pdb_id": "1ABC",
                        "path": str(pdb_path),
                        "status": "downloaded",
                        "checksums": {"sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest()},
                    }
                ],
            )
            inventory_path = root / "inventory.jsonl"
            selected_root = root / "selected"
            tables_root = root / "tables"

            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "corpus",
                    "inventory",
                    "--snapshot-root",
                    str(snapshot),
                    "--domain",
                    "protein-heavy",
                    "-o",
                    str(inventory_path),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["count"], 1)
            row = json.loads(inventory_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["pdb_id"], "1ABC")
            self.assertEqual(row["chain_id"], "A")
            self.assertEqual(row["sequence"], "AG")

            clusters_path = root / "clusters-by-30.txt"
            clusters_path.write_text("1ABC_A\n", encoding="utf-8")
            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "corpus",
                    "select",
                    "--inventory",
                    str(inventory_path),
                    "--output-root",
                    str(selected_root),
                    "--sequence-clusters",
                    str(clusters_path),
                    "--min-length",
                    "1",
                    "--max-length",
                    "10",
                ]
            )
            self.assertEqual(status, 0)
            selected = json.loads((selected_root / "selected.json").read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("training-corpus")).validate(selected)
            self.assertEqual(selected["artifact_id"], "protein-heavy-30id")
            self.assertEqual(selected["artifact_version"], "v1")
            self.assertEqual(selected["filters"]["sequence_identity"], 0.30)
            self.assertEqual(selected["filters"]["sequence_identity_label"], "30id")
            self.assertEqual(selected["selected_count"], 1)
            self.assertEqual(selected["filters"]["cluster_source"], str(clusters_path))
            self.assertEqual(selected["provenance"]["source_snapshot"]["snapshot_id"], "snapshot")
            self.assertEqual(len(selected["selected_entry_sha256"]), 64)

            status, stdout, _stderr = _capture_cli(
                ["training", "corpus", "describe", "--training-set", str(selected_root)]
            )
            self.assertEqual(status, 0)
            corpus_description = json.loads(stdout)
            self.assertEqual(corpus_description["artifact_id"], "protein-heavy-30id")
            self.assertEqual(corpus_description["entry_count"], 1)

            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "tables",
                    "build",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(tables_root),
                    "--models",
                    "pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond",
                    "--burial-method",
                    "contacts",
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads((tables_root / "score-tables.json").read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("score-table-set")).validate(payload)
            self.assertEqual(payload["artifact_id"], "protein-heavy-30id-contacts")
            self.assertEqual(payload["artifact_version"], "v1")
            self.assertEqual(payload["metadata"]["burial_method"], "contacts")
            self.assertEqual(payload["metadata"]["source_corpus"]["artifact_id"], "protein-heavy-30id")
            self.assertEqual(payload["default_profile"], "protein-heavy-30id-contacts")
            contact_profile = payload["profiles"]["protein-heavy-30id-contacts"]
            self.assertIn("pheat-mj", contact_profile["models"])
            self.assertIn("pheat-hydropathy", contact_profile["models"])
            self.assertEqual(
                contact_profile["models"]["pheat-mj"]["input_contract"]["id"],
                "pheat.score-contract.pheat-mj.v1",
            )

            status, stdout, _stderr = _capture_cli(
                ["training", "tables", "describe", "--table-set", str(tables_root / "score-tables.json")]
            )
            self.assertEqual(status, 0)
            table_description = json.loads(stdout)
            self.assertEqual(table_description["artifact_id"], "protein-heavy-30id-contacts")
            self.assertIn("protein-heavy-30id-contacts", table_description["profiles"])

            try:
                import freesasa  # noqa: F401
            except Exception:
                pass
            else:
                sasa_root = root / "tables-sasa"
                status, stdout, _stderr = _capture_cli(
                    [
                        "training",
                        "tables",
                        "build",
                        "--training-set",
                        str(selected_root),
                        "--output-root",
                        str(sasa_root),
                        "--models",
                        "pheat-hydropathy",
                        "--burial-method",
                        "sasa",
                        "--sasa-backend",
                        "freesasa",
                    ]
                )
                self.assertEqual(status, 0)
                sasa_payload = json.loads((sasa_root / "score-tables.json").read_text(encoding="utf-8"))
                self.assertEqual(sasa_payload["metadata"]["burial_method"], "sasa")
                self.assertIn("protein-heavy-30id-sasa", sasa_payload["profiles"])

                both_root = root / "tables-both"
                status, stdout, _stderr = _capture_cli(
                    [
                        "training",
                        "tables",
                        "build",
                        "--training-set",
                        str(selected_root),
                        "--output-root",
                        str(both_root),
                        "--models",
                        "pheat-hydropathy",
                        "--burial-method",
                        "both",
                        "--sasa-backend",
                        "freesasa",
                    ]
                )
                self.assertEqual(status, 0)
                both_payload = json.loads((both_root / "score-tables.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    set(both_payload["profiles"]),
                    {"protein-heavy-30id-contacts", "protein-heavy-30id-sasa"},
                )

            status, stdout, _stderr = _capture_cli(
                ["training", "tables", "validate", "--table-set", str(tables_root / "score-tables.json")]
            )
            self.assertEqual(status, 0)
            validate_payload = json.loads(stdout)
            self.assertTrue(validate_payload["ok"])
            self.assertIn("protein-heavy-30id-contacts", validate_payload["profiles"])

            features_path = root / "features.jsonl"
            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "features",
                    "extract",
                    "--training-set",
                    str(selected_root),
                    "--models",
                    "generic,pheat-dfire,heavy-mm",
                    "-o",
                    str(features_path),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["feature_count"], 1)
            feature_row = json.loads(features_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("generic_total", feature_row)
            self.assertIn("heavy-mm_total", feature_row)

            linear_path = root / "pheat-ml-linear.json"
            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "ml",
                    "train-linear",
                    "--features",
                    str(features_path),
                    "-o",
                    str(linear_path),
                ]
            )
            self.assertEqual(status, 0)
            linear_payload = json.loads(stdout)
            linear_model = linear_payload["profiles"]["protein-heavy-contacts"]["models"]["pheat-ml-linear"]
            self.assertEqual(linear_model["target"], "native_vs_decoy")

    def test_training_corpus_versioning_and_member_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inventory_path = root / "inventory.jsonl"
            rows = [
                {
                    "format": "pheat.training-inventory-row",
                    "version": 1,
                    "status": "ok",
                    "pdb_id": "1ABC",
                    "chain_id": "A",
                    "path": str(TINY_PDB),
                    "domain": "protein-heavy",
                    "sequence": "AG",
                    "length": 2,
                    "canonical_fraction": 1.0,
                    "missing_backbone_atom_count": 0,
                },
                {
                    "format": "pheat.training-inventory-row",
                    "version": 1,
                    "status": "ok",
                    "pdb_id": "2DEF",
                    "chain_id": "A",
                    "path": str(TINY_PDB),
                    "domain": "protein-heavy",
                    "sequence": "AG",
                    "length": 2,
                    "canonical_fraction": 1.0,
                    "missing_backbone_atom_count": 0,
                },
                {
                    "format": "pheat.training-inventory-row",
                    "version": 1,
                    "status": "ok",
                    "pdb_id": "3GHI",
                    "chain_id": "B",
                    "path": str(TINY_PDB),
                    "domain": "protein-heavy",
                    "sequence": "GG",
                    "length": 2,
                    "canonical_fraction": 1.0,
                    "missing_backbone_atom_count": 0,
                },
            ]
            inventory_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            include_path = root / "include.txt"
            exclude_path = root / "exclude.txt"
            holdout_path = root / "holdout.txt"
            include_path.write_text("1ABC_A\n2DEF_A\n3GHI_B\n", encoding="utf-8")
            exclude_path.write_text("2DEF_A\n", encoding="utf-8")
            holdout_path.write_text("3GHI_B\n", encoding="utf-8")
            selected_root = root / "protein-heavy-12p5id-xray-v2"

            status, stdout, _stderr = _capture_cli(
                [
                    "training",
                    "corpus",
                    "select",
                    "--inventory",
                    str(inventory_path),
                    "--output-root",
                    str(selected_root),
                    "--sequence-identity",
                    "12.5%",
                    "--corpus-id",
                    "protein-heavy-12p5id-xray",
                    "--corpus-version",
                    "v2",
                    "--include-file",
                    str(include_path),
                    "--exclude-file",
                    str(exclude_path),
                    "--holdout-file",
                    str(holdout_path),
                    "--min-length",
                    "1",
                    "--max-length",
                    "10",
                ]
            )
            self.assertEqual(status, 0)
            result = json.loads(stdout)
            self.assertEqual(result["artifact_id"], "protein-heavy-12p5id-xray")
            self.assertEqual(result["artifact_version"], "v2")
            self.assertEqual(result["artifact_label"], "protein-heavy-12p5id-xray-v2")
            self.assertEqual(result["filters"]["sequence_identity"], 0.125)
            self.assertEqual(result["filters"]["sequence_identity_label"], "12p5id")
            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(result["excluded_count"], 1)
            self.assertEqual(result["holdout_count"], 1)
            self.assertEqual(result["provenance"]["member_files"]["include"]["member_count"], 3)
            self.assertEqual((selected_root / "holdout.jsonl").read_text(encoding="utf-8").count("\n"), 1)

            status, stdout, _stderr = _capture_cli(
                ["training", "corpus", "describe", "--training-set", str(selected_root)]
            )
            self.assertEqual(status, 0)
            description = json.loads(stdout)
            self.assertEqual(description["artifact_version"], "v2")
            self.assertEqual(description["holdout_count"], 1)

    def test_scoring_domains_table_sets_and_new_models(self):
        structure = load_pdb(TINY_PDB)
        self.assertIn("pheat-mj", available_models())
        self.assertIn("pheat-rg", available_models())
        self.assertIn("pheat-geometry-integrity", available_models())
        self.assertIn("openmm-prepared", supported_models())
        generic = score_structure(structure, model="generic")
        full = score_structure(structure, model="generic", domain="all-heavy")
        self.assertEqual(generic.metadata["domain"], "protein-heavy")
        self.assertEqual(generic.metadata["input_contract"]["id"], "pheat.score-contract.generic.v1")
        self.assertEqual(generic.metadata["coverage"]["input_atom_scope"], "heavy")
        self.assertGreater(full.metadata["coverage"]["scored_atom_count"], generic.metadata["coverage"]["scored_atom_count"])

        table_set = {
            "format": "pheat.score-table-set",
            "version": 1,
            "default_profile": "protein-heavy-contacts",
            "metadata": {"test": True},
            "profiles": {
                "protein-heavy-contacts": {
                    "metadata": {"burial_method": "contacts"},
                    "models": {
                        "pheat-mj": {
                            "type": "pheat-mj",
                            "counts": {"ALA-GLY": 3},
                            "generated_by": "test",
                        },
                        "pheat-hydropathy": {
                            "type": "pheat-hydropathy",
                            "hydropathy_scale": {"ALA": 1.8, "GLY": -0.4},
                            "burial_method": "contacts",
                            "generated_by": "test",
                        },
                        "pheat-rg": {
                            "type": "pheat-rg",
                            "atom_set": "backbone",
                            "mode": "unweighted",
                            "a": 1.5,
                            "b": 0.3333333333333333,
                            "sigma_fraction": 0.5,
                            "fitted": True,
                            "generated_by": "test",
                        },
                    },
                },
                "protein-heavy-sasa": {
                    "metadata": {"burial_method": "sasa"},
                    "models": {
                        "pheat-hydropathy": {
                            "type": "pheat-hydropathy",
                            "hydropathy_scale": {"ALA": 1.8, "GLY": -0.4},
                            "burial_method": "sasa",
                            "generated_by": "test",
                        },
                    },
                },
            },
        }
        mj = score_structure(structure, model="pheat-mj", table_set=table_set)
        hydropathy = score_structure(structure, model="pheat-hydropathy", table_set=table_set)
        pheat_rg = score_structure(structure, model="pheat-rg", table_set=table_set)
        self.assertEqual(mj.model, "pheat-mj")
        self.assertEqual(mj.metadata["profile"], "protein-heavy-contacts")
        self.assertEqual(mj.metadata["input_contract"]["compatible_domains"], ["protein-heavy"])
        self.assertEqual(hydropathy.metadata["burial_method"], "contacts")
        self.assertEqual(pheat_rg.model, "pheat-rg")
        self.assertEqual(pheat_rg.metadata["atom_set"], "backbone")
        self.assertEqual(pheat_rg.metadata["mode"], "unweighted")
        self.assertTrue(pheat_rg.metadata["fitted"])
        self.assertAlmostEqual(pheat_rg.metadata["coefficients"]["a"], 1.5)
        all_heavy_mj = score_structure(structure, model="pheat-mj", domain="all-heavy", table_set=table_set)
        self.assertIn("requested domain all-heavy", "\n".join(all_heavy_mj.warnings))
        self.assertIn("expected_rg", pheat_rg.terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "score-tables.json"
            table_path.write_text(json.dumps(table_set), encoding="utf-8")
            profile_results = _capture_cli(
                [
                    "score",
                    str(AG_HEAVY),
                    "--model",
                    "pheat-hydropathy",
                    "--table-set",
                    str(table_path),
                    "--profiles",
                    "protein-heavy-contacts,protein-heavy-sasa",
                ]
            )
            self.assertEqual(profile_results[0], 0)
            profile_payload = json.loads(profile_results[1])
            self.assertEqual(
                set(profile_payload["profiles"]),
                {"protein-heavy-contacts", "protein-heavy-sasa"},
            )

    def test_packaged_v0_score_table_assets_load_and_are_sanitized(self):
        expected_ids = {
            "pheat-ml-linear-aqueous-v0",
            "pheat-ml-linear-membrane-v0",
            "protein-heavy-30id-xray-aqueous-v0",
            "protein-heavy-30id-xray-membrane-v0",
        }
        manifest = packaged_score_table_manifest()
        self.assertEqual(set(packaged_score_table_ids()), expected_ids)
        self.assertEqual(manifest["artifact_version"], "v0")
        self.assertTrue(manifest["provisional"])
        self.assertEqual(manifest["status"], "initial")
        self.assertEqual(
            {row["pdb_id"] for row in manifest["source_snapshot"]["metadata_failures"]},
            {"4M4C", "9KZM", "9MBW"},
        )

        schema = load_schema("score-table-set")
        text_payloads = [json.dumps(manifest, sort_keys=True)]
        for asset_id in sorted(expected_ids):
            asset = manifest["assets"][asset_id]
            self.assertEqual(asset["compression"], "xz")
            self.assertEqual(asset["compression_preset"], 6)
            packaged_path = REPO_ROOT / "src" / "pheat" / "data" / "scoring" / "v0" / asset["path"]
            self.assertTrue(packaged_path.exists())
            self.assertTrue(packaged_path.name.endswith(".json.xz"))
            compressed = packaged_path.read_bytes()
            self.assertEqual(hashlib.sha256(compressed).hexdigest(), asset["sha256"])
            uncompressed = lzma.decompress(compressed)
            self.assertEqual(len(uncompressed), asset["uncompressed_bytes"])
            self.assertEqual(hashlib.sha256(uncompressed).hexdigest(), asset["uncompressed_sha256"])
            self.assertFalse(packaged_path.with_suffix("").exists())

            payload = load_packaged_score_table_set(asset_id)
            Draft202012Validator(schema).validate(payload)
            self.assertEqual(payload["artifact_version"], "v0")
            self.assertTrue(payload["provisional"])
            self.assertEqual(payload["metadata"]["packaged_asset_id"], asset_id)
            text_payloads.append(json.dumps(payload, sort_keys=True))

        committed_payload_text = "\n".join(text_payloads)
        forbidden_path_markers = [
            "/".join(("", "Users", "")),
            "/".join(("", "mnt", "")),
            "/".join(("", "private", "")),
        ]
        for marker in forbidden_path_markers:
            self.assertNotIn(marker, committed_payload_text)

    def test_score_table_set_loads_and_writes_xz_paths(self):
        payload = load_packaged_score_table_set("pheat-ml-linear-aqueous-v0")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "score-tables.json.xz"
            write_score_table_set(output, payload)
            self.assertTrue(output.exists())
            self.assertEqual(load_score_table_set(output), payload)

    def test_reference_package_scoring_assets_compresses_fixture_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "packaged"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "package-scoring-assets",
                    "--reference-root",
                    str(REFERENCE_ARCHIVE_V0),
                    "--artifact-version",
                    "v0",
                    "--destination-root",
                    str(destination),
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            manifest = json.loads(stdout)
            self.assertEqual(manifest["artifact_version"], "v0")
            self.assertEqual(manifest["format"], "pheat.packaged-score-assets")
            self.assertEqual(
                set(manifest["assets"]),
                {"pheat-ml-linear-aqueous-v0", "protein-heavy-30id-xray-aqueous-v0"},
            )
            self.assertGreaterEqual(len(manifest["skipped_assets"]), 1)
            for asset_id, asset in manifest["assets"].items():
                self.assertEqual(asset["compression"], "xz")
                path = destination / asset["path"]
                self.assertTrue(path.exists())
                self.assertTrue(path.name.endswith(".json.xz"))
                compressed = path.read_bytes()
                self.assertEqual(hashlib.sha256(compressed).hexdigest(), asset["sha256"])
                raw = lzma.decompress(compressed)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), asset["uncompressed_sha256"])
                payload = load_score_table_set(path)
                self.assertEqual(payload["metadata"]["packaged_asset_id"], asset_id)

    def test_cli_uses_packaged_v0_score_table_assets(self):
        status, stdout, _stderr = _capture_cli(
            [
                "training",
                "tables",
                "describe",
                "--table-set",
                "packaged:protein-heavy-30id-xray-aqueous-v0",
            ]
        )
        self.assertEqual(status, 0)
        description = json.loads(stdout)
        self.assertEqual(description["artifact_id"], "protein-heavy-30id-xray-aqueous")
        self.assertEqual(description["artifact_version"], "v0")
        self.assertIn("protein-heavy-30id-xray-aqueous-contacts", description["profiles"])
        self.assertIn("protein-heavy-30id-xray-aqueous-sasa", description["profiles"])

        status, stdout, _stderr = _capture_cli(
            [
                "training",
                "tables",
                "validate",
                "--table-set",
                "packaged:protein-heavy-30id-xray-aqueous-v0",
            ]
        )
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(stdout)["ok"])

        status, stdout, _stderr = _capture_cli(
            [
                "score",
                str(TINY_PDB),
                "--model",
                "pheat-dfire",
                "--table-set",
                "packaged:protein-heavy-30id-xray-aqueous-v0",
            ]
        )
        self.assertEqual(status, 0)
        dfire_score = json.loads(stdout)
        self.assertEqual(dfire_score["model"], "pheat-dfire")
        self.assertEqual(
            dfire_score["metadata"]["profile"],
            "protein-heavy-30id-xray-aqueous-contacts",
        )
        self.assertEqual(dfire_score["metadata"]["table_set"], "pheat training tables build")

        status, stdout, _stderr = _capture_cli(
            [
                "score",
                str(TINY_PDB),
                "--model",
                "pheat-ml-linear",
                "--table-set",
                "packaged:pheat-ml-linear-aqueous-v0",
            ]
        )
        self.assertEqual(status, 0)
        ml_score = json.loads(stdout)
        self.assertEqual(ml_score["model"], "pheat-ml-linear")
        self.assertEqual(ml_score["metadata"]["profile"], "protein-heavy-contacts")
        self.assertEqual(ml_score["metadata"]["table_set"], "linear-feature-combination")
        self.assertIn("pheat-dfire_total", ml_score["metadata"]["features"])

    def test_reference_version_audit_uses_tiny_fixture(self):
        status, stdout, _stderr = _capture_cli(
            [
                "reference",
                "audit-version",
                "--reference-root",
                str(REFERENCE_ARCHIVE_V0),
                "--artifact-version",
                "v0",
            ]
        )
        self.assertEqual(status, 0)
        audit = json.loads(stdout)
        self.assertTrue(audit["ok"])
        self.assertGreaterEqual(audit["checked_file_count"], 5)
        self.assertEqual(audit["issue_count"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_root = Path(tmpdir) / "reference"
            shutil.copytree(REFERENCE_ARCHIVE_V0, bad_root)
            bad_file = bad_root / "runs" / "v0" / "bad-v1.json"
            bad_file.write_text(
                json.dumps({"artifact_version": "v1", "format": "pheat.fixture"}, sort_keys=True),
                encoding="utf-8",
            )
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "audit-version",
                    "--reference-root",
                    str(bad_root),
                    "--artifact-version",
                    "v0",
                ]
            )
            self.assertEqual(status, 1)
            audit = json.loads(stdout)
            self.assertFalse(audit["ok"])
            self.assertGreaterEqual(audit["issue_count"], 1)

    def test_residue_angle_specs_are_pheat_native_and_stable(self):
        specs = residue_angle_specs("GAP", stored_angles="all", angle_units="radians")
        Draft202012Validator(load_schema("residue-angle-specs")).validate(
            {"format": "pheat.residue-angle-specs", "version": 1, "specs": specs}
        )
        keys = [item["key"] for item in specs]
        self.assertEqual(keys[:7], ["0_phi", "0_psi", "1_phi", "1_psi", "2_phi", "2_psi", "2_chi1"])
        self.assertIn("0_omega", keys)
        self.assertIn("0_tau", keys)
        self.assertIn("0_theta", keys)
        self.assertNotIn("2_omega", keys)
        self.assertNotIn("2_theta", keys)
        self.assertIn("2_tau", keys)
        self.assertTrue(all("res" not in item and "type" not in item for item in specs))
        first_phi = specs[0]
        self.assertEqual(first_phi["residue_index"], 0)
        self.assertEqual(first_phi["residue_number"], 1)
        self.assertEqual(first_phi["angle_name"], "phi")
        self.assertFalse(first_phi["coordinate_defined"])
        omega = next(item for item in specs if item["key"] == "0_omega")
        self.assertEqual(omega["applies_to"], "peptide-link-to-next")
        self.assertEqual(omega["peptide_link_to_residue_index"], 1)

        all_chi = residue_angle_specs("KW")
        self.assertIn("0_chi4", [item["key"] for item in all_chi])
        no_chi = residue_angle_specs("KW", max_chi=0)
        self.assertFalse([item for item in no_chi if item["category"] == "sidechain-dihedral"])
        chi1_only = residue_angle_specs("KW", max_chi=1)
        self.assertEqual([item["angle_name"] for item in chi1_only if item["category"] == "sidechain-dihedral"], ["chi1", "chi1"])
        max_chi = residue_angle_specs("KW", max_chi=2)
        self.assertNotIn("chi3", [item["angle_name"] for item in max_chi])
        selective = residue_angle_specs("KW", selective_chi_map={"LYS": ["chi1", "chi3"], "W": ["chi2"]})
        self.assertEqual(
            [item["key"] for item in selective if item["category"] == "sidechain-dihedral"],
            ["0_chi1", "0_chi3", "1_chi2"],
        )
        selective_max_chi = residue_angle_specs(
            "KW",
            selective_chi_map={"LYS": ["chi1", "chi3"], "W": ["chi2"]},
            max_chi=2,
        )
        self.assertEqual(
            [item["key"] for item in selective_max_chi if item["category"] == "sidechain-dihedral"],
            ["0_chi1", "1_chi2"],
        )
        with self.assertRaises(TypeError):
            residue_angle_specs("KW", chi_mode="selective")  # type: ignore[call-arg]

    def test_geometry_integrity_scorer_detects_distortion_robustly(self):
        ideal = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        baseline = score_structure(ideal, model="pheat-geometry-integrity")
        self.assertEqual(baseline.model, "pheat-geometry-integrity")
        self.assertTrue(math.isfinite(baseline.total))
        self.assertLess(baseline.total, 1e-6)
        self.assertEqual(baseline.metadata["input_contract"]["id"], "pheat.score-contract.pheat-geometry-integrity.v1")
        self.assertEqual(baseline.metadata["constants"]["weights"]["peptide_planarity"], 20.0)
        self.assertEqual(baseline.metadata["constants"]["planarity_target"], "cis-or-trans")
        cis = structure_from_residue_geometry(
            residue_geometry_structure_from_sequence("AP", angle_units="degrees", omega=0.0, stored_angles="omega")
        )
        cis_score = score_structure(cis, model="pheat-geometry-integrity")
        self.assertLess(cis_score.terms["peptide_planarity"], 1e-6)

        stretched = copy.deepcopy(ideal)
        stretched.atom_lookup()[(stretched.residue_keys()[0], "CA")].x += 0.5
        stretched_score = score_structure(stretched, model="pheat-geometry-integrity")
        self.assertGreater(stretched_score.terms["backbone_bond_lengths"], baseline.terms["backbone_bond_lengths"])

        inverted = copy.deepcopy(ideal)
        lookup = inverted.atom_lookup()
        key = inverted.residue_keys()[0]
        ca = lookup[(key, "CA")]
        cb = lookup[(key, "CB")]
        cb.coord = (ca.x - (cb.x - ca.x), ca.y - (cb.y - ca.y), ca.z - (cb.z - ca.z))
        inverted_score = score_structure(inverted, model="pheat-geometry-integrity")
        self.assertGreater(inverted_score.terms["ca_chirality"], baseline.terms["ca_chirality"])

        pro_open = copy.deepcopy(ideal)
        pro_key = pro_open.residue_keys()[1]
        pro_open.atom_lookup()[(pro_key, "CD")].x += 1.0
        pro_score = score_structure(pro_open, model="pheat-geometry-integrity")
        self.assertGreater(pro_score.terms["proline_ring_closure"], baseline.terms["proline_ring_closure"])

        moderate = HeavyAtomStructure(
            atoms=[
                Atom("N", "N", 0.0, 0.0, 0.0, "ALA"),
                Atom("CA", "C", 3.458, 0.0, 0.0, "ALA"),
            ],
        )
        severe = HeavyAtomStructure(
            atoms=[
                Atom("N", "N", 0.0, 0.0, 0.0, "ALA"),
                Atom("CA", "C", 5.458, 0.0, 0.0, "ALA"),
            ],
        )
        moderate_score = score_structure(moderate, model="pheat-geometry-integrity")
        severe_score = score_structure(severe, model="pheat-geometry-integrity")
        self.assertGreater(severe_score.terms["backbone_bond_lengths"], moderate_score.terms["backbone_bond_lengths"])
        self.assertLess(severe_score.terms["backbone_bond_lengths"], moderate_score.terms["backbone_bond_lengths"] * 4.0)
        self.assertIn("skipped", "\n".join(moderate_score.warnings))

    def test_physical_integrity_scorer_detects_clashes_and_nonfinite_coordinates(self):
        ideal = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        baseline = score_structure(ideal, model="pheat-physical-integrity")
        self.assertEqual(baseline.model, "pheat-physical-integrity")
        self.assertTrue(math.isfinite(baseline.total))
        self.assertLess(baseline.terms["steric_clash"], 1e-6)
        self.assertEqual(baseline.metadata["input_contract"]["id"], "pheat.score-contract.pheat-physical-integrity.v1")

        clash = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 0.1, 0.0, 0.0, "ALA", resseq=4),
            ],
        )
        clash_score = score_structure(clash, model="pheat-physical-integrity")
        self.assertGreater(clash_score.terms["steric_clash"], baseline.terms["steric_clash"])
        self.assertGreater(clash_score.terms["short_contact"], 0.0)
        self.assertIn("clashes", "\n".join(clash_score.warnings))

        nonfinite = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", float("nan"), 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 5.0, 0.0, 0.0, "ALA", resseq=4),
            ],
        )
        nonfinite_score = score_structure(nonfinite, model="pheat-physical-integrity")
        self.assertTrue(math.isfinite(nonfinite_score.total))
        self.assertGreater(nonfinite_score.terms["nonfinite_coordinates"], 0.0)

    def test_heavy_mm_physical_composite_adds_finite_integrity_barrier(self):
        clash = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 0.1, 0.0, 0.0, "ALA", resseq=4),
            ],
        )
        score = score_structure(clash, model="pheat-heavy-mm-physical", physical_integrity_weight=10.0)
        self.assertEqual(score.model, "pheat-heavy-mm-physical")
        self.assertTrue(math.isfinite(score.total))
        self.assertGreater(score.terms["physical_integrity"], 0.0)
        self.assertEqual(score.metadata["physical_integrity_weight"], 10.0)

    def test_goap_physical_composite_adds_light_integrity_guard(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        goap = score_structure(structure, model="pheat-goap")
        physical = score_structure(structure, model="pheat-physical-integrity")
        composite = score_structure(structure, model="pheat-goap-physical", physical_integrity_weight=0.005)

        self.assertEqual(composite.model, "pheat-goap-physical")
        self.assertTrue(math.isfinite(composite.total))
        self.assertEqual(composite.metadata["physical_integrity_weight"], 0.005)
        self.assertEqual(composite.terms["goap"], goap.total)
        self.assertEqual(composite.terms["physical_integrity"], physical.total)
        self.assertAlmostEqual(composite.total, goap.total + 0.005 * physical.total)
        self.assertIn("component_models", composite.metadata)

    def test_pheat_physics_scores_structure_with_terms_and_options(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        score = score_structure(
            structure,
            model="pheat-physics",
            hydrophobic_gamma=15.0,
            end_to_end_weight=50.0,
            physical_integrity_weight=0.01,
            charge_profile="protein-coarse-charge-v1",
        )

        self.assertEqual(score.model, "pheat-physics")
        self.assertTrue(math.isfinite(score.total))
        self.assertIn("hydrophobic_burial", score.terms)
        self.assertIn("weighted_end_to_end", score.terms)
        self.assertIn("geometry_integrity", score.terms)
        self.assertEqual(score.metadata["input_contract"]["id"], "pheat.score-contract.pheat-physics.v1")
        self.assertEqual(score.metadata["weights"]["physical_integrity"], 0.01)
        self.assertEqual(score.metadata["charge_profile"], "protein-coarse-charge-v1")

        models = {item["model"] for item in model_capabilities()}
        self.assertIn("pheat-physics", models)
        specs = {item["name"]: item for item in score_model_option_specs("pheat-physics")}
        self.assertIn("charge_profile", specs)
        self.assertEqual(specs["charge_profile"]["default"], "protein-coarse-charge-v1")
        self.assertIn("hydrophobic_gamma", specs)
        self.assertIn("end_to_end_weight", specs)

    def test_pheat_coarse_protein_folding_scores_structure_with_decoded_torsions(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        score = score_structure(
            structure,
            model="pheat-coarse-protein-folding-v1",
            hydrophobic_gamma=15.0,
            end_to_end_weight=50.0,
            charge_profile="protein-coarse-charge-v1",
            decoded_torsions={
                "0_phi": -1.0,
                "0_psi": -0.8,
                "1_phi": -2.3,
                "1_psi": 2.4,
                "1_chi1": 0.5,
            },
        )

        self.assertEqual(score.model, "pheat-coarse-protein-folding-v1")
        self.assertTrue(math.isfinite(score.total))
        for term in (
            "end_to_end",
            "hydrophobic_burial",
            "hbond",
            "electrostatic",
            "disulfide",
            "steric",
            "rotamer",
            "aromatic",
            "ramachandran",
            "omega",
            "omega_window_penalty",
            "hard_clash",
            "geometry_integrity",
            "adjacent_heavy_sterics",
            "total",
        ):
            self.assertIn(term, score.terms)
        self.assertEqual(score.metadata["input_contract"]["id"], "pheat.score-contract.pheat-coarse-protein-folding-v1.v1")
        self.assertEqual(score.metadata["charge_profile"], "protein-coarse-charge-v1")
        self.assertEqual(score.metadata["decoded_torsion_count"], 5)
        self.assertEqual(score.terms["omega_window_penalty"], 0.0)
        self.assertIn("hard_clash_min_dist", score.metadata)
        self.assertIn("hard_clash_count", score.metadata)

        models = {item["model"] for item in model_capabilities()}
        self.assertIn("pheat-coarse-protein-folding-v1", models)
        specs = {item["name"]: item for item in score_model_option_specs("pheat-coarse-protein-folding-v1")}
        self.assertIn("decoded_torsions", specs)
        self.assertIn("hydrophobic_gamma", specs)
        self.assertIn("hydrophobic_burial_denominator", specs)
        self.assertIn("hydrophobic_burial_scale", specs)
        self.assertIn("end_to_end_weight", specs)
        self.assertIn("omega_window_scale", specs)
        self.assertIn("hard_clash_scale", specs)

    def test_pheat_coarse_protein_folding_options_and_torsion_cleanup(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        scaled = score_structure(
            structure,
            model="pheat-coarse-protein-folding-v1",
            end_to_end_target=0.0,
            end_to_end_slack=0.0,
            end_to_end_weight=2.0,
            end_to_end_scale=0.25,
            decoded_torsions={
                "0_phi": -1.0,
                "0_psi": float("nan"),
                "1_phi": "bad",
            },
        )
        unscaled = score_structure(
            structure,
            model="pheat-coarse-protein-folding-v1",
            end_to_end_target=0.0,
            end_to_end_slack=0.0,
            end_to_end_weight=2.0,
            end_to_end_scale=1.0,
        )
        disabled = score_structure(
            structure,
            model="pheat-coarse-protein-folding-v1",
            end_to_end_target=0.0,
            end_to_end_slack=0.0,
            end_to_end_weight=2.0,
            use_end_to_end_constraint=False,
        )

        self.assertAlmostEqual(scaled.terms["end_to_end"], unscaled.terms["end_to_end"] * 0.25)
        self.assertEqual(disabled.terms["end_to_end"], 0.0)
        self.assertEqual(scaled.metadata["decoded_torsion_count"], 1)
        self.assertEqual(scaled.metadata["ignored_decoded_torsion_count"], 2)
        self.assertTrue(any("ignored non-numeric or non-finite decoded torsion" in item for item in scaled.warnings))

        hydrophobic_structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("VV"))
        default_burial = score_structure(
            hydrophobic_structure,
            model="pheat-coarse-protein-folding-v1",
            hydrophobic_gamma=15.0,
        )
        qtf_main_burial = score_structure(
            hydrophobic_structure,
            model="pheat-coarse-protein-folding-v1",
            hydrophobic_gamma=15.0,
            hydrophobic_burial_denominator=35.0,
            hydrophobic_burial_scale=0.7,
        )
        self.assertEqual(qtf_main_burial.metadata["hydrophobic_burial_denominator"], 35.0)
        self.assertEqual(qtf_main_burial.metadata["hydrophobic_burial_scale"], 0.7)
        self.assertEqual(qtf_main_burial.metadata["weights"]["hydrophobic_burial"], 10.5)
        self.assertNotEqual(default_burial.terms["hydrophobic_burial"], qtf_main_burial.terms["hydrophobic_burial"])

        capped = score_structure(
            structure,
            model="pheat-coarse-protein-folding-v1",
            decoded_torsions={
                "0_phi": -1.0,
                "1_chi4": 0.5,
                "1_chi5": 1.0,
                "0_omega": math.radians(120.0),
            },
        )
        self.assertEqual(capped.metadata["decoded_torsion_count"], 2)
        self.assertEqual(capped.metadata["ignored_decoded_torsion_count"], 2)
        self.assertTrue(any("ignored non-numeric or non-finite decoded torsion" in item for item in capped.warnings))
        self.assertGreater(capped.terms["omega_window_penalty"], 0.0)

        for omega_degrees in (175.0, 180.0, 185.0):
            in_window = score_structure(
                structure,
                model="pheat-coarse-protein-folding-v1",
                decoded_torsions={"0_omega": math.radians(omega_degrees)},
            )
            self.assertEqual(in_window.terms["omega_window_penalty"], 0.0)

        capped_residue_chis = score_structure(
            structure_from_residue_geometry(residue_geometry_structure_from_sequence("LDE")),
            model="pheat-coarse-protein-folding-v1",
            decoded_torsions={
                "0_chi1": 0.1,
                "0_chi2": 0.2,
                "1_chi1": 0.3,
                "1_chi2": 0.4,
                "2_chi1": 0.5,
                "2_chi2": 0.6,
                "2_chi3": 0.7,
            },
        )
        self.assertEqual(capped_residue_chis.metadata["decoded_torsion_count"], 4)
        self.assertEqual(capped_residue_chis.metadata["ignored_decoded_torsion_count"], 3)

        valid = validate_scoring_options(
            "pheat-coarse-protein-folding-v1",
            {
                "decoded_torsions": {"0_phi": -1.0},
                "use_end_to_end_constraint": "false",
                "end_to_end_scale": "0.5",
            },
        )
        self.assertTrue(valid["ok"])
        self.assertFalse(valid["options"]["use_end_to_end_constraint"])
        self.assertEqual(valid["options"]["end_to_end_scale"], 0.5)

        invalid = validate_scoring_options("pheat-coarse-protein-folding-v1", {"decoded_torsions": [1, 2]})
        self.assertFalse(invalid["ok"])
        self.assertIn("decoded_torsions must be a mapping", "; ".join(invalid["errors"]))

    def test_cli_scores_pheat_coarse_protein_folding_with_decoded_torsions(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            structure_path = tmp / "structure.json"
            torsions_path = tmp / "torsions.json"
            output_path = tmp / "score.json"
            structure_path.write_text(structure.to_json() + "\n", encoding="utf-8")
            torsions_path.write_text(json.dumps({"0_phi": -1.0, "0_psi": -0.8}), encoding="utf-8")

            status, _stdout, _stderr = _capture_cli(
                [
                    "score",
                    str(structure_path),
                    "--model",
                    "pheat-coarse-protein-folding-v1",
                    "--decoded-torsions",
                    str(torsions_path),
                    "--disable-end-to-end-constraint",
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "pheat-coarse-protein-folding-v1")
            self.assertEqual(payload["terms"]["end_to_end"], 0.0)
            self.assertEqual(payload["metadata"]["decoded_torsion_count"], 2)

    def test_hydropathy_and_backbone_physical_composites_add_integrity_guard(self):
        structure = structure_from_residue_geometry(residue_geometry_structure_from_sequence("AP"))
        physical = score_structure(structure, model="pheat-physical-integrity")
        hydropathy = score_structure(structure, model="pheat-hydropathy")
        backbone = score_structure(structure, model="pheat-backbone")

        hydropathy_composite = score_structure(
            structure,
            model="pheat-hydropathy-physical",
            physical_integrity_weight=0.0001,
        )
        backbone_composite = score_structure(
            structure,
            model="pheat-backbone-physical",
            physical_integrity_weight=0.001,
        )

        self.assertEqual(hydropathy_composite.model, "pheat-hydropathy-physical")
        self.assertEqual(backbone_composite.model, "pheat-backbone-physical")
        self.assertAlmostEqual(hydropathy_composite.total, hydropathy.total + 0.0001 * physical.total)
        self.assertAlmostEqual(backbone_composite.total, backbone.total + 0.001 * physical.total)
        self.assertEqual(hydropathy_composite.terms["physical_integrity"], physical.total)
        self.assertEqual(backbone_composite.terms["physical_integrity"], physical.total)
        self.assertIn("component_models", hydropathy_composite.metadata)
        self.assertIn("component_models", backbone_composite.metadata)

    def test_gromacs_mdrun_preflight_rejects_unsafe_coordinates_without_gmx(self):
        clash = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 0.1, 0.0, 0.0, "ALA", resseq=4),
            ],
        )
        score = score_structure(clash, model="gromacs-mdrun")
        self.assertEqual(score.model, "gromacs-mdrun")
        self.assertEqual(score.total, 0.0)
        self.assertEqual(score.units, "unavailable")
        self.assertEqual(score.metadata["status"], "unavailable")
        self.assertIn("preflight", score.metadata)
        self.assertIn("short_contact", score.metadata["preflight"]["terms"])

    def test_gromacs_mdrun_preflight_warn_continues_to_gmx_requirement(self):
        clash = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 0.1, 0.0, 0.0, "ALA", resseq=4),
            ],
        )
        with mock.patch("pheat.scoring.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "GROMACS gmx executable"):
                score_structure(clash, model="gromacs-mdrun", gromacs_preflight="warn")

    def test_score_model_option_specs_and_validation_are_general(self):
        specs = score_model_option_specs()
        Draft202012Validator(load_schema("score-model-option-specs")).validate(
            {"format": "pheat.score-model-option-specs", "version": 1, "models": specs}
        )
        self.assertEqual(set(specs), set(supported_models()))
        self.assertTrue(all(specs[model] for model in supported_models()))
        composite_specs = {item["name"]: item for item in score_model_option_specs("pheat-heavy-mm-physical")}
        self.assertEqual(composite_specs["physical_integrity_weight"]["default"], 1000.0)
        goap_physical_specs = {item["name"]: item for item in score_model_option_specs("pheat-goap-physical")}
        self.assertEqual(goap_physical_specs["physical_integrity_weight"]["default"], 0.005)
        hydropathy_physical_specs = {item["name"]: item for item in score_model_option_specs("pheat-hydropathy-physical")}
        self.assertEqual(hydropathy_physical_specs["physical_integrity_weight"]["default"], 0.0001)
        self.assertIn("burial_method", hydropathy_physical_specs)
        backbone_physical_specs = {item["name"]: item for item in score_model_option_specs("pheat-backbone-physical")}
        self.assertEqual(backbone_physical_specs["physical_integrity_weight"]["default"], 0.001)
        gromacs_specs = {item["name"]: item for item in score_model_option_specs("gromacs-mdrun")}
        self.assertEqual(gromacs_specs["gromacs_run_mode"]["choices"], ["rerun", "minimize", "minimize-rerun"])
        self.assertEqual(gromacs_specs["gromacs_preflight"]["default"], "strict")
        self.assertEqual(gromacs_specs["gromacs_preflight"]["choices"], ["strict", "warn", "off"])

        valid = validate_scoring_options(
            "gromacs-mdrun",
            {
                "gromacs_run_mode": "minimize",
                "gromacs_preflight": "warn",
                "gromacs_run_settings": {"cutoff_nm": 1.2},
                "gromacs_solvate": "true",
            },
        )
        Draft202012Validator(load_schema("scoring-options-validation")).validate(valid)
        self.assertTrue(valid["ok"])
        self.assertEqual(valid["options"]["gromacs_preflight"], "warn")
        self.assertEqual(valid["options"]["gromacs_run_settings"]["cutoff_nm"], 1.2)
        self.assertTrue(valid["options"]["gromacs_solvate"])

        unknown = validate_scoring_options("generic", {"not_an_option": 1})
        self.assertFalse(unknown["ok"])
        self.assertIn("unknown option", unknown["errors"][0])

        invalid = validate_scoring_options("gromacs-mdrun", {"gromacs_run_mode": "bad-mode"})
        self.assertFalse(invalid["ok"])
        self.assertIn("gromacs_run_mode must be one of", "; ".join(invalid["errors"]))

        invalid_preflight = validate_scoring_options("gromacs-mdrun", {"gromacs_preflight": "bad-mode"})
        self.assertFalse(invalid_preflight["ok"])
        self.assertIn("GROMACS preflight", "; ".join(invalid_preflight["errors"]))

    def test_scoring_model_capabilities_separate_supported_from_available(self):
        self.assertIn("openmm-prepared", supported_models())
        self.assertIn("ambertools-sander", supported_models())
        self.assertIn("gromacs-mdrun", supported_models())
        self.assertIn("pheat-rg", supported_models())

        def missing_openmm(name):
            if name == "openmm":
                raise ImportError("no openmm")
            return object()

        with mock.patch("pheat.scoring.importlib.import_module", side_effect=missing_openmm):
            self.assertIn("pheat-rg", available_models())
            self.assertNotIn("openmm-prepared", available_models())
            capabilities = {item["model"]: item for item in model_capabilities()}
            self.assertTrue(capabilities["pheat-rg"]["available"])
            self.assertEqual(capabilities["pheat-rg"]["requires"], [])
            self.assertEqual(capabilities["pheat-rg"]["implementation"]["origin"], "native-pheat")
            self.assertFalse(capabilities["pheat-rg"]["implementation"]["external"])
            self.assertFalse(capabilities["openmm-prepared"]["available"])
            self.assertEqual(capabilities["openmm-prepared"]["requires"], ["openmm"])
            self.assertIn("no openmm", capabilities["openmm-prepared"]["reason"])

        with mock.patch("pheat.scoring.importlib.import_module", return_value=object()):
            self.assertIn("openmm-prepared", available_models())
            capabilities = {item["model"]: item for item in model_capabilities()}
            self.assertTrue(capabilities["openmm-prepared"]["available"])
            self.assertEqual(capabilities["openmm-prepared"]["optional_requires"], ["pdbfixer"])
            self.assertEqual(capabilities["openmm-prepared"]["implementation"]["origin"], "external-python")
            self.assertTrue(capabilities["openmm-prepared"]["implementation"]["external"])

        with mock.patch("pheat.scoring.shutil.which", return_value=None):
            self.assertNotIn("ambertools-sander", available_models())
            self.assertNotIn("gromacs-mdrun", available_models())
            capabilities = {item["model"]: item for item in model_capabilities()}
            self.assertFalse(capabilities["ambertools-sander"]["available"])
            self.assertIn("tleap executable", capabilities["ambertools-sander"]["reason"])
            self.assertFalse(capabilities["gromacs-mdrun"]["available"])
            self.assertIn("gmx executable", capabilities["gromacs-mdrun"]["reason"])

        def fake_which(name):
            if name in {"tleap", "sander"}:
                return f"/opt/ambertools/bin/{name}"
            if name == "gmx":
                return "/opt/gromacs/bin/gmx"
            return None

        with mock.patch("pheat.scoring.shutil.which", side_effect=fake_which):
            self.assertIn("ambertools-sander", available_models())
            self.assertIn("gromacs-mdrun", available_models())
            capabilities = {item["model"]: item for item in model_capabilities()}
            self.assertTrue(capabilities["ambertools-sander"]["available"])
            self.assertEqual(
                capabilities["ambertools-sander"]["requires"],
                ["executable:tleap", "executable:sander"],
            )
            self.assertEqual(capabilities["ambertools-sander"]["implementation"]["backend"], "ambertools")
            self.assertEqual(capabilities["ambertools-sander"]["implementation"]["origin"], "external-executable")
            self.assertTrue(capabilities["gromacs-mdrun"]["available"])
            self.assertEqual(capabilities["gromacs-mdrun"]["requires"], ["executable:gmx"])
            self.assertEqual(capabilities["gromacs-mdrun"]["implementation"]["backend"], "gromacs")
            self.assertEqual(capabilities["gromacs-mdrun"]["implementation"]["origin"], "external-executable")

    def test_ambertools_sander_scoring_is_mockable_and_parses_terms(self):
        terms = parse_sander_energy_terms(
            "   NSTEP       ENERGY          RMS            GMAX         NAME    NUMBER\n"
            "      1       5.8445E+01     1.6964E+01     1.7708E+02     CD        233\n"
            " Etot   =       -12.3400  EKtot   =         0.0000  EPtot      =       -12.3400\n"
            " BOND   =         1.0000  ANGLE   =         2.0000  DIHED      =         3.0000\n"
            " VDWAALS =       -4.0000  EEL     =        -5.0000\n"
        )
        self.assertAlmostEqual(terms["ENERGY"], 58.445)
        self.assertAlmostEqual(terms["Etot"], -12.34)
        self.assertAlmostEqual(terms["VDWAALS"], -4.0)

        def fake_which(name):
            if name in {"tleap", "sander"}:
                return f"/opt/ambertools/bin/{name}"
            return None

        def fake_prepare(structure, output_pdb, *, prepare):
            output_pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "END\n",
                encoding="utf-8",
            )
            return ["test preparation warning"], {
                "mode": prepare,
                "engine": "test",
                "input_atom_count": len(structure.atoms),
                "prepared_atom_count": len(structure.atoms) + 2,
                "hydrogens_added": 2,
            }

        def fake_run(command, *, cwd, check, capture_output, text):
            command_name = Path(command[0]).name
            work_dir = Path(cwd)
            if command_name == "tleap":
                (work_dir / "system.prmtop").write_text("prmtop\n", encoding="utf-8")
                (work_dir / "system.inpcrd").write_text("inpcrd\n", encoding="utf-8")
            elif command_name == "sander":
                (work_dir / "sander.out").write_text(
                    " Etot   =       -12.3400  EKtot   =         0.0000  EPtot      =       -12.3400\n"
                    " VDWAALS =       -4.0000  EEL     =        -5.0000\n",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        structure = load_pdb(TINY_PDB)
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = Path(tmpdir) / "prepared.pdb"
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._prepare_pdb_for_forcefield", side_effect=fake_prepare),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                result = score_structure(
                    structure,
                    model="ambertools-sander",
                    prepare="write",
                    prepared_output=prepared,
                    ambertools_work_dir=Path(tmpdir) / "amber",
                )
                self.assertEqual(result.model, "ambertools-sander")
                self.assertEqual(result.units, "kcal/mol")
                self.assertAlmostEqual(result.total, -12.34)
                self.assertAlmostEqual(result.terms["vdwaals"], -4.0)
                self.assertEqual(result.metadata["amber_forcefield"], "leaprc.protein.ff14SB")
                self.assertIsNone(result.metadata["amber_pbradii"])
                self.assertEqual(result.metadata["preparation"]["engine"], "test")
                self.assertEqual(result.metadata["implementation"]["origin"], "external-executable")
                self.assertEqual(result.metadata["implementation"]["backend"], "ambertools")
                self.assertTrue(prepared.exists())

            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._prepare_pdb_for_forcefield", side_effect=fake_prepare),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                status, stdout, _stderr = _capture_cli(
                    [
                        "score",
                        str(TINY_PDB),
                        "--model",
                        "ambertools-sander",
                        "--prepare",
                        "auto",
                        "--ambertools-work-dir",
                        str(Path(tmpdir) / "amber-cli"),
                    ]
                )
                self.assertEqual(status, 0)
                payload = json.loads(stdout)
                self.assertEqual(payload["model"], "ambertools-sander")
                self.assertAlmostEqual(payload["total"], -12.34)

    def test_ambertools_sander_gb_uses_mbondi3_radii(self):
        def fake_which(name):
            if name in {"tleap", "sander"}:
                return f"/opt/ambertools/bin/{name}"
            return None

        def fake_prepare(structure, output_pdb, *, prepare):
            output_pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "END\n",
                encoding="utf-8",
            )
            return [], {"mode": prepare, "engine": "test"}

        def fake_run(command, *, cwd, check, capture_output, text):
            command_name = Path(command[0]).name
            work_dir = Path(cwd)
            if command_name == "tleap":
                tleap_input = (work_dir / "tleap.in").read_text(encoding="utf-8")
                self.assertIn("set default PBRadii mbondi3", tleap_input)
                (work_dir / "system.prmtop").write_text("prmtop\n", encoding="utf-8")
                (work_dir / "system.inpcrd").write_text("inpcrd\n", encoding="utf-8")
            elif command_name == "sander":
                (work_dir / "sander.out").write_text(
                    " Etot   =       -12.3400  EKtot   =         0.0000  EPtot      =       -12.3400\n",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        structure = load_pdb(TINY_PDB)
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._prepare_pdb_for_forcefield", side_effect=fake_prepare),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                result = score_structure(
                    structure,
                    model="ambertools-sander",
                    amber_solvent="gb",
                    ambertools_work_dir=Path(tmpdir) / "amber-gb",
                )
        self.assertEqual(result.metadata["amber_pbradii"], "mbondi3")

    def test_ambertools_sander_failure_reports_sander_output_tail(self):
        def fake_which(name):
            if name in {"tleap", "sander"}:
                return f"/opt/ambertools/bin/{name}"
            return None

        def fake_prepare(structure, output_pdb, *, prepare):
            output_pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "END\n",
                encoding="utf-8",
            )
            return [], {"mode": prepare, "engine": "test"}

        def fake_run(command, *, cwd, check, capture_output, text):
            command_name = Path(command[0]).name
            work_dir = Path(cwd)
            if command_name == "tleap":
                (work_dir / "system.prmtop").write_text("prmtop\n", encoding="utf-8")
                (work_dir / "system.inpcrd").write_text("inpcrd\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if command_name == "sander":
                (work_dir / "sander.out").write_text(
                    "line before\n"
                    "FATAL: bad atom geometry in generated structure\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        structure = load_pdb(TINY_PDB)
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._prepare_pdb_for_forcefield", side_effect=fake_prepare),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "Last lines from sander.out") as raised:
                    score_structure(
                        structure,
                        model="ambertools-sander",
                        ambertools_work_dir=Path(tmpdir) / "amber-fail",
                    )
        self.assertIn("bad atom geometry", str(raised.exception))

    def test_gromacs_mdrun_scoring_is_mockable_and_parses_xvg_terms(self):
        terms = parse_gromacs_xvg_terms(
            '@    s0 legend "Potential"\n'
            '@    s1 legend "Bond"\n'
            '@    s2 legend "Coulomb-(SR)"\n'
            "0.000000 -42.500000 1.250000 -3.750000\n"
        )
        self.assertAlmostEqual(terms["potential"], -42.5)
        self.assertAlmostEqual(terms["bond"], 1.25)
        self.assertAlmostEqual(terms["coulomb_sr"], -3.75)

        def fake_which(name):
            if name == "gmx":
                return "/opt/gromacs/bin/gmx"
            return None

        def fake_prepare_input(structure, output_pdb, *, prepare):
            output_pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "END\n",
                encoding="utf-8",
            )
            return ["test gromacs preparation warning"], {
                "mode": prepare,
                "engine": "test",
                "input_atom_count": len(structure.atoms),
                "prepared_atom_count": len(structure.atoms),
            }

        def fake_run(command, *, cwd, check, capture_output, text, input=None):
            work_dir = Path(cwd)
            if command[1] == "--version":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="GROMACS version: 2026.0\nData prefix: /opt/gromacs\n",
                    stderr="",
                )
            subcommand = command[1]
            if subcommand == "pdb2gmx":
                (work_dir / "prepared.gro").write_text("prepared\n", encoding="utf-8")
                (work_dir / "topol.top").write_text("topology\n", encoding="utf-8")
            elif subcommand == "editconf":
                (work_dir / "boxed.gro").write_text("boxed\n", encoding="utf-8")
            elif subcommand == "grompp":
                output = Path(command[command.index("-o") + 1])
                (work_dir / output.name).write_text("tpr\n", encoding="utf-8")
            elif subcommand == "mdrun":
                deffnm = command[command.index("-deffnm") + 1]
                (work_dir / f"{deffnm}.edr").write_text("edr\n", encoding="utf-8")
                if deffnm == "minimize":
                    (work_dir / "minimize.gro").write_text("minimized\n", encoding="utf-8")
            elif subcommand == "energy":
                (work_dir / "energy.xvg").write_text(
                    '@    s0 legend "Potential"\n'
                    '@    s1 legend "Bond"\n'
                    '@    s2 legend "Coulomb-(SR)"\n'
                    "0.000000 -42.500000 1.250000 -3.750000\n",
                    encoding="utf-8",
                )
            elif subcommand == "gyrate":
                (work_dir / "gyrate.xvg").write_text(
                    '@    s0 legend "Rg"\n'
                    "0.000000 1.500000\n",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        structure = load_pdb(TINY_PDB)
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = Path(tmpdir) / "prepared.gro"
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._write_pdb_for_gromacs_pdb2gmx", side_effect=fake_prepare_input),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                result = score_structure(
                    structure,
                    model="gromacs-mdrun",
                    prepare="write",
                    prepared_output=prepared,
                    gromacs_work_dir=Path(tmpdir) / "gromacs",
                    gromacs_metrics=True,
                )
                self.assertEqual(result.model, "gromacs-mdrun")
                self.assertEqual(result.units, "kJ/mol")
                self.assertAlmostEqual(result.total, -42.5)
                self.assertAlmostEqual(result.terms["bond"], 1.25)
                self.assertEqual(result.metadata["gromacs_forcefield"], "amber19sb")
                self.assertEqual(result.metadata["gromacs_water"], "none")
                self.assertEqual(result.metadata["preparation"]["engine"], "test")
                self.assertEqual(result.metadata["implementation"]["origin"], "external-executable")
                self.assertEqual(result.metadata["implementation"]["backend"], "gromacs")
                self.assertAlmostEqual(
                    result.metadata["gromacs_metrics"]["radius_of_gyration"]["rg"],
                    1.5,
                )
                self.assertTrue(prepared.exists())

            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._write_pdb_for_gromacs_pdb2gmx", side_effect=fake_prepare_input),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                status, stdout, _stderr = _capture_cli(
                    [
                        "score",
                        str(TINY_PDB),
                        "--model",
                        "gromacs-mdrun",
                        "--prepare",
                        "auto",
                        "--gromacs-work-dir",
                        str(Path(tmpdir) / "gromacs-cli"),
                    ]
                )
                self.assertEqual(status, 0)
                payload = json.loads(stdout)
                self.assertEqual(payload["model"], "gromacs-mdrun")
                self.assertAlmostEqual(payload["total"], -42.5)

            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._write_pdb_for_gromacs_pdb2gmx", side_effect=fake_prepare_input),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                status, stdout, _stderr = _capture_cli(
                    [
                        "gromacs",
                        "validate",
                        str(TINY_PDB),
                        "--gromacs-work-dir",
                        str(Path(tmpdir) / "gromacs-validate"),
                    ]
                )
                self.assertEqual(status, 0)
                payload = json.loads(stdout)
                self.assertEqual(payload["score"]["model"], "gromacs-mdrun")
                self.assertEqual(payload["format"], "pheat.gromacs-validation")

    def test_external_scoring_validation_reports_bad_options_before_running(self):
        with mock.patch("pheat.scoring.shutil.which", return_value=None):
            payload = validate_external_scoring_options(model="gromacs-mdrun")
            self.assertFalse(payload["ok"])
            self.assertIn("gmx executable", "; ".join(payload["errors"]))

        def fake_which(name):
            if name == "gmx":
                return "/opt/gromacs/bin/gmx"
            return None

        with (
            mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
            mock.patch("pheat.scoring._installed_gromacs_forcefields", return_value=["amber99sb-ildn"]),
        ):
            missing_default = validate_external_scoring_options(model="gromacs-mdrun")
            self.assertFalse(missing_default["ok"])
            self.assertIn("amber19sb", "; ".join(missing_default["errors"]))
            valid = validate_external_scoring_options(
                model="gromacs-mdrun",
                gromacs_forcefield="amber99sb-ildn",
                gromacs_preflight="warn",
                gromacs_run_settings=GromacsRunSettings(cutoff_nm=1.2),
            )
            self.assertTrue(valid["ok"])
            self.assertEqual(valid["gromacs_preflight"], "warn")
            self.assertEqual(valid["gromacs_run_settings"]["cutoff_nm"], 1.2)

            status, stdout, _stderr = _capture_cli(
                [
                    "scoring",
                    "validate-options",
                    "--model",
                    "gromacs-mdrun",
                    "--gromacs-forcefield",
                    "amber99sb-ildn",
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout)["ok"])

    def test_external_command_timeout_is_reported(self):
        def fake_which(name):
            if name in {"tleap", "sander"}:
                return f"/opt/ambertools/bin/{name}"
            return None

        def fake_prepare(structure, output_pdb, *, prepare):
            output_pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "END\n",
                encoding="utf-8",
            )
            return [], {"mode": prepare, "engine": "test"}

        def fake_run(command, **kwargs):
            work_dir = Path(kwargs["cwd"])
            if Path(command[0]).name == "tleap":
                (work_dir / "system.prmtop").write_text("prmtop\n", encoding="utf-8")
                (work_dir / "system.inpcrd").write_text("inpcrd\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"), output="running", stderr="slow")

        structure = load_pdb(TINY_PDB)
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring._prepare_pdb_for_forcefield", side_effect=fake_prepare),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out after 1.5 seconds"):
                    score_structure(
                        structure,
                        model="ambertools-sander",
                        ambertools_work_dir=Path(tmpdir) / "amber",
                        external_timeout_seconds=1.5,
                    )

    def test_gromacs_run_settings_and_conservative_prep_cache(self):
        commands = []

        def fake_which(name):
            if name == "gmx":
                return "/opt/gromacs/bin/gmx"
            return None

        def fake_run(command, **kwargs):
            commands.append(list(command))
            work_dir = Path(kwargs["cwd"])
            if command[1] == "--version":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="GROMACS version: 2026.0\nData prefix: /opt/gromacs\n",
                    stderr="",
                )
            subcommand = command[1]
            if subcommand == "pdb2gmx":
                (work_dir / "prepared.gro").write_text("prepared\n", encoding="utf-8")
                (work_dir / "topol.top").write_text("topology\n", encoding="utf-8")
            elif subcommand == "editconf":
                (work_dir / "boxed.gro").write_text("boxed\n", encoding="utf-8")
            elif subcommand == "grompp":
                output = Path(command[command.index("-o") + 1])
                (work_dir / output.name).write_text("tpr\n", encoding="utf-8")
            elif subcommand == "mdrun":
                deffnm = command[command.index("-deffnm") + 1]
                (work_dir / f"{deffnm}.edr").write_text("edr\n", encoding="utf-8")
            elif subcommand == "energy":
                (work_dir / "energy.xvg").write_text(
                    '@    s0 legend "Potential"\n'
                    "0.000000 -7.000000\n",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        structure = load_pdb(TWO_MU7_PDB, hydrogens="preserve")
        settings = GromacsRunSettings(
            minimize_steps=25,
            emtol=500.0,
            box_distance_nm=1.4,
            cutoff_nm=1.2,
            mdrun_flags=("-ntomp", "1"),
            grompp_maxwarn=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            first_work = Path(tmpdir) / "first"
            second_work = Path(tmpdir) / "second"
            with (
                mock.patch("pheat.scoring.shutil.which", side_effect=fake_which),
                mock.patch("pheat.scoring.subprocess.run", side_effect=fake_run),
            ):
                first = score_structure(
                    structure,
                    model="gromacs-mdrun",
                    domain="full",
                    prepare="never",
                    gromacs_forcefield="amber99sb-ildn",
                    gromacs_work_dir=first_work,
                    gromacs_run_settings=settings,
                    prep_cache_dir=cache_dir,
                    prep_cache_mode="readwrite",
                )
                self.assertEqual(first.metadata["preparation"]["prep_cache"]["status"], "miss")
                self.assertIn("box_distance_nm", first.metadata["gromacs_run_settings"])
                rerun_mdp = (first_work / "rerun.mdp").read_text(encoding="utf-8")
                self.assertIn("rcoulomb = 1.2", rerun_mdp)
                self.assertIn("rvdw = 1.2", rerun_mdp)
                self.assertTrue(any(command[1] == "pdb2gmx" for command in commands))
                self.assertTrue(any("-maxwarn" in command for command in commands if command[1] == "grompp"))
                self.assertTrue(any("-ntomp" in command for command in commands if command[1] == "mdrun"))

                commands.clear()
                updated_settings = GromacsRunSettings(
                    minimize_steps=25,
                    emtol=500.0,
                    box_distance_nm=1.4,
                    cutoff_nm=1.1,
                    mdrun_flags=("-ntomp", "2"),
                    grompp_maxwarn=1,
                )
                second = score_structure(
                    structure,
                    model="gromacs-mdrun",
                    domain="full",
                    prepare="never",
                    gromacs_forcefield="amber99sb-ildn",
                    gromacs_work_dir=second_work,
                    gromacs_run_settings=updated_settings,
                    prep_cache_dir=cache_dir,
                    prep_cache_mode="readwrite",
                )
                self.assertEqual(second.metadata["preparation"]["prep_cache"]["status"], "hit")
                self.assertFalse(any(command[1] == "pdb2gmx" for command in commands))
                self.assertTrue((second_work / "topol.top").exists())
                updated_rerun_mdp = (second_work / "rerun.mdp").read_text(encoding="utf-8")
                self.assertIn("rcoulomb = 1.1", updated_rerun_mdp)
                self.assertTrue(any("-ntomp" in command and "2" in command for command in commands if command[1] == "mdrun"))

    def test_geometry_tables_build_validate_and_drive_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selected_root = root / "selected"
            selected_root.mkdir()
            row = {
                "format": "pheat.training-inventory-row",
                "version": 1,
                "status": "ok",
                "pdb_id": "1ABC",
                "chain_id": "A",
                "path": str(TINY_PDB),
                "domain": "protein-heavy",
                "sequence": "AG",
                "length": 2,
            }
            (selected_root / "selected.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            table_root = root / "geometry"
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "build-backbone",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(table_root),
                    "--max-entries",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            result = json.loads(stdout)
            table_path = Path(result["output"])
            payload = json.loads(table_path.read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("geometry-table-set")).validate(payload)
            self.assertEqual(payload["format"], GEOMETRY_TABLE_SET_FORMAT)
            self.assertIn("backbone", payload["profiles"])

            status, stdout, _stderr = _capture_cli(
                ["geometry", "tables", "describe", "--table-set", str(table_path)]
            )
            self.assertEqual(status, 0)
            description = json.loads(stdout)
            self.assertTrue(description["profiles"]["backbone"]["has_backbone"])

            status, stdout, _stderr = _capture_cli(
                ["geometry", "tables", "validate", "--table-set", str(table_path)]
            )
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout)["ok"])

            geometry_table = {
                "format": "pheat.geometry-table-set",
                "version": 1,
                "default_profile": "test",
                "metadata": {"test": True},
                "profiles": {
                    "test": {
                        "metadata": {"kind": "backbone"},
                        "backbone": {
                            "defaults": {
                                "angles": {"N-CA-C": 100.0},
                                "lengths": {},
                            }
                        },
                        "sidechains": {"residues": {}},
                    }
                },
            }
            residue_geometry = {"residues": [{"name": "ALA"}, {"name": "GLY"}]}
            fixed = structure_from_residue_geometry(residue_geometry)
            tabled = structure_from_residue_geometry(residue_geometry, geometry_table=geometry_table)
            self.assertNotAlmostEqual(fixed.atoms[2].y, tabled.atoms[2].y)
            self.assertEqual(tabled.metadata["reconstruction_geometry"]["mode"], "table")
            self.assertGreater(tabled.metadata["reconstruction_geometry"]["fallback_count"], 0)

            binned_geometry_table = {
                **geometry_table,
                "profiles": {
                    "test": {
                        "metadata": {"kind": "backbone"},
                        "backbone": {
                            "defaults": {
                                "angles": {"N-CA-C": 100.0},
                                "lengths": {},
                            },
                            "phi_psi_bins": {
                                "phi:-60:-30|psi:-60:-30": {
                                    "angles": {"N-CA-C": 90.0},
                                    "lengths": {"N-CA": 2.0},
                                }
                            },
                        },
                        "sidechains": {"residues": {}},
                    }
                },
            }
            residue_geometry_with_context = {
                "angle_units": "degrees",
                "residues": [
                    {"name": "ALA", "phi": -60.0, "psi": -45.0},
                    {"name": "GLY", "phi": -60.0, "psi": -45.0},
                ],
            }
            default_tabled = structure_from_residue_geometry(
                residue_geometry_with_context,
                geometry_table=geometry_table,
            )
            binned_tabled = structure_from_residue_geometry(
                residue_geometry_with_context,
                geometry_table=binned_geometry_table,
            )
            self.assertNotAlmostEqual(default_tabled.atoms[2].y, binned_tabled.atoms[2].y)
            first_key = binned_tabled.residue_keys()[0]
            lookup = binned_tabled.atom_lookup()
            self.assertAlmostEqual(distance(lookup[(first_key, "N")].coord, lookup[(first_key, "CA")].coord), 2.0)

            residue_geometry_with_stored_length = {
                "angle_units": "degrees",
                "residues": [
                    {"name": "ALA", "phi": -60.0, "psi": -45.0, "bond_lengths": {"N-CA": 1.7}},
                    {"name": "GLY", "phi": -60.0, "psi": -45.0},
                ],
            }
            stored_length_tabled = structure_from_residue_geometry(
                residue_geometry_with_stored_length,
                geometry_table=binned_geometry_table,
            )
            first_key = stored_length_tabled.residue_keys()[0]
            lookup = stored_length_tabled.atom_lookup()
            self.assertAlmostEqual(distance(lookup[(first_key, "N")].coord, lookup[(first_key, "CA")].coord), 1.7)

    def test_geometry_tables_build_cdl_and_import_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selected_root = root / "selected"
            selected_root.mkdir()
            row = {
                "format": "pheat.training-inventory-row",
                "version": 1,
                "status": "ok",
                "pdb_id": "1ABC",
                "chain_id": "A",
                "path": str(TWO_MU7_PDB),
                "domain": "protein-heavy",
                "sequence": "KLVFFAEDVGSNKGAIIGLM",
                "length": 20,
            }
            (selected_root / "selected.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            table_root = root / "cdl-geometry"
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "build-cdl",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(table_root),
                    "--max-entries",
                    "1",
                    "--phi-psi-bin-size",
                    "30",
                    "--min-bin-count",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads((Path(json.loads(stdout)["output"])).read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("geometry-table-set")).validate(payload)
            profile = payload["profiles"]["backbone-cdl"]
            self.assertEqual(profile["metadata"]["kind"], "backbone-cdl")
            self.assertEqual(profile["metadata"]["phi_psi_bin_size"], 30)
            self.assertIn("residue_classes", profile["backbone"])
            self.assertTrue(profile["backbone"]["phi_psi_bins"])
            bin_payload = next(iter(profile["backbone"]["phi_psi_bins"].values()))
            self.assertIn("angles", bin_payload)
            self.assertIn("lengths", bin_payload)

            import_input = root / "simple-cdl.json"
            import_input.write_text(
                json.dumps(
                    {
                        "phi_psi_bin_size": 30,
                        "phi_psi_bins": {
                            "phi:-60:-30|psi:-60:-30": {
                                "angles": {"N-CA-C": 88.0},
                                "lengths": {"N-CA": 1.9},
                            }
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            imported_root = root / "imported-cdl"
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "import-cdl",
                    "--input",
                    str(import_input),
                    "--output-root",
                    str(imported_root),
                    "--source-license",
                    "test-only",
                ]
            )
            self.assertEqual(status, 0)
            imported = json.loads((Path(json.loads(stdout)["output"])).read_text(encoding="utf-8"))
            self.assertEqual(imported["metadata"]["source"], "imported-cdl-json")
            self.assertEqual(imported["metadata"]["source_license"], "test-only")
            self.assertIn("input_sha256", imported["metadata"])
            self.assertAlmostEqual(
                imported["profiles"]["backbone-cdl"]["backbone"]["phi_psi_bins"][
                    "phi:-60:-30|psi:-60:-30"
                ]["lengths"]["N-CA"],
                1.9,
            )

    def test_geometry_tables_build_sidechain_ccd_for_lysine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "build-sidechain-ccd",
                    "--ccd-dir",
                    str(CCD_DIR),
                    "--residues",
                    "LYS",
                    "--output-root",
                    tmpdir,
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads((Path(json.loads(stdout)["output"])).read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("geometry-table-set")).validate(payload)
            steps = payload["profiles"]["ccd-sidechains"]["sidechains"]["residues"]["LYS"]["steps"]
            nz_step = next(step for step in steps if step["atom"] == "NZ")
            self.assertEqual(nz_step["dihedral"], "chi4")
            self.assertAlmostEqual(nz_step["length"], 1.49)

    def test_packaged_geometry_table_can_be_listed_validated_and_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, _stderr = _capture_cli(["geometry", "tables", "list"])
            self.assertEqual(status, 0)
            table_ids = {table["id"] for table in json.loads(stdout)}
            self.assertIn("ccd-sidechain-geometry-v1", table_ids)

            status, stdout, _stderr = _capture_cli(
                ["geometry", "tables", "describe", "--table-set", "ccd-sidechain-geometry-v1"]
            )
            self.assertEqual(status, 0)
            description = json.loads(stdout)
            self.assertEqual(description["artifact_id"], "ccd-sidechain-geometry")
            self.assertTrue(description["profiles"]["ccd-sidechains"]["has_sidechains"])

            status, stdout, _stderr = _capture_cli(
                ["geometry", "tables", "validate", "--table-set", "ccd-sidechain-geometry-v1"]
            )
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout)["ok"])

            geometry_path = Path(tmpdir) / "lys.geometry.json"
            heavy_path = Path(tmpdir) / "lys.heavy.json"
            geometry = residue_geometry_structure_from_sequence(
                "K",
                angle_units="degrees",
                stored_angles="all",
            )
            geometry_path.write_text(geometry.to_json() + "\n", encoding="utf-8")
            status, _stdout, _stderr = _capture_cli(
                [
                    "geometry-to-structure",
                    str(geometry_path),
                    "-o",
                    str(heavy_path),
                    "--geometry-table",
                    "ccd-sidechain-geometry-v1",
                ]
            )
            self.assertEqual(status, 0)
            heavy = json.loads(heavy_path.read_text(encoding="utf-8"))
            self.assertEqual(heavy["metadata"]["reconstruction_geometry"]["mode"], "table")
            self.assertEqual(heavy["metadata"]["reconstruction_geometry"]["table"], "ccd-sidechain-geometry-v1")

    def test_geometry_tables_build_sidechain_ccd_from_full_components_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_ccd = root / "components.cif.gz"
            lys = (CCD_DIR / "LYS.cif").read_text(encoding="utf-8")
            dummy = "data_DUM\n#\nloop_\n_chem_comp_atom.atom_id\nD1\n#\n"
            with gzip.open(full_ccd, "wt", encoding="utf-8") as handle:
                handle.write(dummy)
                handle.write(lys)
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "build-sidechain-ccd",
                    "--ccd-full",
                    str(full_ccd),
                    "--residues",
                    "LYS",
                    "--output-root",
                    str(root / "out"),
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads((Path(json.loads(stdout)["output"])).read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("geometry-table-set")).validate(payload)
            self.assertEqual(payload["metadata"]["source"], "wwpdb-ccd-full")
            steps = payload["profiles"]["ccd-sidechains"]["sidechains"]["residues"]["LYS"]["steps"]
            nz_step = next(step for step in steps if step["atom"] == "NZ")
            self.assertAlmostEqual(nz_step["length"], 1.49)

    def test_geometry_tables_build_sidechain_ccd_bcif_only_warns_and_uses_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bcif_dir = root / "bcif"
            bcif_dir.mkdir()
            (bcif_dir / "cca.bcif").write_bytes(b"atoms")
            (bcif_dir / "ccb.bcif").write_bytes(b"bonds")
            status, stdout, _stderr = _capture_cli(
                [
                    "geometry",
                    "tables",
                    "build-sidechain-ccd",
                    "--ccd-bcif-dir",
                    str(bcif_dir),
                    "--residues",
                    "LYS",
                    "--output-root",
                    str(root / "out"),
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads((Path(json.loads(stdout)["output"])).read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("geometry-table-set")).validate(payload)
            self.assertEqual(payload["metadata"]["source"], "rcsb-ccd-bcif-connectivity-only")
            self.assertIn("connectivity-only", "\n".join(payload["metadata"]["warnings"]))
            steps = payload["profiles"]["ccd-sidechains"]["sidechains"]["residues"]["LYS"]["steps"]
            nz_step = next(step for step in steps if step["atom"] == "NZ")
            self.assertEqual(nz_step["dihedral"], "chi4")

    def test_training_decoy_commands_are_metadata_only(self):
        datasets = list_decoy_datasets()
        self.assertGreaterEqual(len(datasets), 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            status, stdout, _stderr = _capture_cli(
                ["training", "decoys", "fetch", "3drobot", "--output-root", tmpdir, "--yes"]
            )
            self.assertEqual(status, 0)
            result = json.loads(stdout)
            self.assertEqual(result["status"], "metadata-only")
            self.assertTrue(Path(result["manifest"]).exists())

            status, stdout, _stderr = _capture_cli(
                ["training", "decoys", "verify", "--input-root", tmpdir]
            )
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout)["ok"])

    def test_reference_torsion_decoy_profile_records_quality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native = structure_from_residue_geometry(
                residue_geometry_structure_from_sequence("MAGKTFY", stored_angles="all")
            )
            native_path = root / "native.json"
            native_path.write_text(native.to_json() + "\n", encoding="utf-8")
            selected_root = root / "selected"
            selected_root.mkdir()
            selected_row = {
                "pdb_id": "TEST",
                "chain_id": "A",
                "path": str(native_path),
                "split": "train",
                "sequence": "MAGKTFY",
                "atom_count": len(native.atoms),
                "length": 7,
            }
            (selected_root / "selected.jsonl").write_text(
                json.dumps(selected_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            decoy_root = root / "decoys"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "build-decoys",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(decoy_root),
                    "--recipes",
                    "pheat-torsion-v1",
                    "--attempts-per-decoy",
                    "12",
                    "--workers",
                    "1",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["recipe_profile"], "pheat-torsion-v1")
            self.assertEqual(payload["requested_decoy_count"], 8)
            self.assertEqual(payload["decoy_count"], 8)
            self.assertEqual(payload["rejection_count"], 0)
            rows = [json.loads(line) for line in (decoy_root / "decoys.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["status"] for row in rows}, {"accepted"})
            self.assertIn("torsion-far", {row["recipe"] for row in rows})
            first = rows[0]
            self.assertIn("ca_rmsd", first["quality"])
            self.assertIn("all_heavy_rmsd", first["quality"])
            self.assertIn("geometry_integrity", first["quality"])
            decoy = load_heavy_json(first["structure_path"])
            self.assertEqual(decoy.metadata["decoy_generation"], "pheat-residue-geometry-torsion-perturbation")

    def test_reference_run_unattended_dry_run_is_v0_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_root = root / "reference"
            snapshot_root = root / "snapshot"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "run-unattended",
                    "--reference-root",
                    str(reference_root),
                    "--snapshot-root",
                    str(snapshot_root),
                    "--workers",
                    "1",
                    "--dry-run",
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["status"], "planned")
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["artifact_version"], "v0")
            self.assertTrue(payload["provisional"])
            self.assertIn("pheat-geometry-integrity", payload["feature_models"])
            self.assertIn("decoys_aqueous", payload["paths"])
            self.assertFalse(reference_root.exists())

    def test_reference_workflow_builds_v0_artifacts_and_ml_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_root = root / "reference"
            snapshot = root / "snapshot"
            raw = snapshot / "raw"
            raw.mkdir(parents=True)
            pdb_path = raw / "1abc.pdb"
            pdb_path.write_text(TINY_PDB.read_text(encoding="utf-8"), encoding="utf-8")
            _write_snapshot(
                snapshot,
                [
                    {
                        "pdb_id": "1ABC",
                        "path": str(pdb_path),
                        "status": "downloaded",
                        "checksums": {"sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest()},
                    }
                ],
            )
            metadata_path = snapshot / "manifests" / "metadata.jsonl"
            metadata_path.write_text(
                json.dumps(
                    {
                        "pdb_id": "1ABC",
                        "method": "X-RAY DIFFRACTION",
                        "resolution": 1.2,
                        "environment": {"aqueous_like": True, "membrane": False},
                        "sequence_clusters": {"30": ["tiny-30"]},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            payload_file = root / "3DRobot_set.tar.bz"
            payload_file.write_bytes(b"local decoy payload placeholder\n")

            status, stdout, _stderr = _capture_cli(["reference", "datasets", "list"])
            self.assertEqual(status, 0)
            dataset_ids = {dataset["id"] for dataset in json.loads(stdout)}
            self.assertIn("3drobot", dataset_ids)

            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "fetch",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--snapshot-root",
                    str(snapshot),
                    "--dataset",
                    "3drobot",
                    "--local-file",
                    f"3drobot={payload_file}",
                    "--workers",
                    "1",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            fetch = json.loads(stdout)
            self.assertTrue(fetch["provisional"])
            dataset_manifest = Path(fetch["datasets"][0]["manifest"])
            dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
            self.assertFalse(dataset_payload["packageable"])
            self.assertTrue(dataset_payload["local_use_only"])
            self.assertEqual(dataset_payload["payloads"][0]["sha256"], hashlib.sha256(payload_file.read_bytes()).hexdigest())

            inventory_path = reference_root / "inventories" / "v0" / "inventory.jsonl"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "inventory",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--snapshot-root",
                    str(snapshot),
                    "--workers",
                    "1",
                    "--overwrite",
                    "-o",
                    str(inventory_path),
                ]
            )
            self.assertEqual(status, 0)
            inventory = json.loads(stdout)
            self.assertEqual(inventory["row_count"], 1)
            self.assertEqual(inventory["audit"]["method_counts"]["x-ray diffraction"], 1)
            self.assertEqual(inventory["metadata_jsonl"], str(metadata_path))

            selected_root = reference_root / "sets" / "protein-heavy-30id-xray-aqueous-v0"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "select",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--inventory",
                    str(inventory_path),
                    "--output-root",
                    str(selected_root),
                    "--min-length",
                    "1",
                    "--max-length",
                    "10",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            selected = json.loads((selected_root / "selected.json").read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("training-corpus")).validate(selected)
            self.assertEqual(selected["artifact_id"], "protein-heavy-30id-xray-aqueous")
            self.assertEqual(selected["artifact_version"], "v0")
            self.assertTrue(selected["provisional"])
            self.assertEqual(selected["filters"]["reference_subset"], "aqueous")
            self.assertTrue(selected["filters"]["default_xray_only"])
            self.assertEqual(selected["filters"]["sequence_identity"], 0.30)
            self.assertEqual(selected["filters"]["cluster_source"], "metadata-sequence-identity-30id")
            self.assertEqual(selected["selected_count"], 1)

            decoy_root = reference_root / "decoys" / "pheat-decoys-v0"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "build-decoys",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(decoy_root),
                    "--recipes",
                    "backbone-noise-small",
                    "--workers",
                    "1",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            decoys = json.loads(stdout)
            self.assertEqual(decoys["decoy_count"], 1)
            self.assertTrue((decoy_root / "decoys.jsonl").exists())

            tables_root = reference_root / "tables" / "protein-heavy-30id-xray-aqueous-v0"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "build-scores",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(tables_root),
                    "--models",
                    "pheat-mj,pheat-rg",
                    "--burial-method",
                    "contacts",
                    "--workers",
                    "1",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            table_payload = json.loads((tables_root / "score-tables.json").read_text(encoding="utf-8"))
            Draft202012Validator(load_schema("score-table-set")).validate(table_payload)
            profile = table_payload["profiles"]["protein-heavy-30id-xray-aqueous-contacts"]
            self.assertIn("pheat-rg", profile["models"])
            self.assertEqual(table_payload["metadata"]["workers"], 1)

            features_path = reference_root / "features" / "v0" / "features.jsonl"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "extract-features",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--training-set",
                    str(selected_root),
                    "--decoys",
                    str(decoy_root / "decoys.jsonl"),
                    "--models",
                    "generic,pheat-rg",
                    "--workers",
                    "1",
                    "--overwrite",
                    "-o",
                    str(features_path),
                ]
            )
            self.assertEqual(status, 0)
            features = json.loads(stdout)
            self.assertEqual(features["feature_count"], 2)
            feature_rows = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["label"] for row in feature_rows}, {"native", "pheat-decoy"})
            self.assertIn("generic_total", feature_rows[0]["features"])

            model_path = reference_root / "models" / "v0" / "pheat-ml-linear.json"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "train-ml",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--features",
                    str(features_path),
                    "--overwrite",
                    "-o",
                    str(model_path),
                ]
            )
            self.assertEqual(status, 0)
            model = json.loads(stdout)
            self.assertEqual(model["artifact_version"], "v0")
            self.assertTrue(model["provisional"])
            self.assertIn("pheat-ml-linear", model["profiles"]["protein-heavy-contacts"]["models"])

            validation_path = reference_root / "validation" / "v0" / "validation.json"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "validate",
                    "--reference-root",
                    str(reference_root),
                    "--artifact-version",
                    "v0",
                    "--features",
                    str(features_path),
                    "--overwrite",
                    "-o",
                    str(validation_path),
                ]
            )
            self.assertEqual(status, 0)
            validation = json.loads(stdout)
            self.assertEqual(validation["labels"], {"native": 1, "pheat-decoy": 1})
            self.assertIn("generic_total", validation["metrics"])

            promoted_path = reference_root / "promoted" / "validation-v1.json"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "promote",
                    "--source",
                    str(validation_path),
                    "--destination",
                    str(promoted_path),
                    "--note",
                    "reviewed tiny reference validation",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["destination"], str(promoted_path))
            self.assertTrue(promoted_path.exists())

    def test_reference_decoys_and_features_use_selected_chain_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "multi.pdb"
            source.write_text(MULTICHAIN_PDB_TEXT, encoding="utf-8")
            selected_root = root / "selected"
            selected_root.mkdir()
            selected_row = {
                "pdb_id": "MUL1",
                "chain_id": "B",
                "path": str(source),
                "split": "train",
                "sequence": "G",
                "atom_count": 4,
                "length": 1,
            }
            (selected_root / "selected.jsonl").write_text(
                json.dumps(selected_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            decoy_root = root / "decoys"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "build-decoys",
                    "--training-set",
                    str(selected_root),
                    "--output-root",
                    str(decoy_root),
                    "--recipes",
                    "backbone-noise-small",
                    "--workers",
                    "1",
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["decoy_count"], 1)
            decoy_row = json.loads((decoy_root / "decoys.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(decoy_row["chain_id"], "B")
            self.assertEqual(decoy_row["atom_count"], 4)
            decoy_structure = load_heavy_json(decoy_row["structure_path"])
            self.assertEqual({atom.chain_id for atom in decoy_structure.atoms}, {"B"})
            self.assertEqual(len(decoy_structure.atoms), 4)

            features_path = root / "features.jsonl"
            status, stdout, _stderr = _capture_cli(
                [
                    "reference",
                    "extract-features",
                    "--training-set",
                    str(selected_root),
                    "--decoys",
                    str(decoy_root / "decoys.jsonl"),
                    "--models",
                    "generic",
                    "--workers",
                    "1",
                    "--overwrite",
                    "-o",
                    str(features_path),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["feature_count"], 2)
            rows = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["atom_count"] for row in rows], [4, 4])
            self.assertEqual([row["residue_count"] for row in rows], [1, 1])
            self.assertFalse(any(row["errors"] for row in rows))

    def test_worker_normalization_accepts_auto_and_positive_counts(self):
        self.assertGreaterEqual(normalize_workers("auto"), 1)
        self.assertEqual(normalize_workers("2"), 2)
        self.assertEqual(normalize_workers(1), 1)
        with self.assertRaises(ValueError):
            normalize_workers("0")

    def test_makefile_and_dependency_extras_cover_training(self):
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        environment = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")

        for target in [
            "training-snapshot-ids:",
            "training-tables:",
            "training-tables-contacts:",
            "training-tables-sasa:",
            "training-corpus-describe:",
            "training-tables-describe:",
            "training-all:",
            "reference-fetch:",
            "reference-metadata:",
            "reference-inventory:",
            "reference-select-aqueous:",
            "reference-select-membrane:",
            "reference-decoys:",
            "reference-scores:",
            "reference-features:",
            "reference-ml-linear:",
            "reference-validate:",
            "reference-package-scoring-assets:",
            "reference-audit:",
            "reference-unattended:",
            "reference-all:",
        ]:
            self.assertIn(target, makefile)
        self.assertIn("PHEAT ?= pheat", makefile)
        self.assertIn("SNAPSHOT_ID ?= rcsb-current-bcif", makefile)
        self.assertIn("CORPUS_VERSION ?= v1", makefile)
        self.assertIn("TABLE_SET_VERSION ?= $(CORPUS_VERSION)", makefile)
        self.assertIn("TRAINING_MODELS ?= pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg", makefile)
        self.assertIn("REFERENCE_VERSION ?= v0", makefile)
        self.assertIn("REFERENCE_WORKERS ?= auto", makefile)
        self.assertIn("REFERENCE_PACKAGE_DEST ?= src/pheat/data/scoring/$(REFERENCE_VERSION)", makefile)
        self.assertIn("REFERENCE_DECOY_RECIPES ?= pheat-torsion-v1", makefile)
        self.assertIn("REFERENCE_DECOY_ATTEMPTS ?= 8", makefile)
        self.assertIn("REFERENCE_METADATA ?= $(SNAPSHOT_ROOT)/manifests/metadata.jsonl", makefile)
        self.assertIn("archive snapshots metadata $(SNAPSHOT_ID)", makefile)
        self.assertIn("reference run-unattended", makefile)
        self.assertIn("REFERENCE_OVERWRITE_FLAG", makefile)
        self.assertIn("--method x-ray", makefile)
        self.assertIn("--sequence-identity $(SEQUENCE_IDENTITY)", makefile)
        self.assertIn("[project.optional-dependencies]", pyproject)
        self.assertIn("openmm = [", pyproject)
        self.assertIn("training-full = [", pyproject)
        self.assertIn('"data/**/*.json.xz"', pyproject)
        self.assertIn("openmm>=8.2", pyproject)
        self.assertIn("pdbfixer==1.12.0", pyproject)
        self.assertIn("- openmm", environment)
        self.assertIn("- pdbfixer", environment)
        self.assertIn("- ambertools", environment)
        self.assertIn("- gromacs", environment)


if __name__ == "__main__":
    unittest.main()

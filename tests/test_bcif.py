import builtins
import json
import os
import tempfile
import unittest
from unittest import mock

from pheat.bcif import _deduplicate_atoms_by_occupancy, load_bcif
from pheat.cli import main as cli_main
from pheat.models import Atom


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_BCIF = os.path.join(FIXTURE_DIR, "tiny.bcif.gz")


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class BinaryCifTests(unittest.TestCase):
    maxDiff = None

    def test_bcif_load_drops_hydrogens_and_uses_label_chain_ids(self):
        structure = load_bcif(TINY_BCIF)

        self.assertEqual(len(structure.atoms), 9)
        self.assertEqual(structure.name, "tiny.bcif.gz")
        self.assertEqual({atom.chain_id for atom in structure.atoms}, {"A"})
        self.assertEqual({atom.resseq for atom in structure.atoms}, {101, 102})
        self.assertEqual(structure.metadata["source"], "bcif")
        self.assertEqual(structure.metadata["chain_id_source"], "label")
        self.assertEqual(structure.metadata["dropped_hydrogen_count"], 1)
        self.assertIn("dropped 1 hydrogen atoms", "\n".join(structure.metadata["warnings"]))

        cb = next(atom for atom in structure.atoms if atom.name == "CB")
        self.assertEqual(cb.altloc, "A")
        self.assertEqual(cb.occupancy, 0.6)
        self.assertEqual(cb.metadata["bcif_chain_id_source"], "label_asym_id")

    def test_cli_converts_bcif_and_accepts_bcif_for_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heavy_path = os.path.join(tmpdir, "tiny_bcif.heavy.json")
            geometry_path = os.path.join(tmpdir, "tiny_bcif.geometry.json")
            rg_path = os.path.join(tmpdir, "tiny_bcif.rg.json")
            score_path = os.path.join(tmpdir, "tiny_bcif.score.json")

            self.assertEqual(cli_main(["bcif-to-structure", TINY_BCIF, "-o", heavy_path]), 0)
            heavy = json.loads(_read_text(heavy_path))
            self.assertEqual({atom["chain_id"] for atom in heavy["atoms"]}, {"A"})
            self.assertEqual({atom["resseq"] for atom in heavy["atoms"]}, {101, 102})

            self.assertEqual(
                cli_main(
                    [
                        "bcif-to-geometry",
                        TINY_BCIF,
                        "-o",
                        geometry_path,
                        "--store-angles",
                        "omega,theta",
                    ]
                ),
                0,
            )
            geometry = json.loads(_read_text(geometry_path))
            self.assertEqual({residue["chain_id"] for residue in geometry["residues"]}, {"A"})
            self.assertIn("omega", geometry["residues"][0])
            self.assertIn("theta", geometry["residues"][0])

            self.assertEqual(cli_main(["radius-of-gyration", TINY_BCIF, "-o", rg_path]), 0)
            rg = json.loads(_read_text(rg_path))
            self.assertEqual(rg["input"], TINY_BCIF)
            self.assertEqual(rg["radius_of_gyration"]["atom_count"], 9)

            self.assertEqual(cli_main(["score", TINY_BCIF, "--model", "generic", "-o", score_path]), 0)
            score = json.loads(_read_text(score_path))
            self.assertEqual(score["model"], "generic")

    def test_bcif_duplicate_atom_rows_select_highest_occupancy(self):
        warnings = []
        atoms = [
            Atom("CB", "C", 0.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1, altloc="A", occupancy=0.4),
            Atom("CB", "C", 1.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1, altloc="B", occupancy=0.7),
            Atom("CA", "C", 2.0, 0.0, 0.0, "ALA", chain_id="A", resseq=1),
        ]

        deduplicated = _deduplicate_atoms_by_occupancy(atoms, warnings)

        self.assertEqual(len(deduplicated), 2)
        cb = next(atom for atom in deduplicated if atom.name == "CB")
        self.assertEqual(cb.altloc, "B")
        self.assertEqual(cb.occupancy, 0.7)
        self.assertIn("duplicate BinaryCIF atom-site rows", "\n".join(warnings))

    def test_bcif_dependency_error_mentions_scientific_and_all_extras(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "msgpack":
                raise ImportError("blocked msgpack import")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, r"\.\[scientific\].*\.\[all\].*msgpack.*numpy"):
                load_bcif(TINY_BCIF)

    def test_binarycif_dependencies_are_in_full_install_surfaces(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        pyproject = _read_text(os.path.join(repo_root, "pyproject.toml")).lower()
        environment = _read_text(os.path.join(repo_root, "environment.yml")).lower()
        readme = _read_text(os.path.join(repo_root, "README.md")).lower()

        self.assertIn('"msgpack"', pyproject)
        self.assertIn("msgpack-python", environment)
        self.assertIn("binarycif input through pheat/msgpack/numpy", readme)
        self.assertIn("bcif-to-structure", readme)
        self.assertIn("rcsb-current-bcif", readme)


if __name__ == "__main__":
    unittest.main()

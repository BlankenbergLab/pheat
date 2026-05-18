import json
import os
import tempfile
import unittest
from unittest import mock

from pheat.cli import main as cli_main
from pheat.mmcif import (
    load_mmcif,
    structure_from_mmcif_string,
    structure_to_mmcif_string,
)
from pheat.models import Atom, HeavyAtomStructure
from pheat.pdbio import load_pdb, structure_to_pdb_string, write_heavy_json
from pheat.roundtrip import (
    run_roundtrip_cases,
    single_roundtrip_case_spec,
    write_roundtrip_artifacts,
)


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_MMCIF = os.path.join(FIXTURE_DIR, "tiny_mmcif.cif")
DISULFIDE_PDB = os.path.join(FIXTURE_DIR, "disulfide.pdb")
AG_GEOMETRY = os.path.join(FIXTURE_DIR, "ag.residue-geometry.json")


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class MmcifTests(unittest.TestCase):
    maxDiff = None

    def test_mmcif_load_uses_auth_ids_by_default_and_drops_hydrogens(self):
        structure = load_mmcif(TINY_MMCIF)

        self.assertEqual(len(structure.atoms), 9)
        self.assertEqual(structure.name, "tiny_mmcif.cif")
        self.assertEqual({atom.chain_id for atom in structure.atoms}, {"CHAINA"})
        self.assertEqual({atom.resseq for atom in structure.atoms}, {101, 102})
        self.assertEqual(structure.metadata["source"], "mmcif")
        self.assertEqual(structure.metadata["chain_id_source"], "auth")
        self.assertEqual(structure.metadata["dropped_hydrogen_count"], 1)
        self.assertIn("dropped 1 hydrogen atoms", "\n".join(structure.metadata["warnings"]))

        cb = next(atom for atom in structure.atoms if atom.name == "CB")
        self.assertEqual(cb.altloc, "A")
        self.assertEqual(cb.occupancy, 0.6)
        self.assertEqual(cb.metadata["auth_asym_id"], "CHAINA")
        self.assertEqual(cb.metadata["label_asym_id"], "A")

    def test_mmcif_load_can_use_label_ids(self):
        structure = load_mmcif(TINY_MMCIF, chain_id_source="label")

        self.assertEqual(len(structure.atoms), 9)
        self.assertEqual({atom.chain_id for atom in structure.atoms}, {"A"})
        self.assertEqual({atom.resseq for atom in structure.atoms}, {1, 2})
        self.assertEqual(structure.metadata["chain_id_source"], "label")

    def test_mmcif_parser_is_native_and_handles_quoted_values_without_biopython(self):
        original_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name == "Bio" or name.startswith("Bio."):
                raise ImportError("blocked Bio import")
            return original_import(name, *args, **kwargs)

        text = """
data_quote_test
_entry.id 'quote test'
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C 'CA' . GLY 'A long chain' 1 1 ? 1.0 2.0 3.0 1.00 9.0 10 GLY AUTHCHAIN CA 1
#
"""

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            structure = structure_from_mmcif_string(text)

        self.assertEqual(structure.name, "quote test")
        self.assertEqual(len(structure.atoms), 1)
        self.assertEqual(structure.atoms[0].chain_id, "AUTHCHAIN")
        self.assertEqual(structure.atoms[0].metadata["label_asym_id"], "A long chain")

    def test_mmcif_write_and_reload_preserves_long_chain_ids(self):
        original = load_mmcif(TINY_MMCIF)
        text = structure_to_mmcif_string(original)
        roundtrip = structure_from_mmcif_string(text, name="roundtrip.cif")

        self.assertIn("_atom_site.auth_asym_id", text)
        self.assertIn("CHAINA", text)
        self.assertEqual(len(roundtrip.atoms), len(original.atoms))
        self.assertEqual({atom.chain_id for atom in roundtrip.atoms}, {"CHAINA"})
        self.assertEqual({atom.resseq for atom in roundtrip.atoms}, {101, 102})

    def test_mmcif_write_and_reload_preserves_disulfide_annotations(self):
        original = load_pdb(DISULFIDE_PDB)
        text = structure_to_mmcif_string(original)
        roundtrip = structure_from_mmcif_string(text)

        self.assertIn("_struct_conn.conn_type_id", text)
        self.assertEqual(len(roundtrip.disulfide_bonds), 1)
        self.assertEqual(roundtrip.disulfide_bonds[0].residue_ref_1, ("A", 1, ""))
        self.assertEqual(roundtrip.disulfide_bonds[0].residue_ref_2, ("A", 2, ""))

    def test_pdb_writer_requires_explicit_truncation_for_long_chain_ids(self):
        structure = HeavyAtomStructure(
            atoms=[
                Atom(
                    name="CA",
                    element="C",
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    resname="GLY",
                    chain_id="AB",
                    resseq=101,
                )
            ],
            name="long-chain",
        )

        with self.assertRaisesRegex(ValueError, "one-character chain IDs"):
            structure_to_pdb_string(structure)

        pdb = structure_to_pdb_string(structure, allow_chain_truncation=True)
        self.assertIn("GLY A 101", pdb)

    def test_cli_converts_mmcif_and_accepts_mmcif_for_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heavy_path = os.path.join(tmpdir, "tiny_mmcif.heavy.json")
            mmcif_path = os.path.join(tmpdir, "tiny_mmcif.roundtrip.cif")
            geometry_path = os.path.join(tmpdir, "tiny_mmcif.geometry.json")
            rg_path = os.path.join(tmpdir, "tiny_mmcif.rg.json")

            self.assertEqual(cli_main(["mmcif-to-structure", TINY_MMCIF, "-o", heavy_path]), 0)
            heavy = json.loads(_read_text(heavy_path))
            self.assertEqual({atom["chain_id"] for atom in heavy["atoms"]}, {"CHAINA"})
            self.assertEqual({atom["resseq"] for atom in heavy["atoms"]}, {101, 102})

            self.assertEqual(cli_main(["structure-to-mmcif", heavy_path, "-o", mmcif_path]), 0)
            self.assertEqual({atom.chain_id for atom in load_mmcif(mmcif_path).atoms}, {"CHAINA"})

            self.assertEqual(
                cli_main(
                    [
                        "mmcif-to-geometry",
                        TINY_MMCIF,
                        "-o",
                        geometry_path,
                        "--chain-id-source",
                        "label",
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

            self.assertEqual(cli_main(["radius-of-gyration", TINY_MMCIF, "-o", rg_path]), 0)
            rg = json.loads(_read_text(rg_path))
            self.assertEqual(rg["input"], TINY_MMCIF)
            self.assertEqual(rg["radius_of_gyration"]["atom_count"], 9)

    def test_geometry_to_heavy_can_write_optional_mmcif_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heavy_path = os.path.join(tmpdir, "ag.heavy.json")
            mmcif_path = os.path.join(tmpdir, "ag.cif")

            self.assertEqual(
                cli_main(
                    [
                        "geometry-to-structure",
                        AG_GEOMETRY,
                        "-o",
                        heavy_path,
                        "--mmcif-output",
                        mmcif_path,
                    ]
                ),
                0,
            )

            self.assertTrue(os.path.exists(heavy_path))
            self.assertIn("_atom_site.auth_asym_id", _read_text(mmcif_path))
            self.assertGreater(len(load_mmcif(mmcif_path).atoms), 0)

    def test_roundtrip_artifacts_can_include_mmcif_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = load_mmcif(TINY_MMCIF)
            result = run_roundtrip_cases(
                structure,
                [single_roundtrip_case_spec()],
                score_models=["generic"],
            )
            summary = write_roundtrip_artifacts(result, tmpdir, include_mmcif=True)
            case = summary["cases"][0]

            self.assertIn("original_aligned_mmcif", case["paths"])
            self.assertIn("reconstructed_aligned_mmcif", case["paths"])
            original_mmcif_path = os.path.join(tmpdir, case["paths"]["original_aligned_mmcif"])
            reconstructed_mmcif_path = os.path.join(
                tmpdir,
                case["paths"]["reconstructed_aligned_mmcif"],
            )
            self.assertTrue(os.path.exists(original_mmcif_path))
            self.assertTrue(os.path.exists(reconstructed_mmcif_path))
            self.assertEqual({atom.chain_id for atom in load_mmcif(original_mmcif_path).atoms}, {"CHAINA"})
            self.assertEqual(
                {atom.chain_id for atom in load_mmcif(reconstructed_mmcif_path).atoms},
                {"CHAINA"},
            )

    def test_web_app_accepts_mmcif_uploads_and_writes_optional_mmcif_outputs(self):
        from fastapi.testclient import TestClient

        from pheat.webapp import create_app

        with tempfile.TemporaryDirectory() as tmpdir:
            vendor_dir = os.path.join(tmpdir, "vendor", "molstar")
            os.makedirs(vendor_dir, exist_ok=True)
            for filename, contents in {
                "molstar.css": "/* test Mol* css */\n",
                "molstar.js": "window.molstar = window.molstar || {};\n",
                "LICENSE": "MIT\n",
            }.items():
                with open(os.path.join(vendor_dir, filename), "w", encoding="utf-8") as handle:
                    handle.write(contents)

            work_dir = os.path.join(tmpdir, "web")
            client = TestClient(create_app(work_dir=work_dir, molstar_vendor_dir=vendor_dir))
            with open(TINY_MMCIF, "rb") as handle:
                response = client.post(
                    "/roundtrip",
                    data={
                        "mode": "single",
                        "score_models": "generic",
                        "include_mmcif": "on",
                    },
                    files={"pdb_file": ("tiny_mmcif.cif", handle, "chemical/x-mmcif")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("original mmCIF", response.text)
            self.assertIn("reconstructed mmCIF", response.text)

            start = response.text.index('data-session-id="') + len('data-session-id="')
            session_id = response.text[start:response.text.index('"', start)]
            summary_path = os.path.join(work_dir, session_id, "summary.json")
            summary = json.loads(_read_text(summary_path))
            case = summary["cases"][0]
            self.assertEqual(summary["input_name"], "tiny_mmcif.cif")
            self.assertIn("original_aligned_mmcif", case["paths"])
            self.assertTrue(
                os.path.exists(os.path.join(work_dir, session_id, case["paths"]["original_aligned_mmcif"]))
            )

    def test_heavy_json_from_mmcif_can_feed_mmcif_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = load_mmcif(TINY_MMCIF)
            heavy_path = os.path.join(tmpdir, "tiny_mmcif.heavy.json")
            pdb_path = os.path.join(tmpdir, "tiny_mmcif.pdb")
            truncated_pdb_path = os.path.join(tmpdir, "tiny_mmcif.truncated.pdb")
            write_heavy_json(structure, heavy_path)

            self.assertEqual(cli_main(["structure-to-pdb", heavy_path, "-o", pdb_path]), 2)
            self.assertFalse(os.path.exists(pdb_path))
            self.assertEqual(
                cli_main(
                    [
                        "structure-to-pdb",
                        heavy_path,
                        "-o",
                        truncated_pdb_path,
                        "--allow-pdb-chain-truncation",
                    ]
                ),
                0,
            )
            self.assertIn("CHAINA", structure_to_mmcif_string(structure))
            self.assertIn("ALA C 101", _read_text(truncated_pdb_path))
            self.assertEqual(
                cli_main(["structure-to-mmcif", heavy_path, "-o", os.path.join(tmpdir, "out.cif")]),
                0,
            )


if __name__ == "__main__":
    unittest.main()

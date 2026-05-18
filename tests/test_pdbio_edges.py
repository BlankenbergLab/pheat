import tempfile
import unittest
from pathlib import Path

from pheat.pdbio import structure_from_pdb_string, structure_to_pdb_string, write_pdb


class PdbIoEdgeTests(unittest.TestCase):
    def test_parser_preserves_insertion_codes(self):
        pdb = (
            "ATOM      1  CA  GLY A   7A      1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )

        structure = structure_from_pdb_string(pdb)
        self.assertEqual(len(structure.atoms), 1)
        self.assertEqual(structure.atoms[0].resseq, 7)
        self.assertEqual(structure.atoms[0].icode, "A")
        self.assertIn("A   7A", structure_to_pdb_string(structure))

    def test_parser_warns_and_skips_malformed_atom_records(self):
        pdb = (
            "ATOM      1  CA  GLY A BAD       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )

        structure = structure_from_pdb_string(pdb)
        self.assertEqual(structure.atoms, [])
        warnings = "\n".join(structure.metadata["warnings"])
        self.assertIn("invalid ATOM/HETATM field", warnings)

    def test_parser_selects_requested_model_or_all_models(self):
        pdb = (
            "MODEL        1\n"
            "ATOM      1  CA  GLY A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "ATOM      2  CA  GLY A   1       2.000   0.000   0.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "END\n"
        )

        default_structure = structure_from_pdb_string(pdb)
        model_two_structure = structure_from_pdb_string(pdb, model=2)
        all_model_structure = structure_from_pdb_string(pdb, model=None)

        self.assertEqual([atom.x for atom in default_structure.atoms], [1.0])
        self.assertEqual([atom.x for atom in model_two_structure.atoms], [2.0])
        self.assertEqual([atom.x for atom in all_model_structure.atoms], [1.0, 2.0])
        self.assertEqual(default_structure.metadata["selected_model"], 1)
        self.assertIsNone(all_model_structure.metadata["selected_model"])

    def test_remarks_are_written_before_atom_records(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        output = structure_to_pdb_string(
            structure,
            remarks=["ENERGY: -42.500", "SOURCE: pheat-unit-test"],
        )
        lines = output.splitlines()
        remark_indices = [i for i, line in enumerate(lines) if line.startswith("REMARK")]
        atom_index = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
        self.assertEqual(remark_indices, [0, 1])
        self.assertLess(max(remark_indices), atom_index)
        self.assertEqual(lines[0], "REMARK   1 ENERGY: -42.500")
        self.assertEqual(lines[1], "REMARK   1 SOURCE: pheat-unit-test")

    def test_remarks_default_omits_remark_records(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        self.assertNotIn("REMARK", structure_to_pdb_string(structure))

    def test_remarks_empty_iterable_omits_remark_records(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        self.assertNotIn("REMARK", structure_to_pdb_string(structure, remarks=[]))

    def test_remarks_multiline_text_is_split_into_separate_remark_lines(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        output = structure_to_pdb_string(structure, remarks=["line one\nline two"])
        lines = output.splitlines()
        self.assertEqual(lines[0], "REMARK   1 line one")
        self.assertEqual(lines[1], "REMARK   1 line two")


    def test_pdb_writer_filters_by_protein_heavy_domain(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "ATOM      2  H   GLY A   1       1.100   2.100   3.100  1.00 10.00           H\n"
            "HETATM    3  O   HOH A 101       4.000   5.000   6.000  1.00 10.00           O\n"
            "HETATM    4  C1  BEN A 102       7.000   8.000   9.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb, hydrogens="preserve")

        output = structure_to_pdb_string(structure, domain="protein-heavy")

        self.assertIn(" GLY ", output)
        self.assertNotIn(" HOH ", output)
        self.assertNotIn(" BEN ", output)
        self.assertNotIn(" H   GLY", output)

    def test_pdb_writer_all_heavy_keeps_nonprotein_heavy_atoms(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "ATOM      2  H   GLY A   1       1.100   2.100   3.100  1.00 10.00           H\n"
            "HETATM    3  O   HOH A 101       4.000   5.000   6.000  1.00 10.00           O\n"
            "HETATM    4  C1  BEN A 102       7.000   8.000   9.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb, hydrogens="preserve")

        output = structure_to_pdb_string(structure, domain="all-heavy")

        self.assertIn(" GLY ", output)
        self.assertIn(" HOH ", output)
        self.assertIn(" BEN ", output)
        self.assertNotIn(" H   GLY", output)

    def test_pdb_writer_domain_validates_and_write_pdb_forwards_domain(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "HETATM    2  O   HOH A 101       4.000   5.000   6.000  1.00 10.00           O\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        with self.assertRaisesRegex(ValueError, "domain must be one of"):
            structure_to_pdb_string(structure, domain="not-a-domain")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "filtered.pdb"
            write_pdb(structure, output_path, domain="protein-heavy")
            output = output_path.read_text(encoding="utf-8")
        self.assertIn(" GLY ", output)
        self.assertNotIn(" HOH ", output)

    def test_domain_helpers_are_exported_from_package_root(self):
        from pheat import SCORING_DOMAINS, filter_structure_for_domain, normalize_domain

        self.assertIn("protein-heavy", SCORING_DOMAINS)
        self.assertEqual(normalize_domain("protein-heavy"), "protein-heavy")
        pdb = "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n"
        structure = structure_from_pdb_string(pdb)
        filtered, coverage = filter_structure_for_domain(structure, domain="protein-heavy")
        self.assertEqual(len(filtered.atoms), 1)
        self.assertEqual(coverage["domain"], "protein-heavy")

    def test_remarks_round_trip_through_load_pdb(self):
        pdb = (
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "END\n"
        )
        structure = structure_from_pdb_string(pdb)
        output = structure_to_pdb_string(structure, remarks=["ENERGY: -1.250"])
        reloaded = structure_from_pdb_string(output)
        # REMARK records are dropped silently by the parser; ATOM records survive.
        self.assertEqual(len(reloaded.atoms), 1)
        self.assertEqual(reloaded.atoms[0].name, "CA")

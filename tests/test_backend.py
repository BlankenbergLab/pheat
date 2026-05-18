import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from pheat import __version__
from pheat.centroid import to_centroid_structure
from pheat.cli import main as cli_main
from pheat.examples import list_example_sets, load_example_manifest, run_example_set
from pheat.geometry import (
    apply_kabsch_transform,
    distance,
    dihedral_degrees,
    kabsch_align,
    kabsch_rmsd,
    kabsch_transform,
    radius_of_gyration,
    rotate_around_axis,
)
from pheat.metrics import structure_radius_of_gyration, structure_rmsd, structure_rmsd_result
from pheat.models import Atom, HeavyAtomStructure, ResidueGeometryStructure
from pheat.pdbio import (
    load_heavy_json,
    load_pdb,
    structure_from_pdb_string,
    structure_to_pdb_string,
)
from pheat.schemas import load_schema
from pheat.scoring import score_structure
from pheat.sources import fetch_source, list_sources, verify_sources
from pheat.residue_geometry import (
    load_residue_geometry,
    structure_from_residue_geometry,
    structure_to_residue_geometry,
    write_residue_geometry_json,
)
from pheat.residues import CANONICAL_RESIDUES, SUPPORTED_RESIDUES, one_to_three, three_to_one


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EXPECTED_DIR = os.path.join(FIXTURE_DIR, "expected")
TINY_PDB = os.path.join(FIXTURE_DIR, "tiny.pdb")
ALL_ATOM_TINY_PDB = os.path.join(FIXTURE_DIR, "all_atom_tiny.pdb")
DISULFIDE_PDB = os.path.join(FIXTURE_DIR, "disulfide.pdb")
NONCANONICAL_PDB = os.path.join(FIXTURE_DIR, "noncanonical.pdb")
TWO_MU7_PDB = os.path.join(FIXTURE_DIR, "2mu7.pdb")
AG_GEOMETRY = os.path.join(FIXTURE_DIR, "ag.residue-geometry.json")
AG_GEOMETRY_DEGREES = os.path.join(FIXTURE_DIR, "ag.residue-geometry.degrees.json")
NOTEBOOK_DIR = os.path.join(REPO_ROOT, "examples", "notebook")
MOLSTAR_SOURCE_MAP_SCRIPT = os.path.join(REPO_ROOT, "scripts", "strip_molstar_source_maps.py")
MOLSTAR_NOTEBOOK = os.path.join(NOTEBOOK_DIR, "2mu7_roundtrip_energy_rmsd_molstar.ipynb")
OLD_NGLVIEW_NOTEBOOK = os.path.join(NOTEBOOK_DIR, "2mu7_roundtrip_energy_rmsd_nglview.ipynb")
EXECUTED_MOLSTAR_NOTEBOOK = os.path.join(
    NOTEBOOK_DIR,
    "executed",
    "2mu7_roundtrip_energy_rmsd_molstar.executed.ipynb",
)
OLD_EXECUTED_NGLVIEW_NOTEBOOK = os.path.join(
    NOTEBOOK_DIR,
    "executed",
    "2mu7_roundtrip_energy_rmsd_nglview.executed.ipynb",
)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _expected(name):
    return os.path.join(EXPECTED_DIR, name)


def _atom_key(atom):
    return (
        atom.chain_id or "",
        int(atom.resseq),
        atom.icode or "",
        atom.resname.strip().upper(),
        atom.name.strip().upper(),
    )


def _atoms_by_key(structure):
    return {_atom_key(atom): atom for atom in structure.atoms}


def _load_combinatorial_roundtrip_module():
    path = os.path.join(REPO_ROOT, "examples", "2mu7_combinatorial_roundtrip.py")
    spec = importlib.util.spec_from_file_location("pheat_2mu7_combinatorial_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _openmm_available():
    try:
        import openmm  # noqa: F401
    except Exception:
        return False
    return True


class BackendTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def tearDownClass(cls):
        tmpdir = getattr(cls, "_example_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()
        super().tearDownClass()

    @classmethod
    def generated_example_roundtrip_dir(cls):
        roundtrip_dir = getattr(cls, "_example_roundtrip_dir", None)
        if roundtrip_dir is not None:
            return roundtrip_dir

        cls._example_tmpdir = tempfile.TemporaryDirectory()
        root = cls._example_tmpdir.name
        vendor_dir = os.path.join(root, "vendor", "molstar")
        os.makedirs(vendor_dir, exist_ok=True)
        for filename, contents in {
            "molstar.css": "/* test Mol* css */\n",
            "molstar.js": "window.molstar = window.molstar || {};\n",
            "LICENSE": "MIT\n",
        }.items():
            with open(os.path.join(vendor_dir, filename), "w", encoding="utf-8") as handle:
                handle.write(contents)

        roundtrip_dir = os.path.join(root, "roundtrip", "2mu7_combinatorial")
        module = _load_combinatorial_roundtrip_module()
        result = module.main(
            [
                "--output-root",
                roundtrip_dir,
                "--molstar-vendor-dir",
                vendor_dir,
            ]
        )
        if result != 0:
            raise AssertionError(f"2MU7 combinatorial roundtrip generation failed: {result}")
        cls._example_roundtrip_dir = roundtrip_dir
        cls._example_vendor_dir = vendor_dir
        return roundtrip_dir

    def create_test_web_client(self, tmpdir):
        from fastapi.testclient import TestClient

        from pheat.webapp import create_app

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
        app = create_app(work_dir=work_dir, molstar_vendor_dir=vendor_dir)
        return TestClient(app), work_dir

    def session_id_from_response(self, text):
        match = re.search(r'data-session-id="([0-9a-f]+)"', text)
        if match is None:
            raise AssertionError("response did not include a web session id")
        return match.group(1)

    def assert_file_matches_expected(self, actual_path, expected_name):
        self.assertEqual(_read_text(actual_path), _read_text(_expected(expected_name)))

    def assert_payload_matches_schema(self, schema_name, payload):
        schema = load_schema(schema_name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)

    def assert_real_pdb_chi_coverage(self, residue_geometry):
        residues_by_position = {residue.resseq: residue for residue in residue_geometry.residues}
        chi_lengths = {len(residue.chi) for residue in residue_geometry.residues}

        self.assertIn(1, chi_lengths)
        self.assertTrue(any(length > 1 for length in chi_lengths))
        self.assertEqual(
            (residues_by_position[3].name, len(residues_by_position[3].chi)),
            ("SER", 1),
        )
        self.assertEqual(
            (residues_by_position[7].name, len(residues_by_position[7].chi)),
            ("LYS", 4),
        )
        self.assertEqual(
            (residues_by_position[11].name, len(residues_by_position[11].chi)),
            ("VAL", 1),
        )

    def test_pdb_roundtrip_preserves_heavy_atoms_and_heterogens(self):
        structure = load_pdb(TINY_PDB)
        self.assertEqual(len(structure.atoms), 12)
        self.assertIn("A", {atom.altloc for atom in structure.atoms})
        self.assertIn("B", {atom.altloc for atom in structure.atoms})
        self.assertIn("ZN", {atom.resname for atom in structure.atoms})

        roundtrip = structure_from_pdb_string(structure_to_pdb_string(structure))
        self.assertEqual(len(roundtrip.atoms), 12)
        self.assertIn("HOH", {atom.resname for atom in roundtrip.atoms})

    def test_all_atom_pdb_to_heavy_drops_hydrogens(self):
        structure = load_pdb(ALL_ATOM_TINY_PDB)
        self.assertEqual(structure.metadata["dropped_hydrogen_count"], 2)
        self.assertNotIn("H", {atom.element for atom in structure.atoms})
        self.assertEqual([atom.name for atom in structure.atoms], ["N", "CA", "C", "O", "CB"])

    def test_all_atom_pdb_can_preserve_source_hydrogens(self):
        structure = load_pdb(ALL_ATOM_TINY_PDB, hydrogens="preserve")

        self.assertEqual(structure.atom_scope, "all")
        self.assertEqual(structure.metadata["hydrogen_policy"], "preserved")
        self.assertEqual(structure.metadata["preserved_hydrogen_count"], 2)
        self.assertIn("H", {atom.element for atom in structure.atoms})
        self.assertEqual([atom.name for atom in structure.atoms], ["N", "H", "CA", "HA", "C", "O", "CB"])
        self.assertIn(" H   ALA", structure_to_pdb_string(structure))
        self.assertEqual(HeavyAtomStructure.from_json(structure.to_json()).atom_scope, "all")

    def test_atom_structure_can_store_template_bonds_with_lengths(self):
        structure = load_pdb(TINY_PDB, store_bonds="all")

        self.assertEqual(structure.atom_scope, "heavy")
        self.assertGreaterEqual(len(structure.bonds), 8)
        bond_pairs = {
            (
                structure.atoms[bond.atom_index_1].resseq,
                structure.atoms[bond.atom_index_1].name.strip(),
                structure.atoms[bond.atom_index_2].resseq,
                structure.atoms[bond.atom_index_2].name.strip(),
            )
            for bond in structure.bonds
        }
        self.assertIn((1, "N", 1, "CA"), bond_pairs)
        self.assertIn((1, "C", 2, "N"), bond_pairs)
        self.assertTrue(all(bond.length and bond.length > 0 for bond in structure.bonds))
        self.assert_payload_matches_schema("atom-structure", json.loads(structure.to_json()))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "tiny.bonds.json")
            self.assertEqual(
                cli_main(["pdb-to-structure", TINY_PDB, "-o", output, "--store-bonds", "all"]),
                0,
            )
            payload = json.loads(_read_text(output))
            self.assertGreaterEqual(len(payload["bonds"]), 8)
            self.assert_payload_matches_schema("atom-structure", payload)

    def test_residue_geometry_can_store_and_reuse_bond_lengths(self):
        residue_geometry = structure_to_residue_geometry(load_pdb(TINY_PDB), stored_lengths="all")
        first = residue_geometry.residues[0]

        self.assertEqual(residue_geometry.stored_lengths, ("all",))
        self.assertAlmostEqual(first.bond_lengths["N-CA"], 1.458, places=6)
        self.assertIn("CA-CB", first.bond_lengths)
        self.assertIn("C-N", first.bond_lengths)
        self.assertIn("bond_lengths", residue_geometry.to_dict()["residues"][0])

        first.bond_lengths["N-CA"] = 2.0
        rebuilt = structure_from_residue_geometry(residue_geometry)
        atoms = _atoms_by_key(rebuilt)
        n_atom = atoms[("A", 1, "", "ALA", "N")]
        ca_atom = atoms[("A", 1, "", "ALA", "CA")]
        self.assertAlmostEqual(distance(n_atom.coord, ca_atom.coord), 2.0, places=6)

    def test_cli_can_store_selected_residue_geometry_bond_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "tiny.lengths.json")

            self.assertEqual(
                cli_main(["pdb-to-geometry", TINY_PDB, "-o", output, "--store-lengths", "backbone"]),
                0,
            )
            payload = json.loads(_read_text(output))
            first_lengths = payload["residues"][0]["bond_lengths"]
            self.assertIn("N-CA", first_lengths)
            self.assertIn("C-N", first_lengths)
            self.assertNotIn("CA-CB", first_lengths)
            self.assert_payload_matches_schema("residue-geometry-structure", payload)

    def test_heavy_json_roundtrip(self):
        structure = load_pdb(TINY_PDB)
        encoded = structure.to_json()
        decoded = HeavyAtomStructure.from_json(encoded)
        self.assertEqual(decoded.atoms[0].name, "N")
        self.assertAlmostEqual(decoded.atoms[1].x, 1.458)

    def test_kabsch_alignment_and_rmsd_handle_translation_and_rotation(self):
        reference = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 3.0),
        ]
        translated = [(x + 5.0, y - 3.0, z + 2.0) for x, y, z in reference]
        rotated = [
            rotate_around_axis(point, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
            for point in reference
        ]
        rotated_translated = [(x + 5.0, y - 3.0, z + 2.0) for x, y, z in rotated]
        aligned = kabsch_align(reference, rotated_translated)

        self.assertAlmostEqual(kabsch_rmsd(reference, reference), 0.0)
        self.assertAlmostEqual(kabsch_rmsd(reference, translated), 0.0)
        self.assertAlmostEqual(kabsch_rmsd(reference, rotated_translated), 0.0)
        self.assertAlmostEqual(
            kabsch_rmsd(reference, rotated_translated, aligned_target=aligned),
            0.0,
        )
        with self.assertRaises(ValueError):
            kabsch_align(reference, reference[:-1])
        with self.assertRaises(ValueError):
            kabsch_rmsd(reference, reference[:-1])

    def test_kabsch_transform_can_be_applied_to_additional_coordinates(self):
        reference = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 3.0),
        ]
        rotated = [
            rotate_around_axis(point, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
            for point in reference
        ]
        mobile = [(x + 5.0, y - 3.0, z + 2.0) for x, y, z in rotated]
        extra = [(7.0, -4.0, 2.0)]

        transform = kabsch_transform(reference, mobile)
        aligned = apply_kabsch_transform(mobile, transform)
        aligned_extra = apply_kabsch_transform(extra, transform)

        self.assertAlmostEqual(kabsch_rmsd(reference, mobile, aligned_target=aligned), 0.0)
        self.assertEqual(len(aligned_extra), 1)

    def test_structure_rmsd_supports_ca_atom_set_and_alignment_basis(self):
        reference = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CB", "C", 1.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 3.0, 0.0, 0.0, "GLY", resseq=2),
                Atom("CB", "C", 4.0, 0.0, 0.0, "GLY", resseq=2),
            ],
            name="reference",
        )
        target = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 5.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CB", "C", 7.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 8.0, 0.0, 0.0, "GLY", resseq=2),
                Atom("CB", "C", 11.0, 0.0, 0.0, "GLY", resseq=2),
            ],
            name="target",
        )

        ca = structure_rmsd(reference, target, atom_set="ca")
        all_heavy_after_ca_alignment = structure_rmsd(
            reference,
            target,
            atom_set="all-heavy",
            alignment_atom_set="ca",
        )

        self.assertEqual(ca["atom_set"], "ca")
        self.assertEqual(ca["matched_atoms"], 2)
        self.assertAlmostEqual(ca["value"], 0.0)
        self.assertEqual(all_heavy_after_ca_alignment["alignment_atom_set"], "ca")
        self.assertGreater(all_heavy_after_ca_alignment["value"], 0.0)

    def test_structure_rmsd_result_defaults_to_all_heavy(self):
        structure = load_pdb(TINY_PDB)
        payload = structure_rmsd_result(
            structure,
            structure,
            reference_name="reference",
            target_name="target",
        )

        self.assertEqual(payload["format"], "pheat.rmsd-result")
        self.assertEqual(payload["rmsd"]["atom_set"], "all-heavy")
        self.assertEqual(payload["rmsd"]["alignment_atom_set"], "all-heavy")
        self.assertAlmostEqual(payload["rmsd"]["value"], 0.0)

    def test_radius_of_gyration_handles_unweighted_weighted_and_invariant_cases(self):
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        translated = [(x + 5.0, y - 3.0, z + 2.0) for x, y, z in coords]
        rotated = [
            rotate_around_axis(point, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
            for point in coords
        ]

        self.assertAlmostEqual(radius_of_gyration(coords), 1.0)
        self.assertAlmostEqual(radius_of_gyration(coords, weights=[1.0, 3.0]), math.sqrt(0.75))
        self.assertAlmostEqual(radius_of_gyration(coords), radius_of_gyration(translated))
        self.assertAlmostEqual(radius_of_gyration(coords), radius_of_gyration(rotated))

        with self.assertRaises(ValueError):
            radius_of_gyration([])
        with self.assertRaises(ValueError):
            radius_of_gyration(coords, weights=[1.0])
        with self.assertRaises(ValueError):
            radius_of_gyration(coords, weights=[1.0, -1.0])
        with self.assertRaises(ValueError):
            radius_of_gyration(coords, weights=[0.0, 0.0])

    def test_structure_radius_of_gyration_mass_weighting_and_unknown_elements(self):
        structure = HeavyAtomStructure(
            atoms=[
                Atom("C1", "C", 0.0, 0.0, 0.0, "LIG"),
                Atom("X1", "XX", 2.0, 0.0, 0.0, "LIG"),
            ],
            name="unknown-element",
        )

        payload = structure_radius_of_gyration(structure)

        self.assertEqual(payload["atom_count"], 2)
        self.assertEqual(payload["atom_set"], "all-heavy")
        self.assertEqual(payload["units"], "angstrom")
        self.assertEqual(payload["mode"], "both")
        self.assertAlmostEqual(payload["values"]["unweighted"], 1.0)
        self.assertAlmostEqual(payload["values"]["mass_weighted"], 1.0)
        self.assertEqual(payload["unknown_elements"], ["XX"])
        self.assertEqual(payload["mass_source"], "ciaaw-standard-atomic-weights-2024")

    def test_structure_radius_of_gyration_supports_rmsd_atom_sets(self):
        structure = HeavyAtomStructure(
            atoms=[
                Atom("CA", "C", 0.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CB", "C", 50.0, 0.0, 0.0, "ALA", resseq=1),
                Atom("CA", "C", 3.0, 0.0, 0.0, "GLY", resseq=2),
                Atom("MG", "CA", 200.0, 0.0, 0.0, "CA", record_name="HETATM", resseq=3),
            ],
            name="rg-atom-sets",
        )

        all_heavy = structure_radius_of_gyration(structure, atom_set="all-heavy", mode="unweighted")
        ca = structure_radius_of_gyration(structure, atom_set="ca", mode="unweighted")
        backbone = structure_radius_of_gyration(structure, atom_set="backbone", mode="unweighted")

        self.assertEqual(all_heavy["atom_count"], 4)
        self.assertEqual(ca["atom_set"], "ca")
        self.assertEqual(ca["atom_count"], 2)
        self.assertAlmostEqual(ca["values"]["unweighted"], 1.5)
        self.assertGreater(all_heavy["values"]["unweighted"], ca["values"]["unweighted"])
        self.assertEqual(backbone["atom_set"], "backbone")
        self.assertEqual(backbone["atom_count"], 2)

    def test_dihedral_degrees_uses_standard_signed_convention(self):
        self.assertAlmostEqual(
            dihedral_degrees(
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
            ),
            180.0,
        )
        self.assertAlmostEqual(
            dihedral_degrees(
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 1.0),
            ),
            -90.0,
        )

    def test_real_pdb_reconstruction_kabsch_rmsd_regression(self):
        original_atoms = _atoms_by_key(load_pdb(TWO_MU7_PDB))
        reconstructed_atoms = _atoms_by_key(load_heavy_json(_expected("2mu7.reconstructed.heavy.json")))
        common_keys = sorted(set(original_atoms) & set(reconstructed_atoms))

        self.assertEqual(len(common_keys), 150)
        self.assertEqual(set(original_atoms) - set(reconstructed_atoms), set())
        self.assertEqual(set(reconstructed_atoms) - set(original_atoms), set())
        original_coords = [original_atoms[key].coord for key in common_keys]
        reconstructed_coords = [reconstructed_atoms[key].coord for key in common_keys]
        aligned_coords = kabsch_align(original_coords, reconstructed_coords)

        self.assertAlmostEqual(
            kabsch_rmsd(original_coords, reconstructed_coords),
            1.0049890010274787,
        )
        self.assertAlmostEqual(
            kabsch_rmsd(original_coords, reconstructed_coords, aligned_target=aligned_coords),
            1.0049890010274787,
        )

    def test_in_memory_residue_geometry_reconstruction_matches_serialized_optional_geometry(self):
        original = load_pdb(TWO_MU7_PDB)
        residue_geometry = structure_to_residue_geometry(original)
        direct = structure_from_residue_geometry(residue_geometry)

        with tempfile.TemporaryDirectory() as tmpdir:
            residue_geometry_path = os.path.join(tmpdir, "2mu7.residue-geometry.json")
            write_residue_geometry_json(residue_geometry, residue_geometry_path)
            serialized = structure_from_residue_geometry(residue_geometry_path)

        direct_atoms = _atoms_by_key(direct)
        serialized_atoms = _atoms_by_key(serialized)
        self.assertEqual(set(direct_atoms), set(serialized_atoms))
        for key in direct_atoms:
            for direct_value, serialized_value in zip(direct_atoms[key].coord, serialized_atoms[key].coord):
                self.assertAlmostEqual(direct_value, serialized_value, places=10)

        original_atoms = _atoms_by_key(original)
        common_keys = sorted(set(original_atoms) & set(direct_atoms))
        self.assertAlmostEqual(
            kabsch_rmsd(
                [original_atoms[key].coord for key in common_keys],
                [direct_atoms[key].coord for key in common_keys],
            ),
            1.0049890010274787,
        )

    def test_residue_geometry_extraction_does_not_bridge_residue_number_gaps(self):
        structure = structure_from_residue_geometry({"sequence": "AG"})
        for atom in structure.atoms:
            if atom.resseq == 2:
                atom.resseq = 3

        residue_geometry = structure_to_residue_geometry(structure, angle_units="degrees", stored_angles="all")

        self.assertEqual([residue.resseq for residue in residue_geometry.residues], [1, 3])
        self.assertIsNone(residue_geometry.residues[0].psi)
        self.assertIsNone(residue_geometry.residues[0].omega)
        self.assertIsNone(residue_geometry.residues[0].theta)
        self.assertIsNone(residue_geometry.residues[1].phi)

    def test_residue_geometry_reconstruction_does_not_bridge_residue_number_gaps(self):
        residue_geometry = ResidueGeometryStructure.from_dict(
            {
                "residues": [
                    {"name": "ALA", "chain_id": "A", "resseq": 1},
                    {"name": "GLY", "chain_id": "A", "resseq": 3},
                ]
            }
        )
        reconstructed = structure_from_residue_geometry(residue_geometry)
        atoms = _atoms_by_key(reconstructed)
        c_atom = atoms[("A", 1, "", "ALA", "C")]
        n_atom = atoms[("A", 3, "", "GLY", "N")]

        self.assertGreater(distance(c_atom.coord, n_atom.coord), 2.0)

        with_oxt = structure_from_residue_geometry(
            residue_geometry,
            include_terminal_oxt=True,
        )
        self.assertEqual(
            [(atom.resseq, atom.name) for atom in with_oxt.atoms if atom.name == "OXT"],
            [(1, "OXT"), (3, "OXT")],
        )

    def test_residue_geometry_reconstruction_covers_all_canonical_residues(self):
        residue_geometry = {
            "name": "all canonical",
            "sequence": "ARNDCQEGHILKMFPSTWYV",
            "phi": -math.pi / 3.0,
            "psi": -math.pi / 4.0,
            "omega": math.pi,
        }
        structure = structure_from_residue_geometry(residue_geometry)
        self.assertEqual(len(structure.residue_keys()), 20)
        residue_names = {key[3] for key in structure.residue_keys()}
        self.assertIn("TRP", residue_names)
        self.assertIn("GLY", residue_names)
        self.assertNotIn("H", {atom.element for atom in structure.atoms})

    def test_residue_metadata_distinguishes_canonical_and_supported_modified_residues(self):
        for resname in ["SEC", "PYL", "MSE", "HYP", "LYZ", "SEP", "TPO", "PTR", "PCA"]:
            self.assertIn(resname, SUPPORTED_RESIDUES)
            self.assertNotIn(resname, CANONICAL_RESIDUES)

        self.assertEqual(one_to_three("U"), "SEC")
        self.assertEqual(one_to_three("O"), "PYL")
        self.assertEqual(one_to_three("MSE"), "MSE")
        self.assertEqual(one_to_three("HYL"), "LYZ")
        self.assertEqual(three_to_one("SEC"), "U")
        self.assertEqual(three_to_one("PYL"), "O")
        with self.assertRaises(ValueError):
            three_to_one("MSE")

    def test_residue_geometry_reconstruction_supports_documented_modified_residues(self):
        residue_names = ["SEC", "PYL", "MSE", "HYP", "LYZ", "SEP", "TPO", "PTR", "PCA"]
        residue_geometry = {
            "name": "modified residues",
            "angle_units": "degrees",
            "residues": [
                {
                    "name": resname,
                    "chain_id": "A",
                    "resseq": index + 1,
                    "phi": None if index == 0 else -60.0,
                    "psi": None if index == len(residue_names) - 1 else -45.0,
                    "omega": None if index == len(residue_names) - 1 else 180.0,
                    "chi": [-60.0, -60.0, -90.0, 180.0, 0.0, 0.0],
                }
                for index, resname in enumerate(residue_names)
            ],
        }
        structure = structure_from_residue_geometry(residue_geometry)
        atoms = {
            (atom.resname.strip().upper(), atom.name.strip().upper()): atom
            for atom in structure.atoms
        }

        expected_atoms = {
            "SEC": {"SE"},
            "PYL": {"C2", "O2", "CA2", "N2", "CE2", "CD2", "CG2", "CB2"},
            "MSE": {"SE", "CE"},
            "HYP": {"OD1"},
            "LYZ": {"OH", "NZ"},
            "SEP": {"P", "O1P", "O2P", "O3P"},
            "TPO": {"P", "O1P", "O2P", "O3P", "CG2"},
            "PTR": {"P", "O1P", "O2P", "O3P", "OH"},
            "PCA": {"CD", "OE"},
        }
        for resname, atom_names in expected_atoms.items():
            with self.subTest(residue=resname):
                for atom_name in atom_names:
                    self.assertIn((resname, atom_name), atoms)

        self.assertEqual(atoms[("SEC", "SE")].element, "SE")
        self.assertEqual(atoms[("MSE", "SE")].element, "SE")
        self.assertEqual(atoms[("SEP", "P")].element, "P")

        alias_structure = structure_from_residue_geometry({"residues": [{"name": "HYL"}]})
        self.assertEqual(alias_structure.residue_keys()[0][3], "LYZ")

        shorthand_structure = structure_from_residue_geometry({"sequence": "UO"})
        self.assertEqual([key[3] for key in shorthand_structure.residue_keys()], ["SEC", "PYL"])

    def test_modified_residue_chi_extraction_uses_residue_templates(self):
        residue_names = ["SEC", "MSE", "HYP", "LYZ", "SEP", "TPO", "PTR", "PCA", "PYL"]
        structure = structure_from_residue_geometry(
            {
                "angle_units": "degrees",
                "residues": [
                    {
                        "name": resname,
                        "chain_id": "A",
                        "resseq": index + 1,
                        "phi": None if index == 0 else -60.0,
                        "psi": None if index == len(residue_names) - 1 else -45.0,
                        "omega": None if index == len(residue_names) - 1 else 180.0,
                        "chi": [-60.0, -60.0, -90.0, 180.0, 0.0, 0.0],
                    }
                    for index, resname in enumerate(residue_names)
                ],
            }
        )

        residue_geometry = structure_to_residue_geometry(structure, angle_units="degrees")
        chi_lengths = {residue.name: len(residue.chi) for residue in residue_geometry.residues}

        self.assertEqual(
            chi_lengths,
            {
                "SEC": 1,
                "MSE": 3,
                "HYP": 2,
                "LYZ": 4,
                "SEP": 3,
                "TPO": 3,
                "PTR": 4,
                "PCA": 3,
                "PYL": 6,
            },
        )

    def test_noncanonical_pdb_fixture_extracts_supported_residue_geometry(self):
        structure = load_pdb(NONCANONICAL_PDB)
        residue_geometry = structure_to_residue_geometry(structure, angle_units="degrees")

        self.assertEqual(
            [residue.name for residue in residue_geometry.residues],
            ["SEC", "PYL", "MSE", "HYP", "LYZ", "SEP", "TPO", "PTR", "PCA"],
        )
        self.assertEqual(
            {residue.name: len(residue.chi) for residue in residue_geometry.residues},
            {
                "SEC": 1,
                "PYL": 6,
                "MSE": 3,
                "HYP": 2,
                "LYZ": 4,
                "SEP": 3,
                "TPO": 3,
                "PTR": 4,
                "PCA": 3,
            },
        )
        self.assertNotIn("no side-chain chi template", "\n".join(residue_geometry.metadata["warnings"]))

    def test_canonical_ring_reconstruction_closes_ring_bonds(self):
        closure_checks = {
            "P": ("PRO", [(("N", "CD"), 1.47), (("CD", "CG"), 1.50)]),
            "F": ("PHE", [(("CE2", "CZ"), 1.39)]),
            "Y": ("TYR", [(("CE2", "CZ"), 1.39)]),
            "H": ("HIS", [(("CE1", "NE2"), 1.38)]),
            "W": ("TRP", [(("NE1", "CE2"), 1.40), (("CZ3", "CH2"), 1.40)]),
        }

        for sequence, (resname, checks) in closure_checks.items():
            with self.subTest(residue=resname):
                structure = structure_from_residue_geometry({"sequence": sequence})
                atoms = {
                    atom.name.strip().upper(): atom
                    for atom in structure.atoms
                    if atom.resname == resname
                }
                for (atom_a, atom_b), expected_distance in checks:
                    self.assertIn(atom_a, atoms)
                    self.assertIn(atom_b, atoms)
                    self.assertAlmostEqual(
                        distance(atoms[atom_a].coord, atoms[atom_b].coord),
                        expected_distance,
                        delta=0.08,
                    )

    def test_modified_ring_reconstruction_closes_ring_bonds(self):
        closure_checks = {
            "HYP": [(("N", "CD"), 1.47), (("CD", "CG"), 1.50)],
            "PCA": [(("N", "CD"), 1.35), (("CD", "CG"), 1.52)],
            "PYL": [(("CA2", "CG2"), 1.53), (("CD2", "CG2"), 1.53)],
        }

        for resname, checks in closure_checks.items():
            with self.subTest(residue=resname):
                structure = structure_from_residue_geometry({"residues": [{"name": resname}]})
                atoms = {
                    atom.name.strip().upper(): atom
                    for atom in structure.atoms
                    if atom.resname == resname
                }
                for (atom_a, atom_b), expected_distance in checks:
                    self.assertIn(atom_a, atoms)
                    self.assertIn(atom_b, atoms)
                    self.assertAlmostEqual(
                        distance(atoms[atom_a].coord, atoms[atom_b].coord),
                        expected_distance,
                        delta=0.08,
                    )

    def test_real_pdb_reconstructed_aromatic_rings_are_closed(self):
        structure = load_heavy_json(_expected("2mu7.reconstructed.heavy.json"))
        for residue_key, atoms in structure.atoms_by_residue().items():
            _chain_id, _resseq, _icode, resname, _record_name = residue_key
            atoms_by_name = {atom.name.strip().upper(): atom for atom in atoms}
            if resname in {"PHE", "TYR"}:
                self.assertLess(
                    distance(atoms_by_name["CE2"].coord, atoms_by_name["CZ"].coord),
                    1.55,
                )

    def test_terminal_oxt_reconstruction_is_optional(self):
        default_structure = structure_from_residue_geometry({"sequence": "AG"})
        with_oxt_structure = structure_from_residue_geometry(
            {"sequence": "AG"},
            include_terminal_oxt=True,
        )

        self.assertNotIn("OXT", {atom.name for atom in default_structure.atoms})
        self.assertIn("OXT", {atom.name for atom in with_oxt_structure.atoms})

    def test_residue_geometry_json_residue_format(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "residues": [
                    {
                        "name": "LYS",
                        "psi": -math.pi / 4.0,
                        "chi": [-math.pi / 3.0, math.pi, math.pi / 3.0, math.pi / 2.0],
                    },
                    {
                        "name": "VAL",
                        "phi": -math.pi / 3.0,
                        "psi": -math.pi / 4.0,
                        "chi": [math.pi / 3.0],
                    },
                    {"name": "PHE", "phi": -math.pi / 3.0},
                ]
            }
        )
        structure = structure_from_residue_geometry(residue_geometry_structure)
        self.assertEqual(residue_geometry_structure.to_dict()["angle_units"], "radians")
        self.assertEqual(residue_geometry_structure.to_dict()["chi_order"], "chi1_to_chiN")
        self.assertNotIn("omega", residue_geometry_structure.to_dict()["residues"][0])
        self.assertIn("NZ", {atom.name for atom in structure.atoms})
        self.assertIn("CZ", {atom.name for atom in structure.atoms})

    def test_optional_omega_tau_theta_storage_is_supported(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "residues": [
                    {
                        "name": "ALA",
                        "omega": None,
                        "tau": 111.2,
                        "theta": 116.2,
                    }
                ]
            }
        )

        self.assertIsNone(residue_geometry_structure.residues[0].omega)
        self.assertEqual(residue_geometry_structure.residues[0].tau, 111.2)
        self.assertEqual(residue_geometry_structure.residues[0].theta, 116.2)
        payload = residue_geometry_structure.to_dict()
        self.assertIn("omega", payload["residues"][0])
        self.assertIn("tau", payload["residues"][0])
        self.assertIn("theta", payload["residues"][0])

        compact = ResidueGeometryStructure.from_dict({"residues": [{"name": "ALA"}]})
        compact_payload = compact.to_dict()
        self.assertNotIn("omega", compact_payload["residues"][0])
        self.assertNotIn("tau", compact_payload["residues"][0])
        self.assertNotIn("theta", compact_payload["residues"][0])

    def test_model_json_serialization_rounds_float_noise(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "residues": [
                    {
                        "name": "ALA",
                        "phi": -0.0,
                        "psi": 1.9313321525614318,
                    }
                ]
            }
        )
        encoded = residue_geometry_structure.to_json()
        payload = json.loads(encoded)

        self.assertNotIn("-0.0", encoded)
        self.assertEqual(payload["residues"][0]["phi"], 0.0)
        self.assertEqual(payload["residues"][0]["psi"], 1.931332152561)

    def test_metadata_angle_units_are_not_part_of_current_json_format(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "metadata": {"angle_units": "degrees"},
                "residues": [{"name": "ALA", "omega": 180.0}],
            }
        )

        self.assertEqual(residue_geometry_structure.angle_units, "radians")
        self.assertNotIn("angle_units", residue_geometry_structure.metadata)

    def test_residue_geometry_null_resseq_loads_as_position_index(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "angle_units": "radians",
                "residues": [{"name": "ALA", "resseq": None}],
            }
        )

        self.assertEqual(residue_geometry_structure.residues[0].resseq, 1)

    def test_sequence_shorthand_uses_terminal_null_geometry(self):
        residue_geometry_structure = ResidueGeometryStructure.from_dict(
            {
                "sequence": "AG",
                "phi": -math.pi / 3.0,
                "psi": -math.pi / 4.0,
                "stored_angles": "all",
            }
        )
        payload = residue_geometry_structure.to_dict()

        self.assertIsNone(payload["residues"][0]["phi"])
        self.assertIsNone(payload["residues"][1]["psi"])
        self.assertIsNone(payload["residues"][1]["omega"])
        self.assertIsNone(payload["residues"][1]["theta"])

        null_omega_geometry = ResidueGeometryStructure.from_dict(
            {
                "sequence": "AG",
                "omega": None,
                "stored_angles": "all",
            }
        ).to_dict()
        self.assertIsNone(null_omega_geometry["residues"][0]["omega"])

    def test_heavy_to_geometry_extracts_canonical_angles(self):
        structure = structure_from_residue_geometry({"sequence": "KLV"})
        residue_geometry = structure_to_residue_geometry(structure)
        self.assertEqual(residue_geometry.angle_units, "radians")
        self.assertNotIn("angle_units", residue_geometry.metadata)
        self.assertEqual(len(residue_geometry.residues), 3)
        self.assertNotIn("omega", residue_geometry.to_dict()["residues"][0])
        self.assertIsNone(residue_geometry.residues[0].phi)
        self.assertAlmostEqual(residue_geometry.residues[0].psi, -math.pi / 4.0)
        self.assertAlmostEqual(residue_geometry.residues[0].omega, math.pi)
        self.assertAlmostEqual(residue_geometry.residues[0].tau, math.radians(111.2))
        self.assertAlmostEqual(residue_geometry.residues[0].theta, math.radians(116.2))
        self.assertAlmostEqual(residue_geometry.residues[1].phi, -math.pi / 3.0)
        self.assertEqual(len(residue_geometry.residues[0].chi), 4)
        self.assertEqual(len(residue_geometry.residues[1].chi), 2)

        chi1_geometries = structure_to_residue_geometry(structure, max_chi=1)
        self.assertEqual(len(chi1_geometries.residues[0].chi), 1)
        self.assertEqual(len(chi1_geometries.residues[1].chi), 1)
        self.assertEqual(chi1_geometries.residues[0].chi, residue_geometry.residues[0].chi[:1])

        no_chi_geometries = structure_to_residue_geometry(structure, max_chi=0)
        self.assertEqual(no_chi_geometries.residues[0].chi, [])
        self.assertEqual(no_chi_geometries.residues[1].chi, [])

        with self.assertRaises(ValueError):
            structure_to_residue_geometry(structure, max_chi=-1)

        stored_geometries = structure_to_residue_geometry(structure, stored_angles="all")
        self.assertIn("omega", stored_geometries.to_dict()["residues"][0])
        self.assertIn("tau", stored_geometries.to_dict()["residues"][0])
        self.assertIn("theta", stored_geometries.to_dict()["residues"][0])

        stored_payload = stored_geometries.to_dict(max_chi=1)
        self.assertEqual(len(stored_payload["residues"][0]["chi"]), 1)

        degree_geometries = structure_to_residue_geometry(structure, angle_units="degrees")
        self.assertEqual(degree_geometries.angle_units, "degrees")
        self.assertNotIn("angle_units", degree_geometries.metadata)
        self.assertAlmostEqual(degree_geometries.residues[0].psi, -45.0)
        self.assertAlmostEqual(degree_geometries.residues[0].omega, 180.0)
        self.assertAlmostEqual(degree_geometries.residues[0].tau, 111.2)
        self.assertAlmostEqual(degree_geometries.residues[0].theta, 116.2)
        self.assertAlmostEqual(degree_geometries.residues[1].phi, -60.0)

    def test_real_pdb_chi_extraction_covers_one_and_multiple_chis(self):
        residue_geometry = structure_to_residue_geometry(load_pdb(TWO_MU7_PDB))

        self.assertEqual(residue_geometry.angle_units, "radians")
        self.assertEqual(residue_geometry.metadata["source_name"], "2mu7.pdb")
        self.assert_real_pdb_chi_coverage(residue_geometry)

    def test_real_pdb_omega_extraction_uses_conventional_trans_values(self):
        residue_geometry = structure_to_residue_geometry(
            load_pdb(TWO_MU7_PDB),
            angle_units="degrees",
            stored_angles="omega",
        )
        expected_omegas = [
            179.269141,
            176.798443,
            -175.563267,
            175.872613,
            171.804377,
            177.188456,
            175.825211,
            175.646655,
            169.097657,
            177.175943,
            177.915408,
            179.955132,
            177.729080,
            -179.571334,
            -176.905502,
            -178.135677,
            -179.660298,
            177.351352,
            -179.930081,
            None,
        ]

        self.assertEqual(residue_geometry.angle_units, "degrees")
        self.assertEqual(len(residue_geometry.residues), len(expected_omegas))
        for residue, expected in zip(residue_geometry.residues, expected_omegas):
            if expected is None:
                self.assertIsNone(residue.omega)
            else:
                self.assertAlmostEqual(residue.omega, expected, places=6)

    def test_heavy_to_geometry_handles_heterogens_and_altlocs_best_effort(self):
        residue_geometry = structure_to_residue_geometry(load_pdb(TINY_PDB))
        self.assertEqual(len(residue_geometry.residues), 4)
        names = [residue.name for residue in residue_geometry.residues]
        self.assertEqual(names, ["ALA", "GLY", "HOH", "ZN"])
        self.assertEqual(residue_geometry.residues[2].chi, [])
        warnings = "\n".join(residue_geometry.metadata["warnings"])
        self.assertIn("duplicate atom CB", warnings)
        self.assertIn("heterogen has no extractable residue geometry", warnings)

    def test_geometry_with_unsupported_residues_reconstruct_supported_residues(self):
        residue_geometry = structure_to_residue_geometry(load_pdb(TINY_PDB))
        reconstructed = structure_from_residue_geometry(residue_geometry)
        self.assertEqual({key[3] for key in reconstructed.residue_keys()}, {"ALA", "GLY"})
        self.assertIn("unsupported residue skipped", "\n".join(reconstructed.metadata["warnings"]))

    def test_disulfide_connectivity_preserves_explicit_pdb_annotations(self):
        structure = load_pdb(DISULFIDE_PDB)
        self.assertEqual(len(structure.disulfide_bonds), 1)
        self.assertEqual(
            structure.disulfide_bonds[0].to_dict(),
            {
                "chain_id_1": "A",
                "resseq_1": 1,
                "icode_1": "",
                "chain_id_2": "A",
                "resseq_2": 2,
                "icode_2": "",
                "source": "ssbond",
            },
        )

        decoded_heavy = HeavyAtomStructure.from_json(structure.to_json())
        self.assertEqual(decoded_heavy.disulfide_bonds, structure.disulfide_bonds)

        residue_geometry = structure_to_residue_geometry(structure)
        self.assertEqual(residue_geometry.disulfide_bonds, structure.disulfide_bonds)
        decoded_geometry = ResidueGeometryStructure.from_json(residue_geometry.to_json())
        self.assertEqual(decoded_geometry.disulfide_bonds, structure.disulfide_bonds)

        reconstructed = structure_from_residue_geometry(decoded_geometry)
        self.assertEqual(reconstructed.disulfide_bonds, structure.disulfide_bonds)
        reconstructed_pdb = structure_to_pdb_string(reconstructed)
        self.assertIn("SSBOND", reconstructed_pdb)
        self.assertIn("CONECT", reconstructed_pdb)
        self.assertEqual(
            structure_from_pdb_string(reconstructed_pdb).disulfide_bonds,
            structure.disulfide_bonds,
        )

    def test_disulfide_connectivity_uses_explicit_records_without_distance_inference(self):
        no_annotation = "\n".join(
            line
            for line in _read_text(DISULFIDE_PDB).splitlines()
            if not line.startswith(("SSBOND", "CONECT"))
        ) + "\n"
        no_bond_structure = structure_from_pdb_string(no_annotation)
        ssbond_structure = load_pdb(DISULFIDE_PDB)
        no_bond_geometry = structure_to_residue_geometry(no_bond_structure)
        ssbond_geometry = structure_to_residue_geometry(ssbond_structure)

        self.assertEqual(no_bond_structure.disulfide_bonds, [])
        self.assertEqual(no_bond_geometry.disulfide_bonds, [])
        self.assertEqual(
            [residue.chi for residue in no_bond_geometry.residues],
            [residue.chi for residue in ssbond_geometry.residues],
        )

        conect_only = no_annotation.replace("END\n", "CONECT    6   12\nEND\n")
        conect_structure = structure_from_pdb_string(conect_only)
        self.assertEqual(len(conect_structure.disulfide_bonds), 1)
        self.assertEqual(conect_structure.disulfide_bonds[0].source, "conect")

    def test_reconstruction_uses_optional_backbone_geometry_angles(self):
        custom = {
            "angle_units": "degrees",
            "residues": [
                {
                    "name": "ALA",
                    "psi": -45.0,
                    "omega": 170.0,
                    "tau": 100.0,
                    "theta": 120.0,
                },
                {
                    "name": "GLY",
                    "phi": -60.0,
                    "tau": 105.0,
                },
            ],
        }
        extracted = structure_to_residue_geometry(
            structure_from_residue_geometry(custom),
            angle_units="degrees",
            stored_angles="all",
        )
        self.assertAlmostEqual(extracted.residues[0].omega, 170.0)
        self.assertAlmostEqual(extracted.residues[0].tau, 100.0)
        self.assertAlmostEqual(extracted.residues[0].theta, 120.0)
        self.assertAlmostEqual(extracted.residues[1].tau, 105.0)

        fallback = structure_to_residue_geometry(
            structure_from_residue_geometry({"residues": [{"name": "ALA"}, {"name": "GLY"}]}),
            angle_units="degrees",
            stored_angles="all",
        )
        self.assertAlmostEqual(fallback.residues[0].omega, 180.0)
        self.assertAlmostEqual(fallback.residues[0].tau, 111.2)
        self.assertAlmostEqual(fallback.residues[0].theta, 116.2)

    def test_centroids_single_and_multi(self):
        structure = structure_from_residue_geometry({"sequence": "KLVFFA"})
        single = to_centroid_structure(structure, mode="single")
        multi = to_centroid_structure(structure, mode="multi")
        self.assertGreaterEqual(len(single.centroids), 1)
        self.assertGreaterEqual(len(multi.centroids), len(single.centroids))

    def test_scoring_models_are_deterministic(self):
        structure = load_pdb(TINY_PDB)
        for model in ["generic", "pheat-dfire", "pheat-goap", "pheat-rg", "heavy-mm"]:
            first = score_structure(structure, model=model)
            second = score_structure(structure, model=model)
            self.assertEqual(first.model, model)
            self.assertAlmostEqual(first.total, second.total)
            self.assertIsInstance(first.warnings, list)
            if model == "pheat-rg":
                self.assertEqual(first.metadata["atom_set"], "ca")
                self.assertFalse(first.metadata["fitted"])
                self.assertIn("expected_rg", first.terms)

        for legacy_model in ["dfire", "goap"]:
            with self.assertRaisesRegex(ValueError, "Unknown scoring model"):
                score_structure(structure, model=legacy_model)

    def test_scoring_warns_when_supported_modified_residues_leave_canonical_terms(self):
        structure = structure_from_residue_geometry({"residues": [{"name": "MSE"}]})

        result = score_structure(structure, model="generic")

        self.assertIn(
            "modified/special residues are reconstructable",
            "\n".join(result.warnings),
        )
        self.assertIn("MSE", "\n".join(result.warnings))

    def test_committed_json_outputs_match_schemas(self):
        heavy_outputs = [
            "tiny.heavy.json",
            "all_atom_tiny.heavy.json",
            "ag.heavy.json",
            "ag.degrees.heavy.json",
            "2mu7.reconstructed.heavy.json",
        ]
        residue_geometry_outputs = [
            "ag.residue-geometry.json",
            "ag.residue-geometry.degrees.json",
            "tiny.residue-geometry.json",
            "tiny.residue-geometry.degrees.json",
            "ag.roundtrip.residue-geometry.json",
            "ag.roundtrip.residue-geometry.degrees.json",
            "2mu7.residue-geometry.json",
            "2mu7.residue-geometry.degrees.json",
            "2mu7.residue-geometry.max-chi-1.json",
            "2mu7.roundtrip.residue-geometry.json",
            "2mu7.roundtrip.residue-geometry.degrees.json",
            "tiny.residue-geometry.full.json",
        ]

        for filename in heavy_outputs:
            with self.subTest(filename=filename):
                self.assert_payload_matches_schema(
                    "atom-structure",
                    json.loads(_read_text(_expected(filename))),
                )

        for filename in residue_geometry_outputs:
            path = os.path.join(FIXTURE_DIR, filename)
            if not os.path.exists(path):
                path = _expected(filename)
            with self.subTest(filename=filename):
                self.assert_payload_matches_schema(
                    "residue-geometry-structure",
                    json.loads(_read_text(path)),
                )

    def test_generated_centroid_and_energy_results_match_schemas(self):
        structure = load_pdb(TINY_PDB)
        centroid_payload = to_centroid_structure(structure, mode="single").to_dict()
        energy_payload = score_structure(structure, model="generic").to_dict()
        radius_payload = {
            "format": "pheat.radius-of-gyration-result",
            "version": 1,
            "input": TINY_PDB,
            "radius_of_gyration": structure_radius_of_gyration(structure),
        }

        self.assert_payload_matches_schema("centroid-structure", centroid_payload)
        self.assert_payload_matches_schema("energy-result", energy_payload)
        self.assert_payload_matches_schema("radius-of-gyration-result", radius_payload)

    def test_combinatorial_roundtrip_reports_original_scores_once(self):
        example_roundtrip_dir = self.generated_example_roundtrip_dir()
        summary = json.loads(_read_text(os.path.join(example_roundtrip_dir, "summary.json")))
        self.assertEqual(
            set(summary["original_scores"]),
            {"generic", "pheat-dfire", "pheat-goap", "heavy-mm", "openmm-prepared"},
        )
        openmm_original = summary["original_scores"]["openmm-prepared"]
        if _openmm_available():
            self.assertEqual(openmm_original["status"], "ok")
            self.assertEqual(openmm_original["units"], "kJ/mol")
            self.assertIsNotNone(openmm_original["total"])
        else:
            self.assertEqual(openmm_original["status"], "unavailable")
            self.assertIsNone(openmm_original["total"])
            self.assertIn("openmm-prepared scoring requires", openmm_original["error"])
        self.assertEqual(summary["original_radius_of_gyration"]["units"], "angstrom")
        self.assertEqual(summary["original_radius_of_gyration"]["atom_count"], 150)
        self.assertIn("unweighted", summary["original_radius_of_gyration"]["values"])
        self.assertIn("mass_weighted", summary["original_radius_of_gyration"]["values"])
        for case in summary["cases"]:
            for model in summary["original_scores"]:
                self.assertNotIn("original", case["scores"][model])
                self.assertIn("reconstructed", case["scores"][model])
                self.assertIn("delta", case["scores"][model])
            self.assertIn("radius_of_gyration", case)
            self.assertIn("c_alpha", case["rmsd"])
            self.assertIn("matched_c_alpha_atoms", case["rmsd"])
            self.assertEqual(case["rmsd"]["alignment_atom_set"], "all-heavy")
            self.assertIn("reconstructed", case["radius_of_gyration"])
            self.assertIn("delta", case["radius_of_gyration"])
            self.assertIn("unweighted", case["radius_of_gyration"]["reconstructed"]["values"])
            self.assertIn("mass_weighted", case["radius_of_gyration"]["reconstructed"]["values"])

        with open(os.path.join(example_roundtrip_dir, "summary.csv"), newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        header = list(rows[0])
        self.assertNotIn("generic_original_total", header)
        self.assertIn("generic_reconstructed_total", header)
        self.assertIn("pheat-dfire_reconstructed_total", header)
        self.assertIn("pheat-goap_reconstructed_total", header)
        self.assertNotIn("dfire_reconstructed_total", header)
        self.assertNotIn("goap_reconstructed_total", header)
        self.assertIn("openmm-prepared_reconstructed_total", header)
        self.assertIn("rg_unweighted_reconstructed", header)
        self.assertIn("rg_unweighted_delta", header)
        self.assertIn("rg_mass_weighted_reconstructed", header)
        self.assertIn("rg_mass_weighted_delta", header)
        self.assertIn("geometry", header)
        self.assertIn("geometry_mode", header)
        self.assertIn("geometry_table", header)
        self.assertIn("geometry_profile", header)
        self.assertIn("c_alpha_rmsd", header)
        self.assertIn("matched_c_alpha_atoms", header)
        self.assertEqual(summary["case_count"], 48)
        self.assertEqual(
            [variant["case_id"] for variant in summary["reconstruction_geometries"]],
            ["fixed", "ccd-sidechains"],
        )
        self.assertEqual({row["geometry"] for row in rows}, {"fixed", "CCD side-chain geometry"})
        fixed_rows = [row for row in rows if row["geometry"] == "fixed"]
        ccd_rows = [row for row in rows if row["geometry"] == "CCD side-chain geometry"]
        self.assertEqual(len(fixed_rows), 24)
        self.assertEqual(len(ccd_rows), 24)
        self.assertTrue(all(row["geometry_mode"] == "fixed" for row in fixed_rows))
        self.assertTrue(all(row["geometry_mode"] == "table" for row in ccd_rows))
        self.assertTrue(
            all(row["geometry_table"] == "ccd-sidechain-geometry-v1" for row in ccd_rows)
        )
        ccd_case = next(case for case in summary["cases"] if case["geometry"]["case_id"] == "ccd-sidechains")
        self.assertEqual(ccd_case["reconstruction_geometry"]["mode"], "table")
        self.assertEqual(ccd_case["reconstruction_geometry"]["table"], "ccd-sidechain-geometry-v1")
        if _openmm_available():
            self.assertEqual(rows[0]["openmm-prepared_status"], "ok")
            self.assertNotEqual(rows[0]["openmm-prepared_reconstructed_total"], "")
        else:
            self.assertEqual(rows[0]["openmm-prepared_status"], "unavailable")
            self.assertEqual(rows[0]["openmm-prepared_reconstructed_total"], "")

        report = _read_text(os.path.join(example_roundtrip_dir, "report.html"))
        self.assertIn("Original All-Heavy Scores", report)
        self.assertIn("Original All-Heavy Radius of Gyration", report)
        self.assertIn("Rg unweighted", report)
        self.assertIn("Rg mass-weighted", report)
        self.assertIn("generic reconstructed", report)
        self.assertIn("pheat-dfire reconstructed", report)
        self.assertIn("pheat-goap reconstructed", report)
        self.assertNotIn(">dfire reconstructed<", report)
        self.assertNotIn(">goap reconstructed<", report)
        self.assertIn(">Geometry<", report)
        self.assertIn("CCD side-chain geometry", report)
        self.assertIn("OpenMM reconstructed", report)
        self.assertNotIn("generic delta", report)

    def test_combinatorial_roundtrip_report_uses_self_hosted_molstar_viewer(self):
        example_roundtrip_dir = self.generated_example_roundtrip_dir()
        report_path = os.path.join(example_roundtrip_dir, "report.html")
        report = _read_text(report_path)

        self.assertIn('href="../../vendor/molstar/molstar.css"', report)
        self.assertIn('src="../../vendor/molstar/molstar.js"', report)
        self.assertIn("<title>2MU7 Combinatorial Roundtrip</title>", report)
        self.assertIn('<h2 id="viewer-title">Interactive Mol* Alignment Viewer</h2>', report)
        self.assertIn('href="https://molstar.org/"', report)
        self.assertIn('href="https://doi.org/10.1093/nar/gkab314"', report)
        self.assertIn("Sehnal et al.", report)
        self.assertIn("Nucleic Acids Research", report)
        self.assertIn("PHEAT: Protein Heavy-atom Energy and Analysis Toolkit", report)
        self.assertIn("Blankenberg Lab and contributors", report)
        self.assertIn(f"Version {__version__}", report)
        self.assertIn('href="https://github.com/BlankenbergLab/pheat"', report)
        self.assertEqual(report.count('class="score-table sortable-table"'), 2)
        self.assertIn('class="case-table sortable-table"', report)
        self.assertIn(".sortable-table thead th", report)
        self.assertIn("position: sticky", report)
        self.assertIn("top: 0", report)
        self.assertIn("aria-sort", report)
        self.assertIn("Sort table by this column", report)
        self.assertIn("initializeSortableTables", report)
        self.assertIn("sortTableByColumn", report)
        self.assertIn("compareSortableRows", report)
        self.assertIn("Number.parseFloat", report)
        self.assertIn('id="molstar-viewer"', report)
        self.assertIn('id="case-select"', report)
        self.assertIn('id="roundtrip-viewer-data"', report)
        self.assertIn("viewportBackgroundColor: '#ffffff'", report)
        self.assertNotIn("viewportBackgroundColor: 0xffffff", report)
        self.assertIn("data-view-case=\"fixed__angles-none__chi-all\"", report)
        self.assertIn('data-structure-toggle="original" aria-pressed="true"', report)
        self.assertIn('data-structure-toggle="reconstructed" aria-pressed="true"', report)
        self.assertIn("legend-swatch legend-swatch-original", report)
        self.assertIn("legend-swatch legend-swatch-reconstructed", report)
        self.assertIn('class="display-mode-group" role="group" aria-label="Display mode"', report)
        self.assertIn('data-display-mode="ribbon" aria-pressed="true"', report)
        self.assertIn('data-display-mode="all-atom" aria-pressed="false"', report)
        self.assertIn("legend-toggle display-mode-toggle", report)
        self.assertIn('.legend-toggle[data-structure-toggle][aria-pressed="false"]', report)
        self.assertIn('.display-mode-toggle[aria-pressed="true"]', report)
        self.assertNotIn('.legend-toggle[aria-pressed="false"]', report)
        self.assertIn("data-recolor", report)
        self.assertIn("viewerState.visible.original", report)
        self.assertIn("viewerState.visible.reconstructed", report)
        self.assertIn("viewerState.displayMode", report)
        self.assertIn("loadPdbStructure(caseData.original_pdb, 'original'", report)
        self.assertIn("loadPdbStructure(caseData.reconstructed_pdb, 'reconstructed'", report)
        self.assertIn("recolorCurrentRepresentations", report)
        self.assertIn("recolorButton.addEventListener('click', () => recolorCurrentRepresentations())", report)
        self.assertIn("preset-structure-representation-polymer-cartoon", report)
        self.assertIn("preset-structure-representation-atomic-detail", report)
        self.assertIn("viewportFocusBehavior: 'disabled'", report)
        self.assertIn("layoutShowControls: true", report)
        self.assertIn("layoutShowLeftPanel: true", report)
        self.assertIn("viewportShowControls: true", report)
        self.assertIn("viewportShowSettings: true", report)
        self.assertIn("loadCase(caseSelect.value, { resetCamera: false })", report)
        self.assertIn("Both structures hidden.", report)
        self.assertIn("Loaded selected structures from embedded PDB data", report)
        self.assertIn("0x0072B2", report)
        self.assertIn("0xD55E00", report)
        self.assertIn("globalName: 'uniform'", report)
        self.assertIn("globalColorParams", report)
        self.assertIn('id="viewer-rg-unweighted"', report)
        self.assertIn('id="viewer-rg-mass-weighted"', report)
        self.assertIn('id="viewer-geometry"', report)
        self.assertIn("caseData.geometry_label", report)
        self.assertIn("formatRg(caseData.rg_unweighted_reconstructed", report)
        self.assertNotIn("data-sidechain-toggle", report)
        self.assertNotIn("viewerState.sidechains", report)
        self.assertNotIn("original_sidechain_pdb", report)
        self.assertNotIn("reconstructed_sidechain_pdb", report)
        self.assertNotIn("original_backbone_pdb", report)
        self.assertNotIn("reconstructed_backbone_pdb", report)
        self.assertNotIn("loadStructureParts", report)
        self.assertNotIn('<img src="visualizations/', report)
        self.assertNotIn("https://unpkg", report.lower())
        self.assertNotIn("http://unpkg", report.lower())
        self.assertNotIn("unpkg", report.lower())
        self.assertNotIn("cdn", report.lower())
        self.assertNotIn("datatables", report.lower())
        self.assertNotIn("tablesort", report.lower())

        start_tag = '<script type="application/json" id="roundtrip-viewer-data">'
        start = report.index(start_tag) + len(start_tag)
        end = report.index("</script>", start)
        viewer_cases = json.loads(report[start:end])
        self.assertEqual(len(viewer_cases), 48)
        self.assertEqual(viewer_cases[0]["case_id"], "fixed__angles-none__chi-all")
        self.assertEqual(viewer_cases[0]["geometry_label"], "fixed")
        self.assertTrue(
            any(case["geometry_label"] == "CCD side-chain geometry" for case in viewer_cases)
        )
        self.assertIn("rg_unweighted_reconstructed", viewer_cases[0])
        self.assertIn("rg_mass_weighted_reconstructed", viewer_cases[0])
        self.assertIn("rg_unweighted_delta", viewer_cases[0])
        self.assertIn("rg_mass_weighted_delta", viewer_cases[0])
        self.assertIn("ATOM", viewer_cases[0]["original_pdb"])
        self.assertIn("ATOM", viewer_cases[0]["reconstructed_pdb"])
        self.assertIn(" CB ", viewer_cases[0]["original_pdb"])
        self.assertIn(" CA ", viewer_cases[0]["original_pdb"])
        self.assertIn(" CB ", viewer_cases[0]["reconstructed_pdb"])
        self.assertIn(" CA ", viewer_cases[0]["reconstructed_pdb"])
        self.assertNotIn("original_backbone_pdb", viewer_cases[0])
        self.assertNotIn("original_sidechain_pdb", viewer_cases[0])
        self.assertNotIn("reconstructed_backbone_pdb", viewer_cases[0])
        self.assertNotIn("reconstructed_sidechain_pdb", viewer_cases[0])
        self.assertEqual(
            viewer_cases[0]["original_pdb_path"],
            "pdb/fixed__angles-none__chi-all.original_aligned.pdb",
        )

        summary = json.loads(_read_text(os.path.join(example_roundtrip_dir, "summary.json")))
        self.assertNotIn("visualization_png", summary["cases"][0]["paths"])
        self.assertFalse(os.path.exists(os.path.join(example_roundtrip_dir, "visualizations")))

        vendor_dir = self._example_vendor_dir
        self.assertTrue(os.path.exists(os.path.join(vendor_dir, "molstar.js")))
        self.assertTrue(os.path.exists(os.path.join(vendor_dir, "molstar.css")))
        self.assertTrue(os.path.exists(os.path.join(vendor_dir, "LICENSE")))
        self.assertNotIn("sourceMappingURL", _read_text(os.path.join(vendor_dir, "molstar.js")))
        self.assertNotIn("sourceMappingURL", _read_text(os.path.join(vendor_dir, "molstar.css")))

    def test_combinatorial_roundtrip_parses_packaged_geometry_variant_ids(self):
        module = _load_combinatorial_roundtrip_module()
        args = module._parse_args(["--geometry-variants", "fixed,ccd-sidechain-geometry-v1"])
        variants = module._geometry_variants_from_args(args)

        self.assertEqual([variant["case_id"] for variant in variants], ["fixed", "ccd-sidechains"])
        self.assertEqual(variants[0]["mode"], "fixed")
        self.assertIsNone(variants[0]["table"])
        self.assertEqual(variants[1]["mode"], "table")
        self.assertEqual(variants[1]["table"], "ccd-sidechain-geometry-v1")

    def test_molstar_source_map_stripper_removes_only_trailing_comments(self):
        spec = importlib.util.spec_from_file_location(
            "strip_molstar_source_maps",
            MOLSTAR_SOURCE_MAP_SCRIPT,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            vendor_dir = os.path.join(tmpdir, "molstar")
            os.makedirs(vendor_dir)
            js_path = os.path.join(vendor_dir, "molstar.js")
            css_path = os.path.join(vendor_dir, "molstar.css")
            with open(js_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "const label = 'sourceMappingURL text is data';\n"
                    "window.molstar = {};\n"
                    "//# sourceMappingURL=molstar.js.map\n"
                )
            with open(css_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "/* sourceMappingURL text is data */\n"
                    ".msp-layout { color: black; }\n"
                    "/*# sourceMappingURL=molstar.css.map */\n"
                )

            changed = module.strip_molstar_source_maps(Path(vendor_dir))

            self.assertEqual([str(path) for path in changed], [js_path, css_path])
            self.assertIn("sourceMappingURL text is data", _read_text(js_path))
            self.assertIn("sourceMappingURL text is data", _read_text(css_path))
            self.assertNotIn("//# sourceMappingURL=molstar.js.map", _read_text(js_path))
            self.assertNotIn("/*# sourceMappingURL=molstar.css.map */", _read_text(css_path))
            self.assertTrue(_read_text(js_path).endswith("\n"))

    def test_notebook_uses_ipymolstar_molviewspec_with_local_pdb_data(self):
        self.assertTrue(os.path.exists(MOLSTAR_NOTEBOOK))
        self.assertFalse(os.path.exists(OLD_NGLVIEW_NOTEBOOK))
        executed_paths = [
            os.path.relpath(EXECUTED_MOLSTAR_NOTEBOOK, REPO_ROOT),
            os.path.relpath(OLD_EXECUTED_NGLVIEW_NOTEBOOK, REPO_ROOT),
        ]
        tracked_executed = subprocess.run(
            ["git", "ls-files", "--", *executed_paths],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(tracked_executed.returncode, 0, tracked_executed.stderr)
        self.assertEqual("", tracked_executed.stdout.strip())

        notebook = json.loads(_read_text(MOLSTAR_NOTEBOOK))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("from ipymolstar import MolViewSpec", source)
        self.assertIn("from molviewspec import create_builder", source)
        self.assertIn("structure_radius_of_gyration", source)
        self.assertIn("radius_of_gyration_delta", source)
        self.assertIn("Radius Of Gyration", source)
        self.assertIn("MolViewSpec(msvj_data=msvj_data", source)
        self.assertIn("data:chemical/x-pdb;base64", source)
        self.assertIn('parse(format="pdb")', source)
        self.assertIn('chain_id="A"', source)
        self.assertIn('chain_id="B"', source)
        self.assertIn('selector={"auth_asym_id": "A"}', source)
        self.assertIn('selector={"auth_asym_id": "B"}', source)
        self.assertNotIn('color="royalblue"', source)
        self.assertNotIn('color="tomato"', source)
        self.assertNotIn("shown in blue", source)
        self.assertNotIn("shown in red", source)
        self.assertNotIn(".color(", source)
        self.assertNotIn("PDBeMolstar", source)
        self.assertNotIn("custom_data=", source)
        self.assertNotIn("molecule_id=", source)
        self.assertNotIn("nglview", source.lower())

    def test_makefile_defines_generated_example_artifact_targets(self):
        makefile = _read_text(os.path.join(REPO_ROOT, "Makefile"))
        self.assertIn("PYTHON ?= python", makefile)
        self.assertIn("JUPYTER ?= jupyter", makefile)
        self.assertIn("molstar:", makefile)
        self.assertIn("examples-roundtrip:", makefile)
        self.assertIn("examples-notebook-executed:", makefile)
        self.assertIn("examples:", makefile)
        self.assertIn("clean-examples:", makefile)
        self.assertIn("$(PHEAT) molstar install", makefile)
        self.assertIn("--version $(MOLSTAR_VERSION)", makefile)
        self.assertIn("--timeout $(MOLSTAR_TIMEOUT)", makefile)
        self.assertNotIn("MOLSTAR_VENDOR_DIR", makefile)
        self.assertIn("WEB_HOST ?= 127.0.0.1", makefile)
        self.assertIn("WEB_PORT ?= 8000", makefile)
        self.assertIn("ROUNDTRIP_GEOMETRIES ?= fixed,ccd-sidechain-geometry-v1", makefile)
        self.assertIn("--host $(WEB_HOST)", makefile)
        self.assertIn("--port $(WEB_PORT)", makefile)
        self.assertNotIn("scripts/strip_molstar_source_maps.py", makefile)
        self.assertNotIn("CONDA_PREFIX", makefile)
        self.assertNotIn("MAMBA", makefile)
        self.assertNotIn("npm pack", makefile)
        self.assertNotIn("./.conda", makefile)
        self.assertNotIn("free-energy-calc", makefile)
        self.assertNotIn("perl", makefile.lower())

        dry_run = subprocess.run(
            ["make", "-n", "examples"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("pheat molstar install --version 5.9.0 --timeout 60", dry_run.stdout)
        self.assertIn("python examples/2mu7_combinatorial_roundtrip.py", dry_run.stdout)
        self.assertIn("2mu7_combinatorial_roundtrip.py", dry_run.stdout)
        self.assertIn("--geometry-variants fixed,ccd-sidechain-geometry-v1", dry_run.stdout)
        self.assertNotIn("--molstar-vendor-dir", dry_run.stdout)
        self.assertIn("jupyter nbconvert", dry_run.stdout)
        self.assertIn("nbconvert", dry_run.stdout)
        self.assertNotIn("./.conda", dry_run.stdout)
        self.assertNotIn("free-energy-calc", dry_run.stdout)

        web_dry_run = subprocess.run(
            ["make", "-n", "web"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(web_dry_run.returncode, 0, web_dry_run.stderr)
        self.assertIn("--host 127.0.0.1", web_dry_run.stdout)
        self.assertIn("--port 8000", web_dry_run.stdout)
        self.assertIn("--random-port-if-taken", web_dry_run.stdout)

    def test_notebook_dependencies_use_ipymolstar_not_nglview(self):
        pyproject = _read_text(os.path.join(REPO_ROOT, "pyproject.toml")).lower()
        environment = _read_text(os.path.join(REPO_ROOT, "environment.yml")).lower()
        readme = _read_text(os.path.join(REPO_ROOT, "README.md")).lower()
        examples_readme = _read_text(os.path.join(REPO_ROOT, "examples", "README.md")).lower()

        self.assertIn("ipymolstar==0.1.0", pyproject)
        self.assertIn("molviewspec==1.8.1", pyproject)
        self.assertIn('[project.urls]\nrepository = "https://github.com/blankenberglab/pheat"', pyproject)
        self.assertIn("-e .[all]", environment)
        self.assertIn("nodejs", environment)
        for docs_text in (readme, examples_readme):
            self.assertIn("already active", docs_text)
            self.assertNotIn("./.conda/free-energy-calc", docs_text)
            self.assertNotIn("miniforge3/condabin/mamba", docs_text)
            self.assertNotIn("make env", docs_text)
        self.assertNotIn("nglview", pyproject)
        self.assertNotIn("nglview", environment)
        self.assertNotIn("setuptools<81", pyproject)
        self.assertNotIn("setuptools<81", environment)

    def test_readme_documents_scoring_model_semantics(self):
        readme = _read_text(os.path.join(REPO_ROOT, "README.md"))

        self.assertIn("### Scoring Models", readme)
        for model in [
            "`generic`",
            "`pheat-dfire`",
            "`pheat-goap`",
            "`pheat-rg`",
            "`heavy-mm`",
            "`openmm-prepared`",
        ]:
            self.assertIn(model, readme)
        self.assertIn("Compare original vs reconstructed", readme)
        self.assertIn("scores within the same model", readme)
        self.assertIn("do not compare absolute totals across different", readme)
        self.assertIn("arbitrary", readme)
        self.assertIn("kJ/mol", readme)
        self.assertIn("Does not use the original DFIRE parameter table", readme)
        self.assertIn("Does not use the original GOAP parameter tables", readme)
        self.assertIn("hydrophobicity, element-contact, coarse distance-bin", readme)
        self.assertIn("adds a local orientation-vector term", readme)
        self.assertIn("expected-radius-of-gyration", readme.lower())
        self.assertIn("Heavy-atoms-only approximation", readme)
        self.assertIn("may add hydrogens and missing terminal/heavy atoms internally", readme)

    def test_combinatorial_roundtrip_documents_coupled_backbone_geometry(self):
        example_roundtrip_dir = self.generated_example_roundtrip_dir()
        with open(os.path.join(example_roundtrip_dir, "summary.csv"), newline="", encoding="utf-8") as handle:
            rows = {row["case_id"]: row for row in csv.DictReader(handle)}

        no_optional = float(rows["fixed__angles-none__chi-all"]["backbone_rmsd"])
        omega_only = float(rows["fixed__angles-omega__chi-all"]["backbone_rmsd"])
        omega_tau = float(rows["fixed__angles-omega_tau__chi-all"]["backbone_rmsd"])
        all_geometry = float(rows["fixed__angles-all__chi-all"]["backbone_rmsd"])
        self.assertGreater(omega_only, no_optional)
        self.assertLess(omega_tau, no_optional)
        self.assertLess(all_geometry, no_optional)

        residue_geometry = load_residue_geometry(
            os.path.join(
                example_roundtrip_dir,
                "residue_geometry",
                "fixed__angles-omega__chi-all.residue-geometry.json",
            )
        )
        extracted = structure_to_residue_geometry(
            structure_from_residue_geometry(residue_geometry),
            angle_units="degrees",
            stored_angles="all",
        )
        for stored, rebuilt in zip(residue_geometry.residues, extracted.residues):
            self.assertAlmostEqual(rebuilt.tau, 111.2, places=6)
            if stored.omega is None:
                self.assertIsNone(rebuilt.omega)
                self.assertIsNone(rebuilt.theta)
            else:
                self.assertAlmostEqual(rebuilt.omega, math.degrees(stored.omega), places=6)
                self.assertAlmostEqual(rebuilt.theta, 116.2, places=6)

        report = _read_text(os.path.join(example_roundtrip_dir, "report.html"))
        self.assertNotIn("Geometry Interpretation", report)
        self.assertNotIn("omega-only backbone RMSD", report)

    def test_residue_geometry_schema_requires_canonical_units_and_chi_order(self):
        schema = load_schema("residue-geometry-structure")
        validator = Draft202012Validator(schema)
        payload = json.loads(_read_text(_expected("tiny.residue-geometry.json")))
        validator.validate(payload)

        legacy_units_payload = dict(payload)
        legacy_units_payload.pop("angle_units")
        legacy_units_payload["metadata"] = {
            **legacy_units_payload["metadata"],
            "angle_units": "radians",
        }
        with self.assertRaises(ValidationError):
            validator.validate(legacy_units_payload)

        bad_chi_order_payload = {**payload, "chi_order": "unordered"}
        with self.assertRaises(ValidationError):
            validator.validate(bad_chi_order_payload)

        optional_payload = json.loads(_read_text(_expected("tiny.residue-geometry.full.json")))
        validator.validate(optional_payload)
        optional_payload["residues"][0]["omega"] = None
        optional_payload["residues"][0]["tau"] = None
        optional_payload["residues"][0]["theta"] = None
        validator.validate(optional_payload)

    def test_canonical_json_loaders_reject_wrong_format_or_version(self):
        heavy_payload = json.loads(_read_text(_expected("tiny.heavy.json")))
        for field, value in [("format", "pheat.old-heavy"), ("version", 2)]:
            invalid = {**heavy_payload, field: value}
            with self.subTest(structure="heavy", field=field):
                with self.assertRaises(ValueError):
                    HeavyAtomStructure.from_json(json.dumps(invalid))

        residue_geometry_payload = json.loads(_read_text(_expected("tiny.residue-geometry.json")))
        for field, value in [("format", "pheat.old-residue-geometry"), ("version", 2)]:
            invalid = {**residue_geometry_payload, field: value}
            with self.subTest(structure="residue_geometry", field=field):
                with self.assertRaises(ValueError):
                    ResidueGeometryStructure.from_json(json.dumps(invalid))

        missing_angle_units = dict(residue_geometry_payload)
        missing_angle_units.pop("angle_units")
        with self.assertRaises(ValueError):
            ResidueGeometryStructure.from_json(json.dumps(missing_angle_units))

    def test_openmm_prepared_scoring_when_available(self):
        try:
            import openmm  # noqa: F401
            import pdbfixer  # noqa: F401
        except Exception:
            self.skipTest("OpenMM/PDBFixer is not installed")
        structure = load_pdb(TWO_MU7_PDB)
        result = score_structure(structure, model="openmm-prepared")
        self.assertEqual(result.model, "openmm-prepared")
        self.assertEqual(result.units, "kJ/mol")
        self.assertTrue(math.isfinite(result.total))
        self.assertEqual(result.metadata["preparation"], "pdbfixer")
        self.assertEqual(result.metadata["preparation_seed"], 20260514)
        self.assertEqual(result.metadata["missing_terminal_atoms_added"], 1)
        self.assertIn("hydrogens were added internally", "\n".join(result.warnings))
        self.assertIn("PDBFixer added missing heavy or terminal atoms", "\n".join(result.warnings))
        second = score_structure(structure, model="openmm-prepared")
        self.assertAlmostEqual(result.total, second.total)

    def test_manifests_include_required_sets(self):
        self.assertIn("arxiv-2507-08955", list_example_sets())
        self.assertIn("diversity", list_example_sets())
        manifest = load_example_manifest("arxiv-2507-08955")
        ids = {item["id"] for item in manifest["examples"]}
        self.assertIn("2jof_trp_cage", ids)
        self.assertIn("klvffa_quantum_sequence", ids)
        sources = list_sources()
        self.assertTrue(sources)

        ccd_source = next(source for source in sources if source["id"] == "wwpdb-ccd")
        self.assertFalse(ccd_source["downloadable"])
        self.assertIn("CC0 1.0", ccd_source["license"])
        self.assertIn("creativecommons.org/publicdomain/zero/1.0", ccd_source["license_url"])
        self.assertIn("no ccd cif files are vendored", ccd_source["license_note"].lower())
        self.assertIn("reference data", ccd_source["license_note"])
        self.assertIn("Westbrook", ccd_source["citation"])
        self.assertEqual(
            {
                "https://files.rcsb.org/ligands/download/SEC.cif",
                "https://files.rcsb.org/ligands/download/PYL.cif",
                "https://files.rcsb.org/ligands/download/MSE.cif",
                "https://files.rcsb.org/ligands/download/HYP.cif",
                "https://files.rcsb.org/ligands/download/LYZ.cif",
                "https://files.rcsb.org/ligands/download/SEP.cif",
                "https://files.rcsb.org/ligands/download/TPO.cif",
                "https://files.rcsb.org/ligands/download/PTR.cif",
                "https://files.rcsb.org/ligands/download/PCA.cif",
            },
            {url for url in ccd_source["urls"] if url.endswith(".cif")},
        )
        ccd_full_source = next(source for source in sources if source["id"] == "wwpdb-ccd-full")
        self.assertTrue(ccd_full_source["downloadable"])
        self.assertEqual(ccd_full_source["filename"], "components.cif.gz")
        self.assertIn("components.cif.gz", ccd_full_source["urls"][0])
        ccd_bcif_source = next(source for source in sources if source["id"] == "rcsb-ccd-bcif")
        self.assertTrue(ccd_bcif_source["downloadable"])
        self.assertEqual(
            {"cca.bcif", "ccb.bcif"},
            {file_payload["filename"] for file_payload in ccd_bcif_source["files"]},
        )
        cdl_source = next(source for source in sources if source["id"] == "cdl-phenix-v12")
        self.assertFalse(cdl_source["downloadable"])
        self.assertIn("does not vendor", cdl_source["license_note"].lower())

        ciaaw_source = next(source for source in sources if source["id"] == "ciaaw-standard-atomic-weights-2024")
        nist_source = next(source for source in sources if source["id"] == "nist-atomic-weights")
        self.assertFalse(ciaaw_source["downloadable"])
        self.assertFalse(nist_source["downloadable"])
        self.assertIn("copyrighted", ciaaw_source["license_note"].lower())
        self.assertIn("does not download", ciaaw_source["license_note"].lower())
        self.assertIn("no NIST atomic-weight data files".lower(), nist_source["license_note"].lower())

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "reference-only"):
                fetch_source("ciaaw-standard-atomic-weights-2024", tmpdir)
            verification = verify_sources(tmpdir)
        self.assertEqual(
            verification["ciaaw-standard-atomic-weights-2024"],
            {"status": "reference-only", "downloadable": False},
        )
        self.assertEqual(
            verification["nist-atomic-weights"],
            {"status": "reference-only", "downloadable": False},
        )

    def test_example_run_handles_sequence_geometry_and_missing_pdbs(self):
        payload = run_example_set("arxiv-2507-08955", model="generic", fetch=False)
        statuses = {item["id"]: item["status"] for item in payload["results"]}
        self.assertEqual(statuses["klvffa_quantum_sequence"], "ok")
        self.assertEqual(statuses["jun_qin_integrin_15aa"], "skipped")
        self.assertEqual(statuses["2jof_trp_cage"], "missing")

    def test_cli_uses_backend_for_conversions_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heavy_path = os.path.join(tmpdir, "tiny.json")
            pdb_path = os.path.join(tmpdir, "roundtrip.pdb")
            score_path = os.path.join(tmpdir, "score.json")
            pdb_residue_geometry_path = os.path.join(tmpdir, "tiny-residue-geometry.json")
            pdb_residue_geometry_degrees_path = os.path.join(tmpdir, "tiny-residue-geometry-degrees.json")
            pdb_residue_geometry_full_path = os.path.join(tmpdir, "tiny-residue-geometry-full.json")
            heavy_residue_geometry_path = os.path.join(tmpdir, "heavy-residue-geometry.json")
            heavy_residue_geometry_full_path = os.path.join(tmpdir, "heavy-residue-geometry-full.json")
            pheat_rg_score_path = os.path.join(tmpdir, "pheat-rg-score.json")
            all_atom_heavy_path = os.path.join(tmpdir, "all-atom.json")
            radius_of_gyration_path = os.path.join(tmpdir, "tiny-radius-of-gyration.json")
            radius_of_gyration_unweighted_path = os.path.join(
                tmpdir,
                "tiny-radius-of-gyration-unweighted.json",
            )
            radius_of_gyration_ca_path = os.path.join(tmpdir, "tiny-radius-of-gyration-ca.json")
            radius_of_gyration_backbone_path = os.path.join(
                tmpdir,
                "tiny-radius-of-gyration-backbone.json",
            )
            rmsd_path = os.path.join(tmpdir, "tiny-rmsd.json")
            rmsd_ca_path = os.path.join(tmpdir, "tiny-rmsd-ca.json")
            reconstructed_heavy_path = os.path.join(tmpdir, "reconstructed.json")
            reconstructed_degrees_heavy_path = os.path.join(tmpdir, "reconstructed-degrees.json")
            reconstructed_oxt_heavy_path = os.path.join(tmpdir, "reconstructed-oxt.json")
            reconstructed_pdb_path = os.path.join(tmpdir, "reconstructed.pdb")
            reconstructed_residue_geometry_path = os.path.join(tmpdir, "reconstructed-residue-geometry.json")
            reconstructed_residue_geometry_degrees_path = os.path.join(
                tmpdir,
                "reconstructed-residue-geometry-degrees.json",
            )
            self.assertEqual(cli_main(["pdb-to-structure", TINY_PDB, "-o", heavy_path]), 0)
            self.assert_file_matches_expected(heavy_path, "tiny.heavy.json")

            self.assertEqual(cli_main(["structure-to-pdb", heavy_path, "-o", pdb_path]), 0)
            self.assert_file_matches_expected(pdb_path, "tiny.roundtrip.pdb")

            self.assertEqual(cli_main(["pdb-to-geometry", TINY_PDB, "-o", pdb_residue_geometry_path]), 0)
            self.assert_file_matches_expected(pdb_residue_geometry_path, "tiny.residue-geometry.json")

            self.assertEqual(
                cli_main(
                    [
                        "pdb-to-geometry",
                        TINY_PDB,
                        "-o",
                        pdb_residue_geometry_full_path,
                        "--store-angles",
                        "all",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(pdb_residue_geometry_full_path, "tiny.residue-geometry.full.json")

            self.assertEqual(
                cli_main(
                    [
                        "pdb-to-geometry",
                        TINY_PDB,
                        "-o",
                        pdb_residue_geometry_degrees_path,
                        "--angle-units",
                        "degrees",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(
                pdb_residue_geometry_degrees_path,
                "tiny.residue-geometry.degrees.json",
            )

            self.assertEqual(cli_main(["structure-to-geometry", heavy_path, "-o", heavy_residue_geometry_path]), 0)
            self.assert_file_matches_expected(heavy_residue_geometry_path, "tiny.residue-geometry.json")

            self.assertEqual(
                cli_main(
                    [
                        "structure-to-geometry",
                        heavy_path,
                        "-o",
                        heavy_residue_geometry_full_path,
                        "--store-angles",
                        "omega,tau,theta",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(heavy_residue_geometry_full_path, "tiny.residue-geometry.full.json")

            self.assertEqual(cli_main(["pdb-to-structure", ALL_ATOM_TINY_PDB, "-o", all_atom_heavy_path]), 0)
            self.assert_file_matches_expected(all_atom_heavy_path, "all_atom_tiny.heavy.json")

            self.assertEqual(
                cli_main(
                    [
                        "geometry-to-structure",
                        AG_GEOMETRY,
                        "-o",
                        reconstructed_heavy_path,
                        "--pdb-output",
                        reconstructed_pdb_path,
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(reconstructed_heavy_path, "ag.heavy.json")
            self.assert_file_matches_expected(reconstructed_pdb_path, "ag.pdb")
            self.assertNotIn(
                "OXT",
                {atom.name for atom in HeavyAtomStructure.from_json(_read_text(reconstructed_heavy_path)).atoms},
            )

            self.assertEqual(
                cli_main(
                    [
                        "geometry-to-structure",
                        AG_GEOMETRY,
                        "-o",
                        reconstructed_oxt_heavy_path,
                        "--include-terminal-oxt",
                    ]
                ),
                0,
            )
            self.assertIn(
                "OXT",
                {atom.name for atom in HeavyAtomStructure.from_json(_read_text(reconstructed_oxt_heavy_path)).atoms},
            )

            self.assertEqual(
                cli_main(
                    [
                        "geometry-to-structure",
                        AG_GEOMETRY_DEGREES,
                        "-o",
                        reconstructed_degrees_heavy_path,
                        "--angle-units",
                        "degrees",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(reconstructed_degrees_heavy_path, "ag.degrees.heavy.json")

            self.assertEqual(
                cli_main(["structure-to-geometry", reconstructed_heavy_path, "-o", reconstructed_residue_geometry_path]),
                0,
            )
            self.assert_file_matches_expected(
                reconstructed_residue_geometry_path,
                "ag.roundtrip.residue-geometry.json",
            )

            self.assertEqual(
                cli_main(
                    [
                        "structure-to-geometry",
                        reconstructed_heavy_path,
                        "-o",
                        reconstructed_residue_geometry_degrees_path,
                        "--angle-units",
                        "degrees",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(
                reconstructed_residue_geometry_degrees_path,
                "ag.roundtrip.residue-geometry.degrees.json",
            )

            self.assertEqual(cli_main(["score", heavy_path, "--model", "generic", "-o", score_path]), 0)
            self.assertEqual(cli_main(["score", heavy_path, "--model", "pheat-rg", "-o", pheat_rg_score_path]), 0)
            with open(pheat_rg_score_path, "r", encoding="utf-8") as handle:
                pheat_rg_score = json.load(handle)
            self.assert_payload_matches_schema("energy-result", pheat_rg_score)
            self.assertEqual(pheat_rg_score["model"], "pheat-rg")
            self.assertEqual(pheat_rg_score["metadata"]["atom_set"], "ca")
            self.assertIn("rg_compactness_penalty", pheat_rg_score["terms"])
            self.assertEqual(
                cli_main(
                    [
                        "radius-of-gyration",
                        os.path.relpath(TINY_PDB, REPO_ROOT),
                        "-o",
                        radius_of_gyration_path,
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(radius_of_gyration_path, "tiny.radius-of-gyration.json")
            with open(radius_of_gyration_path, "r", encoding="utf-8") as handle:
                rg_payload = json.load(handle)
            self.assert_payload_matches_schema("radius-of-gyration-result", rg_payload)
            self.assertEqual(rg_payload["radius_of_gyration"]["atom_set"], "all-heavy")
            self.assertIn("unweighted", rg_payload["radius_of_gyration"]["values"])
            self.assertIn("mass_weighted", rg_payload["radius_of_gyration"]["values"])

            self.assertEqual(
                cli_main(
                    [
                        "rg",
                        heavy_path,
                        "--mode",
                        "unweighted",
                        "-o",
                        radius_of_gyration_unweighted_path,
                    ]
                ),
                0,
            )
            with open(radius_of_gyration_unweighted_path, "r", encoding="utf-8") as handle:
                rg_unweighted_payload = json.load(handle)
            self.assertEqual(rg_unweighted_payload["radius_of_gyration"]["mode"], "unweighted")
            self.assertEqual(rg_unweighted_payload["radius_of_gyration"]["atom_set"], "all-heavy")
            self.assertIn("unweighted", rg_unweighted_payload["radius_of_gyration"]["values"])
            self.assertNotIn("mass_weighted", rg_unweighted_payload["radius_of_gyration"]["values"])
            self.assertEqual(
                cli_main(["rg", heavy_path, "--atom-set", "ca", "-o", radius_of_gyration_ca_path]),
                0,
            )
            with open(radius_of_gyration_ca_path, "r", encoding="utf-8") as handle:
                rg_ca_payload = json.load(handle)
            self.assertEqual(rg_ca_payload["radius_of_gyration"]["atom_set"], "ca")
            self.assertEqual(rg_ca_payload["radius_of_gyration"]["atom_count"], 2)
            self.assert_payload_matches_schema("radius-of-gyration-result", rg_ca_payload)
            self.assertEqual(
                cli_main(
                    [
                        "radius-of-gyration",
                        heavy_path,
                        "--atom-set",
                        "backbone",
                        "-o",
                        radius_of_gyration_backbone_path,
                    ]
                ),
                0,
            )
            with open(radius_of_gyration_backbone_path, "r", encoding="utf-8") as handle:
                rg_backbone_payload = json.load(handle)
            self.assertEqual(rg_backbone_payload["radius_of_gyration"]["atom_set"], "backbone")
            self.assertEqual(rg_backbone_payload["radius_of_gyration"]["atom_count"], 9)
            self.assertEqual(cli_main(["rmsd", heavy_path, heavy_path, "-o", rmsd_path]), 0)
            with open(rmsd_path, "r", encoding="utf-8") as handle:
                rmsd_payload = json.load(handle)
            self.assertEqual(rmsd_payload["format"], "pheat.rmsd-result")
            self.assertEqual(rmsd_payload["rmsd"]["atom_set"], "all-heavy")
            self.assertAlmostEqual(rmsd_payload["rmsd"]["value"], 0.0)
            self.assertEqual(
                cli_main(
                    [
                        "rmsd",
                        heavy_path,
                        heavy_path,
                        "--atom-set",
                        "ca",
                        "--alignment-atom-set",
                        "ca",
                        "-o",
                        rmsd_ca_path,
                    ]
                ),
                0,
            )
            with open(rmsd_ca_path, "r", encoding="utf-8") as handle:
                rmsd_ca_payload = json.load(handle)
            self.assertEqual(rmsd_ca_payload["rmsd"]["atom_set"], "ca")
            self.assertEqual(rmsd_ca_payload["rmsd"]["matched_atoms"], 2)
            self.assertEqual(
                cli_main(
                    [
                        "examples",
                        "run",
                        "arxiv-2507-08955",
                        "-o",
                        os.path.join(tmpdir, "examples.json"),
                    ]
                ),
                0,
            )
            with open(score_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["model"], "generic")

    def test_cli_roundtrips_real_pdb_through_residue_geometry_and_heavy_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            residue_geometry_path = os.path.join(tmpdir, "2mu7-residue-geometry.json")
            residue_geometry_degrees_path = os.path.join(tmpdir, "2mu7-residue-geometry-degrees.json")
            residue_geometry_max_chi_path = os.path.join(tmpdir, "2mu7-residue-geometry-max-chi.json")
            reconstructed_heavy_path = os.path.join(tmpdir, "2mu7-reconstructed.json")
            reconstructed_pdb_path = os.path.join(tmpdir, "2mu7-reconstructed.pdb")
            roundtrip_residue_geometry_path = os.path.join(tmpdir, "2mu7-roundtrip-residue-geometry.json")
            roundtrip_residue_geometry_degrees_path = os.path.join(
                tmpdir,
                "2mu7-roundtrip-residue-geometry-degrees.json",
            )

            self.assertEqual(cli_main(["pdb-to-geometry", TWO_MU7_PDB, "-o", residue_geometry_path]), 0)
            self.assert_file_matches_expected(residue_geometry_path, "2mu7.residue-geometry.json")

            self.assertEqual(
                cli_main(
                    [
                        "pdb-to-geometry",
                        TWO_MU7_PDB,
                        "-o",
                        residue_geometry_max_chi_path,
                        "--max-chi",
                        "1",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(residue_geometry_max_chi_path, "2mu7.residue-geometry.max-chi-1.json")

            self.assertEqual(
                cli_main(
                    [
                        "pdb-to-geometry",
                        TWO_MU7_PDB,
                        "-o",
                        residue_geometry_degrees_path,
                        "--angle-units",
                        "degrees",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(residue_geometry_degrees_path, "2mu7.residue-geometry.degrees.json")

            self.assertEqual(
                cli_main(
                    [
                        "geometry-to-structure",
                        residue_geometry_path,
                        "-o",
                        reconstructed_heavy_path,
                        "--pdb-output",
                        reconstructed_pdb_path,
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(
                reconstructed_heavy_path,
                "2mu7.reconstructed.heavy.json",
            )
            self.assert_file_matches_expected(reconstructed_pdb_path, "2mu7.reconstructed.pdb")
            reconstructed_heavy = HeavyAtomStructure.from_json(_read_text(reconstructed_heavy_path))
            reconstructed_atom_names = {atom.name.strip().upper() for atom in reconstructed_heavy.atoms}
            reconstructed_pdb = _read_text(reconstructed_pdb_path)
            self.assertEqual(len(reconstructed_heavy.atoms), 150)
            self.assertTrue({"CB", "CG", "CD"}.issubset(reconstructed_atom_names))
            self.assertIn(" CB ", reconstructed_pdb)
            self.assertIn(" CG ", reconstructed_pdb)
            self.assertIn(" CD ", reconstructed_pdb)

            self.assertEqual(
                cli_main(
                    [
                        "structure-to-geometry",
                        reconstructed_heavy_path,
                        "-o",
                        roundtrip_residue_geometry_path,
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(roundtrip_residue_geometry_path, "2mu7.roundtrip.residue-geometry.json")
            roundtrip_chi1_path = os.path.join(tmpdir, "2mu7-roundtrip-residue-geometry-chi1.json")
            self.assertEqual(
                cli_main(
                    [
                        "structure-to-geometry",
                        reconstructed_heavy_path,
                        "-o",
                        roundtrip_chi1_path,
                        "--max-chi",
                        "1",
                    ]
                ),
                0,
            )
            roundtrip_chi1 = ResidueGeometryStructure.from_json(_read_text(roundtrip_chi1_path))
            self.assertTrue(all(len(residue.chi) <= 1 for residue in roundtrip_chi1.residues))

            self.assertEqual(
                cli_main(
                    [
                        "structure-to-geometry",
                        reconstructed_heavy_path,
                        "-o",
                        roundtrip_residue_geometry_degrees_path,
                        "--angle-units",
                        "degrees",
                    ]
                ),
                0,
            )
            self.assert_file_matches_expected(
                roundtrip_residue_geometry_degrees_path,
                "2mu7.roundtrip.residue-geometry.degrees.json",
            )
            roundtrip_geometries = ResidueGeometryStructure.from_json(_read_text(roundtrip_residue_geometry_path))
            self.assert_real_pdb_chi_coverage(roundtrip_geometries)

    def test_web_port_selector_keeps_available_preferred_port(self):
        from pheat.webapp import select_web_port

        self.assertEqual(
            select_web_port(
                "127.0.0.1",
                8000,
                random_port_if_taken=True,
                bind_checker=lambda _host, _port: True,
            ),
            8000,
        )

    def test_web_port_selector_returns_random_fallback_when_preferred_is_taken(self):
        from pheat.webapp import select_web_port

        selected = select_web_port(
            "127.0.0.1",
            8000,
            random_port_if_taken=True,
            fallback_start=8123,
            fallback_end=8123,
            bind_checker=lambda _host, port: port == 8123,
        )

        self.assertEqual(selected, 8123)

    def test_web_port_selector_reports_invalid_or_exhausted_fallback_ports(self):
        from pheat.webapp import select_web_port

        self.assertEqual(
            select_web_port(
                "127.0.0.1",
                8000,
                random_port_if_taken=False,
                bind_checker=lambda _host, _port: False,
            ),
            8000,
        )
        with self.assertRaisesRegex(ValueError, "fallback_start"):
            select_web_port(
                "127.0.0.1",
                8000,
                random_port_if_taken=True,
                fallback_start=8124,
                fallback_end=8123,
                bind_checker=lambda _host, _port: False,
            )
        with self.assertRaisesRegex(RuntimeError, "No available web port"):
            select_web_port(
                "127.0.0.1",
                8000,
                random_port_if_taken=True,
                fallback_start=8123,
                fallback_end=8123,
                bind_checker=lambda _host, _port: False,
            )

    def test_web_launch_url_is_terminal_friendly_and_browser_clickable(self):
        from pheat.webapp import web_bind_message, web_launch_url

        self.assertEqual(web_launch_url("127.0.0.1", 8000), "http://127.0.0.1:8000/")
        self.assertEqual(web_launch_url("localhost", 8000), "http://localhost:8000/")
        self.assertEqual(web_launch_url("0.0.0.0", 8000), "http://127.0.0.1:8000/")
        self.assertEqual(web_launch_url("::1", 8000), "http://[::1]:8000/")
        self.assertIsNone(web_bind_message("127.0.0.1", 8000))
        self.assertEqual(
            web_bind_message("0.0.0.0", 8000),
            "Serving on 0.0.0.0:8000 (all IPv4 interfaces).",
        )
        self.assertEqual(
            web_bind_message("::", 8000),
            "Serving on [::]:8000 (all IPv6 interfaces).",
        )
        with self.assertRaisesRegex(ValueError, "port"):
            web_launch_url("127.0.0.1", 0)

    def test_web_app_uploads_tiny_pdb_and_writes_single_roundtrip_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, work_dir = self.create_test_web_client(tmpdir)
            with open(TINY_PDB, "rb") as handle:
                response = client.post(
                    "/roundtrip",
                    data={"mode": "single", "score_models": "generic"},
                    files={"pdb_file": ("tiny.pdb", handle, "chemical/x-pdb")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn('src="/molstar/molstar.js"', response.text)
            self.assertIn('href="/molstar/molstar.css"', response.text)
            self.assertIn("<title>PHEAT Roundtrip Results</title>", response.text)
            self.assertIn("<h2>Mol* Aligned Structures</h2>", response.text)
            self.assertIn('href="https://molstar.org/"', response.text)
            self.assertIn('href="https://doi.org/10.1093/nar/gkab314"', response.text)
            self.assertIn("Sehnal et al.", response.text)
            self.assertIn("Nucleic Acids Research", response.text)
            self.assertIn("PHEAT: Protein Heavy-atom Energy and Analysis Toolkit", response.text)
            self.assertIn("Blankenberg Lab and contributors", response.text)
            self.assertIn(f"Version {__version__}", response.text)
            self.assertIn('href="https://github.com/BlankenbergLab/pheat"', response.text)
            self.assertEqual(response.text.count('class="sortable-table"'), 3)
            self.assertIn(".sortable-table thead th", response.text)
            self.assertIn("position: sticky", response.text)
            self.assertIn("top: 0", response.text)
            self.assertIn("aria-sort", response.text)
            self.assertIn("Sort table by this column", response.text)
            self.assertIn("initializeSortableTables", response.text)
            self.assertIn("sortTableByColumn", response.text)
            self.assertIn("compareSortableRows", response.text)
            self.assertIn("Number.parseFloat", response.text)
            self.assertIn('id="molstar-viewer"', response.text)
            self.assertIn('data-structure-toggle="original" aria-pressed="true"', response.text)
            self.assertIn('data-structure-toggle="reconstructed" aria-pressed="true"', response.text)
            self.assertIn("legend-swatch legend-swatch-original", response.text)
            self.assertIn("legend-swatch legend-swatch-reconstructed", response.text)
            self.assertIn(
                'class="display-mode-group" role="group" aria-label="Display mode"',
                response.text,
            )
            self.assertIn('data-display-mode="ribbon" aria-pressed="true"', response.text)
            self.assertIn('data-display-mode="all-atom" aria-pressed="false"', response.text)
            self.assertIn("legend-toggle display-mode-toggle", response.text)
            self.assertIn(
                '.legend-toggle[data-structure-toggle][aria-pressed="false"]',
                response.text,
            )
            self.assertIn('.display-mode-toggle[aria-pressed="true"]', response.text)
            self.assertNotIn('.legend-toggle[aria-pressed="false"]', response.text)
            self.assertIn("data-recolor", response.text)
            self.assertIn("viewerState.visible.original", response.text)
            self.assertIn("viewerState.visible.reconstructed", response.text)
            self.assertIn("viewerState.displayMode", response.text)
            self.assertIn("loadPdbStructure(caseData.original_pdb, 'original'", response.text)
            self.assertIn("loadPdbStructure(caseData.reconstructed_pdb, 'reconstructed'", response.text)
            self.assertIn("recolorCurrentRepresentations", response.text)
            self.assertIn(
                "recolorButton.addEventListener('click', () => recolorCurrentRepresentations())",
                response.text,
            )
            self.assertIn("preset-structure-representation-polymer-cartoon", response.text)
            self.assertIn("preset-structure-representation-atomic-detail", response.text)
            self.assertIn("viewportFocusBehavior: 'disabled'", response.text)
            self.assertIn("layoutShowControls: true", response.text)
            self.assertIn("layoutShowLeftPanel: true", response.text)
            self.assertIn("viewportShowControls: true", response.text)
            self.assertIn("viewportShowSettings: true", response.text)
            self.assertIn("loadCase(caseSelect.value, { resetCamera: false })", response.text)
            self.assertIn("Both structures hidden.", response.text)
            self.assertIn("Loaded selected structures from embedded PDB data", response.text)
            self.assertIn("0x0072B2", response.text)
            self.assertIn("0xD55E00", response.text)
            self.assertIn("globalName: 'uniform'", response.text)
            self.assertIn("globalColorParams", response.text)
            self.assertIn("Original Radius Of Gyration", response.text)
            self.assertIn("C-alpha RMSD", response.text)
            self.assertIn("Rg unweighted", response.text)
            self.assertIn("Rg mass-weighted", response.text)
            self.assertIn('id="viewer-c-alpha-rmsd"', response.text)
            self.assertIn("caseData.c_alpha_rmsd", response.text)
            self.assertIn('id="viewer-rg-unweighted"', response.text)
            self.assertIn('id="viewer-rg-mass-weighted"', response.text)
            self.assertIn("formatRg(caseData.rg_unweighted_reconstructed", response.text)
            self.assertNotIn("data-sidechain-toggle", response.text)
            self.assertNotIn("viewerState.sidechains", response.text)
            self.assertNotIn("original_sidechain_pdb", response.text)
            self.assertNotIn("reconstructed_sidechain_pdb", response.text)
            self.assertNotIn("original_backbone_pdb", response.text)
            self.assertNotIn("reconstructed_backbone_pdb", response.text)
            self.assertNotIn("loadStructureParts", response.text)
            self.assertNotIn("cdn", response.text.lower())
            self.assertNotIn("datatables", response.text.lower())
            self.assertNotIn("tablesort", response.text.lower())

            start_tag = '<script type="application/json" id="roundtrip-viewer-data">'
            start = response.text.index(start_tag) + len(start_tag)
            end = response.text.index("</script>", start)
            viewer_cases = json.loads(response.text[start:end])
            self.assertIn("rg_unweighted_reconstructed", viewer_cases[0])
            self.assertIn("rg_mass_weighted_reconstructed", viewer_cases[0])
            self.assertIn(" CB ", viewer_cases[0]["original_pdb"])
            self.assertIn(" CA ", viewer_cases[0]["original_pdb"])
            self.assertIn(" CB ", viewer_cases[0]["reconstructed_pdb"])
            self.assertIn(" CA ", viewer_cases[0]["reconstructed_pdb"])
            self.assertNotIn("original_backbone_pdb", viewer_cases[0])
            self.assertNotIn("original_sidechain_pdb", viewer_cases[0])
            self.assertNotIn("reconstructed_backbone_pdb", viewer_cases[0])
            self.assertNotIn("reconstructed_sidechain_pdb", viewer_cases[0])

            session_id = self.session_id_from_response(response.text)
            summary_path = os.path.join(work_dir, session_id, "summary.json")
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["input_name"], "tiny.pdb")
            self.assertEqual(summary["score_models"], ["generic"])
            self.assertEqual(summary["case_count"], 1)
            self.assertIn("original_radius_of_gyration", summary)
            self.assertIn("radius_of_gyration", summary["cases"][0])
            self.assertEqual(summary["cases"][0]["case_id"], "single")
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        work_dir,
                        session_id,
                        summary["cases"][0]["paths"]["reconstructed_aligned_pdb"],
                    )
                )
            )

    def test_web_app_configurable_options_feed_roundtrip_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, work_dir = self.create_test_web_client(tmpdir)
            with open(TINY_PDB, "rb") as handle:
                response = client.post(
                    "/roundtrip",
                    data={
                        "mode": "configurable",
                        "angle_units": "degrees",
                        "stored_angles": "omega,theta",
                        "max_chi": "1",
                        "include_terminal_oxt": "on",
                        "score_models": "generic",
                    },
                    files={"pdb_file": ("tiny.pdb", handle, "chemical/x-pdb")},
                )

            self.assertEqual(response.status_code, 200)
            session_id = self.session_id_from_response(response.text)
            with open(os.path.join(work_dir, session_id, "summary.json"), "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            case = summary["cases"][0]

            self.assertEqual(case["stored_angles"], ["omega", "theta"])
            self.assertEqual(case["max_chi"], 1)
            self.assertEqual(case["angle_units"], "degrees")
            self.assertTrue(case["include_terminal_oxt"])

            residue_geometry = json.loads(
                _read_text(os.path.join(work_dir, session_id, case["paths"]["residue_geometry"]))
            )
            self.assertEqual(residue_geometry["angle_units"], "degrees")
            self.assertIn("omega", residue_geometry["residues"][0])
            self.assertIn("theta", residue_geometry["residues"][0])
            self.assertNotIn("tau", residue_geometry["residues"][0])

            reconstructed_pdb = _read_text(
                os.path.join(work_dir, session_id, case["paths"]["reconstructed_aligned_pdb"])
            )
            self.assertIn("OXT", reconstructed_pdb)

    def test_web_app_combinatorial_mode_runs_real_pdb_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, work_dir = self.create_test_web_client(tmpdir)
            with open(TWO_MU7_PDB, "rb") as handle:
                response = client.post(
                    "/roundtrip",
                    data={"mode": "combinatorial", "score_models": "generic"},
                    files={"pdb_file": ("2mu7.pdb", handle, "chemical/x-pdb")},
                )

            self.assertEqual(response.status_code, 200)
            session_id = self.session_id_from_response(response.text)
            with open(os.path.join(work_dir, session_id, "summary.json"), "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["input_name"], "2mu7.pdb")
            self.assertEqual(summary["case_count"], 24)
            self.assertIn("roundtrip-viewer-data", response.text)
            self.assertTrue(all("rmsd" in case for case in summary["cases"]))
            self.assertTrue(all("radius_of_gyration" in case for case in summary["cases"]))
            self.assertTrue(
                any(case["stored_angles"] == ["omega", "tau", "theta"] for case in summary["cases"])
            )

    def test_web_app_reports_invalid_uploads_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, _work_dir = self.create_test_web_client(tmpdir)
            response = client.post(
                "/roundtrip",
                data={"mode": "single", "score_models": "generic"},
                files={"pdb_file": ("not-a-pdb.pdb", b"not pdb data\n", "chemical/x-pdb")},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Uploaded structure did not contain any heavy atoms", response.text)


if __name__ == "__main__":
    unittest.main()

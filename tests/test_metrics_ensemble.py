"""Tests for pheat.metrics.pairwise_rmsd_matrix and ensemble_rmsd_stats."""

import unittest

import numpy as np

from pheat.metrics import ensemble_rmsd_stats, pairwise_rmsd_matrix
from pheat.models import Atom, HeavyAtomStructure


def _make_structure(name, coords):
    atoms = [
        Atom(
            name="CA",
            element="C",
            x=float(coord[0]),
            y=float(coord[1]),
            z=float(coord[2]),
            resname="GLY",
            chain_id="A",
            resseq=i + 1,
            record_name="ATOM",
            serial=i + 1,
        )
        for i, coord in enumerate(coords)
    ]
    return HeavyAtomStructure(atoms=atoms, name=name, metadata={})


class PairwiseRmsdMatrixCoordArraysTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.b = self.a.copy() + np.array([10.0, 0.0, 0.0])  # translation only
        self.c = self.a.copy() + np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]])

    def test_shape_is_square_with_ensemble_size(self):
        matrix = pairwise_rmsd_matrix([self.a, self.b, self.c])
        self.assertEqual(matrix.shape, (3, 3))

    def test_diagonal_is_zero(self):
        matrix = pairwise_rmsd_matrix([self.a, self.b, self.c])
        np.testing.assert_allclose(np.diag(matrix), np.zeros(3), atol=1e-9)

    def test_matrix_is_symmetric(self):
        matrix = pairwise_rmsd_matrix([self.a, self.b, self.c])
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    def test_translation_only_pair_has_zero_rmsd(self):
        matrix = pairwise_rmsd_matrix([self.a, self.b])
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=6)

    def test_distinct_structures_have_positive_rmsd(self):
        matrix = pairwise_rmsd_matrix([self.a, self.c])
        self.assertGreater(matrix[0, 1], 0.0)

    def test_single_structure_returns_zero_matrix(self):
        matrix = pairwise_rmsd_matrix([self.a])
        self.assertEqual(matrix.shape, (1, 1))
        self.assertEqual(matrix[0, 0], 0.0)

    def test_two_identical_structures_have_zero_rmsd(self):
        matrix = pairwise_rmsd_matrix([self.a, self.a.copy()])
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=9)


class PairwiseRmsdMatrixHeavyAtomStructureTests(unittest.TestCase):
    def setUp(self):
        coords_a = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
        coords_b = [[10.0, 0.0, 0.0], [11.5, 0.0, 0.0], [13.0, 0.0, 0.0]]  # translation
        coords_c = [[0.0, 0.0, 0.0], [1.5, 0.5, 0.0], [3.0, 1.0, 0.0]]
        self.structures = [
            _make_structure("a", coords_a),
            _make_structure("b", coords_b),
            _make_structure("c", coords_c),
        ]

    def test_structure_input_produces_symmetric_zero_diagonal_matrix(self):
        matrix = pairwise_rmsd_matrix(self.structures, atom_set="ca")
        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_allclose(np.diag(matrix), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    def test_structure_input_translation_only_pair_is_zero(self):
        matrix = pairwise_rmsd_matrix(self.structures[:2], atom_set="ca")
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=6)


class EnsembleRmsdStatsTests(unittest.TestCase):
    def test_returns_avg_max_and_min_nonzero(self):
        matrix = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
        stats = ensemble_rmsd_stats(matrix)
        self.assertAlmostEqual(stats["avg_pairwise_rmsd"], (1.0 + 3.0 + 2.0) / 3.0, places=6)
        self.assertAlmostEqual(stats["max_pairwise_rmsd"], 3.0, places=6)
        self.assertAlmostEqual(stats["min_nonzero_rmsd"], 1.0, places=6)
        self.assertEqual(stats["ensemble_size"], 3)

    def test_min_nonzero_skips_zeros_from_identical_replicas(self):
        # off-diagonal includes a zero (replicas 0,1 are identical) and 2.0
        matrix = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
        stats = ensemble_rmsd_stats(matrix)
        self.assertAlmostEqual(stats["min_nonzero_rmsd"], 2.0, places=6)

    def test_single_structure_returns_zero_stats(self):
        stats = ensemble_rmsd_stats(np.zeros((1, 1)))
        self.assertEqual(stats["avg_pairwise_rmsd"], 0.0)
        self.assertEqual(stats["max_pairwise_rmsd"], 0.0)
        self.assertEqual(stats["min_nonzero_rmsd"], 0.0)
        self.assertEqual(stats["ensemble_size"], 1)

    def test_rejects_non_square_input(self):
        with self.assertRaises(ValueError):
            ensemble_rmsd_stats(np.zeros((2, 3)))

    def test_all_zero_off_diagonal_returns_zero_min_nonzero(self):
        stats = ensemble_rmsd_stats(np.zeros((3, 3)))
        self.assertEqual(stats["min_nonzero_rmsd"], 0.0)


if __name__ == "__main__":
    unittest.main()

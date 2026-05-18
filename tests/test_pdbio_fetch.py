"""Tests for RCSB fetch helpers in pheat.pdbio."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from pheat import pdbio
from pheat.pdbio import fetch_pdb, load_pdb_by_id, load_pdb_ca_by_id

FIXTURES = Path(__file__).parent / "fixtures"
TINY_PDB = (FIXTURES / "tiny.pdb").read_text(encoding="utf-8")
TINY_CIF = (FIXTURES / "tiny_mmcif.cif").read_text(encoding="utf-8")


class _Recorder:
    """Mock downloader: writes a fixed payload and records each call."""

    def __init__(self, payload: str = TINY_PDB) -> None:
        self.payload = payload
        self.calls: List[Tuple[str, str]] = []

    def __call__(self, url: str, dest: str) -> None:
        self.calls.append((url, dest))
        Path(dest).write_text(self.payload, encoding="utf-8")


class FetchPdbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pheat-fetch-test-"))
        # Reset the module-level default cache so cache_dir=None tests are isolated.
        pdbio._default_cache_dir = None

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        if pdbio._default_cache_dir is not None:
            shutil.rmtree(pdbio._default_cache_dir, ignore_errors=True)
            pdbio._default_cache_dir = None

    def test_fetch_pdb_writes_to_explicit_cache_dir(self):
        rec = _Recorder()
        path = fetch_pdb("1abc", self.tmp, downloader=rec)
        self.assertEqual(path, self.tmp / "1ABC.pdb")
        self.assertTrue(path.exists())
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0][0], "https://files.rcsb.org/download/1ABC.pdb")

    def test_fetch_pdb_skips_download_when_cached(self):
        rec = _Recorder()
        fetch_pdb("1abc", self.tmp, downloader=rec)
        rec.calls.clear()
        path2 = fetch_pdb("1abc", self.tmp, downloader=rec)
        self.assertEqual(path2, self.tmp / "1ABC.pdb")
        self.assertEqual(rec.calls, [])

    def test_fetch_pdb_overwrite_redownloads(self):
        rec = _Recorder()
        fetch_pdb("1abc", self.tmp, downloader=rec)
        rec.calls.clear()
        fetch_pdb("1abc", self.tmp, downloader=rec, overwrite=True)
        self.assertEqual(len(rec.calls), 1)

    def test_fetch_pdb_format_cif_dispatches_extension(self):
        rec = _Recorder(payload=TINY_CIF)
        path = fetch_pdb("2xyz", self.tmp, format="cif", downloader=rec)
        self.assertEqual(path, self.tmp / "2XYZ.cif")
        self.assertEqual(rec.calls[0][0], "https://files.rcsb.org/download/2XYZ.cif")

    def test_fetch_pdb_mmcif_alias_maps_to_cif(self):
        rec = _Recorder(payload=TINY_CIF)
        path = fetch_pdb("2xyz", self.tmp, format="mmcif", downloader=rec)
        self.assertEqual(path.suffix, ".cif")

    def test_fetch_pdb_invalid_id_raises(self):
        rec = _Recorder()
        for bad in ("abc", "abcde", "1@bc", "", None):
            with self.assertRaises((ValueError, TypeError)):
                fetch_pdb(bad, self.tmp, downloader=rec)  # type: ignore[arg-type]

    def test_fetch_pdb_unknown_format_raises(self):
        rec = _Recorder()
        with self.assertRaises(ValueError):
            fetch_pdb("1abc", self.tmp, format="xyz", downloader=rec)

    def test_fetch_pdb_default_cache_dir_creates_and_reuses_tempdir(self):
        rec = _Recorder()
        path1 = fetch_pdb("1abc", downloader=rec)
        self.assertTrue(path1.exists())
        self.assertEqual(path1.name, "1ABC.pdb")
        cache1 = path1.parent
        self.assertTrue(str(cache1).startswith(tempfile.gettempdir()))
        self.assertEqual(len(rec.calls), 1)

        rec.calls.clear()
        path2 = fetch_pdb("1abc", downloader=rec)
        self.assertEqual(path2, path1)
        self.assertEqual(rec.calls, [], "cached call should not re-download")

        path3 = fetch_pdb("2xyz", downloader=rec)
        self.assertEqual(path3.parent, cache1, "default cache dir should be reused")

    def test_fetch_pdb_creates_missing_cache_dir(self):
        nested = self.tmp / "nested" / "cache"
        rec = _Recorder()
        path = fetch_pdb("1abc", nested, downloader=rec)
        self.assertTrue(path.exists())
        self.assertTrue(nested.is_dir())


class LoadByIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pheat-fetch-test-"))
        pdbio._default_cache_dir = None

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        if pdbio._default_cache_dir is not None:
            shutil.rmtree(pdbio._default_cache_dir, ignore_errors=True)
            pdbio._default_cache_dir = None

    def test_load_pdb_by_id_returns_structure(self):
        rec = _Recorder()
        structure = load_pdb_by_id("1abc", self.tmp, downloader=rec)
        self.assertTrue(len(structure.atoms) > 0)

    def test_load_pdb_by_id_cif_routes_to_mmcif(self):
        rec = _Recorder(payload=TINY_CIF)
        structure = load_pdb_by_id("2xyz", self.tmp, format="cif", downloader=rec)
        self.assertTrue(len(structure.atoms) > 0)

    def test_load_pdb_ca_by_id_returns_n_by_3_array(self):
        rec = _Recorder()
        ca = load_pdb_ca_by_id("1abc", self.tmp, downloader=rec)
        self.assertEqual(ca.ndim, 2)
        self.assertEqual(ca.shape[1], 3)
        self.assertGreater(ca.shape[0], 0)


if __name__ == "__main__":
    unittest.main()

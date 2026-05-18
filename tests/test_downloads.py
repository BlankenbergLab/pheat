import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pheat import examples, sources


class DownloadTests(unittest.TestCase):
    def test_fetch_source_downloads_atomically_and_verifies_checksum(self):
        payload = b"verified source payload\n"
        manifest = {
            "sources": [
                {
                    "id": "demo-source",
                    "filename": "demo.dat",
                    "urls": ["https://example.invalid/demo.dat"],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
        }

        def downloader(url, filename):
            self.assertEqual(url, "https://example.invalid/demo.dat")
            Path(filename).write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(sources, "load_source_manifest", return_value=manifest):
                output = sources.fetch_source("demo-source", tmpdir, downloader=downloader)

            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(output.name, "demo.dat")
            self.assertEqual([path.name for path in Path(tmpdir).glob("*.tmp")], [])

    def test_fetch_source_removes_failed_temporary_files_and_preserves_existing_output(self):
        manifest = {
            "sources": [
                {
                    "id": "demo-source",
                    "filename": "demo.dat",
                    "urls": ["https://example.invalid/demo.dat"],
                }
            ]
        }

        def failing_downloader(_url, _filename):
            raise OSError("network unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "demo.dat"
            existing.write_text("old payload\n", encoding="utf-8")
            with mock.patch.object(sources, "load_source_manifest", return_value=manifest):
                with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                    sources.fetch_source("demo-source", tmpdir, downloader=failing_downloader)

            self.assertEqual(existing.read_text(encoding="utf-8"), "old payload\n")
            self.assertEqual([path.name for path in Path(tmpdir).glob("*.tmp")], [])

    def test_fetch_source_does_not_publish_checksum_mismatches(self):
        manifest = {
            "sources": [
                {
                    "id": "demo-source",
                    "filename": "demo.dat",
                    "urls": ["https://example.invalid/demo.dat"],
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                }
            ]
        }

        def downloader(_url, filename):
            Path(filename).write_bytes(b"actual")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(sources, "load_source_manifest", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    sources.fetch_source("demo-source", tmpdir, downloader=downloader)

            self.assertFalse((Path(tmpdir) / "demo.dat").exists())
            self.assertEqual([path.name for path in Path(tmpdir).glob("*.tmp")], [])

    def test_fetch_source_downloads_multi_file_bundle_with_provenance(self):
        payloads = {
            "https://example.invalid/cca.bcif": b"atom subset\n",
            "https://example.invalid/ccb.bcif": b"bond subset\n",
        }
        manifest = {
            "sources": [
                {
                    "id": "demo-bundle",
                    "name": "Demo bundle",
                    "downloadable": True,
                    "license": "CC0 1.0 Universal",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "files": [
                        {
                            "filename": "cca.bcif",
                            "url": "https://example.invalid/cca.bcif",
                            "sha256": hashlib.sha256(payloads["https://example.invalid/cca.bcif"]).hexdigest(),
                        },
                        {
                            "filename": "ccb.bcif",
                            "url": "https://example.invalid/ccb.bcif",
                            "sha256": hashlib.sha256(payloads["https://example.invalid/ccb.bcif"]).hexdigest(),
                        },
                    ],
                }
            ]
        }

        def downloader(url, filename):
            Path(filename).write_bytes(payloads[url])

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "nested" / "bcif"
            with mock.patch.object(sources, "load_source_manifest", return_value=manifest):
                outputs = sources.fetch_source("demo-bundle", destination, downloader=downloader)
                verification = sources.verify_sources(tmpdir)

            self.assertEqual([path.name for path in outputs], ["cca.bcif", "ccb.bcif"])
            provenance = json.loads((destination / sources.PROVENANCE_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(provenance["format"], "pheat.source-cache-provenance")
            source_record = provenance["sources"][0]
            self.assertEqual(source_record["id"], "demo-bundle")
            self.assertEqual(len(source_record["files"]), 2)
            self.assertEqual(verification["demo-bundle"]["status"], "ok")
            self.assertEqual(len(verification["demo-bundle"]["files"]), 2)

    def test_fetch_example_set_downloads_pdbs_atomically(self):
        manifest = {"examples": [{"id": "demo", "pdb_id": "1ABC"}]}
        payload = b"HEADER    TEST PDB\nEND\n"

        def downloader(url, filename):
            self.assertEqual(url, "https://files.rcsb.org/download/1ABC.pdb")
            Path(filename).write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(examples, "load_example_manifest", return_value=manifest):
                fetched = examples.fetch_example_set("demo-set", tmpdir, downloader=downloader)

            self.assertEqual(len(fetched), 1)
            self.assertEqual(fetched[0], Path(tmpdir) / "demo-set" / "1abc.pdb")
            self.assertEqual(fetched[0].read_bytes(), payload)
            self.assertEqual([path.name for path in (Path(tmpdir) / "demo-set").glob("*.tmp")], [])

    def test_fetch_example_set_removes_failed_temporary_files(self):
        manifest = {"examples": [{"id": "demo", "pdb_id": "1ABC"}]}

        def failing_downloader(_url, _filename):
            raise OSError("network unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(examples, "load_example_manifest", return_value=manifest):
                with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                    examples.fetch_example_set("demo-set", tmpdir, downloader=failing_downloader)

            output_dir = Path(tmpdir) / "demo-set"
            self.assertFalse((output_dir / "1abc.pdb").exists())
            self.assertEqual([path.name for path in output_dir.glob("*.tmp")], [])

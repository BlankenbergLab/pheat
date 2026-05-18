import unittest
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError, version
from importlib import resources

import pheat
from pheat.examples import list_example_sets, load_example_manifest
from pheat.geometry_tables import list_packaged_geometry_tables, load_packaged_geometry_table
from pheat.molstar_assets import (
    MOLSTAR_ENV_VAR,
    copy_molstar_assets,
    default_molstar_asset_dir,
    install_molstar_assets,
    molstar_asset_status,
)
from pheat.schemas import available_schemas, load_schema
from pheat.sources import list_sources
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PUBLIC_BASE_URL = "https://pheat.tools.blankenberglab.org/schemas/"


class PackagingTests(unittest.TestCase):
    def test_package_exposes_installed_version(self):
        self.assertTrue(pheat.__version__)
        try:
            installed_version = version("pheat")
        except PackageNotFoundError:
            self.assertEqual(pheat.__version__, "0.0.0+unknown")
        else:
            self.assertEqual(pheat.__version__, installed_version)

    def test_packaged_schema_and_manifest_resources_load(self):
        self.assertIn("arxiv-2507-08955", list_example_sets())
        self.assertTrue(load_example_manifest("arxiv-2507-08955")["examples"])
        self.assertTrue(list_sources())
        for schema_name in available_schemas():
            with self.subTest(schema=schema_name):
                schema = load_schema(schema_name)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertTrue(schema["$id"].startswith(SCHEMA_PUBLIC_BASE_URL))
                self.assertIn("title", schema)

    def test_mkdocs_schema_hook_publishes_packaged_schemas(self):
        package_schema_dir = ROOT / "src" / "pheat" / "schemas"
        package_files = sorted(path.name for path in package_schema_dir.glob("*.schema.json"))

        self.assertTrue(package_files)
        hook_path = ROOT / "scripts" / "mkdocs_schema_assets.py"
        spec = importlib.util.spec_from_file_location("mkdocs_schema_assets", hook_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            copied = module.copy_schema_assets(repo_root=ROOT, site_dir=Path(tmpdir) / "site")
            self.assertEqual(sorted(path.name for path in copied), package_files)
            for filename in package_files:
                with self.subTest(schema=filename):
                    self.assertEqual(
                        (Path(tmpdir) / "site" / "schemas" / filename).read_bytes(),
                        (package_schema_dir / filename).read_bytes(),
                    )

    def test_packaged_geometry_tables_load_and_match_manifest(self):
        tables = list_packaged_geometry_tables()
        table_ids = {table["id"] for table in tables}
        self.assertIn("ccd-sidechain-geometry-v1", table_ids)
        record = next(table for table in tables if table["id"] == "ccd-sidechain-geometry-v1")
        table = load_packaged_geometry_table("ccd-sidechain-geometry-v1")
        Draft202012Validator(load_schema("geometry-table-set")).validate(table)
        self.assertEqual(table["metadata"]["source"], "wwpdb-ccd-full")
        self.assertEqual(table["metadata"]["residue_count"], 29)

        table_bytes = (
            resources.files("pheat")
            .joinpath("data")
            .joinpath("geometry")
            .joinpath(record["filename"])
            .read_bytes()
        )
        self.assertEqual(hashlib.sha256(table_bytes).hexdigest(), record["table_sha256"])
        manifest = json.loads(
            resources.files("pheat")
            .joinpath("data")
            .joinpath("geometry")
            .joinpath("manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["format"], "pheat.packaged-geometry-manifest")

    def test_molstar_assets_use_platform_cache_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pheat.molstar_assets.user_cache_dir", return_value="/tmp/pheat-cache"):
                self.assertEqual(
                    default_molstar_asset_dir("5.9.0"),
                    Path("/tmp/pheat-cache/vendor/molstar/5.9.0"),
                )
                status = molstar_asset_status(version="5.9.0")
                self.assertEqual(status.path, Path("/tmp/pheat-cache/vendor/molstar/5.9.0"))
                self.assertEqual(status.source, "platform-cache")

    def test_molstar_assets_support_environment_override_and_missing_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {MOLSTAR_ENV_VAR: tmpdir}):
                status = molstar_asset_status()
                self.assertEqual(status.path, Path(tmpdir))
                self.assertEqual(status.source, MOLSTAR_ENV_VAR)
                self.assertFalse(status.available)
                with self.assertWarnsRegex(RuntimeWarning, "pheat molstar install"):
                    copied, message = copy_molstar_assets(Path(tmpdir) / "report" / "vendor" / "molstar")
                self.assertIsNone(copied)
                self.assertIn("pheat molstar install", message)

    def test_molstar_install_downloads_and_strips_mocked_npm_package(self):
        def fake_run(cmd, check, capture_output, text, timeout):
            destination = Path(cmd[-1])
            package_path = destination / "molstar-5.9.0.tgz"
            source_root = Path(tempdir) / "source"
            viewer = source_root / "package" / "build" / "viewer"
            viewer.mkdir(parents=True)
            (viewer / "molstar.js").write_text(
                "window.molstar = {};\n//# sourceMappingURL=molstar.js.map\n",
                encoding="utf-8",
            )
            (viewer / "molstar.css").write_text(
                ".msp-layout { color: black; }\n/*# sourceMappingURL=molstar.css.map */\n",
                encoding="utf-8",
            )
            (source_root / "package" / "LICENSE").write_text("MIT\n", encoding="utf-8")
            with tarfile.open(package_path, "w:gz") as archive:
                archive.add(source_root / "package", arcname="package")
            return subprocess.CompletedProcess(cmd, 0, stdout=package_path.name + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "assets"
            with patch("pheat.molstar_assets.subprocess.run", side_effect=fake_run):
                installed = install_molstar_assets(destination=destination, timeout=1, force=True)

            self.assertEqual(installed, destination)
            self.assertIn("window.molstar", (destination / "molstar.js").read_text(encoding="utf-8"))
            self.assertNotIn("sourceMappingURL", (destination / "molstar.js").read_text(encoding="utf-8"))
            self.assertNotIn("sourceMappingURL", (destination / "molstar.css").read_text(encoding="utf-8"))
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "pheat.molstar-assets")
            self.assertEqual(manifest["molstar_version"], "5.9.0")

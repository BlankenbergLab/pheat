import unittest
from unittest import mock

from pheat.report_assets import software_provenance_html
from pheat.software_provenance import collect_software_provenance, installed_distributions


class SoftwareProvenanceTests(unittest.TestCase):
    def test_collects_selected_components_without_legacy_packages(self):
        payload = collect_software_provenance(
            selected_score_models=["generic"],
            selected_features=["web"],
            include_installed_distributions=True,
        )

        names = {item["name"].lower() for item in payload["package_components"]}
        self.assertIn("platformdirs", names)
        self.assertIn("fastapi", names)
        self.assertNotIn("biopython", names)
        self.assertNotIn("mdtraj", names)
        self.assertNotIn("matplotlib", names)
        self.assertIn("installed_distributions", payload)

        html = software_provenance_html(payload)
        self.assertIn("Run Components", html)
        self.assertNotIn("biopython", html.lower())
        self.assertNotIn("mdtraj", html.lower())

    def test_collects_scoring_packages_and_external_tools_from_capabilities(self):
        capabilities = [
            {
                "model": "openmm-prepared",
                "requires": ["openmm"],
                "optional_requires": ["pdbfixer"],
            },
            {
                "model": "ambertools-sander",
                "requires": ["executable:tleap", "executable:sander"],
                "optional_requires": [],
            },
        ]

        def fake_distribution_version(name):
            return {
                "platformdirs": "4.0",
                "openmm": "8.2",
            }.get(name)

        def fake_tool(name, role, required):
            return {
                "name": name,
                "path": f"/fake/bin/{name}",
                "version": f"{name} version",
                "role": role,
                "required": required,
                "selected": True,
                "status": "available",
                "details": None,
            }

        with (
            mock.patch("pheat.software_provenance.model_capabilities", return_value=capabilities),
            mock.patch("pheat.software_provenance.distribution_version", side_effect=fake_distribution_version),
            mock.patch("pheat.software_provenance.external_tool_component", side_effect=fake_tool),
        ):
            payload = collect_software_provenance(
                selected_score_models=["openmm-prepared", "ambertools-sander"],
            )

        packages = {item["name"]: item for item in payload["package_components"]}
        tools = {item["name"]: item for item in payload["external_tools"]}
        self.assertEqual(packages["openmm"]["status"], "available")
        self.assertTrue(packages["openmm"]["required"])
        self.assertEqual(sorted(tools), ["sander", "tleap"])
        self.assertTrue(tools["sander"]["required"])

    def test_optional_requirements_are_opt_in(self):
        capabilities = [
            {
                "model": "openmm-prepared",
                "requires": ["openmm"],
                "optional_requires": ["pdbfixer"],
            },
        ]
        with (
            mock.patch("pheat.software_provenance.model_capabilities", return_value=capabilities),
            mock.patch("pheat.software_provenance.distribution_version", return_value=None),
        ):
            compact = collect_software_provenance(selected_score_models=["openmm-prepared"])
            expanded = collect_software_provenance(
                selected_score_models=["openmm-prepared"],
                include_optional_requirements=True,
            )

        compact_names = {item["name"] for item in compact["package_components"]}
        expanded_names = {item["name"] for item in expanded["package_components"]}
        self.assertNotIn("pdbfixer", compact_names)
        self.assertIn("pdbfixer", expanded_names)

    def test_installed_distributions_uses_metadata_name(self):
        class FakeDistribution:
            metadata = {
                "Name": "example-package",
                "Version": "1.2.3",
            }

        with mock.patch("pheat.software_provenance.importlib_metadata.distributions", return_value=[FakeDistribution()]):
            self.assertEqual(
                installed_distributions(),
                [{"name": "example-package", "version": "1.2.3"}],
            )


if __name__ == "__main__":
    unittest.main()

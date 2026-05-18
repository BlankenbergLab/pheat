import unittest

from pheat.report_assets import molstar_alignment_viewer_script, software_provenance_html, sortable_table_css


class ReportAssetTests(unittest.TestCase):
    def test_sortable_table_css_keeps_sticky_clickable_headers(self):
        css = sortable_table_css()

        self.assertIn(".sortable-table thead th", css)
        self.assertIn("position: sticky", css)
        self.assertIn("top: 0", css)
        self.assertIn(".sort-indicator", css)

    def test_molstar_viewer_script_contains_shared_sorting_and_viewer_controls(self):
        script = molstar_alignment_viewer_script()

        self.assertIn("function initializeSortableTables", script)
        self.assertIn("function sortTableByColumn", script)
        self.assertIn("replace(/\\s+/g, ' ')", script)
        self.assertIn("function loadPdbStructure", script)
        self.assertIn("loadPdbStructure(caseData.original_pdb, 'original'", script)
        self.assertIn("loadPdbStructure(caseData.reconstructed_pdb, 'reconstructed'", script)
        self.assertIn("function recolorCurrentRepresentations", script)
        self.assertIn("viewportBackgroundColor: '#ffffff'", script)
        self.assertIn("layoutShowControls: true", script)
        self.assertIn("loadCase(caseSelect.value, { resetCamera: false })", script)


class SoftwareProvenanceReportAssetTests(unittest.TestCase):
    def test_software_provenance_html_renders_components_and_tools(self):
        html = software_provenance_html({
            "python": {"version": "3.12", "implementation": "CPython", "executable": "/python"},
            "platform": {"platform": "test-platform"},
            "pheat": {"version": "0.1.0"},
            "selected_score_models": ["generic"],
            "selected_features": ["web"],
            "package_components": [
                {
                    "name": "platformdirs",
                    "role": "cache paths",
                    "required": True,
                    "status": "available",
                    "version": "4.0",
                }
            ],
            "external_tools": [
                {
                    "name": "gmx",
                    "role": "GROMACS",
                    "required": False,
                    "status": "missing",
                    "path": None,
                    "details": "not found",
                }
            ],
        })

        self.assertIn("Software Versions", html)
        self.assertIn("Run Components", html)
        self.assertIn("Selected External Tools", html)
        self.assertIn("platformdirs", html)
        self.assertIn("gmx", html)

    def test_software_provenance_html_tolerates_malformed_optional_fields(self):
        html = software_provenance_html({
            "python": "not-a-mapping",
            "platform": None,
            "pheat": [],
            "selected_score_models": "generic",
            "selected_features": {"web": True},
            "package_components": ["platformdirs"],
            "external_tools": {"gmx": "missing"},
        })

        self.assertIn("Software Versions", html)
        self.assertIn("No run components recorded", html)
        self.assertIn("n/a", html)

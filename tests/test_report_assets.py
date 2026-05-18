import unittest

from pheat.report_assets import molstar_alignment_viewer_script, sortable_table_css


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

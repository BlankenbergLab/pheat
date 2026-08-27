"""Shared HTML/CSS/JavaScript snippets for PHEAT reports and local viewers."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version

PHEAT_REPOSITORY_URL = "https://github.com/BlankenbergLab/pheat"
PHEAT_CITATION_TITLE = "PHEAT: Protein Heavy-atom Energy and Analysis Toolkit"
PHEAT_CITATION_AUTHORS = "Blankenberg Lab and contributors"


def pheat_version() -> str:
    """Return the installed PHEAT version for generated report citations."""

    try:
        return version("pheat")
    except PackageNotFoundError:  # pragma: no cover - only hit from an unpackaged source tree
        return "0.0.0+unknown"


def pheat_citation_html(*, package_version: str | None = None) -> str:
    """Return the standard HTML citation for generated PHEAT reports."""

    observed_version = package_version or pheat_version()
    return (
        f'<a href="{html.escape(PHEAT_REPOSITORY_URL)}" rel="noopener">'
        f"{html.escape(PHEAT_CITATION_TITLE)}</a>. "
        f"{html.escape(PHEAT_CITATION_AUTHORS)}. "
        f"Version {html.escape(observed_version)}."
    )


def _software_status_text(status: object) -> str:
    return str(status or "n/a")


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _mapping_rows(payload: Mapping[str, object], key: str) -> Iterable[Mapping[str, object]]:
    value = payload.get(key)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return (item for item in value if isinstance(item, Mapping))


def _string_values(payload: Mapping[str, object], key: str) -> Iterable[str]:
    value = payload.get(key)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return (str(item) for item in value)


def software_provenance_html(provenance: Mapping[str, object]) -> str:
    """Render selected-component software provenance as report HTML."""

    package_rows = []
    for item in _mapping_rows(provenance, "package_components"):
        package_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('name') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('role') or 'n/a'))}</td>"
            f"<td>{'yes' if item.get('required') else 'no'}</td>"
            f"<td>{html.escape(_software_status_text(item.get('status')))}</td>"
            f"<td>{html.escape(str(item.get('version') or 'not installed'))}</td>"
            "</tr>"
        )
    package_body = "\n".join(package_rows) or '<tr><td colspan="5">No run components recorded.</td></tr>'

    tool_rows = []
    for item in _mapping_rows(provenance, "external_tools"):
        tool_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('name') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('role') or 'n/a'))}</td>"
            f"<td>{'yes' if item.get('required') else 'no'}</td>"
            f"<td>{html.escape(_software_status_text(item.get('status')))}</td>"
            f"<td><code>{html.escape(str(item.get('path') or 'not found'))}</code></td>"
            f"<td>{html.escape(str(item.get('version') or item.get('details') or 'n/a'))}</td>"
            "</tr>"
        )
    external_tools_section = ""
    if tool_rows:
        external_tools_section = f"""
    <h3>Selected External Tools</h3>
    <table class="sortable-table software-tools">
      <thead><tr><th>Tool</th><th>Role</th><th>Required</th><th>Status</th><th>Path</th><th>Version/details</th></tr></thead>
      <tbody>{''.join(tool_rows)}</tbody>
    </table>
"""

    python_info = _mapping_value(provenance, "python")
    platform_info = _mapping_value(provenance, "platform")
    pheat_info = _mapping_value(provenance, "pheat")
    selected_models = ", ".join(_string_values(provenance, "selected_score_models")) or "n/a"
    selected_features = ", ".join(_string_values(provenance, "selected_features")) or "n/a"
    return f"""
    <section class="software-provenance">
      <h2>Software Versions</h2>
      <table>
        <tbody>
          <tr><th>Python</th><td>{html.escape(str(python_info.get('version') or 'n/a'))} ({html.escape(str(python_info.get('implementation') or 'n/a'))})</td></tr>
          <tr><th>Python executable</th><td><code>{html.escape(str(python_info.get('executable') or 'n/a'))}</code></td></tr>
          <tr><th>Platform</th><td>{html.escape(str(platform_info.get('platform') or 'n/a'))}</td></tr>
          <tr><th>PHEAT</th><td>version {html.escape(str(pheat_info.get('version') or 'n/a'))}</td></tr>
          <tr><th>Selected score models</th><td><code>{html.escape(selected_models)}</code></td></tr>
          <tr><th>Selected features</th><td><code>{html.escape(selected_features)}</code></td></tr>
        </tbody>
      </table>
      <h3>Run Components</h3>
      <p class="muted">This table highlights software selected by this run path, not every installed package.</p>
      <table class="sortable-table software-components">
        <thead><tr><th>Package</th><th>Role</th><th>Required</th><th>Status</th><th>Version</th></tr></thead>
        <tbody>{package_body}</tbody>
      </table>
      {external_tools_section}
    </section>
"""


def sortable_table_css() -> str:
    """Return CSS for sortable tables with sticky headers."""

    return """
    .sortable-table thead th {
      cursor: pointer;
      position: sticky;
      top: 0;
      user-select: none;
      z-index: 2;
    }
    .sortable-table thead th:focus {
      outline: 2px solid #205fa6;
      outline-offset: -2px;
    }
    .sort-indicator {
      color: var(--muted);
      display: inline-block;
      font-size: 11px;
      margin-left: 4px;
      min-width: 10px;
    }
"""


def molstar_alignment_viewer_script() -> str:
    """Return shared JavaScript for sortable tables and Mol* alignment viewers.

    The page that embeds this script must define these globals first:
    `roundtripCases`, `caseSelect`, `statusNode`, `VIEWER_COLORS`,
    `REPRESENTATION_PRESETS`, `viewerState`, and `setDetails(caseData)`.
    It may optionally define `PHEAT_VIEWER_LABELS` with `original`,
    `reconstructed`, and `loaded` strings.
    """

    return r"""
    function sortableCellValue(cell) {
      const text = (cell?.textContent || '').trim().replace(/\s+/g, ' ');
      if (!text || text.toLowerCase() === 'n/a') {
        return { kind: 'empty', value: Number.POSITIVE_INFINITY, text: '' };
      }
      const numericMatch = text.match(/^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/);
      if (numericMatch) {
        return { kind: 'number', value: Number.parseFloat(numericMatch[0]), text };
      }
      return { kind: 'text', value: text.toLocaleLowerCase(), text };
    }

    function compareSortableRows(left, right, columnIndex, direction) {
      const leftValue = sortableCellValue(left.row.cells[columnIndex]);
      const rightValue = sortableCellValue(right.row.cells[columnIndex]);
      let result = 0;
      if (leftValue.kind === 'empty' && rightValue.kind !== 'empty') result = 1;
      else if (leftValue.kind !== 'empty' && rightValue.kind === 'empty') result = -1;
      else if (leftValue.kind === 'number' && rightValue.kind === 'number') {
        result = leftValue.value - rightValue.value;
      } else {
        result = `${leftValue.value}`.localeCompare(`${rightValue.value}`, undefined, {
          numeric: true,
          sensitivity: 'base',
        });
      }
      if (result === 0) result = left.index - right.index;
      return direction === 'ascending' ? result : -result;
    }

    function sortTableByColumn(table, header, columnIndex) {
      const tbody = table.tBodies[0];
      if (!tbody) return;
      const currentDirection = header.getAttribute('aria-sort');
      const nextDirection = currentDirection === 'ascending' ? 'descending' : 'ascending';
      for (const sortableHeader of table.querySelectorAll('thead th')) {
        sortableHeader.setAttribute('aria-sort', 'none');
        const indicator = sortableHeader.querySelector('.sort-indicator');
        if (indicator) indicator.textContent = '';
      }
      header.setAttribute('aria-sort', nextDirection);
      const activeIndicator = header.querySelector('.sort-indicator');
      if (activeIndicator) activeIndicator.textContent = nextDirection === 'ascending' ? '^' : 'v';
      const rows = Array.from(tbody.rows).map((row, index) => ({ row, index }));
      rows.sort((left, right) => compareSortableRows(left, right, columnIndex, nextDirection));
      for (const entry of rows) tbody.appendChild(entry.row);
    }

    function initializeSortableTables() {
      for (const table of document.querySelectorAll('table.sortable-table')) {
        for (const [columnIndex, header] of Array.from(table.querySelectorAll('thead th')).entries()) {
          header.tabIndex = 0;
          header.setAttribute('aria-sort', 'none');
          header.title = 'Sort table by this column';
          const indicator = document.createElement('span');
          indicator.className = 'sort-indicator';
          indicator.setAttribute('aria-hidden', 'true');
          header.appendChild(indicator);
          header.addEventListener('click', () => sortTableByColumn(table, header, columnIndex));
          header.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              sortTableByColumn(table, header, columnIndex);
            }
          });
        }
      }
    }

    function releaseObjectUrls() {
      for (const url of viewerState.objectUrls) URL.revokeObjectURL(url);
      viewerState.objectUrls = [];
    }

    function pdbBlobUrl(pdbText) {
      const blob = new Blob([pdbText], { type: 'chemical/x-pdb' });
      const url = URL.createObjectURL(blob);
      viewerState.objectUrls.push(url);
      return url;
    }

    function representationParams(key) {
      const color = VIEWER_COLORS[key].value;
      return {
        theme: {
          globalName: 'uniform',
          globalColorParams: { value: color },
        },
      };
    }

    function hasCoordinateRecords(pdbText) {
      return /^(ATOM  |HETATM)/m.test(pdbText);
    }

    async function loadPdbStructure(pdbText, key, label) {
      if (!pdbText || !hasCoordinateRecords(pdbText)) return;
      const plugin = viewerState.viewer.plugin;
      const beforeRefs = representationRefs();
      const data = await plugin.builders.data.download(
        { url: pdbBlobUrl(pdbText), isBinary: false, label },
        { state: { isGhost: true } },
      );
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      const preset = REPRESENTATION_PRESETS[viewerState.displayMode] || REPRESENTATION_PRESETS.ribbon;
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
        representationPreset: preset,
        representationPresetParams: representationParams(key),
      });
      for (const ref of representationRefs()) {
        if (!beforeRefs.has(ref)) viewerState.representationKeys.set(ref, key);
      }
    }

    function formatRg(value, delta) {
      if (typeof value !== 'number') return 'n/a';
      const deltaText = typeof delta === 'number'
        ? ` (delta ${delta >= 0 ? '+' : ''}${delta.toFixed(6)} A)`
        : '';
      return `${value.toFixed(6)} A${deltaText}`;
    }

    function setToggleStates() {
      for (const button of document.querySelectorAll('[data-structure-toggle]')) {
        const key = button.dataset.structureToggle;
        button.setAttribute('aria-pressed', viewerState.visible[key] ? 'true' : 'false');
      }
      for (const button of document.querySelectorAll('[data-display-mode]')) {
        button.setAttribute(
          'aria-pressed',
          button.dataset.displayMode === viewerState.displayMode ? 'true' : 'false',
        );
      }
    }

    function visibleStructureCount() {
      return Number(viewerState.visible.original) + Number(viewerState.visible.reconstructed);
    }

    function stateCells() {
      const cells = viewerState.viewer?.plugin?.state?.data?.cells;
      if (!cells) return [];
      if (typeof cells.values === 'function') return Array.from(cells.values());
      return Object.values(cells);
    }

    function representationRefs() {
      return new Set(
        stateCells()
          .filter((cell) => cell?.transform?.ref && isRepresentationCell(cell))
          .map((cell) => cell.transform.ref),
      );
    }

    function textForCell(cell, cellByRef, seen = new Set()) {
      if (!cell || seen.has(cell)) return '';
      seen.add(cell);
      const parts = [
        cell.obj?.label,
        cell.obj?.data?.label,
        cell.params?.values?.label,
        cell.transform?.params?.label,
      ];
      const parentRef = cell.transform?.parent || cell.parent?.ref || cell.sourceRef;
      if (parentRef && cellByRef.has(parentRef)) {
        parts.push(textForCell(cellByRef.get(parentRef), cellByRef, seen));
      }
      return parts.filter(Boolean).join(' ');
    }

    function colorKeyForCell(cell, cellByRef) {
      const storedKey = viewerState.representationKeys.get(cell?.transform?.ref);
      if (storedKey) return storedKey;
      const text = textForCell(cell, cellByRef).toLowerCase();
      if (text.includes('reconstructed')) return 'reconstructed';
      if (text.includes('original') || text.includes('initial')) return 'original';
      return null;
    }

    function isRepresentationCell(cell) {
      const transformer = `${cell?.transform?.transformer?.id || cell?.transform?.transformer || ''}`.toLowerCase();
      const type = `${cell?.obj?.type?.name || cell?.obj?.type || ''}`.toLowerCase();
      return transformer.includes('representation') || type.includes('representation');
    }

    function recoloredParams(params, key) {
      const color = VIEWER_COLORS[key].value;
      const next = { ...(params || {}) };
      next.theme = {
        ...(next.theme || {}),
        globalName: 'uniform',
        globalColorParams: {
          ...(next.theme?.globalColorParams || {}),
          value: color,
        },
      };
      next.colorTheme = {
        ...(next.colorTheme || {}),
        name: 'uniform',
        params: {
          ...(next.colorTheme?.params || {}),
          value: color,
        },
      };
      if (next.color !== undefined) next.color = 'uniform';
      if (next.colorParams !== undefined) next.colorParams = { ...(next.colorParams || {}), value: color };
      return next;
    }

    async function recolorCurrentRepresentations(options = {}) {
      const plugin = viewerState.viewer?.plugin;
      const builder = plugin?.build ? plugin.build() : null;
      if (!plugin || !builder) {
        if (!options.silent) statusNode.textContent = 'Mol* state is not ready for recoloring.';
        return 0;
      }
      const cells = stateCells();
      const cellByRef = new Map(cells.map((cell) => [cell?.transform?.ref, cell]).filter(([ref]) => ref));
      let changed = 0;
      for (const cell of cells) {
        if (!cell?.transform?.ref || !isRepresentationCell(cell)) continue;
        const key = colorKeyForCell(cell, cellByRef);
        if (!key) continue;
        builder.to(cell.transform.ref).update((old) => recoloredParams(old, key));
        changed += 1;
      }
      if (changed) await builder.commit();
      if (!options.silent) {
        statusNode.textContent = changed
          ? `Recolored ${changed} Mol* representation${changed === 1 ? '' : 's'}.`
          : 'No recolorable Mol* representations were found.';
      }
      return changed;
    }

    function viewerLabel(key, fallback) {
      const labels = typeof PHEAT_VIEWER_LABELS !== 'undefined' ? PHEAT_VIEWER_LABELS : {};
      return labels[key] || fallback;
    }

    async function loadCase(caseId, options = {}) {
      const resetCamera = options.resetCamera !== false;
      const caseData = roundtripCases.find((entry) => entry.case_id === caseId) || roundtripCases[0];
      if (!caseData || !viewerState.viewer) return;
      setDetails(caseData);
      statusNode.textContent = 'Loading aligned PDBs...';
      releaseObjectUrls();
      await viewerState.viewer.plugin.clear(false);
      viewerState.representationKeys.clear();
      if (visibleStructureCount() === 0) {
        statusNode.textContent = 'Both structures hidden.';
        return;
      }
      if (viewerState.visible.original) {
        await loadPdbStructure(caseData.original_pdb, 'original', viewerLabel('original', 'Original heavy atoms'));
      }
      if (viewerState.visible.reconstructed) {
        await loadPdbStructure(caseData.reconstructed_pdb, 'reconstructed', viewerLabel('reconstructed', 'Reconstructed heavy atoms'));
      }
      if (resetCamera) {
        viewerState.viewer.plugin.managers.camera.reset(undefined, 0);
      }
      await recolorCurrentRepresentations({ silent: true });
      statusNode.textContent = viewerLabel(
        'loaded',
        'Loaded selected structures from embedded PDB data.',
      );
    }

    async function initializeViewer() {
      if (!window.molstar || !roundtripCases.length) {
        statusNode.textContent = 'Mol* assets are unavailable.';
        return;
      }
      viewerState.viewer = await molstar.Viewer.create('molstar-viewer', {
        layoutIsExpanded: false,
        layoutShowControls: true,
        layoutShowRemoteState: false,
        layoutShowSequence: true,
        layoutShowLog: true,
        layoutShowLeftPanel: true,
        viewportShowReset: true,
        viewportShowScreenshotControls: true,
        viewportShowControls: true,
        viewportShowExpand: true,
        viewportShowToggleFullscreen: true,
        viewportShowSettings: true,
        viewportShowSelectionMode: true,
        viewportShowAnimation: true,
        viewportShowTrajectoryControls: true,
        viewportFocusBehavior: 'disabled',
        viewportBackgroundColor: '#ffffff',
        volumeStreamingDisabled: true,
      });
      caseSelect.addEventListener('change', () => loadCase(caseSelect.value, { resetCamera: true }));
      for (const button of document.querySelectorAll('[data-view-case]')) {
        button.addEventListener('click', () => {
          caseSelect.value = button.dataset.viewCase;
          loadCase(button.dataset.viewCase, { resetCamera: true });
        });
      }
      for (const button of document.querySelectorAll('[data-structure-toggle]')) {
        button.addEventListener('click', () => {
          const key = button.dataset.structureToggle;
          viewerState.visible[key] = !viewerState.visible[key];
          setToggleStates();
          loadCase(caseSelect.value, { resetCamera: false });
        });
      }
      for (const button of document.querySelectorAll('[data-display-mode]')) {
        button.addEventListener('click', () => {
          viewerState.displayMode = button.dataset.displayMode;
          setToggleStates();
          loadCase(caseSelect.value, { resetCamera: false });
        });
      }
      const recolorButton = document.querySelector('[data-recolor]');
      if (recolorButton) {
        recolorButton.addEventListener('click', () => recolorCurrentRepresentations());
      }
      setToggleStates();
      await loadCase(caseSelect.value || roundtripCases[0].case_id, { resetCamera: true });
    }
"""

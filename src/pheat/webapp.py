"""Optional FastAPI web app for PDB/mmCIF roundtrip comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import socket
import uuid
from typing import Any, Callable, Mapping, Optional, Sequence

from pheat.mmcif import structure_from_mmcif_string
from pheat.molstar_assets import (
    DEFAULT_MOLSTAR_VERSION,
    molstar_asset_status,
    molstar_missing_assets_message,
    resolve_molstar_assets,
)
from pheat.pdbio import structure_from_pdb_string
from pheat.roundtrip import (
    ANGLE_NAMES,
    SCORE_MODELS,
    RoundtripCaseSpec,
    combinatorial_roundtrip_case_specs,
    configurable_roundtrip_case_spec,
    normalize_max_chi,
    normalize_score_models,
    normalize_stored_angles,
    run_roundtrip_cases,
    single_roundtrip_case_spec,
    write_roundtrip_artifacts,
)
from pheat.report_assets import (
    molstar_alignment_viewer_script,
    pheat_citation_html,
    software_provenance_html,
    sortable_table_css,
)
from pheat.software_provenance import collect_software_provenance


DEFAULT_WEB_WORK_DIR = ".pheat-cache/web"
DEFAULT_MOLSTAR_VENDOR_DIR = None
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000
DEFAULT_WEB_FALLBACK_PORT_START = 8001
DEFAULT_WEB_FALLBACK_PORT_END = 8999
DEFAULT_WEB_SCORE_MODELS = tuple(model for model in SCORE_MODELS if model != "openmm-prepared")
MOLSTAR_PROJECT_URL = "https://molstar.org/"
MOLSTAR_DOI_URL = "https://doi.org/10.1093/nar/gkab314"


def create_app(
    *,
    work_dir: str | Path = DEFAULT_WEB_WORK_DIR,
    molstar_vendor_dir: str | Path | None = DEFAULT_MOLSTAR_VENDOR_DIR,
) -> Any:
    """Create the optional local FastAPI app.

    FastAPI is imported lazily so the dependency-light core package can still be
    installed and imported without web dependencies.
    """

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover - depends on optional web stack
        raise RuntimeError("The web app requires `pip install -e '.[web]'` or `'.[all]'`.") from exc

    # FastAPI resolves postponed route annotations against module globals.
    globals()["Request"] = Request
    work_path = Path(work_dir).resolve()
    molstar_path = _resolve_molstar_vendor_dir(molstar_vendor_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="PHEAT Roundtrip Web App")
    app.state.work_dir = work_path
    app.state.molstar_vendor_dir = molstar_path
    app.mount("/molstar", StaticFiles(directory=str(molstar_path)), name="molstar")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _index_page()

    @app.post("/roundtrip", response_class=HTMLResponse)
    async def roundtrip(request: Request) -> Any:
        form = await request.form()
        pdb_file = form.get("pdb_file")
        if pdb_file is None or not hasattr(pdb_file, "read"):
            return HTMLResponse(
                _error_page("Upload a PDB or mmCIF file before running a comparison."),
                status_code=400,
            )
        mode = str(form.get("mode") or "single")
        include_mmcif = _form_bool(form, "include_mmcif")
        score_models = _form_list(form, "score_models")
        if not score_models:
            score_models = list(DEFAULT_WEB_SCORE_MODELS)
        try:
            pdb_text = await _read_upload_text(pdb_file)
            upload_name = getattr(pdb_file, "filename", "") or "uploaded.pdb"
            structure = _structure_from_upload_text(pdb_text, upload_name)
            if not structure.atoms:
                raise ValueError("Uploaded structure did not contain any heavy atoms.")
            specs = _case_specs_from_form(mode, form)
            normalized_score_models = normalize_score_models(score_models)
            result = run_roundtrip_cases(
                structure,
                specs,
                score_models=normalized_score_models,
            )
            result = dict(result)
            result["software_provenance"] = collect_software_provenance(
                selected_score_models=normalized_score_models,
                selected_features=["web"],
            )
            session_id = uuid.uuid4().hex
            session_dir = work_path / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            input_suffix = Path(upload_name).suffix.lower()
            input_name = "input.cif" if input_suffix in {".cif", ".mmcif"} else "input.pdb"
            (session_dir / input_name).write_text(pdb_text, encoding="utf-8")
            summary = write_roundtrip_artifacts(result, session_dir, include_mmcif=include_mmcif)
            return _results_page(
                result,
                summary=summary,
                session_id=session_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                upload_name=upload_name,
            )
        except Exception as exc:
            return HTMLResponse(_error_page(str(exc)), status_code=400)

    @app.get("/download/{session_id}/{file_path:path}")
    async def download(session_id: str, file_path: str) -> Any:
        session_dir = (work_path / session_id).resolve()
        target = (session_dir / file_path).resolve()
        try:
            target.relative_to(session_dir)
        except ValueError:
            return HTMLResponse("Invalid download path.", status_code=404)
        if not target.is_file():
            return HTMLResponse("Download not found.", status_code=404)
        return FileResponse(target)

    return app


def run_app(
    *,
    host: str = DEFAULT_WEB_HOST,
    port: int = DEFAULT_WEB_PORT,
    work_dir: str | Path = DEFAULT_WEB_WORK_DIR,
    molstar_vendor_dir: str | Path | None = DEFAULT_MOLSTAR_VENDOR_DIR,
    random_port_if_taken: bool = False,
) -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - depends on optional web stack
        raise RuntimeError("Running the web app requires uvicorn from `pheat[web]` or `pheat[all]`.") from exc

    app = create_app(work_dir=work_dir, molstar_vendor_dir=molstar_vendor_dir)
    selected_port = select_web_port(
        host,
        port,
        random_port_if_taken=random_port_if_taken,
    )
    if selected_port != port:
        print(
            f"Port {port} is unavailable; selected fallback port {selected_port}.",
            flush=True,
        )
    bind_message = web_bind_message(host, selected_port)
    if bind_message:
        print(bind_message, flush=True)
    print(f"Open PHEAT web app: {web_launch_url(host, selected_port)}", flush=True)
    uvicorn.run(app, host=host, port=selected_port)


def web_bind_message(host: str, port: int) -> Optional[str]:
    """Return an explanatory bind-address message when launch URL differs."""

    bind_host = str(host or DEFAULT_WEB_HOST).strip()
    normalized_port = _normalize_port(port, label="port")
    if bind_host == "0.0.0.0":
        return f"Serving on 0.0.0.0:{normalized_port} (all IPv4 interfaces)."
    if bind_host == "::":
        return f"Serving on [::]:{normalized_port} (all IPv6 interfaces)."
    return None


def web_launch_url(host: str, port: int) -> str:
    """Return a terminal-friendly browser URL for the local web server."""

    launch_host = str(host or DEFAULT_WEB_HOST).strip()
    if launch_host in {"0.0.0.0", "::"}:
        launch_host = DEFAULT_WEB_HOST
    elif ":" in launch_host and not launch_host.startswith("["):
        launch_host = f"[{launch_host}]"
    return f"http://{launch_host}:{_normalize_port(port, label='port')}/"


def select_web_port(
    host: str,
    preferred_port: int,
    *,
    random_port_if_taken: bool = False,
    fallback_start: int = DEFAULT_WEB_FALLBACK_PORT_START,
    fallback_end: int = DEFAULT_WEB_FALLBACK_PORT_END,
    bind_checker: Optional[Callable[[str, int], bool]] = None,
) -> int:
    """Return a bindable web port, optionally falling back from a busy port."""

    can_bind = bind_checker or _can_bind
    preferred_port = _normalize_port(preferred_port, label="preferred port")
    if can_bind(host, preferred_port):
        return preferred_port
    if not random_port_if_taken:
        return preferred_port

    fallback_start = _normalize_port(fallback_start, label="fallback_start")
    fallback_end = _normalize_port(fallback_end, label="fallback_end")
    if fallback_start > fallback_end:
        raise ValueError("fallback_start must be less than or equal to fallback_end.")

    candidates = [port for port in range(fallback_start, fallback_end + 1) if port != preferred_port]
    random.shuffle(candidates)
    for candidate in candidates:
        if can_bind(host, candidate):
            return candidate
    raise RuntimeError(
        f"No available web port found for {host} in range {fallback_start}-{fallback_end}."
    )


def _can_bind(host: str, port: int) -> bool:
    family = socket.AF_INET6 if _is_ipv6_host(host) else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as handle:
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            handle.bind((host, port))
    except OSError:
        return False
    return True


def _is_ipv6_host(host: str) -> bool:
    candidate = (host or "").strip()
    if not candidate:
        return False
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    return ":" in candidate


def _normalize_port(value: int, *, label: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"{label} must be between 1 and 65535.")
    return port


def _validate_molstar_vendor_dir(path: Path) -> None:
    status = molstar_asset_status(path=path)
    if not status.available:
        raise FileNotFoundError(molstar_missing_assets_message(status))


def _resolve_molstar_vendor_dir(path: str | Path | None) -> Path:
    resolved = resolve_molstar_assets(path=path, version=DEFAULT_MOLSTAR_VERSION, warn=True)
    if resolved is None:
        status = molstar_asset_status(path=path, version=DEFAULT_MOLSTAR_VERSION)
        raise FileNotFoundError(molstar_missing_assets_message(status))
    return resolved.resolve()


async def _read_upload_text(upload: Any) -> str:
    raw = await upload.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Uploaded structure file is larger than the 20 MB web-app limit.")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _structure_from_upload_text(text: str, filename: str) -> Any:
    suffix = Path(filename).suffix.lower()
    stripped = text.lstrip()
    if suffix in {".cif", ".mmcif"} or (
        stripped.startswith("data_") and "_atom_site." in text
    ):
        return structure_from_mmcif_string(text, name=filename)
    return structure_from_pdb_string(text, name=filename)


def _case_specs_from_form(mode: str, form: Any) -> list[RoundtripCaseSpec]:
    normalized = mode.strip().lower()
    angle_units = str(form.get("angle_units") or "radians")
    include_terminal_oxt = _form_bool(form, "include_terminal_oxt")
    if normalized == "single":
        return [
            single_roundtrip_case_spec(
                angle_units=angle_units,
                include_terminal_oxt=include_terminal_oxt,
            )
        ]
    if normalized == "configurable":
        stored_angles = _form_list(form, "stored_angles")
        if len(stored_angles) == 1 and "," in stored_angles[0]:
            stored_angles = list(normalize_stored_angles(stored_angles[0]))
        max_chi = normalize_max_chi(form.get("max_chi"))
        return [
            configurable_roundtrip_case_spec(
                stored_angles=stored_angles,
                max_chi=max_chi,
                angle_units=angle_units,
                include_terminal_oxt=include_terminal_oxt,
            )
        ]
    if normalized == "combinatorial":
        return combinatorial_roundtrip_case_specs(
            angle_units=angle_units,
            include_terminal_oxt=include_terminal_oxt,
        )
    raise ValueError("Unknown workflow mode. Select single, configurable, or combinatorial.")


def _form_list(form: Any, name: str) -> list[str]:
    if hasattr(form, "getlist"):
        values = [str(value) for value in form.getlist(name) if str(value).strip()]
    else:
        value = form.get(name)
        values = [] if value is None else [str(value)]
    if len(values) == 1 and "," in values[0]:
        values = [item.strip() for item in values[0].split(",") if item.strip()]
    return values


def _form_bool(form: Any, name: str) -> bool:
    value = form.get(name)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _index_page() -> str:
    score_checkboxes = "\n".join(
        f'<label><input type="checkbox" name="score_models" value="{html.escape(model)}"'
        f"{' checked' if model in DEFAULT_WEB_SCORE_MODELS else ''}> {html.escape(model)}</label>"
        for model in SCORE_MODELS
    )
    angle_checkboxes = "\n".join(
        f'<label><input type="checkbox" name="stored_angles" value="{html.escape(angle)}"> '
        f"{html.escape(angle)}</label>"
        for angle in ANGLE_NAMES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PHEAT Roundtrip Comparison</title>
  {_style_block()}
</head>
<body>
  <main class="tool">
    <h1>PHEAT Roundtrip Comparison</h1>
    <form action="/roundtrip" method="post" enctype="multipart/form-data" class="upload-form">
      <label class="field">
        <span>PDB or mmCIF file</span>
        <input type="file" name="pdb_file" accept=".pdb,.ent,.cif,.mmcif,text/plain,chemical/x-pdb,chemical/x-mmcif" required>
      </label>
      <label class="field">
        <span>Workflow</span>
        <select name="mode" id="mode-select">
          <option value="single">Single roundtrip</option>
          <option value="configurable">Configurable roundtrip</option>
          <option value="combinatorial">Combinatorial sweep</option>
        </select>
      </label>
      <fieldset class="controls">
        <legend>Configurable options</legend>
        <label class="field">
          <span>Angle units</span>
          <select name="angle_units">
            <option value="radians">radians</option>
            <option value="degrees">degrees</option>
          </select>
        </label>
        <div class="check-group">
          <span>Stored angles</span>
          {angle_checkboxes}
        </div>
        <label class="field">
          <span>Max chi</span>
          <input type="number" name="max_chi" min="0" placeholder="all">
        </label>
        <label class="inline">
          <input type="checkbox" name="include_terminal_oxt"> include terminal OXT
        </label>
        <label class="inline">
          <input type="checkbox" name="include_mmcif"> write mmCIF downloads
        </label>
      </fieldset>
      <fieldset class="controls">
        <legend>Scoring models</legend>
        <div class="check-group scores">{score_checkboxes}</div>
      </fieldset>
      <button type="submit">Run comparison</button>
    </form>
  </main>
</body>
</html>
"""


def _results_page(
    result: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    session_id: str,
    generated_at: str,
    upload_name: str,
) -> str:
    viewer_data = _viewer_data_json(result, summary=summary, session_id=session_id)
    original_rows = "\n".join(
        _original_score_row(model, result["original_scores"][model])
        for model in result["score_models"]
    )
    original_rg_rows = _original_radius_of_gyration_rows(result["original_radius_of_gyration"])
    case_rows = "\n".join(
        _case_row(case, summary_case, session_id=session_id, score_models=result["score_models"])
        for case, summary_case in zip(result["cases"], summary["cases"])
    )
    software_section = software_provenance_html(
        summary.get("software_provenance") or result.get("software_provenance") or {}
    )
    options = "\n".join(_case_option(case) for case in result["cases"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PHEAT Roundtrip Results</title>
  <link rel="stylesheet" href="/molstar/molstar.css">
  {_style_block()}
</head>
<body>
  <main class="results" data-session-id="{html.escape(session_id)}">
    <h1>Roundtrip Comparison</h1>
    <p class="muted">
      Generated at {html.escape(generated_at)} from <code>{html.escape(upload_name)}</code>.
      Parsed {result["original_atom_count"]} heavy atoms across {result["original_residue_count"]} residues.
    </p>
    <p><a href="/download/{html.escape(session_id)}/summary.json">Download summary JSON</a></p>
    <h2>Original Scores</h2>
    <table class="sortable-table">
      <thead><tr><th>Model</th><th>Total</th><th>Units</th><th>Status</th></tr></thead>
      <tbody>{original_rows}</tbody>
    </table>
    <h2>Original Radius Of Gyration</h2>
    <table class="sortable-table">
      <thead><tr><th>Mode</th><th>Rg</th><th>Units</th><th>Atom count</th></tr></thead>
      <tbody>{original_rg_rows}</tbody>
    </table>
    <h2>Roundtrip Cases</h2>
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Case</th><th>Angles</th><th>Chi</th><th>All-heavy RMSD</th>
          <th>Backbone RMSD</th><th>C-alpha RMSD</th>
          <th>Rg unweighted (A)</th><th>Rg mass-weighted (A)</th>
          <th>Downloads</th><th>Scores</th>
        </tr>
      </thead>
      <tbody>{case_rows}</tbody>
    </table>
    {software_section}
    <section class="viewer-section">
      <h2>Mol* Aligned Structures</h2>
      <p class="viewer-citations">
        Citations:
        <a href="{MOLSTAR_PROJECT_URL}" rel="noopener">Mol*</a>;
        Sehnal et al.,
        <a href="{MOLSTAR_DOI_URL}" rel="noopener">Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular structures</a>,
        <em>Nucleic Acids Research</em>, 2021;
        {pheat_citation_html()}
      </p>
      <div class="viewer-controls">
        <label for="case-select">Case</label>
        <select id="case-select">{options}</select>
        <div class="legend" aria-label="Viewer toggles">
          <button type="button" class="legend-toggle" data-structure-toggle="original" aria-pressed="true">
            <span class="legend-swatch legend-swatch-original" aria-hidden="true"></span>
            Original
          </button>
          <button type="button" class="legend-toggle" data-structure-toggle="reconstructed" aria-pressed="true">
            <span class="legend-swatch legend-swatch-reconstructed" aria-hidden="true"></span>
            Reconstructed
          </button>
          <span class="display-mode-group" role="group" aria-label="Display mode">
            <button type="button" class="legend-toggle display-mode-toggle" data-display-mode="ribbon" aria-pressed="true">Ribbon</button>
            <button type="button" class="legend-toggle display-mode-toggle" data-display-mode="all-atom" aria-pressed="false">All atom</button>
          </span>
          <button type="button" class="legend-toggle" data-recolor>Recolor</button>
        </div>
      </div>
      <div class="viewer-layout">
        <div id="molstar-viewer"></div>
        <aside>
          <h3 id="viewer-case-id">Loading...</h3>
          <dl>
            <dt>All-heavy RMSD</dt><dd id="viewer-all-heavy-rmsd"></dd>
            <dt>Backbone RMSD</dt><dd id="viewer-backbone-rmsd"></dd>
            <dt>C-alpha RMSD</dt><dd id="viewer-c-alpha-rmsd"></dd>
            <dt>Rg unweighted</dt><dd id="viewer-rg-unweighted"></dd>
            <dt>Rg mass-weighted</dt><dd id="viewer-rg-mass-weighted"></dd>
            <dt>Matched atoms</dt><dd id="viewer-matched-atoms"></dd>
          </dl>
          <p id="viewer-status" class="muted">Initializing Mol*...</p>
        </aside>
      </div>
    </section>
  </main>
  <script type="application/json" id="roundtrip-viewer-data">{viewer_data}</script>
  <script src="/molstar/molstar.js"></script>
  {_viewer_script()}
</body>
</html>
"""


def _error_page(message: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PHEAT Roundtrip Error</title>
  {_style_block()}
</head>
<body>
  <main class="tool">
    <h1>Roundtrip failed</h1>
    <p class="error">{html.escape(message)}</p>
    <p><a href="/">Return to upload</a></p>
  </main>
</body>
</html>
"""


def _original_score_row(model: str, score: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(model)}</code></td>"
        f"<td>{html.escape(_format_score_total(score))}</td>"
        f"<td>{html.escape(str(score.get('units') or ''))}</td>"
        f"<td>{html.escape(str(score.get('status') or 'unknown'))}</td>"
        "</tr>"
    )


def _original_radius_of_gyration_rows(payload: Mapping[str, Any]) -> str:
    values = payload.get("values", {})
    atom_count = payload.get("atom_count", "")
    rows = []
    for key, label in [("unweighted", "unweighted"), ("mass_weighted", "mass-weighted")]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(_format_rg_value(values.get(key)))}</td>"
            f"<td>{html.escape(str(payload.get('units') or ''))}</td>"
            f"<td>{html.escape(str(atom_count))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _case_row(
    case: Mapping[str, Any],
    summary_case: Mapping[str, Any],
    *,
    session_id: str,
    score_models: Sequence[str],
) -> str:
    download_items = [
        ("geometry", summary_case["paths"]["residue_geometry"]),
        ("heavy", summary_case["paths"]["reconstructed_heavy"]),
        ("original pdb", summary_case["paths"]["original_aligned_pdb"]),
        ("reconstructed pdb", summary_case["paths"]["reconstructed_aligned_pdb"]),
    ]
    if "original_aligned_mmcif" in summary_case["paths"]:
        download_items.extend(
            [
                ("original mmCIF", summary_case["paths"]["original_aligned_mmcif"]),
                ("reconstructed mmCIF", summary_case["paths"]["reconstructed_aligned_mmcif"]),
            ]
        )
    download_items.append(("metrics", summary_case["paths"]["metrics"]))
    downloads = " ".join(
        f'<a href="/download/{html.escape(session_id)}/{html.escape(path)}">'
        f"{html.escape(label)}</a>"
        for label, path in download_items
    )
    scores = "<br>".join(
        f"<code>{html.escape(model)}</code>: "
        f"{html.escape(_format_score_total_or_status(case['scores'][model]['reconstructed']))}"
        for model in score_models
    )
    rg_values = case["radius_of_gyration"]["reconstructed"]["values"]
    rg_delta_values = case["radius_of_gyration"]["delta"]["values"]
    return (
        "<tr>"
        f"<td><button type=\"button\" data-view-case=\"{html.escape(case['case_id'])}\">"
        f"{html.escape(case['case_id'])}</button></td>"
        f"<td>{html.escape(_angles_text(case))}</td>"
        f"<td>{html.escape(_chi_text(case))}</td>"
        f"<td>{case['rmsd']['all_heavy']:.6f} A</td>"
        f"<td>{case['rmsd']['backbone']:.6f} A</td>"
        f"<td>{case['rmsd']['c_alpha']:.6f} A</td>"
        f"<td>{html.escape(_format_rg_with_delta(rg_values.get('unweighted'), rg_delta_values.get('unweighted')))}</td>"
        f"<td>{html.escape(_format_rg_with_delta(rg_values.get('mass_weighted'), rg_delta_values.get('mass_weighted')))}</td>"
        f"<td>{downloads}</td>"
        f"<td>{scores}</td>"
        "</tr>"
    )


def _case_option(case: Mapping[str, Any]) -> str:
    label = f"{case['case_id']} | angles: {_angles_text(case)}; chi: {_chi_text(case)}"
    return f"<option value=\"{html.escape(case['case_id'])}\">{html.escape(label)}</option>"


def _viewer_data_json(
    result: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    session_id: str,
) -> str:
    viewer_cases = []
    for case, summary_case in zip(result["cases"], summary["cases"]):
        viewer_cases.append(
            {
                "case_id": case["case_id"],
                "all_heavy_rmsd": case["rmsd"]["all_heavy"],
                "backbone_rmsd": case["rmsd"]["backbone"],
                "c_alpha_rmsd": case["rmsd"]["c_alpha"],
                "matched_heavy_atoms": case["rmsd"]["matched_heavy_atoms"],
                "matched_backbone_atoms": case["rmsd"]["matched_backbone_atoms"],
                "matched_c_alpha_atoms": case["rmsd"]["matched_c_alpha_atoms"],
                "rg_unweighted_reconstructed": _rg_value(case, "unweighted"),
                "rg_unweighted_delta": _rg_delta_value(case, "unweighted"),
                "rg_mass_weighted_reconstructed": _rg_value(case, "mass_weighted"),
                "rg_mass_weighted_delta": _rg_delta_value(case, "mass_weighted"),
                "original_pdb_path": f"/download/{session_id}/"
                f"{summary_case['paths']['original_aligned_pdb']}",
                "reconstructed_pdb_path": f"/download/{session_id}/"
                f"{summary_case['paths']['reconstructed_aligned_pdb']}",
                "original_pdb": case["artifacts"]["original_aligned_pdb"],
                "reconstructed_pdb": case["artifacts"]["reconstructed_aligned_pdb"],
            }
        )
    return json.dumps(viewer_cases, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")


def _angles_text(case: Mapping[str, Any]) -> str:
    return ", ".join(case["stored_angles"]) or "none"


def _chi_text(case: Mapping[str, Any]) -> str:
    return "all" if case["max_chi"] is None else str(case["max_chi"])


def _format_score_total(score: Mapping[str, Any]) -> str:
    total = score.get("total")
    if score.get("status") != "ok" or total is None:
        return "n/a"
    return f"{float(total):.6f}"


def _format_score_total_or_status(score: Mapping[str, Any]) -> str:
    total = score.get("total")
    if score.get("status") == "ok" and total is not None:
        return f"{float(total):.6f}"
    return str(score.get("status") or "n/a")


def _format_rg_value(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _format_rg_with_delta(value: Any, delta: Any) -> str:
    if value is None:
        return "n/a"
    if delta is None:
        return _format_rg_value(value)
    sign = "+" if float(delta) >= 0.0 else ""
    return f"{float(value):.6f} ({sign}{float(delta):.6f})"


def _rg_value(case: Mapping[str, Any], key: str) -> Optional[float]:
    value = case["radius_of_gyration"]["reconstructed"]["values"].get(key)
    return None if value is None else float(value)


def _rg_delta_value(case: Mapping[str, Any], key: str) -> Optional[float]:
    value = case["radius_of_gyration"]["delta"]["values"].get(key)
    return None if value is None else float(value)


def _style_block() -> str:
    return (
        """<style>
    :root {
      --border: #d7dde5;
      --soft: #f6f8fb;
      --text: #1f2937;
      --muted: #5d6876;
    }
    * { box-sizing: border-box; }
    body {
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 28px;
    }
    h1 { margin: 0 0 18px; }
    h2 { margin: 28px 0 12px; }
    table { border-collapse: collapse; font-size: 13px; width: 100%; }
    th, td { border: 1px solid var(--border); padding: 7px 8px; text-align: left; vertical-align: top; }
    th { background: var(--soft); }
"""
        + sortable_table_css()
        + """
    button, input, select {
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text);
      font: inherit;
      padding: 6px 8px;
    }
    button { background: #ffffff; cursor: pointer; }
    button[type="submit"] { background: #1f2937; color: #ffffff; padding: 8px 12px; }
    a { color: #205fa6; }
    code { background: var(--soft); padding: 1px 3px; }
    .tool, .results { max-width: 1200px; }
    .upload-form { display: grid; gap: 18px; max-width: 760px; }
    .field { display: grid; gap: 6px; }
    .field span, legend, .check-group > span { color: var(--muted); font-size: 13px; font-weight: 600; }
    .controls { border: 1px solid var(--border); border-radius: 6px; display: grid; gap: 12px; padding: 14px; }
    .check-group { display: flex; flex-wrap: wrap; gap: 10px 18px; }
    .inline { display: inline-flex; gap: 8px; }
    .muted { color: var(--muted); }
    .error { color: #b42318; }
    .viewer-section { margin: 28px 0; }
    .viewer-citations { color: var(--muted); font-size: 13px; margin: -4px 0 12px; }
    .viewer-controls { align-items: center; display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
    .legend { align-items: center; display: inline-flex; flex-wrap: wrap; gap: 8px; }
    .legend-toggle {
      align-items: center;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 4px;
      display: inline-flex;
      gap: 6px;
      padding: 4px 7px;
    }
    .legend-toggle[data-structure-toggle][aria-pressed="false"] {
      color: var(--muted);
      opacity: 0.55;
    }
    .display-mode-group {
      align-items: center;
      display: inline-flex;
      gap: 6px;
    }
    .display-mode-toggle {
      justify-content: center;
      min-width: 70px;
    }
    .display-mode-toggle[aria-pressed="true"] {
      background: #eef2f7;
      border-color: #1f2937;
      box-shadow: inset 0 0 0 1px #1f2937;
      color: #111827;
    }
    .legend-swatch {
      border: 1px solid rgba(0, 0, 0, 0.18);
      border-radius: 50%;
      display: inline-block;
      flex: 0 0 auto;
      height: 10px;
      width: 10px;
    }
    .legend-swatch-original { background: #0072B2; }
    .legend-swatch-reconstructed { background: #D55E00; }
    .viewer-layout {
      border: 1px solid var(--border);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      min-height: 560px;
    }
    #molstar-viewer { min-height: 560px; position: relative; }
    aside { background: var(--soft); border-left: 1px solid var(--border); padding: 16px; }
    dt { color: var(--muted); font-size: 12px; font-weight: 600; }
    dd { margin: 0 0 10px; }
    @media (max-width: 900px) {
      body { margin: 16px; }
      .viewer-layout { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--border); }
    }
  </style>"""
    )


def _viewer_script() -> str:
    return (
        """<script>
    const roundtripCases = JSON.parse(document.getElementById('roundtrip-viewer-data').textContent);
    const caseSelect = document.getElementById('case-select');
    const statusNode = document.getElementById('viewer-status');
    const VIEWER_COLORS = {
      original: { hex: '#0072B2', value: 0x0072B2 },
      reconstructed: { hex: '#D55E00', value: 0xD55E00 },
    };
    const REPRESENTATION_PRESETS = {
      ribbon: 'preset-structure-representation-polymer-cartoon',
      'all-atom': 'preset-structure-representation-atomic-detail',
    };
    const PHEAT_VIEWER_LABELS = {
      original: 'Original heavy atoms',
      reconstructed: 'Reconstructed heavy atoms',
      loaded: 'Loaded selected structures from embedded PDB data.',
    };
    const viewerState = {
      viewer: null,
      objectUrls: [],
      visible: { original: true, reconstructed: true },
      displayMode: 'ribbon',
      representationKeys: new Map(),
    };

    function setDetails(caseData) {
      document.getElementById('viewer-case-id').textContent = caseData.case_id;
      document.getElementById('viewer-all-heavy-rmsd').textContent =
        `${caseData.all_heavy_rmsd.toFixed(6)} A`;
      document.getElementById('viewer-backbone-rmsd').textContent =
        `${caseData.backbone_rmsd.toFixed(6)} A`;
      document.getElementById('viewer-c-alpha-rmsd').textContent =
        `${caseData.c_alpha_rmsd.toFixed(6)} A`;
      document.getElementById('viewer-rg-unweighted').textContent =
        formatRg(caseData.rg_unweighted_reconstructed, caseData.rg_unweighted_delta);
      document.getElementById('viewer-rg-mass-weighted').textContent =
        formatRg(caseData.rg_mass_weighted_reconstructed, caseData.rg_mass_weighted_delta);
      document.getElementById('viewer-matched-atoms').textContent =
        `${caseData.matched_heavy_atoms} heavy, ${caseData.matched_backbone_atoms} backbone, `
        + `${caseData.matched_c_alpha_atoms} C-alpha`;
    }
"""
        + molstar_alignment_viewer_script()
        + """
    initializeSortableTables();
    initializeViewer().catch((error) => {
      console.error(error);
      statusNode.textContent = `Mol* failed to load: ${error.message || error}`;
    });
    window.addEventListener('unload', releaseObjectUrls);
  </script>"""
    )

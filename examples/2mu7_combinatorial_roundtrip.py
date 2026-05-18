#!/usr/bin/env python3
"""Run 2MU7 residue-geometry roundtrips across optional angle and chi settings.

Generated artifacts are written under ``examples/roundtrip/2mu7_combinatorial``.
The script reconstructs from the serialized residue-geometry JSON for each case so omitted
optional backbone geometry fields exercise the same behavior as the CLI.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence, Union


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pheat import load_pdb  # noqa: E402
from pheat.geometry_tables import GEOMETRY_MODES  # noqa: E402
from pheat.metrics import RMSD_ATOM_SETS, radius_of_gyration_delta  # noqa: E402
from pheat.models import HeavyAtomStructure  # noqa: E402
from pheat.molstar_assets import (  # noqa: E402
    molstar_asset_status,
    molstar_missing_assets_message,
    resolve_molstar_assets,
)
from pheat.mmcif import structure_to_mmcif_string  # noqa: E402
from pheat.pdbio import write_heavy_json, write_pdb  # noqa: E402
from pheat.report_assets import (  # noqa: E402
    molstar_alignment_viewer_script,
    pheat_citation_html,
    sortable_table_css,
)
from pheat.residue_geometry import (  # noqa: E402
    structure_from_residue_geometry,
    structure_to_residue_geometry,
    write_residue_geometry_json,
)
from pheat.roundtrip import (  # noqa: E402
    ANGLE_NAMES,
    CHI_LIMITS,
    SCORE_MODELS,
    align_reconstructed_to_original,
    radius_of_gyration_payload,
    score_comparison,
    score_payload,
)


INPUT_PDB = REPO_ROOT / "tests" / "fixtures" / "2mu7.pdb"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples" / "roundtrip" / "2mu7_combinatorial"
MOLSTAR_PROJECT_URL = "https://molstar.org/"
MOLSTAR_DOI_URL = "https://doi.org/10.1093/nar/gkab314"
DEFAULT_GEOMETRY_VARIANTS = ("fixed", "ccd-sidechain-geometry-v1")
PACKAGED_CCD_GEOMETRY_TABLE = "ccd-sidechain-geometry-v1"
ANGLE_DEFINITIONS = {
    "omega": "peptide-bond dihedral CA(i)-C(i)-N(i+1)-CA(i+1)",
    "tau": "intra-residue bond angle N(i)-CA(i)-C(i)",
    "theta": "peptide-link bond angle CA(i)-C(i)-N(i+1)",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    return run_roundtrip(
        output_root=args.output_root,
        molstar_vendor_dir=args.molstar_vendor_dir,
        write_mmcif=args.write_mmcif,
        alignment_atom_set=args.alignment_atom_set,
        geometry_variants=_geometry_variants_from_args(args),
    )


def run_roundtrip(
    *,
    output_root: Path,
    molstar_vendor_dir: Optional[Path],
    write_mmcif: bool = False,
    alignment_atom_set: str = "all-heavy",
    geometry_variants: Optional[Sequence[Mapping[str, Any]]] = None,
) -> int:
    output_root = output_root.resolve()
    molstar_vendor_dir = _resolve_molstar_vendor_dir(molstar_vendor_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    subdirs = {
        "residue_geometry": output_root / "residue_geometry",
        "heavy": output_root / "heavy",
        "pdb": output_root / "pdb",
        "metrics": output_root / "metrics",
    }
    if write_mmcif:
        subdirs["mmcif"] = output_root / "mmcif"
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    original = load_pdb(INPUT_PDB)
    original_scores = {model: score_payload(original, model) for model in SCORE_MODELS}
    original_radius_of_gyration = radius_of_gyration_payload(original)
    geometry_variant_payloads = list(geometry_variants or _default_geometry_variants())
    cases = []

    for geometry_variant in geometry_variant_payloads:
        for stored_angles in _angle_subsets():
            for max_chi in CHI_LIMITS:
                case = _run_case(
                    original,
                    original_scores,
                    original_radius_of_gyration,
                    stored_angles=stored_angles,
                    max_chi=max_chi,
                    geometry_variant=geometry_variant,
                    subdirs=subdirs,
                    output_root=output_root,
                    write_mmcif=write_mmcif,
                    alignment_atom_set=alignment_atom_set,
                )
                cases.append(case)
                print(
                    f"{case['case_id']}: all-heavy RMSD={case['rmsd']['all_heavy']:.6f} A, "
                    f"backbone RMSD={case['rmsd']['backbone']:.6f} A, "
                    f"C-alpha RMSD={case['rmsd']['c_alpha']:.6f} A"
                )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_pdb": _relative(INPUT_PDB),
        "output_root": _relative(output_root),
        "alignment_atom_set": alignment_atom_set,
        "reconstruction_geometries": [_summary_geometry_variant(variant) for variant in geometry_variant_payloads],
        "angle_definitions": ANGLE_DEFINITIONS,
        "original_scores": original_scores,
        "original_radius_of_gyration": original_radius_of_gyration,
        "case_count": len(cases),
        "cases": cases,
    }
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "summary.csv", cases)
    _write_report(
        output_root / "report.html",
        summary,
        output_root=output_root,
        molstar_vendor_dir=molstar_vendor_dir,
    )
    print(f"Wrote {len(cases)} roundtrip cases to {_relative(output_root)}")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated roundtrip artifacts.",
    )
    parser.add_argument(
        "--molstar-vendor-dir",
        type=Path,
        default=None,
        help="Optional directory containing molstar.js, molstar.css, and LICENSE. Defaults to PHEAT's Mol* cache.",
    )
    parser.add_argument(
        "--write-mmcif",
        action="store_true",
        help="Also write aligned original/reconstructed structures as mmCIF.",
    )
    parser.add_argument(
        "--alignment-atom-set",
        choices=RMSD_ATOM_SETS,
        default="all-heavy",
        help="Matched atoms used to align written structures and the Mol* viewer. Default: all-heavy.",
    )
    parser.add_argument(
        "--geometry-variants",
        help=(
            "Comma-separated reconstruction geometry variants to include. "
            "Use 'fixed' for template geometry and geometry-table IDs or paths for table mode. "
            f"Default: {','.join(DEFAULT_GEOMETRY_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--geometry-mode",
        choices=GEOMETRY_MODES,
        help="Single custom reconstruction geometry mode. Used only when --geometry-variants is omitted.",
    )
    parser.add_argument(
        "--geometry-table",
        help="Single custom pheat.geometry-table-set JSON or packaged table ID. Used only when --geometry-variants is omitted.",
    )
    parser.add_argument("--geometry-profile", help="Profile ID to read from --geometry-table.")
    return parser.parse_args(argv)


def _validate_molstar_vendor_dir(path: Path) -> None:
    status = molstar_asset_status(path=path)
    if not status.available:
        raise FileNotFoundError(molstar_missing_assets_message(status))


def _resolve_molstar_vendor_dir(path: Optional[Path]) -> Path:
    resolved = resolve_molstar_assets(path=path, warn=True)
    if resolved is None:
        status = molstar_asset_status(path=path)
        raise FileNotFoundError(molstar_missing_assets_message(status))
    return resolved.resolve()


def _geometry_variants_from_args(args: argparse.Namespace) -> Sequence[Mapping[str, Any]]:
    if args.geometry_variants:
        return [_geometry_variant_from_token(token) for token in _comma_values(args.geometry_variants)]
    if args.geometry_mode or args.geometry_table or args.geometry_profile:
        return [
            _geometry_variant(
                "custom",
                mode=args.geometry_mode or ("table" if args.geometry_table else "fixed"),
                table=args.geometry_table,
                profile=args.geometry_profile,
            )
        ]
    return _default_geometry_variants()


def _default_geometry_variants() -> Sequence[Mapping[str, Any]]:
    return [_geometry_variant_from_token(token) for token in DEFAULT_GEOMETRY_VARIANTS]


def _geometry_variant_from_token(token: str) -> Mapping[str, Any]:
    normalized = token.strip()
    if not normalized:
        raise ValueError("--geometry-variants cannot contain empty entries")
    if normalized == "fixed":
        return _geometry_variant("fixed", mode="fixed")
    return _geometry_variant(_geometry_variant_case_id(normalized), mode="table", table=normalized)


def _geometry_variant(
    case_id: str,
    *,
    mode: str,
    table: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
) -> Mapping[str, Any]:
    if mode not in GEOMETRY_MODES:
        raise ValueError(f"Unknown geometry mode {mode!r}")
    return {
        "case_id": _case_token(case_id),
        "label": _geometry_variant_label(case_id, mode=mode, table=table),
        "mode": mode,
        "table": table,
        "profile": profile,
    }


def _summary_geometry_variant(variant: Mapping[str, Any]) -> dict[str, Any]:
    table = variant.get("table")
    return {
        "case_id": variant["case_id"],
        "label": variant["label"],
        "mode": variant["mode"],
        "table": _table_label(table),
        "profile": variant.get("profile"),
    }


def _geometry_variant_case_id(value: str) -> str:
    if value == PACKAGED_CCD_GEOMETRY_TABLE:
        return "ccd-sidechains"
    path = Path(value)
    name = path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return name or value


def _geometry_variant_label(
    case_id: str,
    *,
    mode: str,
    table: Optional[Union[str, Path]],
) -> str:
    if mode == "fixed":
        return "fixed"
    table_label = _table_label(table)
    if table_label == PACKAGED_CCD_GEOMETRY_TABLE or case_id == "ccd-sidechains":
        return "CCD side-chain geometry"
    return table_label or case_id


def _table_label(table: Optional[Union[str, Path]]) -> Optional[str]:
    if table is None:
        return None
    if isinstance(table, Path):
        return _relative(table)
    return table


def _case_token(value: str) -> str:
    token = "".join(character.lower() if character.isalnum() else "-" for character in value)
    token = "-".join(part for part in token.split("-") if part)
    if not token:
        raise ValueError("geometry variant case IDs must contain at least one alphanumeric character")
    return token


def _comma_values(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _run_case(
    original: HeavyAtomStructure,
    original_scores: Mapping[str, Mapping[str, Any]],
    original_radius_of_gyration: Mapping[str, Any],
    *,
    stored_angles: Sequence[str],
    max_chi: Optional[int],
    geometry_variant: Mapping[str, Any],
    subdirs: Mapping[str, Path],
    output_root: Path,
    write_mmcif: bool,
    alignment_atom_set: str,
) -> dict[str, Any]:
    case_id = f"{geometry_variant['case_id']}__{_angles_label(stored_angles)}__{_chi_label(max_chi)}"
    residue_geometry_path = subdirs["residue_geometry"] / f"{case_id}.residue-geometry.json"
    reconstructed_heavy_path = subdirs["heavy"] / f"{case_id}.reconstructed.heavy.json"
    original_pdb_path = subdirs["pdb"] / f"{case_id}.original_aligned.pdb"
    reconstructed_pdb_path = subdirs["pdb"] / f"{case_id}.reconstructed_aligned.pdb"
    metrics_path = subdirs["metrics"] / f"{case_id}.metrics.json"
    original_mmcif_path = subdirs["mmcif"] / f"{case_id}.original_aligned.cif" if write_mmcif else None
    reconstructed_mmcif_path = (
        subdirs["mmcif"] / f"{case_id}.reconstructed_aligned.cif" if write_mmcif else None
    )

    residue_geometry = structure_to_residue_geometry(
        original,
        stored_angles=stored_angles,
        max_chi=max_chi,
    )
    write_residue_geometry_json(residue_geometry, residue_geometry_path)

    reconstructed = structure_from_residue_geometry(
        residue_geometry_path,
        geometry_mode=geometry_variant.get("mode"),
        geometry_table=geometry_variant.get("table"),
        geometry_profile=geometry_variant.get("profile"),
    )
    write_heavy_json(reconstructed, reconstructed_heavy_path)

    alignment = align_reconstructed_to_original(
        original,
        reconstructed,
        alignment_atom_set=alignment_atom_set,
    )
    write_pdb(alignment["original_aligned"], original_pdb_path)
    write_pdb(alignment["reconstructed_aligned"], reconstructed_pdb_path)
    if original_mmcif_path is not None and reconstructed_mmcif_path is not None:
        original_mmcif_path.write_text(
            structure_to_mmcif_string(alignment["original_aligned"]),
            encoding="utf-8",
        )
        reconstructed_mmcif_path.write_text(
            structure_to_mmcif_string(alignment["reconstructed_aligned"]),
            encoding="utf-8",
        )

    scores = score_comparison(
        original_scores,
        alignment["reconstructed_aligned"],
        score_models=SCORE_MODELS,
    )
    reconstructed_radius_of_gyration = radius_of_gyration_payload(alignment["reconstructed_aligned"])
    case = {
        "case_id": case_id,
        "geometry": _summary_geometry_variant(geometry_variant),
        "stored_angles": list(stored_angles),
        "max_chi": max_chi,
        "reconstruction_geometry": reconstructed.metadata.get("reconstruction_geometry"),
        "paths": {
            "residue_geometry": _relative_to_root(residue_geometry_path, output_root),
            "reconstructed_heavy": _relative_to_root(reconstructed_heavy_path, output_root),
            "original_aligned_pdb": _relative_to_root(original_pdb_path, output_root),
            "reconstructed_aligned_pdb": _relative_to_root(reconstructed_pdb_path, output_root),
            "metrics": _relative_to_root(metrics_path, output_root),
        },
        "rmsd": {
            "all_heavy": alignment["all_heavy_rmsd"],
            "backbone": alignment["backbone_rmsd"],
            "c_alpha": alignment["c_alpha_rmsd"],
            "alignment_atom_set": alignment["alignment_atom_set"],
            "matched_heavy_atoms": alignment["matched_heavy_atoms"],
            "matched_backbone_atoms": alignment["matched_backbone_atoms"],
            "matched_c_alpha_atoms": alignment["matched_c_alpha_atoms"],
            "matched_alignment_atoms": alignment["matched_alignment_atoms"],
        },
        "scores": scores,
        "radius_of_gyration": {
            "reconstructed": reconstructed_radius_of_gyration,
            "delta": radius_of_gyration_delta(
                original_radius_of_gyration,
                reconstructed_radius_of_gyration,
            ),
        },
    }
    if original_mmcif_path is not None and reconstructed_mmcif_path is not None:
        case["paths"]["original_aligned_mmcif"] = _relative_to_root(original_mmcif_path, output_root)
        case["paths"]["reconstructed_aligned_mmcif"] = _relative_to_root(
            reconstructed_mmcif_path,
            output_root,
        )
    _write_json(metrics_path, case)
    return case


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    *,
    output_root: Path,
    molstar_vendor_dir: Path,
) -> None:
    case_rows = "\n".join(_case_table_row(case) for case in summary["cases"])
    case_options = "\n".join(_case_option(case) for case in summary["cases"])
    viewer_data = _viewer_data_json(summary["cases"], output_root=output_root)
    original_score_rows = "\n".join(
        _original_score_table_row(model, summary["original_scores"][model])
        for model in SCORE_MODELS
    )
    original_rg_rows = _original_radius_of_gyration_rows(summary["original_radius_of_gyration"])
    angle_definitions = "".join(
        f"<li><code>{html.escape(name)}</code>: {html.escape(definition)}</li>"
        for name, definition in summary["angle_definitions"].items()
    )
    molstar_css = _relative_to_output(molstar_vendor_dir / "molstar.css", output_root)
    molstar_js = _relative_to_output(molstar_vendor_dir / "molstar.js", output_root)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2MU7 Combinatorial Roundtrip</title>
  <link rel="stylesheet" href="{html.escape(molstar_css)}">
  <style>
    :root {{
      color-scheme: light;
      --border: #d8d8d8;
      --soft-border: #e7e7e7;
      --table-head: #f4f4f4;
      --text: #1f252d;
      --muted: #5c6673;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 32px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin: 24px 0; font-size: 13px; }}
    th, td {{ border: 1px solid var(--border); padding: 6px 8px; text-align: right; }}
    .case-table th:first-child, .case-table td:first-child,
    .case-table td:nth-child(2), .case-table td:nth-child(3),
    .score-table th:first-child, .score-table td:first-child, .score-table td:nth-child(4) {{
      text-align: left;
    }}
    th {{ background: var(--table-head); }}
{sortable_table_css()}
    button, select {{
      border: 1px solid var(--border);
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    button {{ cursor: pointer; padding: 3px 7px; }}
    button:hover, select:hover {{ border-color: #aab3bd; }}
    .viewer-section {{ margin: 28px 0; }}
    .viewer-controls {{
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin: 12px 0;
    }}
    .viewer-controls label {{ color: var(--muted); font-size: 13px; font-weight: 600; }}
    .viewer-controls select {{ min-width: 300px; padding: 4px 28px 4px 8px; }}
    .legend {{ align-items: center; display: inline-flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
    .legend-toggle {{
      align-items: center;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 4px;
      display: inline-flex;
      gap: 6px;
      padding: 3px 7px;
    }}
    .legend-toggle[data-structure-toggle][aria-pressed="false"] {{
      color: var(--muted);
      opacity: 0.55;
    }}
    .display-mode-group {{
      align-items: center;
      display: inline-flex;
      gap: 6px;
    }}
    .display-mode-toggle {{
      justify-content: center;
      min-width: 70px;
    }}
    .display-mode-toggle[aria-pressed="true"] {{
      background: #eef2f7;
      border-color: #1f2937;
      box-shadow: inset 0 0 0 1px #1f2937;
      color: #111827;
    }}
    .legend-swatch {{
      border: 1px solid rgba(0, 0, 0, 0.18);
      border-radius: 50%;
      display: inline-block;
      flex: 0 0 auto;
      height: 10px;
      width: 10px;
    }}
    .legend-swatch-original {{ background: #0072B2; }}
    .legend-swatch-reconstructed {{ background: #D55E00; }}
    .viewer-layout {{
      border: 1px solid var(--soft-border);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      min-height: 580px;
    }}
    #molstar-viewer {{ min-height: 580px; position: relative; }}
    .viewer-details {{
      border-left: 1px solid var(--soft-border);
      background: #fafafa;
      padding: 16px;
    }}
    .viewer-details h3 {{ font-size: 16px; margin: 0 0 12px; }}
    .viewer-details dl {{ display: grid; gap: 8px; grid-template-columns: 1fr; margin: 0; }}
    .viewer-details dt {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    .viewer-details dd {{ margin: -4px 0 4px; }}
    .viewer-details a {{ margin-right: 10px; }}
    .viewer-status {{ color: var(--muted); font-size: 13px; margin: 8px 0 0; }}
    .viewer-citations {{ color: var(--muted); font-size: 13px; margin: -4px 0 12px; }}
    .case-button {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    code {{ background: #f4f4f4; padding: 1px 3px; }}
    @media (max-width: 900px) {{
      body {{ margin: 18px; }}
      .viewer-layout {{ grid-template-columns: 1fr; }}
      .viewer-details {{ border-left: 0; border-top: 1px solid var(--soft-border); }}
      .viewer-controls select {{ min-width: min(100%, 300px); }}
    }}
  </style>
</head>
<body>
  <h1>2MU7 Combinatorial Roundtrip</h1>
  <p>
    Generated at {html.escape(summary["generated_at"])} from
    <code>{html.escape(summary["input_pdb"])}</code>. Each case stores residue geometry,
    reconstructs from the serialized residue-geometry JSON, Kabsch-aligns the reconstruction
    to the initial 2MU7 heavy atoms, and reports reconstructed score totals plus RMSDs.
    The geometry variants compare fixed PHEAT template reconstruction with the packaged
    CCD-derived side-chain geometry table. The original all-heavy 2MU7 scores are listed
    once below.
  </p>
  <ul>{angle_definitions}</ul>
  <h2>Original All-Heavy Scores</h2>
  <table class="score-table sortable-table">
    <thead>
      <tr>
        <th>Model</th>
        <th>Total</th>
        <th>Units</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>{original_score_rows}</tbody>
  </table>
  <h2>Original All-Heavy Radius of Gyration</h2>
  <table class="score-table sortable-table">
    <thead>
      <tr>
        <th>Mode</th>
        <th>Rg</th>
        <th>Units</th>
        <th>Atom count</th>
      </tr>
    </thead>
    <tbody>{original_rg_rows}</tbody>
  </table>
	  <h2>Roundtrip Cases</h2>
	  <table class="case-table sortable-table">
    <thead>
      <tr>
        <th>Case</th>
        <th>Geometry</th>
        <th>Angles</th>
        <th>Chi limit</th>
        <th>All-heavy RMSD (A)</th>
        <th>Backbone RMSD (A)</th>
        <th>C-alpha RMSD (A)</th>
        <th>Rg unweighted (A)</th>
        <th>Rg mass-weighted (A)</th>
        <th>generic reconstructed</th>
        <th>pheat-dfire reconstructed</th>
        <th>pheat-goap reconstructed</th>
        <th>heavy-mm reconstructed</th>
        <th>OpenMM reconstructed</th>
      </tr>
    </thead>
	    <tbody>{case_rows}</tbody>
	  </table>
	  <section class="viewer-section" aria-labelledby="viewer-title">
	    <h2 id="viewer-title">Interactive Mol* Alignment Viewer</h2>
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
	      <select id="case-select">{case_options}</select>
		      <div class="legend" aria-label="Viewer toggles">
		        <button type="button" class="legend-toggle" data-structure-toggle="original" aria-pressed="true">
		          <span class="legend-swatch legend-swatch-original" aria-hidden="true"></span>
		          Initial 2MU7 heavy atoms
		        </button>
		        <button type="button" class="legend-toggle" data-structure-toggle="reconstructed" aria-pressed="true">
		          <span class="legend-swatch legend-swatch-reconstructed" aria-hidden="true"></span>
		          Reconstructed heavy atoms
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
	      <aside class="viewer-details" id="viewer-details" aria-live="polite">
	        <h3 id="viewer-case-id">Loading...</h3>
	        <dl>
	          <dt>Stored angles</dt>
	          <dd id="viewer-angles"></dd>
	          <dt>Geometry</dt>
	          <dd id="viewer-geometry"></dd>
	          <dt>Chi limit</dt>
	          <dd id="viewer-chi"></dd>
	          <dt>All-heavy RMSD</dt>
	          <dd id="viewer-all-heavy-rmsd"></dd>
	          <dt>Backbone RMSD</dt>
	          <dd id="viewer-backbone-rmsd"></dd>
	          <dt>C-alpha RMSD</dt>
	          <dd id="viewer-c-alpha-rmsd"></dd>
	          <dt>Rg unweighted</dt>
	          <dd id="viewer-rg-unweighted"></dd>
	          <dt>Rg mass-weighted</dt>
	          <dd id="viewer-rg-mass-weighted"></dd>
	          <dt>Aligned PDBs</dt>
	          <dd>
	            <a id="viewer-original-link" href="">initial</a>
	            <a id="viewer-reconstructed-link" href="">reconstructed</a>
	          </dd>
	          <dt id="viewer-mmcif-label">Aligned mmCIFs</dt>
	          <dd id="viewer-mmcif-links">
	            <a id="viewer-original-mmcif-link" href="">initial</a>
	            <a id="viewer-reconstructed-mmcif-link" href="">reconstructed</a>
	          </dd>
	        </dl>
	        <p class="viewer-status" id="viewer-status">Initializing Mol*...</p>
	      </aside>
	    </div>
	  </section>
	  <script type="application/json" id="roundtrip-viewer-data">{viewer_data}</script>
	  <script src="{html.escape(molstar_js)}"></script>
	  <script>
	    const roundtripCases = JSON.parse(document.getElementById('roundtrip-viewer-data').textContent);
	    const caseSelect = document.getElementById('case-select');
	    const statusNode = document.getElementById('viewer-status');
	    const VIEWER_COLORS = {{
	      original: {{ hex: '#0072B2', value: 0x0072B2 }},
	      reconstructed: {{ hex: '#D55E00', value: 0xD55E00 }},
	    }};
	    const REPRESENTATION_PRESETS = {{
	      ribbon: 'preset-structure-representation-polymer-cartoon',
	      'all-atom': 'preset-structure-representation-atomic-detail',
	    }};
	    const PHEAT_VIEWER_LABELS = {{
	      original: 'Initial 2MU7 heavy atoms',
	      reconstructed: 'Reconstructed heavy atoms',
	      loaded: 'Loaded selected structures from embedded PDB data; no network access is required.',
	    }};
	    const detailNodes = {{
	      caseId: document.getElementById('viewer-case-id'),
	      angles: document.getElementById('viewer-angles'),
	      geometry: document.getElementById('viewer-geometry'),
	      chi: document.getElementById('viewer-chi'),
	      allHeavyRmsd: document.getElementById('viewer-all-heavy-rmsd'),
	      backboneRmsd: document.getElementById('viewer-backbone-rmsd'),
	      cAlphaRmsd: document.getElementById('viewer-c-alpha-rmsd'),
	      rgUnweighted: document.getElementById('viewer-rg-unweighted'),
	      rgMassWeighted: document.getElementById('viewer-rg-mass-weighted'),
	      originalLink: document.getElementById('viewer-original-link'),
	      reconstructedLink: document.getElementById('viewer-reconstructed-link'),
	      mmcifLabel: document.getElementById('viewer-mmcif-label'),
	      mmcifLinks: document.getElementById('viewer-mmcif-links'),
	      originalMmcifLink: document.getElementById('viewer-original-mmcif-link'),
	      reconstructedMmcifLink: document.getElementById('viewer-reconstructed-mmcif-link'),
	    }};
	    const viewerState = {{
	      viewer: null,
	      objectUrls: [],
	      visible: {{ original: true, reconstructed: true }},
	      displayMode: 'ribbon',
	      representationKeys: new Map(),
	    }};

	    function setDetails(caseData) {{
      detailNodes.caseId.textContent = caseData.case_id;
      detailNodes.angles.textContent = caseData.angles_label;
      detailNodes.geometry.textContent = caseData.geometry_label;
      detailNodes.chi.textContent = caseData.chi_label;
      detailNodes.allHeavyRmsd.textContent = `${{caseData.all_heavy_rmsd.toFixed(6)}} A`;
      detailNodes.backboneRmsd.textContent = `${{caseData.backbone_rmsd.toFixed(6)}} A`;
      detailNodes.cAlphaRmsd.textContent = `${{caseData.c_alpha_rmsd.toFixed(6)}} A`;
      detailNodes.rgUnweighted.textContent = formatRg(caseData.rg_unweighted_reconstructed, caseData.rg_unweighted_delta);
      detailNodes.rgMassWeighted.textContent = formatRg(caseData.rg_mass_weighted_reconstructed, caseData.rg_mass_weighted_delta);
      detailNodes.originalLink.href = caseData.original_pdb_path;
      detailNodes.reconstructedLink.href = caseData.reconstructed_pdb_path;
      if (caseData.original_mmcif_path && caseData.reconstructed_mmcif_path) {{
        detailNodes.mmcifLabel.style.display = '';
        detailNodes.mmcifLinks.style.display = '';
        detailNodes.originalMmcifLink.href = caseData.original_mmcif_path;
        detailNodes.reconstructedMmcifLink.href = caseData.reconstructed_mmcif_path;
      }} else {{
        detailNodes.mmcifLabel.style.display = 'none';
        detailNodes.mmcifLinks.style.display = 'none';
      }}
    }}

{molstar_alignment_viewer_script()}

    initializeSortableTables();
	    initializeViewer().catch((error) => {{
	      console.error(error);
	      statusNode.textContent = `Mol* failed to load: ${{error.message || error}}`;
	    }});
	    window.addEventListener('unload', releaseObjectUrls);
	  </script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _original_score_table_row(model: str, score: Mapping[str, Any]) -> str:
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


def _case_table_row(case: Mapping[str, Any]) -> str:
    angles, chi_limit = _case_labels(case)
    score = case["scores"]
    case_id = str(case["case_id"])
    geometry_label = str(case["geometry"]["label"])
    rg_values = case["radius_of_gyration"]["reconstructed"]["values"]
    rg_delta_values = case["radius_of_gyration"]["delta"]["values"]
    return (
        "<tr>"
        f"<td><button class=\"case-button\" type=\"button\" data-view-case=\"{html.escape(case_id)}\">"
        f"{html.escape(case_id)}</button></td>"
        f"<td>{html.escape(geometry_label)}</td>"
        f"<td>{html.escape(angles)}</td>"
        f"<td>{html.escape(chi_limit)}</td>"
        f"<td>{case['rmsd']['all_heavy']:.6f}</td>"
        f"<td>{case['rmsd']['backbone']:.6f}</td>"
        f"<td>{case['rmsd']['c_alpha']:.6f}</td>"
        f"<td>{html.escape(_format_rg_with_delta(rg_values.get('unweighted'), rg_delta_values.get('unweighted')))}</td>"
        f"<td>{html.escape(_format_rg_with_delta(rg_values.get('mass_weighted'), rg_delta_values.get('mass_weighted')))}</td>"
        f"<td>{_format_score_total(score['generic']['reconstructed'])}</td>"
        f"<td>{_format_score_total(score['pheat-dfire']['reconstructed'])}</td>"
        f"<td>{_format_score_total(score['pheat-goap']['reconstructed'])}</td>"
        f"<td>{_format_score_total(score['heavy-mm']['reconstructed'])}</td>"
        f"<td>{html.escape(_format_score_total_or_status(score['openmm-prepared']['reconstructed']))}</td>"
        "</tr>"
    )


def _case_option(case: Mapping[str, Any]) -> str:
    angles, chi_limit = _case_labels(case)
    case_id = str(case["case_id"])
    geometry_label = str(case["geometry"]["label"])
    label = f"{case_id} | geometry: {geometry_label}; angles: {angles}; chi: {chi_limit}"
    return f"<option value=\"{html.escape(case_id)}\">{html.escape(label)}</option>"


def _case_labels(case: Mapping[str, Any]) -> tuple[str, str]:
    angles = ", ".join(case["stored_angles"]) or "none"
    chi_limit = "all" if case["max_chi"] is None else str(case["max_chi"])
    return angles, chi_limit


def _viewer_data_json(cases: Sequence[Mapping[str, Any]], *, output_root: Path) -> str:
    viewer_cases = []
    for case in cases:
        angles, chi_limit = _case_labels(case)
        original_pdb_path = _path_from_output_root(case["paths"]["original_aligned_pdb"], output_root)
        reconstructed_pdb_path = _path_from_output_root(
            case["paths"]["reconstructed_aligned_pdb"],
            output_root,
        )
        original_mmcif_path = (
            _path_from_output_root(case["paths"]["original_aligned_mmcif"], output_root)
            if "original_aligned_mmcif" in case["paths"]
            else None
        )
        reconstructed_mmcif_path = (
            _path_from_output_root(case["paths"]["reconstructed_aligned_mmcif"], output_root)
            if "reconstructed_aligned_mmcif" in case["paths"]
            else None
        )
        original_pdb = original_pdb_path.read_text(encoding="utf-8")
        reconstructed_pdb = reconstructed_pdb_path.read_text(encoding="utf-8")
        viewer_case = {
            "case_id": case["case_id"],
            "geometry_label": case["geometry"]["label"],
            "angles_label": angles,
            "chi_label": chi_limit,
            "all_heavy_rmsd": case["rmsd"]["all_heavy"],
            "backbone_rmsd": case["rmsd"]["backbone"],
            "c_alpha_rmsd": case["rmsd"]["c_alpha"],
            "rg_unweighted_reconstructed": _rg_value(case, "unweighted"),
            "rg_unweighted_delta": _rg_delta_value(case, "unweighted"),
            "rg_mass_weighted_reconstructed": _rg_value(case, "mass_weighted"),
            "rg_mass_weighted_delta": _rg_delta_value(case, "mass_weighted"),
            "original_pdb_path": _relative_to_output(original_pdb_path, output_root),
            "reconstructed_pdb_path": _relative_to_output(reconstructed_pdb_path, output_root),
            "original_pdb": original_pdb,
            "reconstructed_pdb": reconstructed_pdb,
        }
        if original_mmcif_path is not None and reconstructed_mmcif_path is not None:
            viewer_case["original_mmcif_path"] = _relative_to_output(original_mmcif_path, output_root)
            viewer_case["reconstructed_mmcif_path"] = _relative_to_output(
                reconstructed_mmcif_path,
                output_root,
            )
        viewer_cases.append(viewer_case)
    return json.dumps(viewer_cases, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")


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


def _format_rg_value(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _format_rg_with_delta(value: object, delta: object) -> str:
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


def _write_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "geometry",
        "geometry_mode",
        "geometry_table",
        "geometry_profile",
        "stored_angles",
        "max_chi",
        "all_heavy_rmsd",
        "backbone_rmsd",
        "c_alpha_rmsd",
        "matched_heavy_atoms",
        "matched_backbone_atoms",
        "matched_c_alpha_atoms",
        "alignment_atom_set",
        "matched_alignment_atoms",
        "rg_unweighted_reconstructed",
        "rg_unweighted_delta",
        "rg_mass_weighted_reconstructed",
        "rg_mass_weighted_delta",
    ]
    for model in SCORE_MODELS:
        fieldnames.extend(
            [
                f"{model}_reconstructed_total",
                f"{model}_delta",
                f"{model}_units",
                f"{model}_status",
            ]
        )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            row = {
                "case_id": case["case_id"],
                "geometry": case["geometry"]["label"],
                "geometry_mode": case["geometry"]["mode"],
                "geometry_table": case["geometry"]["table"],
                "geometry_profile": case["geometry"]["profile"],
                "stored_angles": ",".join(case["stored_angles"]) or "none",
                "max_chi": "all" if case["max_chi"] is None else case["max_chi"],
                "all_heavy_rmsd": case["rmsd"]["all_heavy"],
                "backbone_rmsd": case["rmsd"]["backbone"],
                "c_alpha_rmsd": case["rmsd"]["c_alpha"],
                "matched_heavy_atoms": case["rmsd"]["matched_heavy_atoms"],
                "matched_backbone_atoms": case["rmsd"]["matched_backbone_atoms"],
                "matched_c_alpha_atoms": case["rmsd"]["matched_c_alpha_atoms"],
                "alignment_atom_set": case["rmsd"]["alignment_atom_set"],
                "matched_alignment_atoms": case["rmsd"]["matched_alignment_atoms"],
                "rg_unweighted_reconstructed": _rg_value(case, "unweighted"),
                "rg_unweighted_delta": _rg_delta_value(case, "unweighted"),
                "rg_mass_weighted_reconstructed": _rg_value(case, "mass_weighted"),
                "rg_mass_weighted_delta": _rg_delta_value(case, "mass_weighted"),
            }
            for model in SCORE_MODELS:
                model_score = case["scores"][model]
                row[f"{model}_reconstructed_total"] = model_score["reconstructed"]["total"]
                row[f"{model}_delta"] = model_score["delta"]
                row[f"{model}_units"] = model_score["units"]
                row[f"{model}_status"] = model_score["reconstructed"]["status"]
            writer.writerow(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _angle_subsets() -> Iterable[tuple[str, ...]]:
    for size in range(len(ANGLE_NAMES) + 1):
        yield from itertools.combinations(ANGLE_NAMES, size)


def _angles_label(stored_angles: Sequence[str]) -> str:
    if not stored_angles:
        return "angles-none"
    if tuple(stored_angles) == ANGLE_NAMES:
        return "angles-all"
    return "angles-" + "_".join(stored_angles)


def _chi_label(max_chi: Optional[int]) -> str:
    return "chi-all" if max_chi is None else f"chi-{max_chi}"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return _relative(path)


def _path_from_output_root(path_text: str, output_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return output_root / path


def _relative_to_output(path: Path, output_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_root.resolve())).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())

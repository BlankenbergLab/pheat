from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

from _common import load_structure, runtime_executable, selected_manifest_rows, write_rows
from pheat import __version__
from pheat.sasa import residue_sasa
from pheat.scoring import score_structure


BACKENDS = ("openmm", "gromacs", "ambertools", "freesasa")
DEFAULT_GROMACS_FORCEFIELD = "amber99sb-ildn"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tiny paper-0 external backend score probes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--gromacs-forcefield", default=DEFAULT_GROMACS_FORCEFIELD)
    parser.add_argument("--external-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    _prepend_runtime_bin_to_path()
    rows = []
    backends = [item.strip().lower() for item in args.backends.split(",") if item.strip()]
    for manifest_row in selected_manifest_rows(args.manifest):
        for backend in backends:
            rows.append(
                _backend_status_row(
                    manifest_row,
                    backend,
                    gromacs_forcefield=args.gromacs_forcefield,
                    external_timeout_seconds=args.external_timeout_seconds,
                )
            )
    write_rows(args.output, rows)
    return 0


def _backend_status_row(
    row: dict,
    backend: str,
    *,
    gromacs_forcefield: str,
    external_timeout_seconds: float,
) -> dict:
    start = time.perf_counter()
    if backend == "openmm" and importlib.util.find_spec("openmm") is None:
        return _row(row, backend, start, status="not_installed", reason="openmm Python package is not installed")
    if backend == "freesasa" and importlib.util.find_spec("freesasa") is None:
        return _row(row, backend, start, status="not_installed", reason="freesasa Python package is not installed")
    if backend == "gromacs" and runtime_executable("gmx") is None:
        return _row(row, backend, start, status="not_installed", reason="gmx executable is not available")
    if backend == "ambertools" and runtime_executable("sander") is None:
        return _row(row, backend, start, status="not_installed", reason="sander executable is not available")
    if backend == "ambertools" and runtime_executable("tleap") is None:
        return _row(row, backend, start, status="not_installed", reason="tleap executable is not available")
    if backend in {"openmm", "gromacs"} and _protein_residue_count(row) < 2:
        return _row(
            row,
            backend,
            start,
            status="unsupported_input",
            reason="tiny single-residue fixture is not a valid terminal-template input for this backend probe",
        )

    try:
        if backend == "freesasa":
            return _run_freesasa(row, backend, start)
        if backend == "openmm":
            return _run_score_structure(
                row,
                backend,
                start,
                model="openmm-prepared",
                external_timeout_seconds=external_timeout_seconds,
            )
        if backend == "ambertools":
            return _run_score_structure(
                row,
                backend,
                start,
                model="ambertools-sander",
                external_timeout_seconds=external_timeout_seconds,
            )
        if backend == "gromacs":
            return _run_score_structure(
                row,
                backend,
                start,
                model="gromacs-mdrun",
                external_timeout_seconds=external_timeout_seconds,
                gromacs_forcefield=gromacs_forcefield,
            )
        return _row(row, backend, start, status="unknown_backend", reason=backend)
    except Exception as exc:
        return _row(row, backend, start, status="error", reason=str(exc))


def _protein_residue_count(row: dict) -> int:
    try:
        return int(row.get("residue_count") or 0)
    except Exception:
        return 0


def _run_freesasa(row: dict, backend: str, start: float) -> dict:
    structure = load_structure(row["source_path"], row.get("source_format"))
    residue_values, used_backend = residue_sasa(structure, backend="freesasa")
    return _row(
        row,
        backend,
        start,
        status="ok",
        score_name="freesasa_total_sasa",
        score_value=sum(float(value) for value in residue_values.values()),
        score_units="A^2",
        metadata={"sasa_backend": used_backend, "residue_count": len(residue_values)},
    )


def _run_score_structure(
    row: dict,
    backend: str,
    start: float,
    *,
    model: str,
    external_timeout_seconds: float,
    gromacs_forcefield: str | None = None,
) -> dict:
    structure = load_structure(row["source_path"], row.get("source_format"))
    result = score_structure(
        structure,
        model=model,
        gromacs_forcefield=gromacs_forcefield or DEFAULT_GROMACS_FORCEFIELD,
        external_timeout_seconds=external_timeout_seconds,
    )
    payload = result.to_dict()
    metadata = {
        "model": payload.get("model"),
        "terms": payload.get("terms"),
        "warnings": payload.get("warnings"),
        "citations": payload.get("citations"),
        "metadata": payload.get("metadata"),
    }
    return _row(
        row,
        backend,
        start,
        status="ok",
        score_name=model,
        score_value=float(result.total),
        score_units=result.units,
        metadata=metadata,
    )


def _row(
    source_row: dict,
    backend: str,
    start: float,
    *,
    status: str,
    reason: str | None = None,
    score_name: str | None = None,
    score_value: float | None = None,
    score_units: str | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "record_id": source_row.get("record_id"),
        "pdb_id": source_row.get("pdb_id"),
        "backend": backend,
        "backend_status": status,
        "score_name": score_name,
        "score_value": score_value,
        "score_units": score_units,
        "score_origin": "external-backend",
        "runtime_seconds": time.perf_counter() - start,
        "failure_reason": reason,
        "backend_metadata": metadata or {},
        "pheat_version": __version__,
    }


def _prepend_runtime_bin_to_path() -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


if __name__ == "__main__":
    raise SystemExit(main())

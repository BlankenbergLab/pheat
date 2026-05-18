# PHEAT Examples

This directory is for user-facing walkthroughs and runnable examples. Package and
CLI example manifests live separately under `src/pheat/data/examples/`.

## Current Examples

- `notebook/2mu7_roundtrip_energy_rmsd_molstar.ipynb`: demonstrates the committed
  `2MU7` heavy-atom to residue geometry to heavy-atom roundtrip, energy comparisons,
  radius-of-gyration comparisons, optional OpenMM-prepared scoring, Kabsch RMSDs,
  and Mol* alignment visualization through `ipymolstar` and MolViewSpec. This is
  the clean source notebook without saved outputs.
- `2mu7_combinatorial_roundtrip.py`: runs the 2MU7 residue-geometry roundtrip for every
  subset of optional stored `omega`, `tau`, and `theta` fields, with chi limits of
  all, 1, and 2, across both fixed PHEAT reconstruction geometry and the packaged
  CCD-derived side-chain geometry table. It writes aligned initial/reconstructed
  PDBs, optional aligned mmCIFs, original all-heavy scores, reconstructed score
  totals, radius-of-gyration comparisons, all-heavy/backbone/C-alpha RMSDs, and an
  HTML report under
  `roundtrip/2mu7_combinatorial/`.
  The default sweep produces 48 cases: 8 optional-angle combinations x 3 chi
  limits x 2 reconstruction geometry variants.
  The report uses PHEAT-managed self-hosted Mol* assets from the runtime cache and
  embedded PDB data for interactive offline alignment visualization; it does not
  load a CDN after the assets are installed. The viewer applies stable semantic
  original/reconstructed colors, can switch between ribbon and all-atom Mol*
  representations, and includes a recolor control that reapplies those initial
  colors without reloading embedded PDB data. Hidden structures dim when toggled
  off, while the selected representation mode is outlined and the other mode
  remains fully clickable.
  Pass `--write-mmcif` to also write aligned mmCIF artifacts.
  Pass `--alignment-atom-set ca` to align written structures and the viewer on
  C-alpha atoms instead of the default all-heavy matched atoms.
  Pass `--geometry-variants fixed` to generate only the fixed-geometry cases, or
  provide comma-separated packaged table IDs/paths to compare additional
  reconstruction geometry tables.
  In the local Miniforge environment, the OpenMM-prepared totals are calculated
  after internal PDBFixer preparation with PHEAT's fixed preparation seed.
- `MOLSTAR_ASSETS.md`: records the Mol* asset source, license, and generation
  details used by the HTML report.
- The local `pheat web` app also uses the PHEAT-managed self-hosted Mol* assets for
  uploaded PDB or mmCIF roundtrip comparisons, with the same structure toggles,
  ribbon/all-atom display modes, and in-place recolor behavior.

Build all generated example artifacts from the repository root:

```bash
make examples
```

Makefile targets assume the conda/development environment is already active. They
use `python`, `jupyter`, and `pheat` from `PATH`; `pheat molstar install` uses
`npm` from `PATH` when Mol* assets need to be downloaded. The targets do not
activate or address a repo-local environment prefix.

Useful individual targets:

```bash
make molstar
make examples-roundtrip
make examples-notebook-executed
make web
make clean-examples
```

`make web` is local-only by default: it binds `127.0.0.1:8000` and automatically
chooses a random available port from `8001-8999` if `8000` is already busy. Use
`make web WEB_HOST=0.0.0.0` to bind all IPv4 interfaces, or
`make web WEB_HOST=<ip-address>` to bind a specific interface. Only use
`0.0.0.0` on trusted networks.

## Running Notebooks

Use the full optional dependency set for notebooks from an already-active
environment. With pip:

```bash
python -m pip install -e ".[all]"
```

Or update the active conda/Miniforge environment from the repository root:

```bash
mamba env update -n "$CONDA_DEFAULT_ENV" -f environment.yml
```

Launch JupyterLab from the same environment used for installation so the server,
browser widget manager, and selected Python kernel all see the same widget packages:

```bash
jupyter lab
```

When the notebook opens, select the kernel for the active environment. If the Mol*
widget cell reports a browser-side widget error, first check the diagnostic cell
immediately above the visualization. It prints the kernel executable and package
versions for `ipymolstar`, `molviewspec`, `anywidget`, `ipywidgets`, and
`jupyterlab_widgets`. Then verify the JupyterLab frontend extension state from
the same environment:

```bash
jupyter labextension list
```

The usual fix is to update the active environment with either
`python -m pip install -e ".[all]"` or `environment.yml` and restart JupyterLab from
that same environment, rather than launching a different global or home-directory
JupyterLab server. `ipymolstar` uses `anywidget`, so it avoids a separate
viewer-specific JupyterLab extension/version pairing.

## Useful Future Examples

- `cli_roundtrip.sh`: CLI-only PDB, atom-structure JSON, residue-geometry JSON, and reconstructed
  PDB walkthrough.
- `schema_validation.py`: validate atom-structure, residue-geometry, centroid, and energy JSON against
  bundled schemas.
- `score_models.py`: compare `generic`, `pheat-dfire`, `pheat-goap`, `heavy-mm`, and optional
  `openmm-prepared` scoring outputs.
- `centroid_reduction.py`: build single and multi side-chain centroid structures.
- `fetch_and_run_examples.sh`: fetch and run packaged example manifests through
  `pheat examples`.

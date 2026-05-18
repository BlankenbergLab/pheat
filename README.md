# PHEAT

**PHEAT** is the **Protein Heavy-atom Energy and Analysis Toolkit**. It is a
library-first Python tool for converting PDB, mmCIF, or BinaryCIF structures to a compact atom
representation, reconstructing structures from residue geometry, building
reduced side-chain centroid models, computing radius-of-gyration metrics, and
computing approximate protein energy scores.

The CLI is intentionally thin: every command delegates to importable backend functions
under `pheat`.

## Environment

The package core is dependency-light and can be imported with the system Python:

```bash
PYTHONPATH=src python3 - <<'PY'
from pheat import structure_from_residue_geometry
print(len(structure_from_residue_geometry({"sequence": "AG"}).atoms))
PY
```

For editable local development without optional scientific or notebook dependencies:

```bash
python -m pip install -e .
```

For the full pip-managed development stack, install the `all` extra:

```bash
python -m pip install -e ".[all]"
python -m pytest
```

The `all` extra is intended to cover the same feature and test surface as the
Miniforge environment where pip has a satisfiable package set: scientific
conversion/scoring dependencies, BinaryCIF input through PHEAT/msgpack/numpy,
OpenMM/PDBFixer, executable-backed scoring hooks for `ambertools-sander` and `gromacs-mdrun`,
training/table-generation dependencies, JupyterLab/Mol* widget
support, the optional FastAPI web app, tests, and linting tools. On Python 3.9, the pip
extras skip OpenMM/PDBFixer because current pip packages do not provide a
compatible `pdbfixer`/`openmm>=8.2` set; use Python 3.10+ or the Miniforge
environment for the `openmm-prepared` path. AmberTools and GROMACS executables are
available through the conda environment, not pip extras.

Common local checks are available through `make`:

```bash
make test
make coverage
make coverage-html
make lint
make typecheck
make package
make check
```

`make coverage` writes terminal coverage and `coverage.xml`. HTML coverage is
generated only when requested with `make coverage-html`. `make package` builds the
source distribution and wheel so packaged schemas and manifests can be checked
outside an editable install.

## Documentation

The public documentation is built with MkDocs, Material for MkDocs, and
mkdocstrings. Its styling is matched to the Blankenberg Lab Astro site while
remaining self-contained in this repository.

Preview the docs locally with:

```bash
make docs-deps
make docs-serve
```

Build the static site with `make docs`, which runs `mkdocs build --strict`.
Direct `mkdocs serve` also works when the docs requirements are installed.

The Cloudflare Workers static-assets deployment uses the generated `site/`
directory. Configure the Cloudflare build with:

```text
Build command:
python -m pip install --upgrade pip && python -m pip install -r docs/requirements.txt && python -m mkdocs build --strict

Deploy command:
npx wrangler@4 deploy
```

The production documentation URL is:

https://pheat.tools.blankenberglab.org/

For conda/Miniforge development, activate the environment first and then update it
from the repository root:

```bash
mamba env update -n "$CONDA_DEFAULT_ENV" -f environment.yml
python -m pytest
```

The Makefile assumes the development environment is already active and uses
`python`, `jupyter`, and `npm` from `PATH`. The conda environment installs the
package as `-e ".[all]"` after resolving the scientific stack from conda-forge,
including OpenMM/PDBFixer support used by the deterministic `openmm-prepared`
development scoring path.

Example reports, executed notebooks, and Mol* browser assets are generated artifacts
rather than committed files:

```bash
make examples
```

Run these targets from an already-active development environment. The environment
includes `nodejs`, and `make molstar` wraps `pheat molstar install` to download
the pinned Mol* viewer bundle into PHEAT's platform-aware runtime cache. Use
`pheat molstar status` to inspect the active asset location.

## Local Web App

The optional web app accepts a PDB or mmCIF upload and runs the same residue-geometry
roundtrip comparison used by the backend tests and 2MU7 report. Install the web
extra, build local Mol* assets, and start the app:

```bash
python -m pip install -e ".[web]"
make molstar
pheat web
```

For the full development environment, `pip install -e ".[all]"` or
`environment.yml` includes the web dependencies. `make web` installs the pinned
local Mol* assets in PHEAT's runtime cache and starts the app with the active environment's `python`. By
default it binds local-only on `127.0.0.1`, tries `http://127.0.0.1:8000/` first,
and chooses a random available port in `8001-8999` if `8000` is already in use.
Override the preferred port with `make web WEB_PORT=8888`.

Bind a different interface explicitly when needed:

```bash
pheat web --host 127.0.0.1
pheat web --host 0.0.0.0
make web WEB_HOST=0.0.0.0
make web WEB_HOST=192.168.1.25
```

`0.0.0.0` means all IPv4 interfaces and can make the app reachable from other
computers on the network; use it only on trusted networks. The server prints an
`Open PHEAT web app:` URL that most terminals render as a clickable browser link.

The first screen is the upload tool. It has a workflow selector for a single
default roundtrip, one configurable roundtrip, or the combinatorial omega/tau/theta
and chi-limit sweep. Built-in deterministic scorers are selected by default, and
the optional OpenMM-prepared scorer can be selected explicitly. Results include
original score totals, reconstructed score totals, Kabsch RMSDs, radius-of-gyration
comparisons, aligned original/reconstructed PDB downloads, optional mmCIF downloads,
residue-geometry JSON, reconstructed atom-structure JSON, metrics JSON, and an embedded
Mol* alignment viewer. Generated
upload artifacts are written under `.pheat-cache/web/`. The viewer uses Mol*
semantic original/reconstructed coloring, includes quick toggles for each
structure, can switch between ribbon and all-atom Mol* representations, and has a
recolor control that reapplies the initial colors without reloading the PDB data.
Hidden structures dim when toggled off; the selected representation mode is
outlined while the other mode remains fully clickable.

## CLI Examples

```bash
pheat pdb-to-structure input.pdb -o structure.json
pheat pdb-to-structure input.pdb -o all-atom-structure.json --hydrogens preserve
pheat pdb-to-structure input.pdb -o bonded-structure.json --store-bonds all
pheat mmcif-to-structure input.cif -o structure.json
pheat bcif-to-structure input.bcif.gz -o structure.json
pheat structure-to-pdb structure.json -o roundtrip.pdb
pheat structure-to-mmcif structure.json -o roundtrip.cif
pheat structure-to-geometry structure.json -o residue-geometry.json
pheat pdb-to-geometry input.pdb -o residue-geometry.json
pheat mmcif-to-geometry input.cif -o residue-geometry.json
pheat bcif-to-geometry input.bcif.gz -o residue-geometry.json
pheat pdb-to-geometry input.pdb -o residue-geometry-degrees.json --angle-units degrees
pheat pdb-to-geometry input.pdb -o residue-geometry-chi1.json --max-chi 1
pheat pdb-to-geometry input.pdb -o residue-geometry-full.json --store-angles all
pheat pdb-to-geometry input.pdb -o residue-geometry-lengths.json --store-lengths all
pheat structure-to-geometry structure.json -o residue-geometry-full.json --store-angles omega,tau,theta
pheat geometry-to-structure residue-geometry.json -o structure.json --pdb-output rebuilt.pdb
pheat geometry-to-structure residue-geometry.json -o all-atom-structure.json --hydrogens generate
pheat geometry-to-structure residue-geometry.json -o structure.json --mmcif-output rebuilt.cif
pheat geometry-to-structure residue-geometry.json -o structure.json --geometry-table geometry-tables.json
pheat geometry-to-structure residue-geometry.json -o structure.json --geometry-table ccd-sidechain-geometry-v1
pheat geometry-to-structure residue-geometry.json -o structure.json --include-terminal-oxt
pheat geometry-to-structure residue-geometry-degrees.json -o structure.json --angle-units degrees
pheat score input.pdb --model generic
pheat score input.pdb --model pheat-dfire --table-set packaged:protein-heavy-30id-xray-aqueous-v0
pheat training tables describe --table-set packaged:protein-heavy-30id-xray-aqueous-v0
pheat scoring validate-options --model gromacs-mdrun
pheat gromacs validate-options --gromacs-forcefield amber19sb
pheat radius-of-gyration input.pdb
pheat rg structure.json --mode unweighted
pheat rg structure.json --atom-set ca
pheat rmsd original.pdb reconstructed.pdb
pheat rmsd original.pdb reconstructed.pdb --atom-set ca --alignment-atom-set ca
pheat examples list
pheat sources list
pheat sources fetch wwpdb-ccd-full --destination .pheat-cache/sources/ccd/full
pheat sources fetch rcsb-ccd-bcif --destination .pheat-cache/sources/ccd/bcif
pheat archive download --ids-file ids.txt --dry-run
pheat archive snapshots list
pheat archive snapshots ids rcsb-current-bcif --output-root .pheat-cache/pdb-archive/rcsb-current-bcif
pheat training corpus inventory --snapshot-root .pheat-cache/pdb-archive/rcsb-current-bcif -o .pheat-cache/training/inventory.jsonl
pheat training corpus select --inventory .pheat-cache/training/inventory.jsonl --output-root .pheat-cache/training/sets/protein-heavy-30id --corpus-id protein-heavy-30id --corpus-version v1
pheat training corpus describe --training-set .pheat-cache/training/sets/protein-heavy-30id
pheat training tables build --training-set .pheat-cache/training/sets/protein-heavy-30id --output-root .pheat-cache/training/tables/protein-heavy --burial-method both --table-set-id protein-heavy-30id --table-set-version v1
pheat training tables describe --table-set .pheat-cache/training/tables/protein-heavy/score-tables.json
pheat training tables build --training-set .pheat-cache/training/sets/protein-heavy-30id --output-root .pheat-cache/training/tables/protein-heavy-contacts --burial-method contacts --table-set-id protein-heavy-30id-contacts --table-set-version v1
pheat training tables build --training-set .pheat-cache/training/sets/protein-heavy-30id --output-root .pheat-cache/training/tables/protein-heavy-sasa --burial-method sasa --table-set-id protein-heavy-30id-sasa --table-set-version v1
pheat reference run-unattended --reference-root .pheat-cache/reference-builds --snapshot-root .pheat-cache/pdb-archive/rcsb-current-bcif --workers auto --overwrite
pheat reference build-decoys --training-set .pheat-cache/reference-builds/sets/protein-heavy-30id-xray-aqueous-v0 --output-root .pheat-cache/reference-builds/decoys/protein-heavy-30id-xray-aqueous-v0-pheat-torsion --recipes pheat-torsion-v1
pheat reference build-scores --training-set .pheat-cache/reference-builds/sets/protein-heavy-30id-xray-aqueous-v0 --output-root .pheat-cache/reference-builds/tables/protein-heavy-30id-xray-aqueous-v0
pheat reference extract-features --training-set .pheat-cache/reference-builds/sets/protein-heavy-30id-xray-aqueous-v0 --decoys .pheat-cache/reference-builds/decoys/protein-heavy-30id-xray-aqueous-v0-pheat-torsion/decoys.jsonl -o .pheat-cache/reference-builds/features/v0/aqueous-features.jsonl
pheat reference train-ml --features .pheat-cache/reference-builds/features/v0/aqueous-features.jsonl -o .pheat-cache/reference-builds/models/v0/pheat-ml-linear-aqueous.json
pheat reference package-scoring-assets --reference-root .pheat-cache/reference-builds --artifact-version v0 --destination-root src/pheat/data/scoring/v0 --overwrite
pheat geometry tables list
pheat geometry tables build-backbone --training-set .pheat-cache/training/sets/protein-heavy-30id --output-root .pheat-cache/training/geometry/protein-heavy-30id-backbone
pheat geometry tables build-cdl --training-set .pheat-cache/training/sets/protein-heavy-30id --output-root .pheat-cache/training/geometry/protein-heavy-30id-cdl --phi-psi-bin-size 10 --min-bin-count 20
pheat geometry tables import-cdl --input cdl-like-table.json --output-root .pheat-cache/training/geometry/imported-cdl
pheat geometry tables build-sidechain-ccd --ccd-full .pheat-cache/sources/ccd/full/components.cif.gz --output-root .pheat-cache/training/geometry/ccd-sidechains
pheat geometry tables validate --table-set .pheat-cache/training/geometry/protein-heavy-30id-backbone/geometry-tables.json
pheat web
```

CLI commands print immediate human-readable status to `stderr`, including the
PHEAT version, command path, and key inputs/outputs. Structured command results
remain on `stdout` so JSON output can still be piped to other tools. Use
`--quiet` before the subcommand to suppress terminal startup/progress status,
`--log FILE` to append timestamped status/progress/error lines to a text log,
and `-v` or `-vv` for more diagnostics:

```bash
pheat --quiet archive snapshots list
pheat --log pheat-download.log archive snapshots download rcsb-current-bcif -y
pheat -vv archive download --ids-file ids.txt --dry-run
pheat --version
```

## PDB Archive Corpus Utility

`pheat archive download` builds local RCSB/wwPDB coordinate corpora with
provenance manifests. The default output root is ignored local cache space:

```bash
pheat archive download --ids-file ids.txt --dry-run
pheat archive download --ids-file ids.txt --yes
pheat archive download --ids-file ids.txt --yes --no-progress-redraw
pheat archive download --all-current --format cif --max-entries 100 --dry-run
pheat archive snapshots list
pheat archive snapshots describe wwpdb-current-mmcif
pheat archive snapshots download wwpdb-current-mmcif --max-entries 100 --dry-run
pheat archive snapshots download rcsb-current-bcif --prefetch-metadata --metadata-source rcsb-api --yes
pheat archive snapshots verify wwpdb-current-mmcif
pheat archive snapshots relocate rcsb-current-bcif --output-root .pheat-cache/pdb-archive/rcsb-current-bcif --write
pheat archive snapshots metadata rcsb-current-bcif --output-root .pheat-cache/pdb-archive/rcsb-current-bcif
```

By default it writes under `.pheat-cache/pdb-archive/`:

- `raw/` for coordinate files.
- `processed/`, `analysis/`, and `failed/` for later corpus-building stages.
- `manifests/` for `ids.txt`, `filters.json`, `files.jsonl`, and `api-schemas.json`.

The command can reuse existing coordinate files from another directory without
redownloading them:

```bash
pheat archive download --ids-file ids.txt \
  --reuse-raw-dir /data/pdb/mmcif \
  --dry-run
```

By default, reused files are copied into the new archive so the snapshot is
self-contained. Use `--reuse-mode reference` only when a manifest that points at
another raw-file location is acceptable.

Schema/API provenance is recorded without storing remote schema bodies. The
`api-schemas.json` manifest records the inspected RCSB schema URL, retrieval
timestamp, content SHA-256, content length, HTTP cache headers when present,
embedded version/license metadata when present, and `stored: false`. PHEAT uses
direct HTTP JSON requests for this utility; it does not depend on the external
`rcsb-api` package. Coordinate files are written atomically and local SHA-256
checksums are recorded in `files.jsonl`.

Archive downloads report processed files, failures, elapsed time, average
files/sec, average downloaded bytes/sec, and estimated time remaining. Interactive
terminals use a dynamic redraw line by default; redirected output, CI logs, and
`--no-progress-redraw` use append-only progress lines. `--progress-interval`
controls file-count updates, and `--progress-seconds` controls elapsed-time
updates. `--log FILE` always records append-only timestamped lines even when the
terminal display uses redraw.

Snapshot presets are named download plans for common public coordinate archives.
PHEAT currently includes current-holdings presets for gzipped mmCIF
(`wwpdb-current-mmcif`), gzipped legacy PDB (`wwpdb-current-pdb`), and gzipped
BinaryCIF (`rcsb-current-bcif`). Snapshot downloads use per-snapshot default
roots under `.pheat-cache/pdb-archive/`, record the snapshot ID in
`filters.json`, and can be verified later against the SHA-256 checksums in
`files.jsonl`. The `rcsb-current-bcif` files can be consumed directly by PHEAT
when the scientific or all extras are installed. BinaryCIF coordinate input uses
PHEAT's native atom-site decoder and label asym IDs for chain identifiers.
If a snapshot directory is moved, `pheat archive snapshots relocate` resolves
files by basename under the local `raw/` directory and can rewrite
`files.jsonl`/`filters.json` with relative paths. `pheat archive snapshots
metadata` writes compact normalized entry metadata for downstream training and
reference selection. Snapshot downloads can also use `--prefetch-metadata` to
populate that metadata cache immediately after coordinate files are downloaded;
this is the preferred one-pass archival mode for snapshots intended to be reused
for reference builds. Metadata extraction reports batch progress to stderr and
the existing global `--log FILE` captures the same progress lines for later
auditing.

For a reusable BinaryCIF snapshot on network-attached storage while downloading
through local staging space in the current directory:

```bash
pheat archive snapshots download rcsb-current-bcif \
  --output-root /path/to/pheat-archive/rcsb-current-bcif \
  --staging-dir ./pheat-bcif-staging \
  --cleanup-staging \
  -y
```

This downloads pending files into `./pheat-bcif-staging`, computes SHA-256,
promotes each verified file into `raw/` under the snapshot root, verifies the
promoted copy, records the final paths and checksums in `manifests/files.jsonl`,
and removes successfully promoted staged files. Existing final files are skipped
when they match a prior manifest checksum; stale files that disagree with the
manifest are downloaded again. Verify the reusable snapshot later with:

```bash
pheat archive snapshots verify rcsb-current-bcif \
  --output-root /path/to/pheat-archive/rcsb-current-bcif
```

## Training Score Tables

PHEAT's training commands default to all-heavy protein scoring (`--domain
protein-heavy`) and to the reusable `rcsb-current-bcif` snapshot. Broader
heavy-atom domains are opt-in with `--domain all-heavy` or `--domain full` for
models that support them. Full-corpus table outputs are not bundled in this pass;
the commands below create reproducible artifacts under local cache or a
user-selected output root.

```bash
make training-snapshot-ids
make training-decoys
make training-inventory
make training-select
make training-tables
make training-tables-contacts
make training-tables-sasa
make training-validate
make training-validate-contacts
make training-validate-sasa
```

The corresponding CLI commands are available without Make:

```bash
pheat training decoys list
pheat training decoys fetch 3drobot --output-root .pheat-cache/training/decoys --yes
pheat training corpus inventory \
  --snapshot-root .pheat-cache/pdb-archive/rcsb-current-bcif \
  --domain protein-heavy \
  -o .pheat-cache/training/inventory.jsonl
pheat training corpus select \
  --inventory .pheat-cache/training/inventory.jsonl \
  --output-root .pheat-cache/training/sets/protein-heavy-30id \
  --sequence-identity 0.30 \
  --corpus-id protein-heavy-30id \
  --corpus-version v1
pheat training corpus describe \
  --training-set .pheat-cache/training/sets/protein-heavy-30id
pheat training tables build \
  --training-set .pheat-cache/training/sets/protein-heavy-30id \
  --output-root .pheat-cache/training/tables/protein-heavy \
  --models pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg \
  --burial-method both \
  --sasa-backend auto \
  --table-set-id protein-heavy-30id \
  --table-set-version v1
pheat training tables describe \
  --table-set .pheat-cache/training/tables/protein-heavy/score-tables.json
pheat training tables build \
  --training-set .pheat-cache/training/sets/protein-heavy-30id \
  --output-root .pheat-cache/training/tables/protein-heavy-contacts \
  --models pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg \
  --burial-method contacts \
  --table-set-id protein-heavy-30id-contacts \
  --table-set-version v1
pheat training tables build \
  --training-set .pheat-cache/training/sets/protein-heavy-30id \
  --output-root .pheat-cache/training/tables/protein-heavy-sasa \
  --models pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg \
  --burial-method sasa \
  --sasa-backend auto \
  --table-set-id protein-heavy-30id-sasa \
  --table-set-version v1
```

`pheat training tables build` writes `score-tables.json` with
`format: "pheat.score-table-set"`. The table format is profile-native:
`--burial-method both` writes both `protein-heavy-30id-contacts` and
`protein-heavy-30id-sasa` into one file, and single-method builds write one profile
with the same shape. Score both profiles side by side with:

```bash
pheat score input.pdb \
  --model pheat-hydropathy \
  --table-set .pheat-cache/training/tables/protein-heavy/score-tables.json \
  --profiles protein-heavy-30id-contacts,protein-heavy-30id-sasa
```

SASA builds use `--sasa-backend auto`, which uses MIT-licensed FreeSASA.
Contact builds remain dependency-light and are useful for systems where SASA
packages are unavailable.

The source tree includes compressed JSON.xz initial `v0` score assets under
`src/pheat/data/scoring/v0` so installed packages can exercise trained scoring
without an external reference-build archive. Use them with `packaged:<id>`:
`protein-heavy-30id-xray-aqueous-v0`, `protein-heavy-30id-xray-membrane-v0`,
`pheat-ml-linear-aqueous-v0`, or `pheat-ml-linear-membrane-v0`. The packaged
assets are about 1.7 MiB total and are functional testing artifacts, not final
scientific defaults.

`pheat training corpus select` can use a local RCSB-style sequence-cluster file
with `--sequence-clusters`; otherwise it falls back to deterministic internal
sequence-identity clustering. `--sequence-identity` accepts fractions or
percent-style values such as `0.30`, `30`, or `30%`; generated labels use the
safe `30id`/`12p5id` form for paths and profile IDs.
Generated corpus and score-table JSON separate schema `version` from artifact
identity. Corpus manifests include `artifact_id`, `artifact_version`, a safe
`artifact_label`, selected-entry checksums, source snapshot provenance when the
inventory came from a PHEAT archive snapshot, and optional `--include-file`,
`--exclude-file`, and `--holdout-file` member-list checksums. Score-table sets
record the source corpus artifact and entry checksum; generated profile IDs use
the corpus ID plus burial method, for example `protein-heavy-30id-contacts`.
Create a new corpus or table artifact version whenever the PDB snapshot,
selection filters, inclusion/exclusion lists, clustering threshold, scoring
definitions, or PHEAT build changes in a way that should remain distinguishable.

## Reference And ML Builds

The `pheat reference` commands wrap the full local reference-build workflow used
to derive future PHEAT-owned scoring assets. Defaults are deliberately strict for
the current protein-heavy build: the reusable `rcsb-current-bcif` snapshot,
X-ray structures only (`--method x-ray`), 30% sequence-identity clustering
(`--sequence-identity 0.30`), maximum resolution 2.5 A, and artifact version
`v0`. This initial `v0` label is intentional while the first packaged scoring
assets are functional but still awaiting broader manual and scientific review.

```bash
make reference-unattended \
  SNAPSHOT_ROOT=.pheat-cache/pdb-archive/rcsb-current-bcif \
  REFERENCE_ROOT=.pheat-cache/reference-builds \
  REFERENCE_WORKERS=auto \
  REFERENCE_OVERWRITE=1
```

The unattended target runs the complete `pheat reference run-unattended` workflow:
input registration, snapshot metadata extraction, inventory, aqueous/membrane
selection, canary decoy/feature validation, full decoy generation, score-table
builds, feature extraction, ML fitting, and validation. It defaults to artifact
version `v0`, writes stage logs under `$(REFERENCE_ROOT)/runs/v0/logs`, and can
move previous outputs into `backups/<timestamp>/` with
`REFERENCE_BACKUP_EXISTING=1`.

The currently bundled initial `v0` assets were packaged from a completed local
reference run. That run selected 21,721 aqueous and 2,373 membrane X-ray
protein-heavy chains, accepted 5,454 aqueous and 554 membrane torsion-space
decoys, and extracted 27,175 aqueous and 2,927 membrane native/decoy feature
rows. Snapshot metadata had three missing RCSB metadata records (`4M4C`, `9KZM`,
and `9MBW`). Large source snapshots, decoys, feature JSONL files, and logs remain
external archive artifacts; only the packageable score/model JSON is committed as
compressed `.json.xz` payloads with compressed and uncompressed SHA-256 checksums
recorded in the packaged scoring manifest.
Use `pheat reference audit-version --reference-root .pheat-cache/reference-builds
--artifact-version v0` or `make reference-audit` to check that active manifests
and packageable outputs consistently use the expected artifact version. See
[docs/reference-build-v0.md](docs/reference-build-v0.md) for the current build
settings and comparison against the previous comparable run.

Individual stages remain available for debugging:

```bash
make reference-fetch
make reference-metadata
make reference-inventory
make reference-select-aqueous
make reference-select-membrane
make reference-decoys
make reference-scores
make reference-features
make reference-ml-linear
make reference-validate
```

`reference-fetch` records the selected coordinate snapshot and decoy benchmark
metadata. External benchmark payloads such as 3DRobot, CASP, I-TASSER, and
Rosetta decoy files are treated as local-use-only unless their license is
reviewed; PHEAT records source URLs, command settings, SHA-256 checksums, byte
counts, and registration/download dates, but does not redistribute unclear
license payloads. Local payloads can be registered with `--local-file
DATASET=PATH` or `--local-dir DATASET=PATH`; direct downloads require
`--include-payloads --payload-url DATASET=URL`.

`reference-metadata` writes compact normalized snapshot metadata to
`$(SNAPSHOT_ROOT)/manifests/metadata.jsonl` before inventory. It records method,
resolution in Angstroms, deposition/revision dates, X-ray refinement and
validation summaries, composition counts, protein entity/chain sequence
metadata, RCSB sequence-cluster IDs when available, and normalized
aqueous/membrane/computed-model flags. The default source is `auto`: PHEAT uses
the RCSB Data API when available and falls back to local BinaryCIF metadata for
offline snapshots. Metadata manifests record source URLs, build dates, checksums,
and the fields intentionally omitted by default, such as coordinates, raw API
payloads, full citations, full crystallization text, and per-residue validation
details.

`reference-inventory` builds a JSONL inventory from a local snapshot using
multiple workers (`--workers auto` by default) and automatically consumes the
snapshot metadata file when present. `reference-select` writes a training-corpus
manifest plus `selected.jsonl`, `holdout.jsonl`, split files, and an audit
report. The aqueous subset keeps non-membrane entries and warns about entries
without explicit solvent metadata during this prototype phase; the membrane
subset only keeps entries with membrane annotations. RCSB sequence-cluster
metadata is used for the selected sequence-identity threshold when present;
otherwise PHEAT falls back to deterministic internal sequence comparison. Both
subsets keep relative, relocatable paths where possible and record input
manifest checksums.

`reference build-decoys` creates PHEAT-owned deterministic decoys from selected
native chains. The default `pheat-torsion-v1` profile perturbs PHEAT
residue-geometry degrees of freedom, reconstructs heavy atoms, aligns decoys to
their native chain for inspection, and records all-heavy RMSD, C-alpha RMSD,
radius-of-gyration ratio, geometry-integrity score, acceptance status, seed, and
SHA-256 for every accepted or rejected candidate. Older coordinate-noise recipes
remain available as smoke-test recipes but are not the default for reference
training. `reference build-scores` builds native PHEAT score tables, including
contact and optional SASA profiles. `reference
extract-features`, `reference train-ml`, and `reference validate` create
feature rows, a lightweight `pheat-ml-linear` baseline, and native-vs-decoy
separation summaries. For provisional ML experiments, `reference
extract-features --max-entries N` samples the first `N` selected native entries
and matching decoys while recording that cap in the feature manifest. Use
`reference package-scoring-assets` to refresh bundled compressed score/model
assets from a completed reference build. Use `reference promote` with a review note to copy a packageable generated artifact
into a reviewed destination; manifests containing local-use-only or
unpackageable payload metadata are blocked from promotion.

## Source Data Licensing

PHEAT source code is licensed under MIT. The wwPDB/RCSB Chemical Component
Dictionary definition CIFs consulted for modified residue templates are PDB
archive data files made available under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)
according to the [RCSB PDB usage policy](https://www.rcsb.org/pages/usage-policy).
The consulted files are the CCD definition CIFs for
[`SEC`](https://files.rcsb.org/ligands/download/SEC.cif),
[`PYL`](https://files.rcsb.org/ligands/download/PYL.cif),
[`MSE`](https://files.rcsb.org/ligands/download/MSE.cif),
[`HYP`](https://files.rcsb.org/ligands/download/HYP.cif),
[`LYZ`](https://files.rcsb.org/ligands/download/LYZ.cif),
[`SEP`](https://files.rcsb.org/ligands/download/SEP.cif),
[`TPO`](https://files.rcsb.org/ligands/download/TPO.cif),
[`PTR`](https://files.rcsb.org/ligands/download/PTR.cif), and
[`PCA`](https://files.rcsb.org/ligands/download/PCA.cif), plus the
[wwPDB CCD documentation](https://wwpdb-beta.rcsb.org/data/ccd) and
[RCSB download documentation](https://www.rcsb.org/docs/programmatic-access/file-download-services).
Those CCD files are not vendored, redistributed, or packaged with PHEAT; they
were used as reference data for component IDs, parent relationships, atom names,
connectivity, and rounded idealized residue templates. Use of those references
does not imply wwPDB/RCSB endorsement.

PHEAT can also fetch local CCD caches for user-generated geometry tables:
`wwpdb-ccd-full` downloads the full `components.cif.gz` CCD file, and
`rcsb-ccd-bcif` downloads the compact `cca.bcif`/`ccb.bcif` atom and bond
subsets. The full CCD mmCIF file is the preferred source for deriving
side-chain reconstruction geometry because it contains ideal/model-coordinate
and bond-distance fields used to compute lengths and angles. The compact
BinaryCIF subsets are useful for lightweight atom/bond connectivity validation,
but they do not replace the full CCD geometry fields. Fetch commands write a
`pheat-source-provenance.json` file with URLs, timestamps, SHA-256 checksums,
file sizes, license metadata, and PHEAT version.

Published conformation-dependent library (CDL) references are documented as
non-downloadable literature/source references. PHEAT's current context-dependent
backbone tables are generated from selected local corpora; they are not the
official Phenix/CCTBX CDL tables.

Mass-weighted radius of gyration uses a compact built-in representative atomic-mass
table for common PDB heavy elements. The values are derived from CIAAW Standard
Atomic Weights 2024 and cross-checked against the NIST Atomic Weights and Isotopic
Compositions reference database. CIAAW/IUPAC website content is copyright-marked
with attribution conditions for republication and commercial-use restrictions;
NIST notes that Standard Reference Data and other NIST works can carry different
copyright/licensing terms. PHEAT treats both atomic-weight references as citation-only
inputs for this purpose: no CIAAW or NIST atomic-weight pages or data files are
downloaded, vendored, redistributed, or packaged. `pheat sources list` records
those entries as reference-only, and `pheat sources fetch` refuses to fetch them.

RCSB Search API schema documents used by the archive corpus utility are not
vendored or packaged. When inspected, PHEAT records only provenance such as the
URL, retrieval timestamp, content SHA-256, and embedded metadata. The Search API
OpenAPI document declares Apache 2.0 in its own `info.license` field; RCSB API
data and PDB archive data remain governed by the RCSB usage policy and its CC0
statement plus external-resource caveats.

Snapshot metadata extraction uses the RCSB Data API GraphQL endpoint for compact
entry/entity/validation/cluster fields when network access is available. PHEAT
stores normalized metadata rows and provenance, not raw RCSB API responses or
downloaded API schemas.

PHEAT cites Miyazawa-Jernigan contact potentials, Kyte-Doolittle hydropathy,
FreeSASA, Zhang Lab decoy datasets, and CASP download areas as method or
benchmark references. The implementation does not vendor original MJ tables,
external hydropathy data files, decoy payloads, or CASP payloads. The generated
`pheat-mj` and other trained score-table outputs are PHEAT-owned artifacts built
from user-selected corpora and record their own provenance in
`pheat.score-table-set` metadata.

## Backend Examples

```python
from pheat import (
    filter_structure_for_domain,
    kabsch_align,
    kabsch_rmsd,
    load_mmcif,
    load_pdb,
    residue_angle_specs,
    score_model_option_specs,
    score_structure,
    validate_external_scoring_options,
    validate_scoring_options,
    write_pdb,
)
from pheat.metrics import structure_radius_of_gyration, structure_rmsd

structure = load_pdb("input.pdb")
mmcif_structure = load_mmcif("input.cif")
protein_heavy, coverage = filter_structure_for_domain(structure, domain="protein-heavy")
write_pdb(structure, "protein-heavy.pdb", domain="protein-heavy")
result = score_structure(structure, model="generic")
external_check = validate_external_scoring_options(model="gromacs-mdrun")
generic_options = validate_scoring_options("generic", {"domain": "protein-heavy"})
angle_specs = residue_angle_specs("MAG", stored_angles="omega")
print(result.total)
print(external_check["ok"])
print(generic_options["ok"])
print(coverage["scored_atom_count"], len(protein_heavy.atoms))
print(angle_specs[0]["angle_name"])
print(score_model_option_specs("gromacs-mdrun")[0]["name"])
print(len(mmcif_structure.atoms))
print(structure_radius_of_gyration(structure)["values"])
print(structure_rmsd(structure, structure)["value"])
coords = [atom.coord for atom in structure.atoms]
aligned = kabsch_align(coords, coords)
print(kabsch_rmsd(coords, coords, aligned_target=aligned))
```

`pheat rmsd` and `structure_rmsd` default to all matched heavy atoms. Use
`--atom-set ca` for C-alpha-only RMSD; this matches atom name `CA`, not calcium
element records. `--alignment-atom-set` controls which matched atoms define the
Kabsch superposition, so callers can align on C-alpha atoms and measure
all-heavy RMSD, or align and measure on the same atom set.

## Notebook Example

`examples/notebook/2mu7_roundtrip_energy_rmsd_molstar.ipynb` demonstrates the committed
`2MU7` heavy-atom to residue geometry to heavy-atom roundtrip. It computes energy comparisons,
radius-of-gyration comparisons, optional OpenMM-prepared scores, all-heavy,
backbone, and C-alpha Kabsch RMSDs, and a Mol* alignment visualization through `ipymolstar`.
Run `make examples-notebook-executed` to create an executed copy under
`examples/notebook/executed/`.
`examples/2mu7_combinatorial_roundtrip.py` runs the same 2MU7 roundtrip across every
subset of stored `omega`, `tau`, and `theta` fields, with chi limits of all, 1, and
2, across both fixed PHEAT reconstruction geometry and the packaged
CCD-derived side-chain geometry table. It writes aligned initial/reconstructed
PDBs, optional aligned mmCIFs, energy comparisons, radius-of-gyration comparisons, RMSDs,
`summary.json`, `summary.csv`, and `report.html` under
`examples/roundtrip/2mu7_combinatorial/`.
The default sweep produces 48 cases: 8 optional-angle combinations x 3 chi
limits x 2 reconstruction geometry variants.
The HTML report lists the original all-heavy scores once and reports reconstructed
score totals for each roundtrip case. It also embeds the aligned PDB pairs into an
interactive Mol* viewer loaded from PHEAT-managed local assets installed by
`pheat molstar install`, so the report can be opened directly from disk without a
CDN or runtime network dependency after `make examples`. The viewer uses semantic original/reconstructed
coloring, can switch between ribbon and all-atom Mol* representations, and
includes a recolor control that reapplies the initial colors without reloading
the embedded PDB data. Hidden structures dim when toggled off; the selected
representation mode is outlined while the other mode remains fully clickable.
Pass `--write-mmcif` to the example script when aligned mmCIF artifacts should be
written alongside the default PDB artifacts. Pass `--geometry-variants fixed` to
generate only the fixed-geometry cases, or provide comma-separated packaged table
IDs/paths to compare additional reconstruction geometry tables.
Use either `pip install -e ".[all]"` or the active conda environment from
`environment.yml` for JupyterLab and Mol* notebook widget support. Both install
paths include `ipymolstar` and `molviewspec`, which provide a Mol* `anywidget`
Jupyter viewer for local molecular data.

## Scientific Scope

The current implementation distinguishes between production plumbing and approximate
scoring. PDB, mmCIF, and BinaryCIF parsing writes canonical `atom-structure` JSON.
By default PHEAT drops hydrogens and records the dropped count so artifacts stay
heavy-atom compact. Use `--hydrogens preserve` to keep source H/D/T atoms, or
`--hydrogens generate` on supported workflows to add hydrogens through the optional
OpenMM path. The JSON `atom_scope` field reports whether an artifact is `heavy` or
`all`.

Optional top-level bond storage is off by default. Use `--store-bonds declared`,
`--store-bonds template`, or `--store-bonds all` to include zero-based atom-index
bond records with coordinate-measured Angstrom lengths. Declared bonds come from
source connectivity such as PDB `CONECT` and mmCIF `struct_conn`; template bonds use
PHEAT's supported protein/CCD residue templates. PHEAT does not infer generic bonds
by distance in this pass.

Atom-structure JSON preserves heterogens, record metadata, and explicit disulfide
connectivity from `SSBOND`, CYS `SG`-to-`SG` `CONECT` records, or mmCIF
`struct_conn` disulfide annotations. Disulfides are preserved as
connectivity annotations only: PHEAT does not infer them from sulfur distance and
does not fit sulfur atoms to disulfide geometry during residue-geometry reconstruction.
Atom-structure JSON can be converted back to
PDB or mmCIF, extracted to best-effort residue-geometry JSON, or reconstructed from
residue-geometry JSON into atom-structure JSON plus optional PDB or mmCIF output.
Residue Geometry JSON uses radians by default; pass
`--angle-units degrees` for degree-valued residue-geometry input or output. Optional backbone
geometry storage is compact by default; pass `--store-angles omega,tau,theta` or
`--store-angles all` when exporting residue-geometry JSON if those fields should be stored.
Pass `--store-lengths all`, `backbone`, `sidechain`, or explicit `ATOM-ATOM` keys
to store per-residue measured bond lengths in Angstroms; reconstruction uses stored
lengths before geometry tables or built-in defaults.
Pass `--max-chi N` to keep only the first N side-chain chi angles per residue;
`max_chi=0` suppresses chi angles, `max_chi=1` keeps only chi1, and the default
has no chi limit.
The Python API function `residue_angle_specs(...)` reports the PHEAT residue-angle
fields available for a sequence, including phi/psi, residue-template chi angles,
and optional omega/tau/theta fields. It returns PHEAT-native metadata such as
`residue_index`, `residue_name`, `angle_name`, `category`, `applies_to`, and
`required_atoms`; it does not expose optimizer-specific aliases. The optional
`selective_chi_map` argument can restrict named chi angles by residue, and
`max_chi` is then applied as a numeric ceiling.
Residue-geometry extraction and reconstruction supports all 20 canonical amino
acids plus `SEC`, `PYL`, `MSE`, `HYP`, `LYZ`, `SEP`, `TPO`, `PTR`, and `PCA`.
Hydroxylysine uses the wwPDB Chemical Component Dictionary code `LYZ`; `HYL` is
accepted as an input alias and normalizes to `LYZ`. One-letter shorthand is
available for `SEC` (`U`) and `PYL` (`O`); other modified residues require their
three-letter CCD names. Ring templates are closed for canonical `PRO`, `PHE`,
`TYR`, `HIS`, and `TRP`, and for modified `HYP`, `PCA`, and `PYL`. Modified
residue side-chain templates are idealized CCD/PDB-name-compatible heavy-atom
reconstructions, not rotamer-library or force-field minimization.
Modified residues are reconstructable, but remain outside canonical residue-specific
statistical terms; generic and heavy-mm paths use element-level terms where available.

### Scoring Models

PHEAT includes deterministic built-in scorers for testing pipelines and comparative
experiments, plus optional OpenMM, AmberTools, and GROMACS-backed paths. Compare original vs reconstructed
scores within the same model; do not compare absolute totals across different
models because their scales and terms are different. In the Python API,
`supported_models()` lists every recognized model ID, while `available_models()`
lists only models runnable in the active environment. `model_capabilities()`
reports the same distinction with optional dependency details; for example,
`openmm-prepared` is supported everywhere but available only when OpenMM can be
imported, and `ambertools-sander` is available only when `tleap` and `sander`
are on `PATH`; `gromacs-mdrun` is available only when `gmx` is on `PATH`.
Capability records and every energy-result metadata payload include an
`implementation` block that states whether the model is native PHEAT code,
an optional Python backend, or an external executable backend.
Use `score_model_option_specs(model)` to inspect accepted scorer options and
`validate_scoring_options(model, options, require_executables=False)` to validate
API option dictionaries without running scoring or requiring external executables.
The `pheat-geometry-integrity` scorer reports its diagnostic tolerances, per-term
weights, Huber delta, and cis-or-trans planarity target in result metadata.

| Model | Implementation | What it computes | Units | Main caveat |
| --- | --- | --- | --- | --- |
| `generic` | Native PHEAT | Element-based steric clash and short-range contact score for broad PDB/mmCIF coverage. | arbitrary | Smoke-test score, not a physical or statistical potential. |
| `pheat-dfire` | Native PHEAT | PHEAT canonical-residue distance-contact heuristic plus side-chain burial, inspired by DFIRE. | arbitrary | Does not use the original DFIRE parameter table or reference-state calculation. |
| `pheat-goap` | Native PHEAT | `pheat-dfire` base score plus a PHEAT residue-orientation heuristic from CA-CB or N-CA vectors, inspired by GOAP. | arbitrary | Does not use the original GOAP parameter tables. |
| `pheat-mj` | Native PHEAT | PHEAT-generated Miyazawa-Jernigan-style residue contact score. | arbitrary | Valid for supported protein residues; original MJ parameter tables are not redistributed. |
| `pheat-hydropathy` | Native PHEAT | Kyte-Doolittle hydropathy/burial compatibility score using contact density or optional SASA. | arbitrary | SASA scoring requires a SASA backend; contact-density scoring is an approximation. |
| `pheat-backbone` | Native PHEAT | Backbone torsion plausibility from extracted phi/psi/omega geometry. | arbitrary | Requires ordered protein backbone atoms. |
| `pheat-rotamer` | Native PHEAT | Side-chain chi/rotamer plausibility by residue type. | arbitrary | Gly/Ala have no side-chain rotamer term; incomplete side chains score partially. |
| `pheat-hbond` | Native PHEAT | Heavy-atom donor/acceptor contact geometry and buried-polar term. | arbitrary | Protonation is inferred from heavy atoms and remains ambiguous. |
| `pheat-rg` | Native PHEAT | Expected-radius-of-gyration compactness penalty. Defaults to C-alpha, unweighted Rg with placeholder coefficients. | arbitrary | Shape score only; fit coefficients from an in-domain corpus before interpreting as a calibrated potential. |
| `pheat-ml-linear` | Native PHEAT | Lightweight linear combination of PHEAT score features. | arbitrary | Only meaningful with a trained table set from an in-domain corpus. |
| `pheat-coarse-protein-folding-v1` | Native PHEAT | Coarse folding objective with end-to-end compactness, hydrophobic burial, contact, decoded torsion, aromatic, disulfide, steric, and geometry-integrity terms. | arbitrary | Heuristic lower-is-better objective for staged folding/reranking; not a physical free energy or trained statistical potential. |
| `pheat-geometry-integrity` | Native PHEAT | Robust coordinate-geometry plausibility score for backbone bonds, peptide C-N links, C-alpha chirality, peptide planarity, and proline ring closure. | arbitrary | Geometry-quality diagnostic only; missing atoms are skipped with warnings and the score is not a thermodynamic energy. |
| `heavy-mm` | Native PHEAT | Heavy-atom Lennard-Jones-like, simple charge, and backbone bond-length penalty terms. | arbitrary | Heavy-atoms-only approximation, not AMBER/OpenMM force-field energy. |
| `openmm-prepared` | External Python backend | OpenMM AMBER potential after internal OpenMM/PDBFixer preparation. | kJ/mol | Optional dependency path; requires OpenMM to run, uses PDBFixer when available, and may add hydrogens and missing terminal/heavy atoms internally for scoring without modifying input artifacts. |
| `ambertools-sander` | External executable backend | AmberTools `tleap` plus `sander` single-point AMBER molecular mechanics energy after preparation. | kcal/mol | Requires AmberTools executables and a parameterizable prepared protein; not a folding free energy. |
| `gromacs-mdrun` | External executable backend | GROMACS `pdb2gmx`, `grompp`, `mdrun -rerun`, and `energy` validation/reranking energy after topology preparation. | kJ/mol | Requires the `gmx` executable and a parameterizable protein; defaults to `amber19sb`, unsolvated rerun scoring, and is not a folding free energy. |

The built-in `generic`, `pheat-dfire`, `pheat-goap`, `pheat-mj`,
`pheat-hydropathy`, `pheat-backbone`, `pheat-rotamer`, `pheat-hbond`,
`pheat-rg`, `pheat-ml-linear`, `pheat-coarse-protein-folding-v1`,
`pheat-geometry-integrity`, and `heavy-mm` result metadata labels
their scale as arbitrary unless an exact external parameter source is added and
verified. The `pheat-dfire` score is generated from PHEAT's built-in
hydrophobicity, element-contact, coarse distance-bin, and side-chain burial
constants; `pheat-goap` adds a local orientation-vector term. Original DFIRE
and GOAP papers are cited as method inspiration only. `pheat-rg` currently uses
the placeholder form `expected_rg = a * residue_count ** b` and reports the
squared standardized deviation from that expectation; table sets can override
`atom_set`, `mode`, `a`, `b`, and `sigma_fraction` once fitted coefficients are
available.
`pheat-coarse-protein-folding-v1` accepts optional decoded torsion angles in
radians from the Python API as `decoded_torsions={"0_phi": -1.0, "1_chi1": 0.5}`
or from the CLI with `--decoded-torsions torsions.json`, where the file is a JSON
object keyed by zero-based residue index and angle name. Non-numeric or
non-finite torsion values are ignored and counted in result metadata.

Scoring defaults to `--domain protein-heavy`, which ignores waters, ions,
ligands, nucleic acids, and hydrogens for PHEAT's protein-oriented scores.
Use `--domain all-heavy` or `--domain full` explicitly for broader heavy-atom
experiments. The same domain names are available from Python for explicit
structure filtering and PDB serialization: `protein-heavy` writes supported
protein heavy atoms, `all-heavy` keeps nonprotein heavy atoms, and `full` keeps
all atoms already present in the PHEAT structure object. Every energy-result
metadata payload reports the selected domain and atom/residue coverage. It also
reports an `input_contract` for the selected score model: the expected structure
type, accepted atom scopes, compatible domains, required atom families, hydrogen
handling, table usage, burial dependence, and whether the scorer operates
directly on coordinates, derived torsions, feature vectors, or an internally
prepared force-field system. Torsion or residue-geometry workflows should
reconstruct an atom structure first and score that coordinate structure unless a
future torsion-native scorer explicitly declares a different contract.

PHEAT's internal `chain_id` field is a string and can preserve full mmCIF chain
identifiers in atom-structure JSON, residue-geometry JSON, and mmCIF output. Legacy
PDB files have a one-character chain ID column. For that reason, direct PDB output
rejects chain IDs longer than one character unless `--allow-pdb-chain-truncation`
is selected; prefer mmCIF output when preserving full author or label chain IDs
matters. mmCIF input uses author chain/residue IDs by default and can read label
IDs with `--chain-id-source label`.

Radius-of-gyration calculations are geometric summary metrics, not energy terms.
Unweighted Rg measures the root-mean-square distance of supplied heavy atoms from
their coordinate centroid. Mass-weighted Rg uses the same coordinates with a
center of mass and mass-weighted squared distances. Unknown elements fall back to
carbon mass and are reported in the JSON payload's `unknown_elements` list.
Rg accepts the same atom-set names as RMSD: `all-heavy` by default, `backbone`,
or `ca`. The `ca` atom set matches atom name `CA`, not calcium element records,
and is useful for backbone-trace compactness.

## Residue Geometry JSON

Residue-geometry files are versioned with `format: "pheat.residue-geometry-structure"` and carry
`angle_units` as a required top-level field. Supported values are `radians` and
`degrees`; radians are emitted by default. Dihedrals are stored as conventional
signed torsion angles, so trans peptide `omega` values are near `+/-180` degrees
rather than near zero. Per-residue `chi` arrays are ordered as
`[chi1, chi2, ...]`, recorded by the top-level `chi_order: "chi1_to_chiN"` field.
When exporting residue geometry, `--max-chi N` truncates each residue's `chi` array to the
first N entries in that order; omitting it stores every extractable chi angle.
For supported modified residues, chi arrays follow the same template order and may
include template-specific torsions for the modification, such as phosphate or
pyrrolysine extension atoms.
Per-residue `omega`, `tau`, and `theta` are optional stored fields:

- `omega`: peptide-bond dihedral `CA(i)-C(i)-N(i+1)-CA(i+1)`.
- `tau`: intra-residue bond angle `N(i)-CA(i)-C(i)`.
- `theta`: peptide-link bond angle `CA(i)-C(i)-N(i+1)`.

When those fields are absent during reconstruction, PHEAT falls back to its idealized
backbone geometry constants.

Reconstruction uses the fixed Engh-Huber-style geometry profile by default. An
opt-in `pheat.geometry-table-set` can provide replacement reconstruction targets:

```bash
pheat geometry tables list
pheat geometry-to-structure residue-geometry.json \
  -o structure.json \
  --geometry-table ccd-sidechain-geometry-v1
pheat geometry tables build-backbone \
  --training-set .pheat-cache/training/sets/protein-heavy-30id \
  --output-root .pheat-cache/training/geometry/protein-heavy-30id-backbone
pheat geometry tables build-cdl \
  --training-set .pheat-cache/training/sets/protein-heavy-30id \
  --output-root .pheat-cache/training/geometry/protein-heavy-30id-cdl \
  --phi-psi-bin-size 10 \
  --min-bin-count 20
pheat geometry tables import-cdl \
  --input cdl-like-table.json \
  --output-root .pheat-cache/training/geometry/imported-cdl
pheat geometry-to-structure residue-geometry.json \
  -o structure.json \
  --geometry-table .pheat-cache/training/geometry/protein-heavy-30id-backbone/geometry-tables.json
```

Backbone geometry tables are PHEAT-owned artifacts generated from a selected local
corpus and record source corpus checksums, filters, PHEAT version, and command
arguments. They store default/residue-level bond targets and phi/psi-binned
tau/theta targets; table-mode reconstruction uses those binned targets only when
the residue supplies phi and psi and tau/theta were not stored explicitly.
`build-cdl` creates a PHEAT-generated conformation-dependent backbone profile
from the same selected local corpus. It bins residue phi/psi space, records
backbone bond-length and bond-angle targets, and can group observations as
`gly-pro-general`, `canonical`, or `per-residue`; stored per-residue bond lengths
and stored tau/theta still take precedence during reconstruction. The builder
does not vendor the official Phenix/CCTBX CDL tables. `import-cdl` accepts a
JSON CDL-like bin table and writes a normal PHEAT geometry-table-set while
recording the input path, SHA-256 checksum, and optional source-license string.
The `--smoothing kernel` option is recorded for generated table provenance, but
current runtime lookup uses the nearest matching phi/psi bin. CCD side-chain
geometry tables can be generated from the full wwPDB CCD
`components.cif.gz` file or from per-component CCD CIF files with
`pheat geometry tables build-sidechain-ccd`; the current builder uses PHEAT's
placement order and fills or validates bond lengths, angles, and element symbols
from CCD bond/model-coordinate data. The compact CCD BinaryCIF atom/bond subsets
are accepted as a connectivity-only input and warn that PHEAT template geometry
defaults are being used. PHEAT bundles the small derived
`ccd-sidechain-geometry-v1` table as packaged runtime data under
`src/pheat/data/geometry`; raw CCD source files remain external cache/archive
artifacts and are not packaged.

Top-level `disulfide_bonds` entries preserve explicit CYS-CYS connectivity across
atom-structure and residue-geometry JSON. They do not add chi values or disulfide-specific
torsions; cysteine still stores its normal `chi1` side-chain angle.

These optional backbone fields are coupled. Storing only `omega` can make a
roundtrip RMSD worse than using the ideal trans fallback because the real peptide
twist is then applied inside an otherwise idealized `tau`/`theta` frame. In the
committed 2MU7 combinatorial example, all-chi backbone RMSD is `0.9321 A` with no
optional geometry, `1.0169 A` with `omega` alone, `0.4932 A` with `omega,tau`,
and `0.4047 A` with `omega,tau,theta`. The omega values are still preserved
correctly; the difference reflects mixed real/ideal internal-coordinate geometry.

## JSON Schemas

Draft 2020-12 schemas are bundled for the canonical `atom-structure`,
`residue-geometry-structure`, `centroid-structure`, `energy-result`,
`radius-of-gyration-result`, `residue-angle-specs`,
`score-model-option-specs`, `scoring-options-validation`, `score-table-set`,
`geometry-table-set`, and `training-corpus` JSON
formats:

```python
from pheat.schemas import load_schema

residue_geometry_schema = load_schema("residue-geometry-structure")
```

The bundled schema `$id` values use stable public URLs under
`https://pheat.tools.blankenberglab.org/schemas/`. The same schema files are
published with the documentation site, for example
`https://pheat.tools.blankenberglab.org/schemas/residue-geometry-structure.schema.json`.

Saved atom-structure and residue-geometry JSON artifacts must use the current `format`
string and `version: 1`; file and JSON-string loaders reject other versions. Python
dictionary shorthand, such as `{"sequence": "AG"}`, remains available for direct API
construction.

Model JSON serialization rounds floating-point values to 12 decimal places to keep
committed artifacts stable across supported platforms without changing in-memory
geometry or scoring calculations.

OpenMM remains optional for the dependency-light core. The `openmm`,
`training-full`, `dev`, and `all` extras include OpenMM/PDBFixer on Python
3.10+, and the Python 3.11 Miniforge environment includes the same path for
local development. `training` intentionally omits OpenMM/PDBFixer for lighter
corpus/table workflows. The explicit `openmm-prepared` path may add missing
terminal atoms and hydrogens internally for scoring without modifying input
artifacts. PHEAT uses a fixed preparation seed for this path so regenerated
example artifacts are reproducible within a given OpenMM/PDBFixer version.
Successful OpenMM-prepared scores are reported in kJ/mol.

AmberTools and GROMACS scoring are executable-based and should be installed through
conda or another system distribution, not pip extras. The repository
`environment.yml` includes `ambertools` and `gromacs`; `pip install .[all]` installs the
Python optional dependencies but cannot provide `tleap`, `sander`, or `gmx`. Score a
heavy-atom or partial structure through AMBER preparation with:

```bash
pheat score input.pdb --model ambertools-sander --prepare auto
pheat score input.pdb --model ambertools-sander --prepare write \
  --prepared-output prepared.pdb --ambertools-work-dir ambertools-run \
  --external-timeout 300
```

AmberTools solvent mode defaults to `vacuum`. When `--amber-solvent gb` is
selected, PHEAT writes `set default PBRadii mbondi3` into the generated `tleap`
input and records `amber_pbradii: "mbondi3"` in result metadata so GB setup is
auditable and reproducible.

GROMACS scoring is available as `gromacs-mdrun`. The default force field is
`amber19sb`, selected as the current native GROMACS protein-oriented default for
PHEAT validation/reranking; `--gromacs-water auto` resolves to `none` for the
default unsolvated score and to `opc` when `--gromacs-solvate` is selected.
The default run mode is `rerun`, which evaluates the prepared coordinates with
`gmx mdrun -rerun` instead of treating zero-step MD as a single-point score.
Unsolvated scoring still centers the prepared molecule in a GROMACS box so the
Verlet cutoff/PBC machinery is valid; it does not add water unless
`--gromacs-solvate` is selected.
PHEAT checks the active GROMACS force-field directory before running `pdb2gmx`
and reports the installed force-field names when the requested one is missing.
Some conda-forge GROMACS builds may not yet bundle `amber19sb`; in that case,
install a GROMACS/GMXLIB force-field set that provides it or select an installed
alternative such as `--gromacs-forcefield amber99sb-ildn`.

```bash
pheat score input.pdb --model gromacs-mdrun --prepare auto
pheat score input.pdb --model gromacs-mdrun \
  --gromacs-forcefield amber19sb \
  --gromacs-run-mode rerun \
  --external-timeout 300 \
  --gromacs-work-dir gromacs-run \
  --keep-gromacs-files
pheat score all-atom.pdb --model gromacs-mdrun \
  --domain full \
  --hydrogens preserve \
  --prepare never \
  --prep-cache-dir .pheat-cache/external-prep \
  --prep-cache-mode readwrite
pheat gromacs prepare input.pdb -o prepared.gro --topology topol.top
pheat gromacs minimize input.pdb -o minimized.gro --score-output minimize-score.json
pheat gromacs validate input.pdb --json gromacs-validation.json
```

GROMACS can also be used with `--gromacs-run-mode minimize` or
`minimize-rerun`; those modes intentionally change coordinates and should be
interpreted separately from pure rerun validation. GROMACS totals are comparable
only when the structures use the same force field, water/solvation setting,
termini/protonation policy, preparation path, and run mode.

External AmberTools and GROMACS commands accept `--external-timeout SECONDS`,
which applies to each subprocess invocation and fails with the captured stdout/stderr
tail when a command exceeds the limit. AmberTools command failures also include
the tail of `leap.log` or `sander.out` when those files were written, which helps
diagnose parameterization and geometry problems without preserving the whole
working directory. Use `pheat scoring validate-options` or `pheat gromacs
validate-options` to check selected options before launching a run; validation
catches unsupported enum values, missing executables, missing GROMACS force
fields in the active installation, and invalid cache configuration. The same
validation is available to Python callers through
`validate_external_scoring_options(...)`.

GROMACS run settings are exposed for validation and reranking experiments:
`--gromacs-minimize-steps`, `--gromacs-emtol`, `--gromacs-emstep`,
`--gromacs-box-distance`, `--gromacs-cutoff`, `--gromacs-coulombtype`,
`--gromacs-vdwtype`, `--gromacs-nstlist`, `--gromacs-pbc`,
`--gromacs-comm-mode`, `--gromacs-grompp-maxwarn`, and repeated
`--gromacs-mdrun-flag` values. For example, pass
`--gromacs-mdrun-flag=-ntomp --gromacs-mdrun-flag 4` to request four OpenMP
threads from `mdrun`.

`--prep-cache-dir` plus `--prep-cache-mode off|readwrite|readonly|refresh`
records and optionally reuses external preparation artifacts. AmberTools records
cache metadata but still runs `tleap`, because its coordinate file is
candidate-specific. GROMACS can reuse a cached topology only for
`--prepare never` inputs that already include hydrogens and keep the same atom
order; runtime MDP and `mdrun` settings are intentionally not part of the topology
cache key. Use `--domain full` if the input contains hydrogens that must be retained.
Because PHEAT readers drop hydrogens by default, CLI cache-reuse runs also need
`--hydrogens preserve`. For heavy-atom default scoring, auto-preparation remains
the safer path and the cache reports itself as disabled rather than silently
reusing an incompatible topology.

## Reference Corpus Specs

PHEAT can validate corpus specs and build small local reference-corpus manifests
from ID lists, local archives, or dry-run archived snapshot templates. The tiny
demo uses only local fixtures and is intended as a workflow check:

```bash
pheat reference validate-spec examples/corpora/user_defined_ids_demo.yml
pheat reference build --corpus-spec examples/corpora/user_defined_ids_demo.yml --output-root .pheat-cache/corpora/user-defined-demo --overwrite
```

Related docs:

- `docs/corpus-specs.md`
- `docs/reference-manifests.md`
- `docs/ccd-heterogen-annotation.md`

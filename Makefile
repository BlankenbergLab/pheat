SHELL := /bin/bash

PYTHON ?= python
PHEAT ?= pheat
JUPYTER ?= jupyter
MKDOCS ?= mkdocs
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 8000
DOCS_HOST ?= 127.0.0.1
DOCS_PORT ?= 8000
SNAPSHOT_ID ?= rcsb-current-bcif
SNAPSHOT_ROOT ?= .pheat-cache/pdb-archive/$(SNAPSHOT_ID)
TRAINING_ROOT ?= .pheat-cache/training
TRAINING_SET ?= protein-heavy-30id
CORPUS_ID ?= protein-heavy-30id
CORPUS_VERSION ?= v1
TABLE_SET_ID ?= $(CORPUS_ID)
TABLE_SET_VERSION ?= $(CORPUS_VERSION)
DOMAIN ?= protein-heavy
SEQUENCE_IDENTITY ?= 0.30
TRAINING_MODELS ?= pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg
SASA_BACKEND ?= auto
DECOY_DATASET ?= 3drobot
REFERENCE_ROOT ?= .pheat-cache/reference-builds
REFERENCE_VERSION ?= v0
REFERENCE_WORKERS ?= auto
REFERENCE_SEED ?= 20260522
REFERENCE_MAX_RESOLUTION ?= 2.5
REFERENCE_MIN_LENGTH ?= 50
REFERENCE_MAX_LENGTH ?= 800
REFERENCE_DATASET ?= 3drobot
REFERENCE_DECOY_RECIPES ?= pheat-torsion-v1
REFERENCE_DECOY_ATTEMPTS ?= 8
REFERENCE_MODELS ?= pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg
REFERENCE_FEATURE_MODELS ?= generic,pheat-dfire,pheat-goap,pheat-mj,pheat-hydropathy,pheat-backbone,pheat-rotamer,pheat-hbond,pheat-rg,heavy-mm,pheat-geometry-integrity
REFERENCE_FEATURE_MAX_ENTRIES ?=
REFERENCE_METADATA ?= $(SNAPSHOT_ROOT)/manifests/metadata.jsonl
REFERENCE_METADATA_SOURCE ?= auto
REFERENCE_CANARY_ENTRIES ?= 25
REFERENCE_BACKUP_EXISTING ?= 0
REFERENCE_BACKUP_FLAG := $(if $(filter 1 true yes,$(REFERENCE_BACKUP_EXISTING)),--backup-existing,)
REFERENCE_AQUEOUS_SET ?= protein-heavy-30id-xray-aqueous-$(REFERENCE_VERSION)
REFERENCE_MEMBRANE_SET ?= protein-heavy-30id-xray-membrane-$(REFERENCE_VERSION)
REFERENCE_SET ?= $(REFERENCE_AQUEOUS_SET)
REFERENCE_OVERWRITE ?= 0
REFERENCE_OVERWRITE_FLAG := $(if $(filter 1 true yes,$(REFERENCE_OVERWRITE)),--overwrite,)
REFERENCE_PACKAGE_DEST ?= src/pheat/data/scoring/$(REFERENCE_VERSION)
CCD_ROOT ?= .pheat-cache/sources/ccd
CCD_FULL_DIR ?= $(CCD_ROOT)/full
CCD_BCIF_DIR ?= $(CCD_ROOT)/bcif
CCD_FULL ?= $(CCD_FULL_DIR)/components.cif.gz
CCD_DIR ?= $(CCD_ROOT)/components

MOLSTAR_VERSION ?= 5.9.0
MOLSTAR_TIMEOUT ?= 60
ROUNDTRIP_OUTPUT ?= examples/roundtrip/2mu7_combinatorial
ROUNDTRIP_GEOMETRIES ?= fixed,ccd-sidechain-geometry-v1
NOTEBOOK_SOURCE ?= examples/notebook/2mu7_roundtrip_energy_rmsd_molstar.ipynb
NOTEBOOK_EXECUTED ?= examples/notebook/executed/2mu7_roundtrip_energy_rmsd_molstar.executed.ipynb

.PHONY: test coverage coverage-html lint typecheck package check docs-deps docs docs-serve molstar web examples-roundtrip examples-notebook-executed examples clean-examples sources-ccd-full sources-ccd-bcif sources-ccd training-snapshot-ids training-decoys training-inventory training-select training-corpus-describe training-tables training-tables-contacts training-tables-sasa training-tables-describe training-validate training-validate-contacts training-validate-sasa training-features training-ml-linear training-all reference-datasets reference-fetch reference-metadata reference-inventory reference-select reference-select-aqueous reference-select-membrane reference-decoys reference-scores reference-features reference-ml-linear reference-validate reference-promote reference-package-scoring-assets reference-audit reference-all reference-unattended geometry-backbone-tables geometry-sidechain-ccd-tables geometry-tables-validate

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=pheat --cov-report=term-missing --cov-report=xml

coverage-html:
	$(PYTHON) -m pytest --cov=pheat --cov-report=term-missing --cov-report=xml --cov-report=html

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/pheat

package:
	$(PYTHON) -m build

check: lint typecheck test

docs-deps:
	$(PYTHON) -m pip install -r docs/requirements.txt

docs:
	$(MKDOCS) build --strict

docs-serve:
	$(MKDOCS) serve -a $(DOCS_HOST):$(DOCS_PORT)

molstar:
	$(PHEAT) molstar install --version $(MOLSTAR_VERSION) --timeout $(MOLSTAR_TIMEOUT)

web: molstar
	$(PYTHON) -m pheat.cli web \
	  --host $(WEB_HOST) \
	  --port $(WEB_PORT) \
	  --random-port-if-taken \
	  --work-dir .pheat-cache/web

examples-roundtrip: molstar
	$(PYTHON) examples/2mu7_combinatorial_roundtrip.py \
	  --output-root $(ROUNDTRIP_OUTPUT) \
	  --geometry-variants $(ROUNDTRIP_GEOMETRIES)

examples-notebook-executed:
	mkdir -p $(dir $(NOTEBOOK_EXECUTED))
	$(JUPYTER) nbconvert --to notebook --execute "$(NOTEBOOK_SOURCE)" \
	  --output-dir "$(dir $(NOTEBOOK_EXECUTED))" \
	  --output "$(notdir $(NOTEBOOK_EXECUTED))" \
	  --ExecutePreprocessor.timeout=600

examples: examples-roundtrip examples-notebook-executed

clean-examples:
	rm -rf examples/roundtrip examples/vendor examples/notebook/executed

training-snapshot-ids:
	$(PHEAT) archive snapshots ids $(SNAPSHOT_ID) \
	  --output-root $(SNAPSHOT_ROOT) \
	  -o $(TRAINING_ROOT)/snapshot-ids.txt

training-decoys:
	$(PHEAT) training decoys fetch $(DECOY_DATASET) \
	  --output-root $(TRAINING_ROOT)/decoys \
	  --yes

training-inventory:
	$(PHEAT) training corpus inventory \
	  --snapshot-root $(SNAPSHOT_ROOT) \
	  --domain $(DOMAIN) \
	  -o $(TRAINING_ROOT)/inventory.jsonl

training-select:
	$(PHEAT) training corpus select \
	  --inventory $(TRAINING_ROOT)/inventory.jsonl \
	  --output-root $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --sequence-identity $(SEQUENCE_IDENTITY) \
	  --corpus-id $(CORPUS_ID) \
	  --corpus-version $(CORPUS_VERSION)

training-corpus-describe:
	$(PHEAT) training corpus describe \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET)

training-tables:
	$(PHEAT) training tables build \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --output-root $(TRAINING_ROOT)/tables/$(TRAINING_SET) \
	  --models $(TRAINING_MODELS) \
	  --domain $(DOMAIN) \
	  --burial-method both \
	  --sasa-backend $(SASA_BACKEND) \
	  --table-set-id $(TABLE_SET_ID) \
	  --table-set-version $(TABLE_SET_VERSION)

training-tables-contacts:
	$(PHEAT) training tables build \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --output-root $(TRAINING_ROOT)/tables/$(TRAINING_SET)-contacts \
	  --models $(TRAINING_MODELS) \
	  --domain $(DOMAIN) \
	  --burial-method contacts \
	  --table-set-id $(TABLE_SET_ID)-contacts \
	  --table-set-version $(TABLE_SET_VERSION)

training-tables-sasa:
	$(PHEAT) training tables build \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --output-root $(TRAINING_ROOT)/tables/$(TRAINING_SET)-sasa \
	  --models $(TRAINING_MODELS) \
	  --domain $(DOMAIN) \
	  --burial-method sasa \
	  --sasa-backend $(SASA_BACKEND) \
	  --table-set-id $(TABLE_SET_ID)-sasa \
	  --table-set-version $(TABLE_SET_VERSION)

training-tables-describe:
	$(PHEAT) training tables describe \
	  --table-set $(TRAINING_ROOT)/tables/$(TRAINING_SET)/score-tables.json

training-validate-contacts:
	$(PHEAT) training tables validate \
	  --table-set $(TRAINING_ROOT)/tables/$(TRAINING_SET)-contacts/score-tables.json \
	  --decoy-root $(TRAINING_ROOT)/decoys

training-validate-sasa:
	$(PHEAT) training tables validate \
	  --table-set $(TRAINING_ROOT)/tables/$(TRAINING_SET)-sasa/score-tables.json \
	  --decoy-root $(TRAINING_ROOT)/decoys

training-validate:
	$(PHEAT) training tables validate \
	  --table-set $(TRAINING_ROOT)/tables/$(TRAINING_SET)/score-tables.json \
	  --decoy-root $(TRAINING_ROOT)/decoys

training-features:
	$(PHEAT) training features extract \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --models generic,pheat-dfire,heavy-mm,pheat-hydropathy \
	  --domain $(DOMAIN) \
	  -o $(TRAINING_ROOT)/features.jsonl

training-ml-linear:
	$(PHEAT) training ml train-linear \
	  --features $(TRAINING_ROOT)/features.jsonl \
	  -o $(TRAINING_ROOT)/tables/pheat-ml-linear.json

training-all: training-snapshot-ids training-decoys training-inventory training-select training-tables training-tables-contacts training-tables-sasa training-validate training-validate-contacts training-validate-sasa training-features training-ml-linear

reference-datasets:
	$(PHEAT) reference datasets list

reference-fetch:
	$(PHEAT) reference fetch \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --snapshot-id $(SNAPSHOT_ID) \
	  --snapshot-root $(SNAPSHOT_ROOT) \
	  --dataset $(REFERENCE_DATASET) \
	  --workers $(REFERENCE_WORKERS) \
	  --seed $(REFERENCE_SEED) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-metadata:
	$(PHEAT) archive snapshots metadata $(SNAPSHOT_ID) \
	  --output-root $(SNAPSHOT_ROOT) \
	  --source $(REFERENCE_METADATA_SOURCE) \
	  --workers $(REFERENCE_WORKERS) \
	  $(REFERENCE_OVERWRITE_FLAG) \
	  -o $(REFERENCE_METADATA)

reference-inventory: reference-metadata
	$(PHEAT) reference inventory \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --snapshot-root $(SNAPSHOT_ROOT) \
	  --domain $(DOMAIN) \
	  --metadata-jsonl $(REFERENCE_METADATA) \
	  --workers $(REFERENCE_WORKERS) \
	  $(REFERENCE_OVERWRITE_FLAG) \
	  -o $(REFERENCE_ROOT)/inventories/$(REFERENCE_VERSION)/inventory.jsonl

reference-select: reference-select-aqueous

reference-select-aqueous:
	$(PHEAT) reference select \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --inventory $(REFERENCE_ROOT)/inventories/$(REFERENCE_VERSION)/inventory.jsonl \
	  --output-root $(REFERENCE_ROOT)/sets/$(REFERENCE_AQUEOUS_SET) \
	  --subset aqueous \
	  --domain $(DOMAIN) \
	  --method x-ray \
	  --max-resolution $(REFERENCE_MAX_RESOLUTION) \
	  --sequence-identity $(SEQUENCE_IDENTITY) \
	  --min-length $(REFERENCE_MIN_LENGTH) \
	  --max-length $(REFERENCE_MAX_LENGTH) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-select-membrane:
	$(PHEAT) reference select \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --inventory $(REFERENCE_ROOT)/inventories/$(REFERENCE_VERSION)/inventory.jsonl \
	  --output-root $(REFERENCE_ROOT)/sets/$(REFERENCE_MEMBRANE_SET) \
	  --subset membrane \
	  --domain $(DOMAIN) \
	  --method x-ray \
	  --max-resolution $(REFERENCE_MAX_RESOLUTION) \
	  --sequence-identity $(SEQUENCE_IDENTITY) \
	  --min-length $(REFERENCE_MIN_LENGTH) \
	  --max-length $(REFERENCE_MAX_LENGTH) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-decoys:
	$(PHEAT) reference build-decoys \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --training-set $(REFERENCE_ROOT)/sets/$(REFERENCE_SET) \
	  --output-root $(REFERENCE_ROOT)/decoys/pheat-decoys-$(REFERENCE_VERSION) \
	  --recipes $(REFERENCE_DECOY_RECIPES) \
	  --attempts-per-decoy $(REFERENCE_DECOY_ATTEMPTS) \
	  --seed $(REFERENCE_SEED) \
	  --workers $(REFERENCE_WORKERS) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-scores:
	$(PHEAT) reference build-scores \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --training-set $(REFERENCE_ROOT)/sets/$(REFERENCE_SET) \
	  --output-root $(REFERENCE_ROOT)/tables/$(REFERENCE_SET) \
	  --models $(REFERENCE_MODELS) \
	  --domain $(DOMAIN) \
	  --burial-method both \
	  --sasa-backend $(SASA_BACKEND) \
	  --table-set-id $(REFERENCE_SET) \
	  --workers $(REFERENCE_WORKERS) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-features:
	$(PHEAT) reference extract-features \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --training-set $(REFERENCE_ROOT)/sets/$(REFERENCE_SET) \
	  --decoys $(REFERENCE_ROOT)/decoys/pheat-decoys-$(REFERENCE_VERSION)/decoys.jsonl \
	  --models $(REFERENCE_FEATURE_MODELS) \
	  --domain $(DOMAIN) \
	  $(if $(REFERENCE_FEATURE_MAX_ENTRIES),--max-entries $(REFERENCE_FEATURE_MAX_ENTRIES),) \
	  --workers $(REFERENCE_WORKERS) \
	  $(REFERENCE_OVERWRITE_FLAG) \
	  -o $(REFERENCE_ROOT)/features/$(REFERENCE_VERSION)/features.jsonl

reference-ml-linear:
	$(PHEAT) reference train-ml \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --features $(REFERENCE_ROOT)/features/$(REFERENCE_VERSION)/features.jsonl \
	  $(REFERENCE_OVERWRITE_FLAG) \
	  -o $(REFERENCE_ROOT)/models/$(REFERENCE_VERSION)/pheat-ml-linear.json

reference-validate:
	$(PHEAT) reference validate \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --features $(REFERENCE_ROOT)/features/$(REFERENCE_VERSION)/features.jsonl \
	  $(REFERENCE_OVERWRITE_FLAG) \
	  -o $(REFERENCE_ROOT)/validation/$(REFERENCE_VERSION)/validation.json

reference-promote:
	$(PHEAT) reference promote \
	  --source $(PROMOTE_SOURCE) \
	  --destination $(PROMOTE_DESTINATION) \
	  --note "$(PROMOTE_NOTE)" \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-package-scoring-assets:
	$(PHEAT) reference package-scoring-assets \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --destination-root $(REFERENCE_PACKAGE_DEST) \
	  $(REFERENCE_OVERWRITE_FLAG)

reference-audit:
	$(PHEAT) reference audit-version \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION)

reference-all: reference-fetch reference-metadata reference-inventory reference-select-aqueous reference-select-membrane reference-decoys reference-scores reference-features reference-ml-linear reference-validate

reference-unattended:
	$(PHEAT) reference run-unattended \
	  --reference-root $(REFERENCE_ROOT) \
	  --artifact-version $(REFERENCE_VERSION) \
	  --snapshot-id $(SNAPSHOT_ID) \
	  --snapshot-root $(SNAPSHOT_ROOT) \
	  --domain $(DOMAIN) \
	  --method x-ray \
	  --max-resolution $(REFERENCE_MAX_RESOLUTION) \
	  --sequence-identity $(SEQUENCE_IDENTITY) \
	  --min-length $(REFERENCE_MIN_LENGTH) \
	  --max-length $(REFERENCE_MAX_LENGTH) \
	  --decoy-recipes $(REFERENCE_DECOY_RECIPES) \
	  --attempts-per-decoy $(REFERENCE_DECOY_ATTEMPTS) \
	  --models $(REFERENCE_MODELS) \
	  --feature-models $(REFERENCE_FEATURE_MODELS) \
	  --sasa-backend $(SASA_BACKEND) \
	  --metadata-source $(REFERENCE_METADATA_SOURCE) \
	  --workers $(REFERENCE_WORKERS) \
	  --seed $(REFERENCE_SEED) \
	  --canary-entries $(REFERENCE_CANARY_ENTRIES) \
	  $(REFERENCE_BACKUP_FLAG) \
	  $(REFERENCE_OVERWRITE_FLAG)

sources-ccd-full:
	$(PHEAT) sources fetch wwpdb-ccd-full --destination $(CCD_FULL_DIR)

sources-ccd-bcif:
	$(PHEAT) sources fetch rcsb-ccd-bcif --destination $(CCD_BCIF_DIR)

sources-ccd: sources-ccd-full sources-ccd-bcif

geometry-backbone-tables:
	$(PHEAT) geometry tables build-backbone \
	  --training-set $(TRAINING_ROOT)/sets/$(TRAINING_SET) \
	  --output-root $(TRAINING_ROOT)/geometry/$(TRAINING_SET)-backbone \
	  --domain $(DOMAIN) \
	  --table-set-id $(TRAINING_SET)-backbone-geometry \
	  --table-set-version $(TABLE_SET_VERSION)

geometry-sidechain-ccd-tables:
	$(PHEAT) geometry tables build-sidechain-ccd \
	  --ccd-full $(CCD_FULL) \
	  --output-root $(TRAINING_ROOT)/geometry/ccd-sidechains \
	  --table-set-id ccd-sidechain-geometry \
	  --table-set-version $(TABLE_SET_VERSION)

geometry-tables-validate:
	$(PHEAT) geometry tables validate \
	  --table-set $(TRAINING_ROOT)/geometry/$(TRAINING_SET)-backbone/geometry-tables.json

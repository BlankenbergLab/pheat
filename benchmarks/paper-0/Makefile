PYTHON ?= python
ROOT := ../..
PYTHONPATH := $(ROOT)/src
SPEC := corpus_specs/user_defined_ids_demo.yml
RESULTS := results/demo
TABLES := tables/demo
FIGURES := figures/demo
PDF := paper/paper-0-draft.pdf
PHEAT := PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pheat.cli --quiet
SCRIPT := PYTHONPATH=$(PYTHONPATH) $(PYTHON)

.PHONY: help validate-specs build-demo-corpus benchmark-io-demo benchmark-reconstruction-demo generate-decoys-demo score-demo summarize-demo tables-demo figures-demo pdf-demo reproduce-demo clean-demo clean-pdf

help:
	@echo "Targets: validate-specs build-demo-corpus benchmark-io-demo benchmark-reconstruction-demo generate-decoys-demo score-demo summarize-demo tables-demo figures-demo pdf-demo reproduce-demo clean-demo clean-pdf"

validate-specs:
	$(PHEAT) reference validate-spec $(SPEC)

build-demo-corpus:
	$(PHEAT) reference build --corpus-spec $(SPEC) --output-root $(RESULTS)/corpus --overwrite

benchmark-io-demo: build-demo-corpus
	$(SCRIPT) scripts/02_compare_io_tools.py --manifest $(RESULTS)/corpus/manifest.jsonl --output $(RESULTS)/io-results.jsonl

benchmark-reconstruction-demo: build-demo-corpus
	$(SCRIPT) scripts/03_compare_reconstruction.py --manifest $(RESULTS)/corpus/manifest.jsonl --output $(RESULTS)/reconstruction-results.jsonl

generate-decoys-demo: build-demo-corpus
	$(SCRIPT) scripts/04_generate_decoys.py --manifest $(RESULTS)/corpus/manifest.jsonl --recipe pheat-coordinate-noise-v1 --n-per-native 2 --seed 20260529 --output-root $(RESULTS)/decoys

score-demo: generate-decoys-demo
	$(SCRIPT) scripts/05_score_native_and_decoys.py --native-manifest $(RESULTS)/corpus/manifest.jsonl --decoy-manifest $(RESULTS)/decoys/decoy-set-manifest.jsonl --output $(RESULTS)/scores.jsonl
	$(SCRIPT) scripts/06_external_backend_scores.py --manifest $(RESULTS)/corpus/manifest.jsonl --output $(RESULTS)/external-backend-status.jsonl

summarize-demo: benchmark-io-demo benchmark-reconstruction-demo score-demo
	$(SCRIPT) scripts/07_summarize_results.py --inputs $(RESULTS)/io-results.jsonl $(RESULTS)/reconstruction-results.jsonl $(RESULTS)/scores.jsonl $(RESULTS)/external-backend-status.jsonl --output $(RESULTS)/summary.json

tables-demo: summarize-demo
	$(SCRIPT) scripts/09_make_tables.py --summary $(RESULTS)/summary.json --output-root $(TABLES)

figures-demo: summarize-demo
	$(SCRIPT) scripts/08_make_figures.py --summary $(RESULTS)/summary.json --output-root $(FIGURES)

pdf-demo: tables-demo figures-demo
	$(SCRIPT) scripts/10_assemble_pdf.py --summary $(RESULTS)/summary.json --table $(TABLES)/demo-summary.csv --figure $(FIGURES)/demo-workflow-summary.png --output $(PDF)

reproduce-demo: validate-specs build-demo-corpus benchmark-io-demo benchmark-reconstruction-demo generate-decoys-demo score-demo summarize-demo tables-demo figures-demo

clean-demo:
	rm -rf $(RESULTS) $(TABLES) $(FIGURES)

clean-pdf:
	rm -f $(PDF)

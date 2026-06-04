# PHEAT

**PHEAT** is the **Protein Heavy-atom Energy and Analysis Toolkit**. It
provides importable Python APIs and a CLI for protein structure conversion,
residue-geometry reconstruction, roundtrip comparison, radius-of-gyration and
RMSD metrics, and approximate heavy-atom scoring.

PHEAT is built as a library-first package. The command line interface delegates
to the same backend functions exposed by the `pheat` Python package.

## Installation

For a minimal editable install from a local checkout:

```bash
python -m pip install -e .
```

For the full development and optional feature surface:

```bash
python -m pip install -e ".[all]"
```

For documentation work:

```bash
make docs-deps
make docs-serve
```

## Start Here

- Follow the [Quick Start](quickstart.md) for common conversion and scoring
  examples.
- Browse the [Basic API](api.md) for the import-root interface.
- Use the [Advanced API](advanced-api.md) when you need module-level details.
- Review the [Initial v0 Reference Build](reference-build-v0.md) for provenance
  on the packaged scoring assets.
- Review [Corpus Specs](corpus-specs.md), [Reference Manifests](reference-manifests.md),
  and [CCD Heterogen Annotation](ccd-heterogen-annotation.md)
  for schema-validated reference-corpus workflows.
- Review the [paper-0 readiness checklist](paper-0-submission-readiness.md)
  for the paper-specific draft benchmark workstream.
- Review [Deployment](deployment.md) for the Cloudflare Workers static-assets
  configuration.

## Reference Corpus Specs

PHEAT includes a small offline vertical slice for schema-validated corpus specs,
reference manifests, and CCD-aware component summaries. These demo outputs are
workflow checks, not calibrated scientific benchmark results.

## paper-0 Draft

The `paper-0-draft` branch includes paper-specific benchmark scaffolding,
draft outline material, and PDF assembly support. Demo outputs are workflow
checks only, not manuscript-scale scientific results.

## Blankenberg Lab

PHEAT is developed by the
[Blankenberg Lab](https://www.blankenberglab.org/). Related tools are available
through [tools.blankenberglab.org](https://tools.blankenberglab.org/).

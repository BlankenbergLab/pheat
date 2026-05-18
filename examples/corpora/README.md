# PHEAT Corpus Spec Examples

These examples are starter templates and offline demos for local reference-corpus
workflows. They are not final scientific benchmark datasets.

Use the tiny local demo first:

```bash
pheat reference validate-spec examples/corpora/user_defined_ids_demo.yml
pheat reference build --corpus-spec examples/corpora/user_defined_ids_demo.yml --output-root .pheat-cache/corpora/user-defined-demo --overwrite
```

The X-ray aqueous and membrane specs describe larger archived-snapshot corpus
templates. Run them with `--dry-run` until a local archived snapshot and final
filters are available.

The ligand-bound CCD demo shows where a user-provided CCD-like CSV or JSONL
table can be supplied with `pheat reference build --ccd ccd-table.csv`.

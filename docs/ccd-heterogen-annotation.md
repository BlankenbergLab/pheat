# CCD And Heterogen Annotation

PHEAT can add lightweight component summaries to corpus manifests. This is not a
full vendored copy of the wwPDB Chemical Component Dictionary. The built-in
fallback rows are demo/test metadata only.

Use a user CCD-like table during corpus build:

```bash
pheat reference build \
  --corpus-spec examples/corpora/ligand_bound_ccd_demo_v1.yml \
  --ccd my-components.csv \
  --output-root .pheat-cache/corpora/ligand-demo \
  --overwrite
```

Supported CSV or JSONL fields:

- `component_id`
- `component_type`
- `name`
- `formula`
- `atom_count`
- `classification`

Classifications include `standard_residue`, `modified_residue`, `water`, `ion`,
`ligand`, and `unknown`. Unknown component IDs are reported in manifest rows and
are not fatal.

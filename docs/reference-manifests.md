# Reference Manifests

`pheat reference build` writes a `manifest.jsonl` file with one selected record
per JSON line. Rows include:

- corpus ID and version
- record and PDB identifiers
- source format/path/checksum
- parser and PHEAT versions
- selected atom domain
- atom and residue counts
- generated artifacts and checksums
- CCD-like heterogen, ligand, water, ion, modified-residue, and unknown-component summaries
- creation timestamp

The command also writes:

- `excluded-records.jsonl`
- `structure-summary.csv`
- `build-metadata.json`
- `checksums.sha256`

Build the offline demo:

```bash
pheat reference build --corpus-spec examples/corpora/user_defined_ids_demo.yml --output-root .pheat-cache/corpora/user-defined-demo --overwrite
```

Failures are recorded in `excluded-records.jsonl`; one malformed structure does
not stop the whole build.

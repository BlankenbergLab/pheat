# Corpus Specs

Corpus specs describe how PHEAT should build a reference dataset from local
files, ID lists, archived snapshots, or future mixed sources.

Validate a spec:

```bash
pheat reference validate-spec examples/corpora/user_defined_ids_demo.yml
```

Summarize a spec:

```bash
pheat reference summarize-spec examples/corpora/user_defined_ids_demo.yml
```

Initialize a template:

```bash
pheat reference init-spec --template user-defined-ids --output my-corpus.yml
```

Required top-level fields are `corpus_id`, `version`, `source`, and
`selection`. The schema also supports geometry, scoring, decoy, output, and
provenance sections. YAML examples require PyYAML in the active environment;
JSON specs work with the same schema.

The current functional builder supports local offline demos for `id_list` and
`local_archive` sources. `rcsb_snapshot` specs validate and can be dry-run, but
larger downloaded corpora should use archived snapshot workflows and durable source
manifests.

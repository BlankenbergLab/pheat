# paper-0 Publication Roadmap

PHEAT should be framed as an open, reproducible toolkit for generating,
transforming, validating, and scoring user-defined protein heavy-atom reference
corpora with explicit provenance and reusable benchmark artifacts.

## Repository Additions

- Corpus specification schema: `src/pheat/schemas/corpus-spec.schema.json`
- Reference manifest schema: `src/pheat/schemas/reference-dataset-manifest.schema.json`
- Benchmark and decoy schemas under `src/pheat/schemas/`
- Local/user-defined corpus build command: `pheat reference build`
- Spec commands: `validate-spec`, `summarize-spec`, and `init-spec`
- Optional CCD-like component annotation: `src/pheat/ccd.py`
- Benchmark bundle: `benchmarks/paper-0/`
- Manuscript planning docs: `docs/paper/`

## Required Evidence Before Submission

- Rebuild named paper corpora from archived RCSB/wwPDB snapshots.
- Publish manifests, source checksums, generated artifact checksums, and build
  metadata.
- Run structure I/O comparisons against Biopython, Gemmi, MDAnalysis, and MDTraj.
- Run reconstruction comparisons where input/output contracts are comparable.
- Run decoy/scoring benchmarks with PHEAT-native, baseline, and external-backend
  scores clearly separated.
- Archive source code and data with release DOI placeholders replaced by real
  identifiers.

## Non-goals And Claims Discipline

- AG is smoke-test-only and must not be used as scientific evidence.
- PHEAT does not replace Biopython, Gemmi, MDAnalysis, MDTraj, OpenMM, GROMACS,
  AmberTools, FreeSASA, PISCES, or SidechainNet.
- PHEAT-native scores must not be described as DFIRE, DOPE, GOAP, Rosetta,
  OpenMM, GROMACS, or AmberTools unless exact published methods or external
  backends are actually used.
- Demo outputs in `benchmarks/paper-0/results/demo/` are workflow checks, not
  manuscript results.

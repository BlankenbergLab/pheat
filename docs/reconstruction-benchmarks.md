# Reconstruction Benchmarks

`benchmarks/paper-0/scripts/03_compare_reconstruction.py` runs a tiny PHEAT
geometry/reconstruction check from a manifest and records optional comparator
status rows. It also emits concrete `reconstruction_comparison` rows for PHEAT
and PULCHRA:

- `reduced-input`: both tools start from the same C-alpha-only PDB. PHEAT builds
  a residue-geometry artifact from that reduced input, so this task measures the
  current sequence/reduced-geometry reconstruction path rather than full
  coordinate preservation.
- `artifact-assisted`: PHEAT starts from its richer residue-geometry artifact
  derived from the original structure, while PULCHRA starts from the same
  C-alpha-only PDB. This asks how PHEAT's artifact-assisted reconstruction
  differs from a reduced-representation rebuild.

For each task, the demo writes the reduced input PDB, the PHEAT geometry JSON,
the PHEAT rebuilt PDB, and the PULCHRA rebuilt PDB under
`benchmarks/paper-0/results/demo/reconstruction/`. RMSD fields compare PHEAT and
PULCHRA outputs against the original PDB and against each other for C-alpha,
backbone, all-heavy, and sidechain-heavy atom sets. Missing Biopython
internal-coordinate, PeptideBuilder, or PULCHRA support does not fail the demo.

The paper-scale benchmark still needs reviewed atom correspondence rules,
matched input contracts, and large archived corpora before reporting scientific
metrics.

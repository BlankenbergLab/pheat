# paper-0 Benchmark Scaffold

This directory contains the reproducible demo and benchmark scaffolding for the
PHEAT paper-0 preparation workstream.

The demo workflow is intentionally tiny and offline. It uses local repository
fixtures to prove that corpus specs, manifests, checksums, benchmark rows, and
paper-table/figure plumbing work. Demo outputs are not scientific benchmark
results.

Run the offline demo from the repository root:

```bash
make -C benchmarks/paper-0 reproduce-demo
```

Optional comparators such as Biopython, Gemmi, MDAnalysis, MDTraj, OpenMM,
GROMACS, AmberTools, and FreeSASA are installed through the Conda paper
environment. PeptideBuilder and PULCHRA are not available as Conda packages for
this local setup, so the paper environment installs them with pip inside the same
Conda prefix.

Paper-scale results should be generated outside CI with archived source
snapshots, locked environments, and reviewed corpus specs.

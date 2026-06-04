# Scoring Benchmarks

The paper-0 scaffold separates:

- PHEAT-native scores
- baseline scores
- external-backend scores
- unsupported or missing backend status rows

The demo does not fabricate external scores. `06_external_backend_scores.py`
runs installed OpenMM, GROMACS, AmberTools, and FreeSASA probes where the tiny
fixture is compatible, and reports unsupported tiny-fixture inputs separately
from real failures.

Paper-scale scoring claims require real native/decoy datasets, reviewed score
definitions, and archived outputs.

# Initial v0 Reference Build

This document records the active initial `v0` reference-build artifacts used to
package the current PHEAT scoring assets. The label is intentionally `v0`
because the assets are functional and reproducible enough for package testing,
but they have not yet had the manual scientific review expected before a stable
reference release.

## Active Archive Layout

The external archive root is relocatable; repository examples use
`.pheat-cache/reference-builds`. The local build used the same relative layout
inside an external archive root.

Active `v0` paths:

- `runs/v0/summary.json`
- `runs/v0/version-audit.json`
- `features/v0/aqueous-features.jsonl`
- `features/v0/membrane-features.jsonl`
- `models/v0/pheat-ml-linear-aqueous.json`
- `models/v0/pheat-ml-linear-membrane.json`
- `validation/v0/aqueous-validation.json`
- `validation/v0/membrane-validation.json`
- `sets/protein-heavy-30id-xray-aqueous-v0`
- `sets/protein-heavy-30id-xray-membrane-v0`
- `decoys/protein-heavy-30id-xray-aqueous-v0-pheat-torsion`
- `decoys/protein-heavy-30id-xray-membrane-v0-pheat-torsion`
- `tables/protein-heavy-30id-xray-aqueous-v0`
- `tables/protein-heavy-30id-xray-membrane-v0`

Older prototype `v0` paths were moved to
`backups/v0-legacy-20260528T124634Z`. The immediately preceding comparable run
is retained externally as `backups/v1-20260527T211709Z`; it was generated before
the current decision to relabel the active initial assets as `v0`.

## Build Settings

The active build used these settings:

- Snapshot: `rcsb-current-bcif`
- Structure method: X-ray only
- Domain: `protein-heavy`
- Sequence-identity clustering threshold: 30%
- Maximum resolution: 2.5 A
- Length filter: 50 to 800 residues
- Subsets: aqueous and membrane
- Workers: 15
- Seed: `20260522`
- Decoy profile: `pheat-torsion-v1`
- Decoy attempts per selected chain: 8
- Feature set: native and accepted decoy rows
- SASA mode: automatic FreeSASA-backed SASA when available

`pheat-torsion-v1` is a decoy recipe/profile identifier, not the reference
artifact version.

## Run Comparison

The active build is substantially similar to the previous comparable run. The
main difference is that the active aqueous decoy/features set accepted additional
torsion-space decoys.

| Quantity | Previous Comparable Run | Active Initial v0 |
| --- | ---: | ---: |
| Snapshot metadata entries | 254,183 | 254,183 |
| Snapshot metadata failures | 3 | 3 |
| Inventory rows | 1,062,606 | 1,062,606 |
| Aqueous selected chains | 21,721 | 21,721 |
| Membrane selected chains | 2,373 | 2,373 |
| Aqueous accepted decoys | 5,242 | 5,454 |
| Membrane accepted decoys | 554 | 554 |
| Aqueous feature rows | 26,963 | 27,175 |
| Membrane feature rows | 2,927 | 2,927 |

The selected aqueous and membrane chain sets match by `pdb_id:chain_id`. Score
table definitions are logically the same between the two comparable runs. File
checksums differ where provenance, timestamps, rewritten relative paths, or the
expanded aqueous decoy/features rows changed.

## Packaged Assets

Only small packageable JSON assets are committed. Score/model payloads are
stored as compressed JSON.xz; the manifest remains plain JSON and records both
compressed and uncompressed SHA-256 checksums.

- `src/pheat/data/scoring/v0/manifest.json`
- `src/pheat/data/scoring/v0/pheat-ml-linear-aqueous.json.xz`
- `src/pheat/data/scoring/v0/pheat-ml-linear-membrane.json.xz`
- `src/pheat/data/scoring/v0/protein-heavy-30id-xray-aqueous/score-tables.json.xz`
- `src/pheat/data/scoring/v0/protein-heavy-30id-xray-membrane/score-tables.json.xz`

Large source snapshots, inventories, selected-chain JSONL files, decoy JSONL
files, reconstructed structures, feature JSONL files, logs, and validation
working files remain external archive artifacts.

## Version Audit

Run this check after relabeling or rebuilding an archive:

```bash
pheat reference audit-version \
  --reference-root .pheat-cache/reference-builds \
  --artifact-version v0
```

The active local archive was audited successfully with:

- `issue_count`: 0
- `checked_file_count`: 26
- `checked_path_count`: 101

The Makefile shortcut is:

```bash
make reference-audit REFERENCE_ROOT=.pheat-cache/reference-builds REFERENCE_VERSION=v0
```

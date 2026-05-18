# 2026-05-19 mmCIF Support Plan

## Goal

Add mmCIF as a first-class coordinate input/output path while preserving PHEAT's
canonical atom-structure JSON and residue-geometry JSON formats.

## Decisions

- Preserve full chain identifiers internally. PHEAT's `chain_id` is a string, so
  mmCIF author or label chain IDs can remain intact in atom-structure JSON,
  residue-geometry JSON, and mmCIF output.
- Keep legacy PDB output explicit about its format limit. PDB has a
  one-character chain ID column, so direct PDB writing rejects longer chain IDs
  unless the caller opts into truncation with `allow_chain_truncation=True` or
  `--allow-pdb-chain-truncation`.
- Default mmCIF parsing to author identifiers because those generally match
  deposited PDB-style residue numbering. Expose `--chain-id-source label` for
  workflows that prefer label asym/sequence IDs.
- Preserve explicit disulfide connectivity through mmCIF `_struct_conn` records
  where available, matching the existing PDB `SSBOND`/`CONECT` behavior.
- Keep mmCIF output optional in generated roundtrip artifacts to avoid growing
  default example outputs.

## Implementation Checklist

- Add `pheat.mmcif` with lazy Biopython-backed mmCIF parsing and compact mmCIF
  writing.
- Add CLI commands for `mmcif-to-structure`, `structure-to-mmcif`, and
  `mmcif-to-geometry`.
- Allow `geometry-to-structure --mmcif-output` and mmCIF inputs for scoring and
  radius-of-gyration commands.
- Allow web uploads of PDB or mmCIF files, with an option to write mmCIF
  downloads for aligned structures.
- Add roundtrip artifact support for optional aligned mmCIF files.
- Add tests for mmCIF identifier namespaces, hydrogen dropping, disulfide
  preservation, CLI conversions, long-chain PDB truncation behavior, roundtrip
  artifacts, and web upload handling.
- Update README and example documentation to describe the PDB chain ID limit and
  the mmCIF preservation path.

## Questions And Answers

- Why did PHEAT appear to use one-character chain IDs?

  The original coordinate writer was PDB-only, and PDB reserves one character for
  the chain ID. The internal model does not require one-character chains; mmCIF
  support preserves full IDs, while PDB output now requires explicit truncation.

- Should mmCIF change residue-geometry JSON?

  No. mmCIF is a coordinate container. Residue-geometry JSON already stores the
  residue names, chain IDs, residue sequence numbers, optional backbone geometry,
  chi angles, and disulfide annotations needed by PHEAT.

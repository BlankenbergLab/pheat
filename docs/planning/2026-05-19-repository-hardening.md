# 2026-05-19 Repository Hardening

## Prompt

Examine the repository, identify what is missing or can be improved, and implement
the resulting hardening plan.

## Decisions

- Keep current PHEAT file formats, CLI commands, scoring behavior, and generated
  report semantics stable.
- Add build/wheel smoke validation because editable installs can hide missing
  package-data problems.
- Keep the core package dependency-light; remove optional dependencies that are
  not used directly by PHEAT source code.
- Harden network fetch paths with atomic writes and test-injectable downloaders.
- Share report/web table sorting and Mol* viewer JavaScript from package code so
  the two HTML surfaces do not drift.
- Add focused tests for new hardening behavior while preserving the existing
  backend regression coverage.

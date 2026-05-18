# Mol* Report Assets

The generated 2MU7 combinatorial roundtrip report uses browser viewer assets from
`molstar@5.9.0`.

- Source package: https://www.npmjs.com/package/molstar/v/5.9.0
- Upstream repository: https://github.com/molstar/molstar
- License: MIT; the generated `LICENSE` file is copied from the npm package.
- Generated files: `molstar.js`, `molstar.css`, `LICENSE`, and `manifest.json`
  under PHEAT's platform-aware runtime cache.

These files are intentionally generated instead of committed. Build them with:

```bash
pheat molstar install
```

`make molstar` wraps the same command for development workflows. The install
command downloads the pinned npm package, writes the viewer bundle to the active
PHEAT Mol* asset directory, and removes upstream `sourceMappingURL` comments
because source maps are not generated for local reports. Use:

```bash
pheat molstar status
```

to show the active platform cache path or `PHEAT_MOLSTAR_DIR` override.

# Deployment

The public documentation is built as a static MkDocs site and deployed through
Cloudflare Workers static assets.

## Production

- Production domain: `pheat.tools.blankenberglab.org`
- Worker name: `pheat-tools-blankenberglab-org`
- Production branch: `main`
- Static asset directory: `site`

Use these Cloudflare build settings:

```text
Build command:
python -m pip install --upgrade pip && python -m pip install -r docs/requirements.txt && python -m mkdocs build --strict

Deploy command:
npx wrangler@4 deploy

Environment variable:
PYTHON_VERSION=3.12
```

The `wrangler.jsonc` file declares `site/` as the static asset directory and
attaches the Worker custom domain.

## PyPI Package Releases

Python package releases are published by the GitHub Actions workflow in
`.github/workflows/publish.yml`.

Production uploads use PyPI Trusted Publishing. Configure a PyPI pending
publisher before the first release, or a normal publisher after the project
exists, with these values:

- Project name: `pheat`
- Owner: `BlankenbergLab`
- Repository: `pheat`
- Workflow: `publish.yml`
- Environment name: `pypi`

The production publish job runs only when a GitHub Release is published. The
release tag must match the version in `pyproject.toml`, either as `0.1.0` or
`v0.1.0`, and the tagged commit must be present on `main`.

To publish a release:

1. Update `project.version` in `pyproject.toml`.
2. Merge the release commit to `main` and wait for CI to pass.
3. Create a matching tag, for example `v0.1.0`.
4. Create and publish a GitHub Release from that tag.
5. Confirm that the `Publish` workflow succeeds and that the release appears on
   `https://pypi.org/p/pheat`.

The same workflow can be run manually from the GitHub Actions UI to publish to
TestPyPI. Configure a TestPyPI Trusted Publisher with the same repository and
workflow values, but with environment name `testpypi`. TestPyPI install checks
can use:

```bash
python -m pip install -i https://test.pypi.org/simple/ pheat
```

## Local Preview

Install only the documentation toolchain:

```bash
make docs-deps
```

Serve the site locally:

```bash
mkdocs serve -a 127.0.0.1:8000
```

Build the static output:

```bash
make docs
```

The generated `site/` directory is build output and should not be committed.

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
npx wrangler deploy

Environment variable:
PYTHON_VERSION=3.12
```

The `wrangler.jsonc` file declares `site/` as the static asset directory and
attaches the Worker custom domain.

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

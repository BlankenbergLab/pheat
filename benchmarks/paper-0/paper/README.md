# paper-0 Draft PDF

This directory is the output location for assembled paper-0 draft PDFs.

The generated PDF is for internal review. It combines manuscript planning notes
with demo workflow artifacts, and the demo outputs remain workflow checks rather
than scientific benchmark evidence.

Build it from the repository root with:

```bash
make -C benchmarks/paper-0 pdf-demo PYTHON=../../.conda/free-energy-calc/bin/python
```

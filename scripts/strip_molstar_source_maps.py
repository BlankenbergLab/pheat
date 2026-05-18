#!/usr/bin/env python3
"""Strip trailing Mol* source-map comments from generated viewer assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from pheat.molstar_assets import molstar_asset_status
from pheat.molstar_assets import strip_molstar_source_maps as _strip_molstar_source_maps


def strip_molstar_source_maps(vendor_dir: Path) -> list[Path]:
    """Remove source-map references for the Mol* files PHEAT vendors locally."""
    return _strip_molstar_source_maps(vendor_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip trailing sourceMappingURL comments from local Mol* viewer assets."
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help="Directory containing molstar.js and molstar.css. Defaults to PHEAT's Mol* cache.",
    )
    args = parser.parse_args(argv)

    strip_molstar_source_maps(args.vendor_dir or molstar_asset_status().path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

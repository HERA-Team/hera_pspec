"""Write the legacy HDF5 fixtures in tests/data/legacy/.

These files record the HDF5 layout written by hera_pspec at commit f942d1a,
before the format gained a version number and unit attributes. They must not
be regenerated with newer code: a newer writer produces the new format, and the
tests that read them would then no longer exercise the legacy reader. See
tests/data/legacy/README.md.

Usage::

    uv run python tests/data/regenerate_legacy_fixtures.py           # write missing files only
    uv run python tests/data/regenerate_legacy_fixtures.py --force   # overwrite (do not, see README)

The builders live in tests/legacy_fixtures.py, which the tests also import.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.legacy_fixtures import LEGACY_DIR, LEGACY_FIXTURES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "files",
        nargs="*",
        help="Names of the files to write (default: all). "
        f"Available: {', '.join(LEGACY_FIXTURES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist. See tests/data/legacy/README.md.",
    )
    args = parser.parse_args(argv)

    names = args.files or list(LEGACY_FIXTURES)
    unknown = [name for name in names if name not in LEGACY_FIXTURES]
    if unknown:
        parser.error(f"unknown file(s): {', '.join(unknown)}")

    existing = [name for name in names if (LEGACY_DIR / name).exists()]
    if existing and not args.force:
        print(
            "Refusing to overwrite existing legacy fixtures without --force:\n  "
            + "\n  ".join(str(LEGACY_DIR / name) for name in existing)
            + "\nThey were written by hera_pspec f942d1a and must not be "
            "regenerated with newer code; see tests/data/legacy/README.md.",
            file=sys.stderr,
        )
        return 1

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = LEGACY_DIR / name
        if path.exists():
            path.unlink()
        LEGACY_FIXTURES[name](path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

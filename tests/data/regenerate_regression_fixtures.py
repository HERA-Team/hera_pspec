"""Regenerate the expected values used by tests/test_regression_numerics.py.

Do not run this to make a failing regression test pass. The fixtures pin the
numbers hera_pspec produced before the astropy-units migration; changing them
needs explicit approval from the maintainer and an explanation in the commit
body (see tests/data/README.md).

Usage::

    uv run python tests/data/regenerate_regression_fixtures.py            # write missing fixtures only
    uv run python tests/data/regenerate_regression_fixtures.py CASE ...   # selected cases
    uv run python tests/data/regenerate_regression_fixtures.py --force    # overwrite existing fixtures

The inputs are built by tests/regression_cases.py, the same module the tests
use, so a fixture always matches the test that reads it.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from tests.regression_cases import CASES, FIXTURE_DIR, compute_case  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cases",
        nargs="*",
        help="Names of the cases to regenerate (default: all). "
        f"Available: {', '.join(CASES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite fixtures that already exist. Needs maintainer approval.",
    )
    args = parser.parse_args(argv)

    names = args.cases or list(CASES)
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {', '.join(unknown)}")

    existing = [name for name in names if CASES[name].fixture_path.exists()]
    if existing and not args.force:
        print(
            "Refusing to overwrite existing regression fixtures without --force:\n  "
            + "\n  ".join(str(CASES[name].fixture_path) for name in existing)
            + "\nRegenerating needs explicit maintainer approval; see "
            "tests/data/README.md.",
            file=sys.stderr,
        )
        return 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        outputs = compute_case(name)
        np.savez_compressed(CASES[name].fixture_path, **outputs)
        print(f"wrote {CASES[name].fixture_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

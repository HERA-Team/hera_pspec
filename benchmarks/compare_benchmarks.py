"""Compare two run_benchmarks.py JSON outputs.

Usage::

    uv run python benchmarks/compare_benchmarks.py BASELINE.json NEW.json

Prints the median time of every workflow in both runs and the relative change
``(new - baseline) / baseline``. Exits with status 1 when any workflow is more
than ``--threshold`` (default 0.10, i.e. 10%) slower than the baseline, and
with status 0 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def describe_run(label: str, run: dict[str, Any]) -> str:
    env = run.get("environment", {})
    git = env.get("git", {}) or {}
    commit = (git.get("commit") or "unknown")[:10]
    dirty = " (dirty)" if git.get("dirty") else ""
    repeat = run.get("settings", {}).get("repeat", "?")
    return (
        f"{label}: {commit}{dirty} on {git.get('branch')}, "
        f"{env.get('timestamp')}, repeat={repeat}, python {env.get('python')}, "
        f"numpy {env.get('numpy')}, astropy {env.get('astropy')}, "
        f"pyuvdata {env.get('pyuvdata')}"
    )


def environment_differences(baseline: dict[str, Any], new: dict[str, Any]) -> list[str]:
    keys = ["platform", "machine", "cpu_count", "python", "numpy", "scipy"]
    keys += ["astropy", "pyuvdata", "blas", "thread_env"]
    base_env = baseline.get("environment", {})
    new_env = new.get("environment", {})
    return [
        f"{key}: {base_env.get(key)} -> {new_env.get(key)}"
        for key in keys
        if base_env.get(key) != new_env.get(key)
    ]


def compare(
    baseline: dict[str, Any], new: dict[str, Any], threshold: float
) -> tuple[list[str], list[str]]:
    """Return the table lines and the names of workflows over the threshold."""
    base_workflows = baseline["workflows"]
    new_workflows = new["workflows"]
    header = (
        f"{'workflow':<24} {'baseline':>10} {'new':>10} {'delta':>8} "
        f"{'new std%':>9}  status"
    )
    lines = [header, "-" * len(header)]
    slower = []
    for name, base_result in base_workflows.items():
        if name not in new_workflows:
            lines.append(
                f"{name:<24} {base_result['median_s']:10.3f} {'-':>10}  not run"
            )
            continue
        new_result = new_workflows[name]
        base_median = base_result["median_s"]
        new_median = new_result["median_s"]
        delta = (new_median - base_median) / base_median
        std_pct = 100 * new_result["std_s"] / new_result["mean_s"]
        if delta > threshold:
            status = "SLOWER"
            slower.append(name)
        elif delta < -threshold:
            status = "faster"
        else:
            status = "ok"
        if base_result.get("sizes") != new_result.get("sizes"):
            status += " (input sizes differ)"
        lines.append(
            f"{name:<24} {base_median:10.3f} {new_median:10.3f} "
            f"{100 * delta:+7.1f}% {std_pct:8.1f}%  {status}"
        )
    lines.extend(
        f"{name:<24} {'-':>10} {result['median_s']:10.3f}  new workflow"
        for name, result in new_workflows.items()
        if name not in base_workflows
    )
    lines.append("(median times in seconds)")
    return lines, slower


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline", type=Path, help="baseline JSON")
    parser.add_argument("new", type=Path, help="JSON to compare against it")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="relative slowdown that counts as a regression (default 0.10)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = load(args.baseline)
    new = load(args.new)

    print(describe_run("baseline", baseline))
    print(describe_run("new     ", new))
    differences = environment_differences(baseline, new)
    if differences:
        print("environment differs, timings may not be comparable:")
        for difference in differences:
            print(f"  {difference}")
    print()

    lines, slower = compare(baseline, new, args.threshold)
    print("\n".join(lines))
    print()
    if slower:
        print(
            f"FAIL: {len(slower)} workflow(s) more than {100 * args.threshold:.0f}% "
            f"slower: {', '.join(slower)}"
        )
        return 1
    print(f"OK: no workflow more than {100 * args.threshold:.0f}% slower")
    return 0


if __name__ == "__main__":
    sys.exit(main())

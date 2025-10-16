"""Command-line tool for plotting pytest-jax-bench results.

Usage examples:
  pjax-plot --bench-dir .benchmarks --mode all --save png
  pjax-plot --bench-dir .benchmarks --mode each --save png --xaxis run
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import plots


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pjax-plot", description="Plot pytest-jax-bench results")
    p.add_argument("-d", "--bench-dir", default=".benchmarks", help="Directory containing benchmark .csv files")
    p.add_argument("-m", "--mode", choices=("all", "each"), default="each", help="Plot all benchmarks together ('all') or create individual plots ('each')")
    p.add_argument("-x", "--xaxis", choices=("commit", "run"), default="commit", help="x-axis for plots")
    p.add_argument("-s", "--save", choices=("png", "pdf", "show"), default="png", help="Save plots to files (png/pdf) or 'show' to only show")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    save = None if args.save == "show" else args.save

    if args.mode == "all":
        plots.plot_all_benchmarks_together(bench_dir=args.bench_dir, xaxis=args.xaxis, save=save)
    else:
        plots.plot_all_benchmarks_individually(bench_dir=args.bench_dir, xaxis=args.xaxis, save=save)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line tool for plotting pytest-jax-bench results.

Usage examples:
  pjax-plot -d .benchmarks
  pjax-plot --bench-dir .benchmarks --save pdf --xaxis commit
"""
from __future__ import annotations

import argparse
from typing import Optional

from . import plots


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pjax-plot", description="Plot pytest-jax-bench results")
    p.add_argument("-d", "--bench-dir", default=".benchmarks", help="Directory containing benchmark .csv files")
    p.add_argument("-x", "--xaxis", choices=("commit", "run"), default="run", help="x-axis for plots")
    p.add_argument("-s", "--save", choices=("png", "pdf"), default="png", help="Save plots to files (png/pdf) or 'show' to only show")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    plots.plot_all_benchmarks(bench_dir=args.bench_dir, xaxis=args.xaxis, save=args.save, trep="print")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

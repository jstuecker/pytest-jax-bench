from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from .data import BenchData, load_bench_data
from .utils import folded_constants_bytes

import pytest
import subprocess
import numpy as np

import warnings

import jax

# ---------------------------
# Pytest plugin configuration
# ---------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("Pytest-Jax-Bench (ptjb)")
    group.addoption(
        "--ptjb-output-dir",
        action="store",
        default=".benchmarks",
        help="Directory for output files (default: .benchmarks)",
    )
    group.addoption(
        "--ptjb-basetag",
        action="store",
        default="base",
        help="Modify basetag for this benchmark run (default: base)",
    )
    group.addoption(
        "--ptjb-no-compare",
        action="store_true",
        default=False,
        help="Do not compare to previous runs of same commit (default: False)",
    )
    group.addoption(
        "--ptjb-plot-all",
        action="store_true",
        default=False,
        help="Generate a joint summary plot after the test run (default: False)",
    )
    group.addoption(
        "--ptjb-plot-each",
        action="store_true",
        default=False,
        help="Generate plots for each benchmark after the test run (default: False)",
    )
    group.addoption(
        "--ptjb-plot-xaxis",
        action="store",
        default="run",
        help="X-axis for plots - can be 'commit' or 'run' (default: run)",
    )


def select_commit_runs(data, commit, tag=None):
    commit_base = commit.rstrip("+")
    mask = (data["commit"] == commit_base) | (data["commit"] == commit_base + "+")
    if tag is not None:
        mask &= (data["tag"] == tag)
    return data[mask]

def get_comparison_data(data, tag="base"):
    if len(data) == 0:
        return None, None

    new_data = data[data["tag"] == tag][-1]
    comparison_data = select_commit_runs(data, new_data["commit"], new_data["tag"])[0]

    return new_data, comparison_data

def _colored(s: str, color: str) -> str:
    # Respect NO_COLOR or dumb terminals
    if os.getenv("NO_COLOR") is not None or os.getenv("TERM") == "dumb":
        return s
    colors = {"green": "\x1b[32m", "yellow": "\x1b[33m", "red": "\x1b[31m", "reset": "\x1b[0m",
              "grey": "\x1b[90m"}
    return f"{colors.get(color, '')}{s}{colors['reset']}"

def _colored_diff(txt, v1, v2, tol=0.):
    if v2 < v1 - tol:
        color = "green"
    elif v2 > v1 + tol:
        color = "red"
    else:
        color = "grey"
    return _colored(txt, color)

def nodeid_to_path(test_nodeid: str, output_dir: str = ".") -> str:
    test_file, test_name = test_nodeid.split("::")
    test_file = re.sub(r"[^A-Za-z0-9._-]+", ":", test_file)
    test_file = re.sub(r"\.py", "", test_file)

    return os.path.join(output_dir, f"{test_file}::{test_name}.csv")

def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus : pytest.ExitCode, config: pytest.Config) -> None:
    if config.getoption("-v") >= 0:
        forked = config.getoption("--forked", False)

        output_dir = config.getoption("--ptjb-output-dir")
        no_compare = config.getoption("--ptjb-no-compare", False)

        if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
            return

        entries = []
        def add_entry(data, nodeid, tag=None):
            new, old = get_comparison_data(data, tag=tag)
            if new is None: 
                return
            
            same = (new["run_id"] == old["run_id"]) or no_compare

            entry = {}

            def str_and_len(s):
                return s, len(s)

            entry["Test"] = str_and_len(nodeid)
            entry["C.Run"] = str_and_len(f'{new["commit_run"]}' if same else f'{old["commit_run"]}->{new["commit_run"]}')
            entry["Tag"] = str_and_len(new["tag"])

            def compare_perf(key, std=0., tol=None, only_different=False):
                if tol is None: tol = 2.*std
                v1, v2 = old[key], new[key]
                if same:
                    txt = f"{v2:.1f}"
                elif only_different and np.abs(v1 - v2) <= std*2.:
                    txt = ""
                elif np.isnan(v1) and np.isnan(v2):
                    txt = ""
                elif std > 0.:
                    txt = f"{v1:.1f}->{v2:.1f}+-{std:.1f}"
                else:
                    txt = f"{v1:.1f}->{v2:.1f}"
                return _colored_diff(txt, v1, v2, tol=tol), len(txt)

            entry["Compile(ms)"] = compare_perf("compile_ms", tol=np.maximum(new["compile_ms"]*0.2, 20.))
            entry["Jit-Run(ms)"] = compare_perf("jit_mean_ms", std=np.sqrt(new["jit_std_ms"]**2+old["jit_std_ms"]**2))

            def compare_mem(key, only_different=False, min_mb=0.):
                v1 = old[key]/1024.**2 if old[key] >= 0 else np.nan
                v2 = new[key]/1024.**2 if new[key] >= 0 else np.nan
                if same:
                    txt = f"{v2:.3g}"
                elif only_different and v1 == v2:
                    txt = ""
                elif np.isnan(v1) and np.isnan(v2):
                    txt = ""
                elif max(np.nan_to_num(v1,np.inf), np.nan_to_num(v2,np.inf)) >= min_mb:
                    txt = f"{v1:.3g}->{v2:.3g}"
                else:
                    txt = ""
                return _colored_diff(txt, v1, v2), len(txt)

            entry["Eager-Run(ms)"] = compare_perf("eager_mean_ms", std=np.sqrt(new["eager_std_ms"]**2+old["eager_std_ms"]**2))
            entry["Jit-Peak(MB)"] = compare_mem("jit_peak_bytes")
            entry["Jit-Const(MB)"] = compare_mem("jit_constants_bytes", min_mb=0.1)
            eager_str = "Eager-Peak(MB)" #if forked else _colored("Eager-Mem(MB) (invalid!)", "yellow")
            entry[eager_str] = compare_mem("eager_peak_bytes")

            entries.append(entry)

        for report in terminalreporter.getreports("passed"):
            path = nodeid_to_path(report.nodeid, output_dir=output_dir)
            if not os.path.exists(path):
                continue
            data = load_bench_data(path, report.nodeid)
            run = data[-1]["run_id"]
            tags = np.unique(data["tag"][data["run_id"] == run])
            for tag in tags:
                add_entry(data, report.nodeid, tag=tag)

        if len(entries) == 0:
            return
        
        terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) results")
        if not forked:
            terminalreporter.write_line(_colored("Warning: Eager mode memory report is only valid when using --forked!", "yellow"))

        allkeys = entries[0].keys()
        
        lines = ["" for _ in range(len(entries) + 1)]
        for key in allkeys:
            maxlen = max(entry[key][1] for entry in entries)
            if maxlen == 0:
                continue
            maxlen = max(maxlen, len(key))

            lines[0] += f"{key:^{maxlen+2}}"

            for i,entry in enumerate(entries):
                corrected_len = maxlen + len(entry[key][0]) - entry[key][1]
                lines[i+1] += f"{entry[key][0]:<{corrected_len + 2}}"

        for line in lines:
            terminalreporter.write_line(line)
    if config.getoption("--ptjb-plot-all") or config.getoption("--ptjb-plot-each"):
        try:
            from . import plots
        except ImportError as e:
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) plotting skipped")
            terminalreporter.write_line("Failed to import plotting extension.")
            terminalreporter.write_line(str(e))
            return

        xaxis = config.getoption("--ptjb-plot-xaxis")
        if xaxis not in ("commit", "run"):
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) plotting skipped")
            terminalreporter.write_line(f"Unknown xaxis {xaxis}, must be 'commit', 'run' or 'tag'")
            return

        if config.getoption("--ptjb-plot-all"):
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) summary plot")

            plots.plot_all_benchmarks_together(bench_dir=output_dir, xaxis=xaxis, save="png")
            terminalreporter.write_line(f"Summary plot saved to {os.path.join(output_dir, 'all_benchmarks.png')}")

        if config.getoption("--ptjb-plot-each"):
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) all benchmarks plots")
            
            plots.plot_all_benchmarks_individually(bench_dir=output_dir, xaxis=xaxis, save="png")
            terminalreporter.write_line(f"All benchmarks plots saved to {os.path.join(output_dir)}")

# ---------------------------
# StableHLO memory estimate utils (very rough)
# ---------------------------

def _now_iso() -> str:
    # Use timezone-aware UTC datetime to avoid deprecation of utcnow()
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _git_commit_short() -> str:
    """Return short git commit id (7 chars) or 'unknown' if not available."""
    try:
        # Use git directly to avoid adding heavy deps. This will fail cleanly if not a git repo.
        out = subprocess.check_output(["git", "describe", "--always", "--dirty"], stderr=subprocess.DEVNULL)
        out = out.decode("utf-8").strip()
        if out[-6:] == "-dirty":
            out = out[:-6] + "+"
        return out
    except Exception:
        return "unknown"

def _get_run_info(path):
    if os.path.exists(path):
        data = load_bench_data(path)
    else:
        data = np.zeros((0,), dtype=BenchData.data_type())

    if len(data) > 0:
        run_id = np.max(data["run_id"]) + 1
    else:
        run_id = 0

    current_commit = _git_commit_short()
    runs = select_commit_runs(data, current_commit)
    if len(runs) > 0:
        commit_run = np.max(runs["commit_run"]) + 1
    else:
        commit_run = 0

    return run_id, current_commit, commit_run

def _get_backend() -> str:
    try:
        return jax.default_backend()  # "cpu", "gpu", "tpu"
    except Exception:
        return "unknown"

# ---------------------------
# The JaxBench core object
# ---------------------------

class JaxBench:
    def __init__(self, request: pytest.FixtureRequest,
                 jit_rounds: int = 20, jit_warmup: int = 1, eager_rounds = 5, eager_warmup = 1) -> None:
        self.request = request
        self.config = request.config
        self.forked = self.config.getoption("--forked", False)
        self.output_dir = self.config.getoption("--ptjb-output-dir")
        self.tag = self.config.getoption("--ptjb-basetag", "default")
        self.jit_rounds = int(jit_rounds)
        self.jit_warmup = int(jit_warmup)
        self.eager_rounds = int(eager_rounds)
        self.eager_warmup = int(eager_warmup)

        node_id = self.request.node.nodeid
        path = nodeid_to_path(node_id, output_dir=self.output_dir)
        self.run_id, self.commit, self.commit_run = _get_run_info(path)

        self.measurement = 0

    def compile_time_ms(self, fn_jit: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
        """Return compilation time in milliseconds (excluding lowering)."""
        jax.clear_caches()
        t1 = time.perf_counter()
        lowered = fn_jit.lower(*args, **kwargs)
        compiled = lowered.compile()
        t2 = time.perf_counter()
        return (t2 - t1) * 1000.0, lowered, compiled

    def profile_jit(self, fn_jit: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return (mean_ms, std_ms, rounds, warmup)."""
        # warmup
        for _ in range(self.jit_warmup):
            fn_jit(*args, **kwargs)
        jax.block_until_ready(fn_jit(*args, **kwargs))

        times = []
        for _ in range(self.jit_rounds):
            t0 = time.perf_counter()
            out = fn_jit(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        if self.jit_rounds >= 1:
            return np.mean(times), np.std(times)
        else:
            return np.nan, np.nan
    
    def profile_eager(self, fn, *args, **kwargs):
        # Capture memory on first run
        if self.eager_warmup > 0:
            jax.block_until_ready(fn(*args, **kwargs))
            eager_peak_bytes = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]
        else:
            eager_peak_bytes = -1

        for _ in range(self.eager_warmup - 1):
            fn(*args, **kwargs)
        jax.block_until_ready(fn(*args, **kwargs))

        times = []
        for _ in range(self.eager_rounds):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            if eager_peak_bytes < 0:
                eager_peak_bytes = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]

        if (not self.forked) or (self.measurement > 0):
            # Profile is invalid without forking, because peakr memory may be inherited from other
            # processes.
            eager_peak_bytes = -1

        if self.eager_rounds >= 1:
            return np.mean(times), np.std(times), eager_peak_bytes
        else:
            return np.nan, np.nan, eager_peak_bytes

    # ---------- high-level orchestration ----------

    def measure(
        self,
        fn: Optional[Callable[..., Any]] = None,
        fn_jit: Optional[Callable[..., Any]] = None,
        tag = None,
        *args: Any,
        write = True,
        **kwargs: Any,
    ) -> BenchData:
        """Run selected measurements and write one numeric row per call."""

        res = BenchData(jit_rounds=self.jit_rounds, jit_warmup=self.jit_warmup,
                        eager_rounds=self.eager_rounds, eager_warmup=self.eager_warmup)
        
        # First do eager profiling, to get a good idea of the memory
        if fn is not None:
            res.eager_mean_ms, res.eager_std_ms, res.eager_peak_bytes = self.profile_eager(fn, *args, **kwargs)

        if fn_jit is not None:
            res.compile_ms, lowered, fn_compiled = self.compile_time_ms(fn_jit, *args, **kwargs)
            graph_mem = fn_compiled.memory_analysis()

            res.jit_constants_bytes = graph_mem.generated_code_size_in_bytes
            res.jit_peak_bytes = graph_mem.peak_memory_in_bytes
            res.jit_temporary_bytes = graph_mem.temp_size_in_bytes
            try:
                # It seems likely the handling of jax's folded constants will change in the future
                # Therefore, we catch exceptions here for now...
                res.jit_constants_bytes = folded_constants_bytes(lowered)
            except Exception as e:
                warnings.warn(f"Failed to compute folded_constants_bytes ({e})", RuntimeWarning)
                res.jit_constants_bytes = 0

            if self.jit_rounds > 0:
                res.jit_mean_ms, res.jit_std_ms = self.profile_jit(fn_jit, *args, **kwargs)

        if write:
            self._write_row(res=res, tag=tag)
        
        self.measurement += 1

        return res

    # ---------- IO ----------

    def _write_row(self, res: BenchData, tag: str = "default") -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        node_id = self.request.node.nodeid
        path = nodeid_to_path(node_id, output_dir=self.output_dir)
        res.run_id, res.commit, res.commit_run = self.run_id, self.commit, self.commit_run
        res.tag = self.tag if tag is None else tag

        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        with open(path, "a", encoding="utf-8") as f:
            if file_size == 0:
                print("Run data will be written to:", path)
                # Add a header
                f.write(f"# pytest-jax-bench\n")
                f.write(f"# created: {_now_iso()}\n")
                f.write(f"# test_nodeid: {node_id}\n")
                f.write(f"# backend: {_get_backend()}\n")
                f.write(f"# device: {jax.devices()[0].device_kind}\n")
                f.write(f"# First commit: {res.commit}\n")
                f.write(res.get_column_header())
                f.write("\n")
            
            # Write our test results:
            f.write(res.formatted_line() + "\n")

@pytest.fixture
def jax_bench(request: pytest.FixtureRequest) -> Callable[..., JaxBench]:
    def _factory(*args, **kwargs) -> JaxBench:
        return JaxBench(request, *args, **kwargs)

    return _factory
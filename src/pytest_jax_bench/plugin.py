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

import math
import warnings

import jax

# ---------------------------
# Pytest plugin configuration
# ---------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("bench-jax")
    group.addoption(
        "--bench-jax-output-dir",
        action="store",
        default=".benchmarks",
        help="Directory for output files (default: .benchmarks)",
    )

benchmarks = {}
def pytest_configure(config: pytest.Config) -> None:
    global benchmarks
    benchmarks = {}
    outdir = config.getoption("--bench-jax-output-dir")
    os.makedirs(outdir, exist_ok=True)

def select_commit_runs(data, commit):
    commit_base = commit.rstrip("+")
    mask = np.where((data["commit"] == commit_base) | (data["commit"] == commit_base + "+"))[0]
    return data[mask]

def summary_select_commits(data):
    if len(data) == 0:
        return None, None

    new_data = data[-1]

    # Find the comparison data
    # If there is no other entry with the same commit hash, we consider the last entry of previous commit
    # If the commit is dirty, we use the first entry with the same commit hash
    commit = new_data["commit"]

    data_same_commit = select_commit_runs(data, commit)

    if len(data_same_commit) >= 1:
        comparison_data = data_same_commit[0]
    elif len(data) >= 2:
        data_last_commit = data[-2]["commit"]
    else: # as a fallback compare with self (to keep code simple)
        data_last_commit = new_data

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

        output_dir = config.getoption("--bench-jax-output-dir")

        terminalreporter.write_sep("=", "JAX benchmark results in ms and MB")
        if not forked:
            terminalreporter.write_line(_colored("Warning: Eager mode memory report is only valid when using --forked!", "yellow"))

        entries = []
        for report in terminalreporter.getreports("passed"):
            path = nodeid_to_path(report.nodeid, output_dir=output_dir)
            data = load_bench_data(path)
            new, old = summary_select_commits(data)
            if new is None: 
                continue

            entry = {}

            def str_and_len(s):
                return s, len(s)

            entry["Test"] = str_and_len(report.nodeid)
            entry["Commit(Run)"] = str_and_len(f'{old["commit"]}({old["commit_run"]})->{new["commit"]}({new["commit_run"]})')

            def compare_perf(key, std=0., tol=None, only_different=False):
                if tol is None: tol = 2.*std
                v1, v2 = old[key], new[key]
                if only_different and np.abs(v1 - v2) <= std*2.:
                    txt = ""
                if std > 0.:
                    txt = f"{v1:.2f}->{v2:.2f}+-{std:.2f}"
                else:
                    txt = f"{v1:.2f}->{v2:.2f}"
                return _colored_diff(txt, v1, v2, tol=tol), len(txt)
            
            def compare_mem(key, only_different=False, min_mb=0.):
                v1, v2 = old[key], new[key]
                if only_different and v1 == v2:
                    txt = ""
                elif max(np.nan_to_num(v1,np.inf), np.nan_to_num(v2,np.inf)) >= min_mb:
                    txt = f"{v1:.3g}->{v2:.3g}"
                else:
                    txt = ""
                return _colored_diff(txt, v1, v2), len(txt)

            entry["Compile(ms)"] = compare_perf("compile_ms", tol=new["compile_ms"]*0.1)
            entry["Jit-Run(ms)"] = compare_perf("run_mean_ms", std=np.sqrt(new["run_std_ms"]**2+old["run_std_ms"]**2))
            entry["Eager-Run(ms)"] = compare_perf("eager_mean_ms", std=np.sqrt(new["eager_std_ms"]**2+old["eager_std_ms"]**2), only_different=True)
            entry["Jit-Mem(MB)"] = compare_mem("graph_peak_memory_mb")
            entry["Constants(MB)"] = compare_mem("graph_constants", min_mb=0.1)
            eager_str = "Eager-Mem(MB)" if forked else _colored("Eager-Mem(MB) (invalid!)", "yellow")
            entry[eager_str] = compare_mem("eager_peak_memory_mb")

            entries.append(entry)

        if len(entries) == 0:
            terminalreporter.write_line("No benchmark data collected.")
            return

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

def _get_backend() -> str:
    try:
        return jax.default_backend()  # "cpu", "gpu", "tpu"
    except Exception:
        return "unknown"

# ---------------------------
# The BenchJax core object
# ---------------------------

class BenchJax:
    def __init__(self, request: pytest.FixtureRequest, config: pytest.Config,
                 run_rounds: int = 20, run_warmup: int = 3, eager_rounds = 5, eager_warmup = 1) -> None:
        self.request = request
        self.forked = config.getoption("--forked", False)
        self.config = config
        self.output_dir = config.getoption("--bench-jax-output-dir")
        self.run_rounds = int(run_rounds)
        self.run_warmup = int(run_warmup)
        self.eager_rounds = int(eager_rounds)
        self.eager_warmup = int(eager_warmup)
        self._fn = None
        self._jitted = None
        self._lowered = None
        self._compiled = None  # may be set by compile_time()

    # ---------- measurement pieces ----------

    def compile_time_ms(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
        """Return compilation time in milliseconds (excluding lowering)."""
        jitted = jax.jit(fn)
        jax.clear_caches()
        t1 = time.perf_counter()
        lowered = jitted.lower(*args, **kwargs)
        compiled = lowered.compile()
        t2 = time.perf_counter()
        return (t2 - t1) * 1000.0, lowered, compiled

    def profile_run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return (mean_ms, std_ms, rounds, warmup)."""
        compiled = self._compiled if self._compiled is not None else jax.jit(fn)

        # warmup
        for _ in range(self.run_warmup):
            compiled(*args, **kwargs)
        jax.block_until_ready(compiled(*args, **kwargs))

        times = []
        for _ in range(self.run_rounds):
            t0 = time.perf_counter()
            out = compiled(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return np.mean(times), np.std(times) if self.run_rounds > 1 else np.nan
    
    def profile_eager(self, fn, *args, **kwargs):
        # Capture memory on first run
        if self.eager_warmup > 0:
            fn(*args, **kwargs).block_until_ready()
            eager_peak_mem = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]
        else:
            eager_peak_mem = None

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
            if eager_peak_mem is None:
                eager_peak_mem = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]

        if not self.forked:
            # Profile is invalid without forking, because peakr memory may be inherited from other
            # processes.
            eager_peak_mem = np.nan 

        return np.mean(times), np.std(times), eager_peak_mem if self.run_rounds > 1 else np.nan

    # ---------- high-level orchestration ----------

    def measure(
        self,
        fn: Callable[..., Any],
        *args: Any,
        write = True,
        **kwargs: Any,
    ) -> BenchData:
        """Run selected measurements and write one numeric row per call."""
        backend = _get_backend()
        name = getattr(fn, "__name__", "anonymous")

        res = BenchData(jit_rounds=self.run_rounds, jit_warmup=self.run_warmup,
                        eager_rounds=self.eager_rounds, eager_warmup=self.eager_warmup)
        
        # First do eager profiling, to get a good idea of the memory
        if self.eager_rounds > 0:
            res.eager_mean_ms, res.eager_std_ms, res.eager_peak_memory = self.profile_eager(fn, *args, **kwargs)

        res.compile_ms, lowered, compiled = self.compile_time_ms(fn, *args, **kwargs)
        graph_mem = compiled.memory_analysis()

        res.graph_constants = graph_mem.generated_code_size_in_bytes
        res.graph_peak_memory = graph_mem.peak_memory_in_bytes
        res.graph_temp_size = graph_mem.temp_size_in_bytes
        try:
            # It seems likely the handling of jax's folded constants will change in the future
            # Therefore, we catch exceptions here for now...
            res.graph_constant_memory = folded_constants_bytes(lowered)
        except Exception as e:
            warnings.warn(f"Failed to compute folded_constants_bytes ({e})", RuntimeWarning)
            res.graph_constant_memory = 0

        if self.run_rounds > 0:
            res.jit_mean_ms, res.jit_std_ms = self.profile_run(fn, *args, **kwargs)

        if write:
            self._write_row(name=name, backend=backend, row=res)

        return res
    
    def _get_run_data(self, path):
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

    # ---------- IO ----------


    def _write_row(self, *, name: str, backend: str, row: BenchData) -> None:
        row.node_id = self.request.node.nodeid
        path = nodeid_to_path(row.node_id, output_dir=self.output_dir)

        row.run_id, row.commit, row.commit_run = self._get_run_data(path)

        with open(path, "a", encoding="utf-8") as f:
            if row.run_id == 0:
                print("Run data will be written to:", path)
                # Add a header
                f.write(f"# pytest-jax-bench\n")
                f.write(f"# created: {_now_iso()}\n")
                f.write(f"# test_nodeid: {row.node_id}\n")
                f.write(f"# function: {name}\n")
                f.write(f"# backend: {backend}\n")
                f.write(f"# First commit: {row.commit}\n")
                for i,c in enumerate(row.column_descriptions()):
                    f.write(f"# ({i+1:2d}) {c}\n")
                f.write(f"#")
                for i in range(0, len(row.column_descriptions())):
                    f.write(f"     ({i+1:2d}) ")
                f.write("\n")
            
            # Add a new line
            f.write(row.formatted_line() + "\n")
        
        benchmarks[row.node_id] = path

@pytest.fixture
def bench_jax(request: pytest.FixtureRequest):
    config = request.config

    def _factory(*, rounds: int = 20, warmup: int = 3, gpu_memory: bool = False) -> BenchJax:
        return BenchJax(request, config, run_rounds=rounds, run_warmup=warmup)

    return _factory
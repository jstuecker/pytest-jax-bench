from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from .data import BenchData, load_bench_data
from .utils import folded_constants_bytes
from dataclasses import dataclass

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
        "--ptjb-graphs-svg",
        action="store_true",
        default=False,
        help="Save the compiled graph as SVG (if differing from previous)",
    )
    group.addoption(
        "--ptjb-plot",
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

    return os.path.join(output_dir, f"{test_file}::{test_name}")

def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus : pytest.ExitCode, config: pytest.Config) -> None:
    output_dir = config.getoption("--ptjb-output-dir")
    
    if config.getoption("-v") >= 0:
        forked = config.getoption("--forked", False)

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
            path = nodeid_to_path(report.nodeid, output_dir=output_dir) + ".csv"
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

    if config.getoption("--ptjb-plot"):
        nodeids = [r.nodeid for r in terminalreporter.getreports("passed")]
        paths = [nodeid_to_path(nid, output_dir=output_dir) for nid in nodeids]

        report : pytest.TestReport = terminalreporter.getreports("passed")[0]

        try:
            from . import plots
        except ImportError as e:
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) plotting skipped")
            terminalreporter.write_line("Failed to import plotting extension.")
            terminalreporter.write_line(str(e))
            return

        xaxis = config.getoption("--ptjb-plot-xaxis")
        if xaxis not in ("commit", "run"):
            terminalreporter.write_line(f"Unknown xaxis {xaxis}, must be 'commit', 'run' or 'tag'")
            return
        
        if config.getoption("-v") >= 1:
            terminalreporter.write_sep("=", "Pytest Jax Benchmark (PTJB) plots")
        plots.plot_all_benchmarks(
            paths=paths, bench_dir=output_dir, xaxis=xaxis, save="png",
            trep= terminalreporter if config.getoption("-v") >= 1 else None
        )
        if config.getoption("-v") == 0:
            terminalreporter.write_line(f"All PTJB benchmarks plots saved to {os.path.join(output_dir)}")

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
    except subprocess.CalledProcessError:
        return "unknown"

def _get_run_info(path):
    if path is not None and os.path.exists(path):
        data = load_bench_data(path)
    else:
        data = np.zeros((0,), dtype=BenchData.data_type())

    if len(data) > 0:
        run_id = int(np.max(data["run_id"]) + 1)
    else:
        run_id = 0

    current_commit = _git_commit_short()
    runs = select_commit_runs(data, current_commit)
    if len(runs) > 0:
        commit_run = int(np.max(runs["commit_run"]) + 1)
    else:
        commit_run = 0

    return run_id, current_commit, commit_run

def _get_peak_bytes() -> int:
    dev = jax.local_devices()[0]
    if dev.platform == "gpu":
        return dev.memory_stats()["peak_bytes_in_use"]
    else:
        return -1 # Forn now only GPU is supported for peak memory measurement

# ---------------------------
# The JaxBench core object
# ---------------------------

class JaxBench:
    def __init__(self, request: pytest.FixtureRequest | None = None, path = None,
                 jit_rounds: int = 20, jit_warmup: int = 1, eager_rounds = 5, eager_warmup = 1) -> None:
        
        if request is not None:
            self.forked = request.config.getoption("--forked", False)
            self.output_dir = request.config.getoption("--ptjb-output-dir")
            self.tag = request.config.getoption("--ptjb-basetag", "base")
            self.node_id = request.node.nodeid
            if path is not None:
                raise ValueError("Path is set through request. Only pass path outside of pytest.")
            self.path = nodeid_to_path(self.node_id, output_dir=self.output_dir)
            self.save_graph_svg = request.config.getoption("--ptjb-graphs-svg")

            os.makedirs(self.output_dir, exist_ok=True)
        else: # Usage outside of pytest, some aspects will be missing
            self.forked = False
            self.tag = "base"
            self.node_id = None
            self.path = path
            self.output_dir = os.path.dirname(path) if path is not None else None
            self.save_graph_svg = False

        self.jit_rounds = int(jit_rounds)
        self.jit_warmup = int(jit_warmup)
        self.eager_rounds = int(eager_rounds)
        self.eager_warmup = int(eager_warmup)

        self.run_id, self.commit, self.commit_run = _get_run_info(self.path + ".csv" if self.path is not None else None)

        self.measurement = 0

    def compile_time_ms(self, fn_jit: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
        """Return compilation time in milliseconds (excluding lowering)."""
        jax.clear_caches()
        t1 = time.perf_counter()
        lowered = fn_jit.lower(*args, **kwargs)
        compiled = lowered.compile()
        t2 = time.perf_counter()
        return float(np.round((t2 - t1) * 1000.0, 3)), lowered, compiled

    def profile_jit(self, fn_jit: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return (mean_ms, std_ms, rounds, warmup)."""
        out = None
        for _ in range(self.jit_warmup):
            out = fn_jit(*args, **kwargs)
        jax.block_until_ready(out)

        times = []
        for _ in range(self.jit_rounds):
            t0 = time.perf_counter()
            out = fn_jit(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        if self.jit_rounds >= 1:
            return out, float(np.round(np.mean(times), 3)), float(np.round(np.std(times), 3))
        else:
            return out, float('nan'), float('nan')
    
    def profile_eager(self, fn, *args, **kwargs):
        out = None
        # Capture memory on first run
        if self.eager_warmup > 0:
            out = jax.block_until_ready(fn(*args, **kwargs))
            eager_peak_bytes = _get_peak_bytes()
        else:
            eager_peak_bytes = -1

        for _ in range(self.eager_warmup - 1):
            out = fn(*args, **kwargs)
        jax.block_until_ready(out)

        times = []
        for _ in range(self.eager_rounds):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            if eager_peak_bytes < 0:
                eager_peak_bytes = _get_peak_bytes()

        if (not self.forked) or (self.measurement > 0):
            # Profile is invalid without forking, because peak memory may be inherited from other
            # tests.
            eager_peak_bytes = -1

        if self.eager_rounds >= 1:
            return out, float(np.round(np.mean(times), 3)), float(np.round(np.std(times), 3)), eager_peak_bytes
        else:
            return out, float("nan"), float("nan"), eager_peak_bytes

    # ---------- high-level orchestration ----------

    def measure(
        self,
        fn: Optional[Callable[..., Any]] = None,
        fn_jit: Optional[Callable[..., Any]] = None,
        *args: Any,
        tag = None,
        write = True,
        **kwargs: Any
    ) -> BenchData:
        """Run selected measurements and write one numeric row per call."""
        out = None
        res = BenchData(jit_rounds=self.jit_rounds, jit_warmup=self.jit_warmup,
                        eager_rounds=self.eager_rounds, eager_warmup=self.eager_warmup)
        
        res.run_id, res.commit, res.commit_run = self.run_id, self.commit, self.commit_run
        res.tag = self.tag if tag is None else tag
        
        # First do eager profiling, to get a good idea of the memory
        if fn is not None and (self.eager_rounds > 0 or self.eager_warmup > 0):
            out, res.eager_mean_ms, res.eager_std_ms, res.eager_peak_bytes = self.profile_eager(fn, *args, **kwargs)

        if fn_jit is not None and (self.jit_rounds > 0 or self.jit_warmup > 0):
            res.compile_ms, lowered, fn_compiled = self.compile_time_ms(fn_jit, *args, **kwargs)
            graph_mem = fn_compiled.memory_analysis()

            res.jit_constants_bytes = graph_mem.generated_code_size_in_bytes
            res.jit_peak_bytes = graph_mem.peak_memory_in_bytes
            res.jit_temporary_bytes = graph_mem.temp_size_in_bytes
            try:
                # It seems likely the handling of jax's folded constants will change in the future
                # Therefore, we catch arbitrary exceptions here for now...
                res.jit_constants_bytes = folded_constants_bytes(lowered)
            except Exception as e:
                warnings.warn(f"Failed to compute folded_constants_bytes ({e})", RuntimeWarning)
                res.jit_constants_bytes = 0

            if self.jit_rounds > 0:
                out, res.jit_mean_ms, res.jit_std_ms = self.profile_jit(fn_jit, *args, **kwargs)

            if self.save_graph_svg:
                from .utils import save_graph_svg
                filename = f"{self.path}-{self.run_id}-{res.tag}.svg"

                # find last svg with same tag and different run id
                # We'll only write ours if it is different
                last_svg = None
                for run in range(self.run_id-1, -1, -1):
                    fn_candidate = f"{self.path}-{run}-{res.tag}.svg"
                    if os.path.exists(fn_candidate):
                        last_svg = fn_candidate
                        break

                save_graph_svg(fn_compiled, filename, only_if_different=last_svg)

        if write and self.path is not None:
            self._write_row(res)
        elif write:
            raise ValueError("Please either provide a path on creation or set write=False.")
        
        self.measurement += 1

        return res, out

    def _write_row(self, res: BenchData) -> None:
        file = self.path + ".csv"
        file_size = os.path.getsize(file) if os.path.exists(file) else 0
        with open(file, "a", encoding="utf-8") as f:
            if file_size == 0:
                # Add a header
                f.write(f"# pytest-jax-bench\n")
                f.write(f"# created: {_now_iso()}\n")
                f.write(f"# test_nodeid: {self.node_id}\n")
                f.write(f"# backend: {jax.default_backend()}\n")
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
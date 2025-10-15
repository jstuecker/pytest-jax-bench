from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional
# from my_jax_utils import folded_constants_bytes 
from .data import BenchData, load_bench_data

import pytest
import subprocess
import numpy as np

# Optional runtime deps
try:  # process memory (RSS)
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# Optional GPU memory via NVML
try:
    import pynvml  # type: ignore
    _NVML_OK = True
except Exception:  # pragma: no cover
    pynvml = None
    _NVML_OK = False

try:
    import jax
except Exception as e:
    raise RuntimeError("pytest-jax-bench requires JAX to be installed") from e


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


def select_commit_runs(arr, commit):
    commit_base = commit.rstrip("+")
    mask = np.where((arr["commit"] == commit_base) | (arr["commit"] == commit_base + "+"))[0]
    return arr[mask]

def summary_select_commits(arr):
    if len(arr) == 0:
        return None, None

    new_data = arr[-1]

    # Find the comparison data
    # If there is no other entry with the same commit hash, we consider the last entry of previous commit
    # If the commit is dirty, we use the first entry with the same commit hash
    commit = new_data["commit"]

    data_same_commit = select_commit_runs(arr, commit)

    if len(data_same_commit) >= 1:
        comparison_data = data_same_commit[0]
    elif len(arr) >= 2:
        data_last_commit = arr[-2]["commit"]
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

def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus : pytest.ExitCode, config: pytest.Config) -> None:
    outdir = config.getoption("--bench-jax-output-dir")
    terminalreporter.write_line(f"{config.getoption("-v")}")
    if config.getoption("-v") >= 0:
        terminalreporter.write_sep("=", "JAX benchmark results in ms and MB")

        for test_nodeid, path in benchmarks.items():
            arr = load_bench_data(path)
            new, old = summary_select_commits(arr)
            if new is None: 
                continue
            
            txt = f'{test_nodeid} com:{old["commit"]}({old["commit_run"]})->{new["commit"]}({new["commit_run"]})'

            def str_compare_perf(key, std=0., tol=None, only_different=False, label=None):
                if tol is None: tol = 2.*std
                label = key if label is None else label
                v1, v2 = old[key], new[key]
                if only_different and np.abs(v1 - v2) <= std*2.:
                    return ""
                if std > 0.:
                    txt = f"{label}({v1:.2f}->{v2:.2f}+-{std:.2f})"
                else:
                    txt = f"{label}({v1:.2f}->{v2:.2f})"
                return _colored_diff(txt, v1, v2, tol=tol)
            
            def str_compare_mem(key, only_different=False, label=None):
                label = key if label is None else label
                v1, v2 = old[key], new[key]
                if only_different and np.abs(v1 - v2) <= 1024*1024:
                    return ""
                txt = f"{label}({v1:.2f}->{v2:.2f})"
                return _colored_diff(txt, v1, v2)

            txt += " " + str_compare_perf("compile_ms", tol=new["compile_ms"]*0.1, label="compile")
            txt += " " + str_compare_perf("run_mean_ms", std=np.sqrt(new["run_std_ms"]**2+old["run_std_ms"]**2), label="run")
            txt += " " + str_compare_mem("graph_peak_memory_mb", label="mem_mb")

            terminalreporter.write_line(txt)


        #     terminalreporter.write_line(f"{test_nodeid} -> {path}")

    # terminalreporter.write_sep("-", f"JAX benchmark results in {outdir}")
    # terminalreporter.write_line(f"To compare results across commits, use e.g.: {benchmarks}")




# ---------------------------
# StableHLO memory estimate utils (very rough)
# ---------------------------

_DTYPE_BYTES = {
    "pred": 1,
    "i1": 1, "i8": 1, "u8": 1,
    "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
    "i32": 4, "u32": 4, "f32": 4,
    "i64": 8, "u64": 8, "f64": 8,
}

_COMPLEX_BYTES = {
    "complex<f32>": 8,
    "complex<f64>": 16,
}

_TENSOR_RE = re.compile(r"tensor<([^>]+)>")


def _dtype_nbytes(type_str: str) -> Optional[int]:
    if type_str.startswith("complex<"):
        base = type_str.split("x")[-1]
        return _COMPLEX_BYTES.get(base, None)
    base = type_str.split("x")[-1]
    return _DTYPE_BYTES.get(base, None)

def _num_elements(type_str: str) -> Optional[int]:
    parts = type_str.split("x")
    dims = parts[:-1]
    if not dims:
        return 1
    prod = 1
    for d in dims:
        d = d.strip()
        if d in {"*", "?"}:  # dynamic/unknown
            return None
        try:
            prod *= int(d)
        except ValueError:
            return None
    return prod

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

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

class _GpuTracker:
    """Helper to read per-process GPU memory via NVML if available and requested."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and _NVML_OK
        self.pid = os.getpid()
        self._started = False
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
                self._started = True
            except Exception:
                self.enabled = False
                self._handles = []
        else:
            self._handles = []

    def _pid_mem_bytes(self) -> int:
        if not self._started:
            return 0
        total = 0
        for h in self._handles:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h)
            except Exception:
                procs = []
            for p in procs:
                if getattr(p, "pid", None) == self.pid:
                    used = getattr(p, "usedGpuMemory", 0) or 0
                    if used > 0:
                        total += int(used)
        return total

    def peak_during(self, fn: Callable[[], Any]) -> int:
        if not self.enabled:
            fn()
            return 0
        # simple pre/post probe; cheap and avoids background threads
        peak = self._pid_mem_bytes()
        fn()
        peak = max(peak, self._pid_mem_bytes())
        return peak


# ---------------------------
# The BenchJax core object
# ---------------------------

class BenchJax:
    """Provides measurement helpers. Defaults can be set per-test via the factory fixture.

    Parameters
    ----------
    default_rounds : int
        Default number of timed iterations when `profile_run=True`.
    default_warmup : int
        Default number of warmup iterations before timing.
    default_gpu_memory : bool
        Whether to attempt GPU memory measurement by default when `profile_run_memory=True`.
    """

    def __init__(self, request: pytest.FixtureRequest, config: pytest.Config,
                 default_rounds: int = 20, default_warmup: int = 3, default_gpu_memory: bool = False) -> None:
        self.request = request
        self.config = config
        self.output_dir = config.getoption("--bench-jax-output-dir")
        self.default_rounds = int(default_rounds)
        self.default_warmup = int(default_warmup)
        self.default_gpu_memory = bool(default_gpu_memory)
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

    def run_ms(self, fn: Callable[..., Any], *args: Any,
               rounds: Optional[int] = None, warmup: Optional[int] = None, **kwargs: Any) -> tuple[float, float, int, int]:
        """Return (mean_ms, std_ms, rounds, warmup)."""
        rounds = int(rounds if rounds is not None else self.default_rounds)
        warmup = int(warmup if warmup is not None else self.default_warmup)

        compiled = self._compiled if self._compiled is not None else jax.jit(fn)

        # warmup
        for _ in range(warmup):
            compiled(*args, **kwargs)
        jax.block_until_ready(compiled(*args, **kwargs))

        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            out = compiled(*args, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        import math
        n = len(times) or 1
        mean = sum(times) / n
        var = sum((x - mean) ** 2 for x in times) / (n - 1 if n > 1 else 1)
        std = math.sqrt(var)
        return mean, std, rounds, warmup

    def practical_memory(self, run_callable: Callable[[], Any], gpu_memory: bool = False) -> tuple[int, int]:
        pid = os.getpid()
        proc = psutil.Process(pid) if psutil is not None else None

        def _rss() -> int:
            if proc is None:
                return 0
            try:
                return int(proc.memory_info().rss)
            except Exception:
                return 0

        rss_start = _rss()
        gpu = _GpuTracker(enabled=gpu_memory)
        gpu_peak = gpu.peak_during(run_callable)
        rss_end = _rss()
        return max(rss_end - rss_start, 0), gpu_peak

    # ---------- high-level orchestration ----------

    def measure(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: Optional[str] = None,
        # choose what to profile
        profile_run: bool = True,
        profile_run_memory: bool = True,
        # knobs declared per-test
        rounds: Optional[int] = None,
        warmup: Optional[int] = None,
        gpu_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> BenchData:
        """Run selected measurements and write one numeric row per call.

        All numeric columns are always present; if a measurement was skipped its value is 0.
        Times are in **milliseconds**.
        """
        backend = _get_backend()
        name = name or getattr(fn, "__name__", "anonymous")

        # defaults for this instance
        gpu_memory = self.default_gpu_memory if gpu_memory is None else bool(gpu_memory)

        res = BenchData()

        res.compile_ms, lowered, compiled = self.compile_time_ms(fn, *args, **kwargs)
        graph_mem = compiled.memory_analysis()

        res.graph_generated_code_size = graph_mem.generated_code_size_in_bytes
        res.graph_peak_memory = graph_mem.peak_memory_in_bytes
        res.graph_temp_size = graph_mem.temp_size_in_bytes

        if profile_run:
            res.run_mean_ms, res.run_std_ms, res.rounds, res.warmup = self.run_ms(fn, *args, rounds=rounds, warmup=warmup, **kwargs)

        if profile_run_memory:
            def _work():
                # A few runs to capture realistic peak usage
                for _ in range(5):
                    out = jax.jit(fn)(*args, **kwargs)
                    jax.block_until_ready(out)
            rss_peak, gpu_peak = self.practical_memory(_work, gpu_memory=gpu_memory)

            res.rss_peak_delta_bytes = int(rss_peak)
            res.gpu_peak_bytes = int(gpu_peak)

        self._write_row(name=name, backend=backend, row=res)
        # self._print_console(name=name, backend=backend, row=res)

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

        print("data:", data)

        return run_id, current_commit, commit_run

    # ---------- IO ----------

    def _outfile(self, test_nodeid: str, name: str) -> str:
        test_file, test_name = test_nodeid.split("::")
        node = re.sub(r"[^A-Za-z0-9._-]+", "_", test_name)
        # node = re.sub(r"\.py", "", node)
        nm = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return os.path.join(self.output_dir, f"{node}_{nm}.csv")

    def _write_row(self, *, name: str, backend: str, row: BenchData) -> None:
        row.node_id = self.request.node.nodeid
        path = self._outfile(row.node_id, name)

        row.run_id, row.commit, row.commit_run = self._get_run_data(path)

        with open(path, "a", encoding="utf-8") as f:
            if row.run_id == 0:
                print("Run data will be written to:", path)
                # Add a header
                f.write(f"# pytest-jax-bench\n")
                f.write(f"# created: {_now_iso()}\n")
                f.write(f"# test_nodeid: {row.node_id}\n")
                f.write(f"# name: {name}\n")
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

    # def _print_console(self, *, name: str, backend: str, row: BenchJaxRow) -> None:
    #     # Print a summary when -v/--verbose is used
    #     try:
    #         verbose = int(self.config.getoption("verbose") or 0)
    #     except Exception:
    #         verbose = 0
    #     if verbose <= 0:
    #         return
    #     rep = self.config.pluginmanager.getplugin("terminalreporter")
    #     msg = (
    #         f"[bench_jax] {name} (backend={backend}) "
    #         f"compile={row.compile_ms:.2f}ms, "
    #         f"run={row.run_mean_ms:.2f}±{row.run_std_ms:.2f}ms, "
    #         f"rounds={row.rounds}, warmup={row.warmup}, "
    #         f"graph_mem={row.graph_mem_bytes_est}B, "
    #         f"rss+={row.rss_peak_delta_bytes}B, gpu_peak={row.gpu_peak_bytes}B"
    #     )
    #     if rep is not None:
    #         rep.write_line(msg)
    #     else:
    #         print(msg)


@pytest.fixture
def bench_jax(request: pytest.FixtureRequest):
    """Factory fixture: call `bench_jax(...)` in your test to configure defaults.

    Examples
    --------
    >>> def test_demo(bench_jax):
    ...     import jax.numpy as jnp
    ...     def f(x):
    ...         return (jnp.sin(x) * jnp.cos(x)).sum()
    ...     x = jnp.ones((4096, 4096), dtype=jnp.float32)
    ...     # Configure per-test defaults here
    ...     jb = bench_jax(rounds=25, warmup=5, gpu_memory=True)
    ...     # Choose which parts to profile per call (defaults: all True)
    ...     jb.measure(f, x, name="trig_sum", profile_graph=True, profile_run=True)

    You can also skip configuration and use built-in defaults:

    >>> def test_demo2(bench_jax):
    ...     import jax.numpy as jnp
    ...     def f(x):
    ...         return jnp.tanh(x @ x.T).sum()
    ...     x = jnp.ones((1024, 1024), dtype=jnp.float32)
    ...     bench = bench_jax()  # defaults: rounds=20, warmup=3, gpu_memory=False
    ...     bench.measure(f, x, name="matmul_tanh", profile_run_memory=False)
    """
    config = request.config

    def _factory(*, rounds: int = 20, warmup: int = 3, gpu_memory: bool = False) -> BenchJax:
        return BenchJax(request, config, default_rounds=rounds, default_warmup=warmup, default_gpu_memory=gpu_memory)

    return _factory

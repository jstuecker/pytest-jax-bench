from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import pytest
import subprocess

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
    # Only keep output-dir as a CLI option. All other knobs are per-test via the fixture.
    group = parser.getgroup("bench-jax")
    group.addoption(
        "--bench-jax-output-dir",
        action="store",
        default=".benchmarks",
        help="Directory for output files (default: .benchmarks)",
    )


def pytest_configure(config: pytest.Config) -> None:
    outdir = config.getoption("--bench-jax-output-dir")
    os.makedirs(outdir, exist_ok=True)


# ---------------------------
# Core measurement container
# ---------------------------

@dataclass
class BenchJaxRow:
    # Stored in **milliseconds** with .2f formatting on write
    compile_ms: float
    run_mean_ms: float
    run_std_ms: float
    rounds: int
    warmup: int
    graph_mem_bytes_est: int
    rss_peak_delta_bytes: int
    gpu_peak_bytes: int


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


def estimate_stablehlo_memory_bytes(compiled_ir_str: str) -> int:
    seen: set[str] = set()
    total = 0
    for m in _TENSOR_RE.finditer(compiled_ir_str):
        t = m.group(1)
        if t in seen:
            continue
        seen.add(t)
        n_el = _num_elements(t)
        bpe = _dtype_nbytes(t)
        if n_el is None or bpe is None:
            continue
        total += n_el * bpe
    return total


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _git_commit_short() -> str:
    """Return short git commit id (7 chars) or 'unknown' if not available."""
    try:
        # Use git directly to avoid adding heavy deps. This will fail cleanly if not a git repo.
        out = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
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
        self._compiled = None  # may be set by compile_time()

    # ---------- measurement pieces ----------

    def compile_time_ms(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
        """Return compilation time in milliseconds (excluding lowering)."""
        jitted = jax.jit(fn)
        t1 = time.perf_counter()
        lowered = jitted.lower(*args, **kwargs)
        compiled = lowered.compile()
        t2 = time.perf_counter()
        self._compiled = compiled
        return (t2 - t1) * 1000.0

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

    def graph_memory_estimate(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> int:
        lowered = jax.jit(fn).lower(*args, **kwargs)
        try:
            ir = lowered.compiler_ir(dialect="stablehlo")
        except TypeError:
            ir = lowered.compiler_ir(dialect="hlo")
        return estimate_stablehlo_memory_bytes(str(ir))

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
        profile_compile: bool = True,
        profile_graph: bool = True,
        profile_run: bool = True,
        profile_run_memory: bool = True,
        # knobs declared per-test
        rounds: Optional[int] = None,
        warmup: Optional[int] = None,
        gpu_memory: Optional[bool] = None,
        **kwargs: Any,
    ) -> BenchJaxRow:
        """Run selected measurements and write one numeric row per call.

        All numeric columns are always present; if a measurement was skipped its value is 0.
        Times are in **milliseconds**.
        """
        test_nodeid = self.request.node.nodeid
        backend = _get_backend()
        name = name or getattr(fn, "__name__", "anonymous")

        # defaults for this instance
        gpu_memory = self.default_gpu_memory if gpu_memory is None else bool(gpu_memory)

        compile_ms = 0.0
        run_mean_ms = 0.0
        run_std_ms = 0.0
        _rounds = int(rounds) if rounds is not None else 0
        _warmup = int(warmup) if warmup is not None else 0
        graph_mem = 0
        rss_peak = 0
        gpu_peak = 0

        if profile_compile:
            compile_ms = self.compile_time_ms(fn, *args, **kwargs)

        if profile_graph:
            try:
                graph_mem = self.graph_memory_estimate(fn, *args, **kwargs)
            except Exception:
                graph_mem = 0

        if profile_run:
            run_mean_ms, run_std_ms, _rounds, _warmup = self.run_ms(fn, *args, rounds=rounds, warmup=warmup, **kwargs)

        if profile_run_memory:
            def _work():
                # A few runs to capture realistic peak usage
                for _ in range(5):
                    out = jax.jit(fn)(*args, **kwargs)
                    jax.block_until_ready(out)
            rss_peak, gpu_peak = self.practical_memory(_work, gpu_memory=gpu_memory)

        row = BenchJaxRow(
            compile_ms=float(compile_ms),
            run_mean_ms=float(run_mean_ms),
            run_std_ms=float(run_std_ms),
            rounds=int(_rounds),
            warmup=int(_warmup),
            graph_mem_bytes_est=int(graph_mem),
            rss_peak_delta_bytes=int(rss_peak),
            gpu_peak_bytes=int(gpu_peak),
        )

        self._write_row(name=name, test_nodeid=test_nodeid, backend=backend, row=row)
        self._print_console(name=name, backend=backend, row=row)
        return row

    # ---------- IO ----------

    def _outfile(self, test_nodeid: str, name: str) -> str:
        test_file, test_name = test_nodeid.split("::")
        node = re.sub(r"[^A-Za-z0-9._-]+", "_", test_name)
        # node = re.sub(r"\.py", "", node)
        nm = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return os.path.join(self.output_dir, f"{node}_{nm}.csv")

    def _write_row(self, *, name: str, test_nodeid: str, backend: str, row: BenchJaxRow) -> None:
        path = self._outfile(test_nodeid, name)
        # Read existing file (if any) to determine run-id and per-commit run count
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as _f:
                    _lines = _f.readlines()
            except Exception:
                _lines = []
        else:
            _lines = []

        # Data lines are those that are not comments (#) and not the column name lines (starting with '(')
        existing_data_lines = [l for l in _lines if l.strip() and not l.lstrip().startswith("#") and not l.lstrip().startswith("(")]
        run_id = len(existing_data_lines)

        current_commit = _git_commit_short()
        # count occurences of current_commit in existing data lines
        commit_run = sum(1 for l in existing_data_lines if re.search(rf"\b{current_commit}\b", l))

        is_new = len(_lines) == 0
        # Header: human-friendly comments, then a single commented column line.
        # Data: tab-separated numeric values (suitable for np.loadtxt with delimiter='	').
        columns = [
            "run_id",
            "commit",
            "commit_run",
            "compile_ms",
            "run_mean_ms",
            "run_std_ms",
            "rounds",
            "warmup",
            "graph_mem_bytes_est",
            "rss_peak_delta_bytes",
            "gpu_peak_bytes",
        ]

        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(f"# pytest-jax-bench\n")
                f.write(f"# created: {_now_iso()}\n")
                f.write(f"# test_nodeid: {test_nodeid}\n")
                f.write(f"# name: {name}\n")
                f.write(f"# backend: {backend}\n")
                # include commit in the header metadata for clarity
                f.write(f"# First commit: {current_commit}\n")
                # f.write("# " + "\t".join(columns) + "\n")
                for i,c in enumerate(columns):
                    f.write(f"({i+1}) {c}\n")
                f.write(f"#      (1) ")
                for i in range(1, len(columns)):
                    f.write(f"      ({i+1}) ")
                f.write("\n")
            # format: ms with .2f, ints as decimal
            # Layout: run_id (int), commit (7-char), commit_run (int), then numeric fields.
            # Keep similar field widths to the original layout for readability.
            
            commit = current_commit
            line = (
                f"{run_id:10d}"
                f"{commit:>10s}"
                f"{commit_run:10}"
                f"{row.compile_ms:10.2f}"
                f"{row.run_mean_ms:10.2f}"
                f"{row.run_std_ms:10.2f}"
                f"{row.rounds:10}"
                f"{row.warmup:10}"
                f"{row.graph_mem_bytes_est:10}"
                f"{row.rss_peak_delta_bytes:10}"
                f"{row.gpu_peak_bytes:10}"
            )
            f.write(line + "\n")

    def _print_console(self, *, name: str, backend: str, row: BenchJaxRow) -> None:
        # Print a summary when -v/--verbose is used
        try:
            verbose = int(self.config.getoption("verbose") or 0)
        except Exception:
            verbose = 0
        if verbose <= 0:
            return
        rep = self.config.pluginmanager.getplugin("terminalreporter")
        msg = (
            f"[bench_jax] {name} (backend={backend}) "
            f"compile={row.compile_ms:.2f}ms, "
            f"run={row.run_mean_ms:.2f}±{row.run_std_ms:.2f}ms, "
            f"rounds={row.rounds}, warmup={row.warmup}, "
            f"graph_mem={row.graph_mem_bytes_est}B, "
            f"rss+={row.rss_peak_delta_bytes}B, gpu_peak={row.gpu_peak_bytes}B"
        )
        if rep is not None:
            rep.write_line(msg)
        else:
            print(msg)


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

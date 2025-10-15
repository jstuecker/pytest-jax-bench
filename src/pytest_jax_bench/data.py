from dataclasses import dataclass
import numpy as np

# ---------------------------
# Core measurement container
# ---------------------------

@dataclass
class BenchData:
    node_id: str = ""

    run_id: int = 0
    commit: str = "unknown"
    commit_run: int = 0
    compile_ms: float = np.nan

    jit_mean_ms: float = np.nan
    jit_std_ms: float = np.nan
    eager_mean_ms: float = np.nan
    eager_std_ms: float = np.nan

    graph_constants: int = np.nan
    graph_peak_memory: int = np.nan
    graph_temp_size: int = np.nan
    eager_peak_memory: int = np.nan

    jit_rounds: int = 0
    jit_warmup: int = 0
    eager_rounds: int = 0
    eager_warmup: int = 0

    def column_descriptions(self) -> tuple[str]:
        return (
            "Run ID",
            "Commit ('+' means with local changes)",
            "Commit Run",
            "Compile Time (ms)",

            "Mean Jitted Run Time (ms)",
            "Stddev. Jitted Run Time (ms)",
            "Mean Eager Run Time (ms)",
            "Stddev. Eager Run Time (ms)",

            "Graph Peak Memory (MB)",
            "Graph Constants Size (MB)",
            "Graph Temp Size (MB)",
            "Eager Peak Memory (MB)",

            "Jitted Run Rounds",
            "Jitted Warmup Rounds",
            "Eager Run Rounds",
            "Eager Warmup Rounds",
        )
    
    def formatted_line(self) -> str:
        return (
                f"{self.run_id:10d}"
                f"{self.commit:>10s}"
                f"{self.commit_run:10}"
                f"{self.compile_ms:10.2f}"

                f"{self.jit_mean_ms:10.2f}"
                f"{self.jit_std_ms:10.2f}"
                f"{self.eager_mean_ms:10.2f}"
                f"{self.eager_std_ms:10.2f}"

                f"{self.graph_peak_memory/1024.**2:10.2f}"
                f"{self.graph_constants/1024.**2:10.2f}"
                f"{self.graph_temp_size/1024.**2:10.2f}"
                f"{self.eager_peak_memory/1024.**2:10.2f}"

                f"{self.jit_rounds:10}"
                f"{self.jit_warmup:10}"
                f"{self.eager_rounds:10}"
                f"{self.eager_warmup:10}"
        )
    
    @classmethod
    def data_type(cls) -> str:
        dt = np.dtype([
            ("run_id", np.int64),
            ("commit", "U32"),
            ("commit_run", np.int64),
            ("compile_ms", np.float64),

            ("run_mean_ms", np.float64),
            ("run_std_ms", np.float64),
            ("eager_mean_ms", np.float64),
            ("eager_std_ms", np.float64),

            ("graph_peak_memory_mb", np.float64),
            ("graph_constants", np.float64),
            ("graph_temp_size_mb", np.float64),
            ("eager_peak_memory_mb", np.float64),

            ("jit_rounds", np.int64),
            ("jit_warmup", np.int64),
            ("eager_rounds", np.int64),
            ("eager_warmup", np.int64)
        ])
        return dt

def load_bench_data(file: str) -> np.ndarray:
    """Load benchmark data from a CSV file into a structured numpy array."""
    arr = np.genfromtxt(file, dtype=BenchData.data_type(), comments="#", ndmin=1)
    return arr
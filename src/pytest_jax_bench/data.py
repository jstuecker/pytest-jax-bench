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
    compile_ms: float = 0.
    run_mean_ms: float = 0.
    run_std_ms: float = 0.
    rounds: int = 0
    warmup: int = 0
    graph_generated_code_size: int = 0
    graph_peak_memory: int = 0
    graph_temp_size: int = 0
    rss_peak_delta_bytes: int = 0
    gpu_peak_bytes: int = 0

    def column_descriptions(self) -> tuple[str]:
        return (
            "Run ID",
            "Commit ('+' means with local changes)",
            "Commit Run",
            "Compile Time (ms)",
            "Mean Run Time (ms)",
            "Std. Dev. Run Time (ms)",
            "Run Rounds",
            "Warmup Rounds",
            "Graph Peak Memory (MB)",
            "Graph Generated Code Size (MB)",
            "Graph Temp Size (MB)",
            "rss_peak_delta_bytes (MB)",
            "gpu_peak_bytes (MB)",
        )
    
    def formatted_line(self) -> str:
        return (
                f"{self.run_id:10d}"
                f"{self.commit:>10s}"
                f"{self.commit_run:10}"
                f"{self.compile_ms:10.2f}"
                f"{self.run_mean_ms:10.2f}"
                f"{self.run_std_ms:10.2f}"
                f"{self.rounds:10}"
                f"{self.warmup:10}"
                f"{self.graph_peak_memory/1024.**2:10.2f}"
                f"{self.graph_generated_code_size/1024.**2:10.2f}"
                f"{self.graph_temp_size/1024.**2:10.2f}"
                f"{self.rss_peak_delta_bytes/1024.**2:10.2f}"
                f"{self.gpu_peak_bytes/1024.**2:10.2f}"
        )
    
    @classmethod
    def data_type(cls) -> str:
        dt = np.dtype([
            ("run_id", np.int64),
            ("commit", "U32"),
            ("commit_run", np.int64),
            ("compile_ms", np.float64),
            ("mean_ms", np.float64),
            ("std_ms", np.float64),
            ("run_rounds", np.int64),
            ("warmup_rounds", np.int64),
            ("graph_peak_mb", np.float64),
            ("code_size_mb", np.float64),
            ("temp_size_mb", np.float64),
            ("rss_peak_mb", np.float64),
            ("gpu_peak_mb", np.float64),
        ])
        return dt

def load_bench_data(file: str) -> np.ndarray:
    """Load benchmark data from a CSV file into a structured numpy array."""
    arr = np.genfromtxt(file, dtype=BenchData.data_type(), comments="#")
    return arr
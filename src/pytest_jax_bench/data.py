from dataclasses import dataclass
import numpy as np

# ---------------------------
# Core measurement container
# ---------------------------

@dataclass
class BenchData:
    run_id: int = 0
    commit: str = "unknown"
    commit_run: int = 0
    tag: str = "base"
    compile_ms: float = np.nan

    jit_mean_ms: float = np.nan
    jit_std_ms: float = np.nan
    eager_mean_ms: float = np.nan
    eager_std_ms: float = np.nan

    jit_peak_bytes: int = -1
    jit_constants_bytes: int = -1
    jit_temporary_bytes: int = -1
    eager_peak_memory: int = -1

    jit_rounds: int = 0
    jit_warmup: int = 0
    eager_rounds: int = 0
    eager_warmup: int = 0
    
    def get_column_header(self):
        txt = ""
        for i,name in enumerate(self.__dataclass_fields__):
            txt += f"# {f'({i+1})':>4} {name}\n"
        txt += "#"
        for i,name in enumerate(self.__dataclass_fields__):
            txt += f"{f'({i+1}) ':>12}"
        return txt
    
    def formatted_line(self) -> str:
        line = ""
        for i, field in enumerate(self.__dataclass_fields__.values()):
            value = getattr(self, field.name)
            if isinstance(value, (int, np.integer)):
                fmt = f"{value:12d}"
            elif isinstance(value, (float, np.floating)):
                fmt = f"{value:12.2f}"
            elif isinstance(value, (str, np.str_)):
                fmt = f"{value:>12s}"
            else:
                raise TypeError(f"Unsupported type {type(value)} for field {field.name}")
            
            line += fmt
        return line
    
    @classmethod
    def data_type(cls) -> str:
        """Return a numpy dtype string matching the fields of BenchData."""
        fields = []
        for f in cls.__dataclass_fields__.values():
            if f.type == str:
                fields.append((f.name, 'U32'))
            else:
                fields.append((f.name, np.dtype(f.type).str))
        return np.dtype(fields)

def load_bench_data(file: str, remove_dirt_mark=True) -> np.ndarray:
    """Load benchmark data from a CSV file into a structured numpy array."""
    data = np.genfromtxt(file, dtype=BenchData.data_type(), comments="#", ndmin=1)

    if remove_dirt_mark:
        data["commit"] = np.char.rstrip(data["commit"], "+")

    return data
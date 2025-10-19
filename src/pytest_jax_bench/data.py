from dataclasses import dataclass
import numpy as np
from numpy.lib import recfunctions as rfn
import ast

# ---------------------------
# Core measurement container
# ---------------------------

@dataclass
class BenchData:
    run_id: int = 0
    commit: str = "unknown"
    commit_run: int = 0
    tag: str = "base"
    parameters: str = ""
    
    compile_ms: float = float('nan')
    jit_mean_ms: float = float('nan')
    jit_std_ms: float = float('nan')
    eager_mean_ms: float = float('nan')
    eager_std_ms: float = float('nan')

    jit_peak_bytes: int = -1
    jit_constants_bytes: int = -1
    jit_temporary_bytes: int = -1
    eager_peak_bytes: int = -1

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
                fmt = f" {value:11d}"
            elif isinstance(value, (float, np.floating)):
                fmt = f" {value:11.2f}"
            elif isinstance(value, (str, np.str_)):
                fmt = f" {value:>11s}"
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

def encode_pardict(par_dict: dict) -> str:
    """Encode a parameter dictionary as a string for storage in BenchData."""
    if par_dict is None or len(par_dict) == 0:
        return "None"
    txt = ",".join([f"{k}:{repr(v)}" for k,v in par_dict.items()])
    return txt

def decode_pardict(par_str: str) -> dict:
    """Decode a parameter dictionary string from BenchData back into a dictionary."""
    par_dict = {}
    if par_str is None or par_str == "None" or par_str == "":
        return par_dict

    if par_str:
        items = par_str.split(",")
        for item in items:
            k, v = item.split(":", 1)
            par_dict[k] = ast.literal_eval(v)
    return par_dict

def load_bench_data(file: str, remove_dirty_mark=True, interprete_parameters=False, merge_with_par=False) -> np.ndarray:
    """Load benchmark data from a CSV file into a structured numpy array."""
    try:
        data = np.genfromtxt(file, dtype=BenchData.data_type(), comments="#", ndmin=1)
    except ValueError as e:
        raise ValueError(f"Found file with an outdated format. Best to recreate {file}: {e}")
    
    if remove_dirty_mark:
        data["commit"] = np.char.rstrip(data["commit"], "+")

    if interprete_parameters:
        pars = {}
        for i in range(len(data)):
            par_dict = decode_pardict(data["parameters"][i])
            for k,v in par_dict.items():
                if k not in pars:
                    pars[k] = []
                pars[k].append(v)
        for k in pars:
            pars[k] = np.array(pars[k])
        # create a structured data array with the parameters
        pars = np.core.records.fromarrays(list(pars.values()), names=list(pars.keys()))

        if merge_with_par:
            data = rfn.merge_arrays((data, pars), flatten=True, usemask=False)
        else:
            return data, pars

    return data
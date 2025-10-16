__all__ = ["__version__"]
__version__ = "0.1.0"

from .data import BenchData, load_bench_data
from . import utils
from .plugin import JaxBench
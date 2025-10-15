# pytest-jax-bench
A pytest plugin to benchmark memory usage, compilation and runtime of jitted JAX functions

## Install

```bash
pip install -e . # from repo root
# optional memory extras
pip install .[mem]
```

## Useful
```bash
pip install pytest-forked
pytest --forked
```

## Todo:
* Add GPU to header

* Measure Constant folding
* Fix memory usage calculation
* Add nice stdout output
* Add helper functions for reading the csv
* Make nice plots
* cli for plots?
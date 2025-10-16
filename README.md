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
pytest --help | grep ptjb -A 2
```

## Todo
Readme
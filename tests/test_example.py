import jax.numpy as jnp
import pytest

def test_demo(bench_jax):
    def f(x): 
        return (jnp.sin(x) * jnp.cos(x)).sum()
    x = jnp.ones((4096, 4096), dtype=jnp.float32)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(
        f, x,
        profile_compile=True,
        # profile_graph=True,
        profile_run=True,
        # profile_run_memory=True,
    )

@pytest.mark.parametrize("offset", [0, 1])
def test_param(bench_jax, offset):
    def f(x): 
        return (jnp.sin(x) * jnp.cos(x)).sum() + offset
    x = jnp.ones((4096, 4096), dtype=jnp.float32)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(
        f, x,
        profile_compile=True,
        # profile_graph=True,
        profile_run=True,
        # profile_run_memory=True,
    )
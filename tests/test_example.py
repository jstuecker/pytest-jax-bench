import jax.numpy as jnp
import pytest

def myfft(x):
    return jnp.fft.fft2(x).real

# @pytest.mark.parametrize("offset", [0, 1])
def test_fft(bench_jax):
    x = jnp.ones((4096, 4096), dtype=jnp.float32)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(myfft, x)
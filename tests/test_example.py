import jax.numpy as jnp
import pytest
import numpy as np

def myfft(x):
    return jnp.fft.fft2(x).real

def test_fft(bench_jax):
    x = jnp.ones((4096, 4096), dtype=jnp.float32)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(myfft, x)

@pytest.mark.parametrize("n", [1,4])
def test_with_cst(bench_jax, n):
    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    add = np.float32(np.random.uniform(0., 1., x.shape + (n,)))

    def my_sin(x):
        return jnp.sin(x) + jnp.sum(add + x[0,0], axis=-1)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(my_sin, x)
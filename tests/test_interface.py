import jax
import jax.numpy as jnp
import pytest
import numpy as np
from pytest_jax_bench import JaxBench

def test_no_bench():
    pass

def rfft(x):
    return jnp.fft.irfftn(jnp.fft.rfftn(x*2.))

def test_full(request):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = JaxBench(request, jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

def test_fixture(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

def test_only_jit(request):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = JaxBench(request, jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn_jit=jax.jit(rfft), x=x)

def test_only_eager(request):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = JaxBench(request, jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, x=x)

def test_no_warmup(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=0, eager_rounds=5, eager_warmup=0)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

def test_no_rounds(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=0, jit_warmup=1, eager_rounds=0, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

def test_nothing(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=0, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

@pytest.mark.parametrize("n", [128, 256])
def test_parameters(request, n):
    x = jnp.ones((n, n, n), dtype=jnp.float32)

    jb = JaxBench(request, jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)
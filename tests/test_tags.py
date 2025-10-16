import jax
import jax.numpy as jnp
from pytest_jax_bench import JaxBench

def rfft(x):
    return jnp.fft.irfftn(jnp.fft.rfftn(x*2.))

def fft(x):
    return jnp.fft.ifftn(jnp.fft.fftn(x*2.))

def test_tags(request):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = JaxBench(request, jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x, tag="fft")
    jb.measure(fn=fft, fn_jit=jax.jit(fft), x=x, tag="rfft")
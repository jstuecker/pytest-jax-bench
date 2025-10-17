import jax
import jax.numpy as jnp
from pytest_jax_bench import JaxBench

def rfft(x):
    return jnp.fft.irfftn(jnp.fft.rfftn(x*2.))

def fft(x):
    return jnp.fft.ifftn(jnp.fft.fftn(x*2.))

# ----------------------- The two standard ways of calling full benchmarks ----------------------- #

def test_standard(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x)

def test_with_tags(jax_bench):
    x = jnp.ones((256, 256, 256), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=1, eager_rounds=5, eager_warmup=1)

    jb.measure(fn=fft, fn_jit=jax.jit(fft), x=x, tag="fft")
    jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x, tag="rfft")

# -------------------------------- Ways of getting reduced outputs ------------------------------- #

def test_compile_only(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=0, jit_warmup=0)
    jb.measure(fn_jit=jax.jit(rfft), x=x)

def test_eager_only(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(eager_rounds=10, eager_warmup=1)
    jb.measure(fn=rfft, x=x)

def test_jit_only(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=1)
    jb.measure(fn_jit=jax.jit(rfft), x=x)

# -------------------------------- Working independently of pytest ------------------------------- #

def test_standalone():
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    # You can work without pytest or fixtures by instantiating JaxBench directly
    # In this case outputs will not be written, but only returned
    # (unless you provide a path to JaxBench)
    jb = JaxBench(jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    res = jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x, write=False)
    print(res)
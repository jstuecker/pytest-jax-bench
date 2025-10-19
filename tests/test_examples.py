import jax
import jax.numpy as jnp
from pytest_jax_bench import JaxBench
import matplotlib.pyplot as plt
import numpy as np
# from pytest_jax_bench.plugin import ptjb_with_custom_plot
import pytest

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

def custom_plot(data):
    fig = plt.figure()
    plt.xlabel("run_id")
    plt.plot(data["run_id"], data["compile_ms"])
    plt.ylabel("compile_ms")
    return fig

@pytest.mark.ptjb(plot=custom_plot)
def test_with_custom_plot(jax_bench):
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=1)
    jb.measure(fn_jit=jax.jit(rfft), x=x)

def custom_plot_par(data):
    fig = plt.figure()
    plt.xlabel("n")
    plt.ylabel("jit_mean_ms")
    plt.plot(data["n"], data["jit_mean_ms"])
    return fig

@pytest.mark.ptjb(plot_summary=custom_plot_par, only_last=True)
@pytest.mark.parametrize("n", [128, 170, 220, 270])
def test_pars_with_custom_plot(jax_bench, n):
    x = jnp.ones((n, n, n), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=10, jit_warmup=1)
    jb.measure(fn_jit=jax.jit(rfft), x=x)

# -------------------------------- Working independently of pytest ------------------------------- #

def test_standalone():
    x = jnp.ones((128, 128, 128), dtype=jnp.float32)

    # You can work without pytest or fixtures by instantiating JaxBench directly
    # In this case outputs will not be written, but only returned
    # (unless you provide a path to JaxBench)
    jb = JaxBench(jit_rounds=10, jit_warmup=2, eager_rounds=5, eager_warmup=1)
    res, out = jb.measure(fn=rfft, fn_jit=jax.jit(rfft), x=x, write=False)
    print(res)
import jax.numpy as jnp

def test_demo(bench_jax):
    import jax.numpy as jnp
    def f(x): return (jnp.sin(x) * jnp.cos(x)).sum()
    x = jnp.ones((4096, 4096), dtype=jnp.float32)

    jb = bench_jax(rounds=30, warmup=5, gpu_memory=True)
    jb.measure(
        f, x, name="trig_sum",
        profile_compile=True,
        # profile_graph=True,
        profile_run=True,
        # profile_run_memory=True,
    )
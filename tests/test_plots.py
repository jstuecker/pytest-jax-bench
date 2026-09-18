import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pytest_jax_bench.data import BenchData
from pytest_jax_bench.plots import plot_parametrized_benchmark, prepare_xaxis


@pytest.mark.parametrize('tags', [('base',), ('lpt', 'simulation')])
def test_named_parameters_generate_plots(tmp_path, tags):
    for setup in ('normal', 'lightcone', 'without_ngenic'):
        rows = [BenchData(parameters=f'setup-{setup}', tag=tag,
                          compile_ms=1, jit_mean_ms=2, jit_std_ms=0.1,
                          jit_peak_bytes=1024) for tag in tags]
        (tmp_path / f'setup-{setup}.csv').write_text(
            rows[0].get_column_header() + '\n'
            + '\n'.join(row.formatted_line() for row in rows) + '\n')
    plot_parametrized_benchmark(str(tmp_path))
    assert (tmp_path / 'setup.png').stat().st_size > 0


@pytest.mark.parametrize('values,scale', [
    ([1, 100], 'log'), ([1., 2.], 'linear'),
    ([0, 100], 'linear'), ([-1, 100], 'linear'),
    ([False, True], 'linear'),
])
def test_parameter_axis_scale(values, scale):
    values = np.asarray(values)
    data = np.empty(len(values), dtype=[('parameter', values.dtype)])
    data['parameter'] = values
    fig, ax = plt.subplots()
    try:
        with np.errstate(divide='raise', invalid='raise'):
            x, _, _ = prepare_xaxis(data, xaxis='parameter', ax=ax)
        np.testing.assert_array_equal(x, values)
        assert ax.get_xscale() == scale
    finally:
        plt.close(fig)

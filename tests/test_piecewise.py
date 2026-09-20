import numpy as np

from extension.core_math import (
    F, G, clipping, clipping_parameters, clipping_region_count,
    compose_pieces, iterate_pieces, parameter_count, shallow_pieces,
)


def test_piecewise_region_counts():
    assert [len(iterate_pieces(n)[0]) for n in range(1, 6)] == [3, 9, 27, 81, 243]
    assert [len(iterate_pieces(n)[1]) for n in range(1, 6)] == [12, 36, 108, 324, 972]
    half = dict(F)
    half["phi"] = (1.0, -1.0, -3.0, 9.3)
    assert len(compose_pieces(shallow_pieces(half), G)) == 10


def test_clipping_and_parameter_counts():
    phi, psi, theta = clipping_parameters()
    x = np.linspace(0, 1, 100)
    y, z, *_ = clipping(x, phi, psi, theta)
    assert y.shape == x.shape and z.shape == (3, 100)
    assert len(clipping_region_count(phi, psi, theta)) == 11
    assert parameter_count([1, 32, 32, 32, 1]) == parameter_count([1, 736, 1]) == 2209

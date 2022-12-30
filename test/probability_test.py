import math

import pytest

import dsfs as scratch


@pytest.mark.parametrize(
    'x,expected',
    [
        (0 - 1e-6, 0),
        (0, 0),
        (0.5, 0.5),
        (1 - 1e-6, 1 - 1e-6),
        (1, 0),
    ]
)
def test_uniform_pdf(x, expected):
    assert scratch.probability.uniform_pdf(x) == expected


@pytest.mark.parametrize(
    'x,expected',
    [
        (0 - 1e-6, 0),
        (0, 0),
        (0.5, 0.5),
        (1 - 1e-6, 1 - 1e-6),
        (1, 1),
    ]
)
def test_uniform_cdf(x, expected):
    assert scratch.probability.uniform_cdf(x) == expected


@pytest.mark.parametrize(
    'mu, sigma, x, expected',
    [
        (0, 1, 0, 1 / ((math.sqrt(2 * math.pi)) * 1)),
        (0, 2, 0, 1 / ((math.sqrt(2 * math.pi)) * 2)),
        (0, 0.5, 0, 1 / ((math.sqrt(2 * math.pi)) * 0.5)),
        (-1, 1, -1, 1 / ((math.sqrt(2 * math.pi)) * 1)),
    ]
)
def test_normal_pdf(mu, sigma, x, expected):
    assert scratch.probability.normal_pdf(x, mu=mu, sigma=sigma) == pytest.approx(expected)


@pytest.mark.parametrize(
    'mu, sigma, p, expected',
    [
        (0, 1, -3, 0.001350),
        (0, 1, -2, 0.022750),
        (0, 1, -1, 0.158655),
        (0, 1, 0, 0.5),
        (0, 1, 1, 0.841345),
        (0, 1, 2, 0.977250),
        (0, 1, 3, 0.998650),
        (0, 2, 4.154418, 0.981109),
        (0, 0.5, -1.694350, 0.000351),
        (-1, 1, 1.390205, 0.991580),
    ]
)
def test_normal_cdf(mu, sigma, p, expected):
    assert scratch.probability.normal_cdf(p, mu=mu, sigma=sigma) == pytest.approx(expected, abs=1e-6)

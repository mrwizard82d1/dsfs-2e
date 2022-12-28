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

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

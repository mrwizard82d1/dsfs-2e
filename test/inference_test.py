import pytest

import dsfs as scratch


def test_normal_approximation_to_binomial():
    assert scratch.inference.normal_approximation_to_binomial(100, 0.5) == \
           pytest.approx(scratch.inference.NormalDistributionParameters(50, 5))


# In many of the tests of a normal distribution, I simply test a standard normal distribution. I assume other tests
# have considered different normal distribution parameters.


@pytest.mark.parametrize(
    'x, expected, abs_tol',
    [
        (-3, 0.00135, 1e-5),
        (0, 0.5, 1e-9),
        (1, 0.841, 1e-3),
    ]
)
def test_normal_probability_below(x, expected, abs_tol):
    assert scratch.inference.normal_probability_below(x) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'x, expected, abs_tol',
    [
        (-3, 0.99865, 1e-5),
        (0, 0.5, 1e-9),
        (1, 0.159, 1e-3),
    ]
)
def test_normal_probability_above(x, expected, abs_tol):
    assert scratch.inference.normal_probability_above(x) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'lo, hi, expected, abs_tol',
    [
        (-3, -2, 0.0214, 1e-4),
        (-1, 1, 0.683, 1e-3),
        (1, 3, 0.15730, 1e-5),
    ]
)
def test_normal_probability_between(lo, hi, expected, abs_tol):
    assert scratch.inference.normal_probability_between(lo, hi) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'lo, hi, expected, abs_tol',
    [
        (-3, -2, 0.9786, 1e-4),
        (-1, 1, 0.317, 1e-3),
        (1, 3, 0.84270, 1e-5),
    ]
)
def test_normal_probability_outside(lo, hi, expected, abs_tol):
    assert scratch.inference.normal_probability_outside(lo, hi) == pytest.approx(expected, abs=abs_tol)

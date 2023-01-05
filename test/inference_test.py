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


@pytest.mark.parametrize(
    'p, expected, abs_tol',
    [
        (0.99865, 3.00000, 2e-5),
        (0.5, 0, 1e-5),
        (0.159, -1, 2e-3),
    ]
)
def test_normal_upper_bound(p, expected, abs_tol):
    assert scratch.inference.normal_upper_bound(p) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'p, expected, abs_tol',
    [
        (0.00135, 3.00000, 2e-5),
        (0.5, 0, 1e-5),
        (0.841, -1, 2e-3),
    ]
)
def test_normal_lower_bound(p, expected, abs_tol):
    assert scratch.inference.normal_lower_bound(p) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'p, expected, abs_tol',
    [
        (0.00135, 3.00000, 2e-5),
        (0.5, 0, 1e-5),
        (0.841, -1, 2e-3),
    ]
)
def test_normal_lower_bound(p, expected, abs_tol):
    assert scratch.inference.normal_lower_bound(p) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'probability, expected, abs_tol',
    [
        (0.68269, (-1, 1), 1e-5),
        (0.9545, (-2, 2), 1e-4),
        (0.997, (-3, 3), 4e-2),
    ]
)
def test_normal_two_sided_bounds(probability, expected, abs_tol):
    assert scratch.inference.normal_two_sided_bounds(probability) == pytest.approx(expected, abs=abs_tol)


@pytest.mark.parametrize(
    'x, mu, sigma, expected',
    [
        (529.5, 500, 15.8, 0.062),
        (470.5, 500, 15.8, 0.062)
    ]
)
def test_two_sided_p_value(x, mu, sigma, expected):
    assert scratch.inference.two_sided_p_value(x, mu, sigma) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize(
    'x, mu, sigma, expected',
    [
        (524.5, 500, 15.8, 0.061),
        (526.5, 500, 15.8, 0.047),
    ]
)
def test_upper_p_value(x, mu, sigma, expected):
    assert scratch.inference.upper_p_value(x, mu, sigma) == pytest.approx(expected, abs=0.001)

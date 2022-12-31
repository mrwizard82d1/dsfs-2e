import dataclasses
import math


from dsfs import probability as dp


@dataclasses.dataclass
class NormalDistributionParameters:
    mu: float = 0
    sigma: float = 1


def normal_approximation_to_binomial(n: int, p: float) -> NormalDistributionParameters:
    """Calculate the parameters for the normal distribution approximating `Bin(n, p)`"""
    return NormalDistributionParameters(mu=n * p, sigma=math.sqrt(n * p * (1 - p)))


# Whenever a random variable follows a normal distribution, we can use `normal_cdf` to determine the probability that
# its realized value lies within or outside a particular interval.


"""The normal CDF **is** the probability that a random variable is **below** a threshold."""
normal_probability_below = dp.normal_cdf


def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """Calculate the probability from an `N(mu, sigma)` distribution **above** a threshold, `lo`"""

    # The probability above a threshold **is** the probability that it is **not** below.
    return 1 - dp.normal_cdf(lo, mu, sigma)


def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """Calculate the probability of a `N(mu, sigma)` distribution between the values `lo` and `hi`"""

    # The result is the probability **less than** `hi` but **not less than** `lo`
    return dp.normal_cdf(hi, mu, sigma) - dp.normal_cdf(lo, mu, sigma)


def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """Calculate the probability of o `N(mu, sigma)` distribution outside of `lo` and `hi`."""

    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Calculate the value from an `N(mu, sigma)` distribution that has `probability` probability **below** it.

    In other words, calculate the value of the one-sided, left-hand tail.
    """
    return dp.inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Calculate the value from an `N(mu, sigma)` distribution that has `probability` probability **above** it.

    In other words, calculate the value of the one-sided, right-hand tail.
    """
    return dp.inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> tuple[float, float]:
    """Calculate the symmetric values about `mu` from an `N(mu, sigma)` distribution containing `probability`."""
    tail_probability = (1 - probability) / 2

    # The upper bound has the one-sided tail probability **above** it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # The lower bound has the one-sided tail probability **below** it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

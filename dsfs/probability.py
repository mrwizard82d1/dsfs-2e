import math


def uniform_pdf(x: float) -> float:
    """Calculate the uniform probability density function at `x`"""
    return x if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """Calculate the amount of probability in the uniform PDF less than or equal to `x`."""
    match x:
        # No probability is below 0
        case _ if x < 0:
            return 0
        # The probability below a value between 0 and 1 is the value itself
        case _ if x < 1:
            return x
        # All the probability is contained in region whose values are less than or equal 1
        case _:
            return 1


SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Calculate the normal probability density at `x`"""
    return (1 / (SQRT_TWO_PI * sigma)) * math.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Calculate the probability that a random variable distributed as N(`mu`, `sigma`) is less than or equal to `x`"""
    return (1 + math.erf((x - mu) / (math.sqrt(2) * sigma))) / 2

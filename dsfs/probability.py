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


def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 1e-5) -> float:
    """Calculate the approximate inverse of the normal CDF.

    The inverse of the normal CDF has no closed form solution even though approximations using infinite series and
    continued fractions do exist. We implement this **computer** function by performing a binary search of the normal
    CDF function for the value of interest.
    """
    # If the distribution is not standard (mean = 0, standard deviation = 1), transform it to standard.
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    # normal_cdf(-10) ~= 0; normal_cdf(10) ~= 1
    lo_z = -10
    hi_z = 10
    mid_z = (hi_z + lo_z) / 2  # initialize the midpoint so it always has a value (edge case)
    # while the range [hi_z, lo_z] is larger than the tolerance
    while hi_z - lo_z > tolerance:
        mid_z = (hi_z + lo_z) / 2  # calculate the midpoint
        mid_p = normal_cdf(mid_z)  # test the CDF at the midpoint
        if mid_p < p:  # if midpoint is less than the desired probability
            lo_z = mid_z  # the midpoint is too low; search above it
        else:
            hi_z = mid_z  # the midpoint is too high; search below it

    # Since we've exited the loop, the range [hi_z, lo_z] is within the tolerance so return the midpoint
    return mid_z

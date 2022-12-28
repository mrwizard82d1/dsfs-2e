def uniform_pdf(x):
    """Calculate the uniform probability density function at `x`"""
    return x if 0 <= x < 1 else 0


def uniform_cdf(x):
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

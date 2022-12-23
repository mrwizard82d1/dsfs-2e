"""
Implementing statistics.

Code from chapter 05, "Statistics," of _Data Science from Scratch_, 2nd edition.

In production, one might use either the `statistics` package (available for Python 3.10 and above), the `numpy` or
the `scipy` package.
"""

from collections import Counter
from numbers import Real
from typing import List, Set


def mean(xs: List[Real]) -> Real:
    """The mean is simply a sum of the values divided by the number of values."""
    return sum(xs) / len(xs)


def median(xs: List[Real]) -> Real:
    """Calculate the median of `xs`"""
    def median_odd(ys):
        return ys[len(ys) // 2]

    def median_even(ys):
        right_middle = xs[len(xs) // 2]  # the "right" item of the middle two items
        left_middle = xs[(len(xs) // 2) - 1]  # the "left" item of the middle two
        return (left_middle + right_middle) / 2  # return mean of middle two items

    sorted_xs = sorted(xs)
    return median_even(sorted_xs) if len(sorted_xs) % 2 == 0 else median_odd(sorted_xs)


def quantile(xs: List[Real], p: Real) -> Real:
    """Calculate the value under which the fraction, `p`, of the data lie."""
    p_index = int(p * len(xs))  # The index of the `p`-th fraction of all the data
    return sorted(xs)[p_index]


def mode(xs: List[Real]) -> Set[Real]:
    """Calculate the most common value(s) of `xs`."""
    # If no `xs`, return an empty set
    if not xs:
        return set()

    counts = Counter(xs)
    max_counts = max(counts.values())
    return {x_i for x_i, count in counts.items() if count == max_counts}

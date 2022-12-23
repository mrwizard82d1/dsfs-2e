"""
Code from chapter 05, "Statistics," of _Data Science from Scratch_, 2nd edition.
"""

from numbers import Real

from typing import List


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

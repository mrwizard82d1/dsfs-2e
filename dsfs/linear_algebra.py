"""
Define what we need to support linear algebra 'from scratch'.

This code is not production code but is useful for teaching.
"""

import functools
from numbers import Real
from typing import List

# A vector is just a list of real numbers. Using the `Real` type includes
# - int
# - float
# - fraction.Fraction
# - decimal.Decimal
Vector = List[Real]


def add(v: Vector, w: Vector) -> Vector:
    """Add two vectors, `v` and `w`, producing a third."""
    assert len(v) == len(w)  # Cannot add Vectors of different rank

    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtract vector `w` from vector `v` producing a third vector."""
    assert len(v) == len(w)  # Cannot subtract Vectors of different rank

    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def zero(rank: int) -> Vector:
    """Return a 'zero' vector of `rank`"""

    return [0] * rank


def vector_sum(vs: List[Vector]) -> Vector:
    """Sum a `list` of `Vector`s, `vs` producing a new vector."""
    # The sum of an empty list is undefinedi
    assert len(vs) != 0

    # Cannot sum vectors of different ranks
    length_of_first_vector = len(vs[0])
    assert all([len(v) == length_of_first_vector for v in vs])

    return functools.reduce(lambda item, so_far: add(so_far, item), vs, zero(length_of_first_vector))


assert vector_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [12, 15, 18]

"""
Define what we need to support linear algebra 'from scratch'.

This code is not production code but is useful for teaching. In production code, you want to use the `numpy` package.
"""

import math
from numbers import Real
from typing import Callable, List, Tuple

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


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtract vector `w` from vector `v` producing a third vector."""
    assert len(v) == len(w)  # Cannot subtract Vectors of different rank

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def zero(rank: int) -> Vector:
    """Return a 'zero' vector of `rank`"""

    return [0] * rank


def vector_sum(vs: List[Vector]) -> Vector:
    """Sum a `list` of `Vector`s, `vs` producing a new vector."""
    # The sum of an empty list is undefined
    assert len(vs) != 0

    # Cannot sum vectors of different ranks
    rank = len(vs[0])
    assert all([len(v) == rank for v in vs])

    return [sum(v[i] for v in vs) for i in range(rank)]
    # Equivalent to the following. Perhaps clearer.
    # return functools.reduce(lambda item, so_far: add(so_far, item), vs, zero(rank))


def scalar_multiply(c: Real, v: Vector) -> Vector:
    """Multiply vector, `v`, by the scalar, `c` producing another vector"""
    return [c * v_i for v_i in v]


# Defining `scalar_multiply` allows us to calculate a component-wise mean of vectors of the same rank
def vector_mean(vs: List[Vector]) -> Vector:
    """Calculate the 'component-wise' mean of a `list` of vectors"""
    return scalar_multiply(1 / len(vs), vector_sum(vs))


def dot(v: Vector, w: Vector) -> Real:
    """Calculate the dot product of two vectors, `v` and `w`"""
    assert len(v) == len(w), 'Dot product only defined on Vectors of the same rank'

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> Real:
    """Calculate the component wise sum of squares of vector, `v`"""
    return dot(v, v)


def magnitude(v: Vector) -> Real:
    """Calculate the magnitude of the vector, `v`"""
    return math.sqrt(sum_of_squares(v))


def distance(v: Vector, w: Vector) -> Real:
    """Calculate the distance between two vectors, `v` and `w`"""
    return magnitude(subtract(v, w))


# A matrix is just a list of identically sized lists of real numbers. Using the `Real` type includes
# - int
# - float
# - fraction.Fraction
# - decimal.Decimal
Matrix = List[List[Real]]

# Per mathematical convention, we frequently use capital letters to represent matrices.
A = [[1, 2, 3],  # A has 2 rows
     [4, 5, 6]]  # and 3 columns

B = [[1, 2],  # B has 3 rows
     [3, 4],  # and 2 columns
     [5, 6]]

# Although in mathematics, we number rows and columns of matrices beginning with 1. Because we are using Python, we use
# the Python convention beginning with 0 (z).

# The shape of a (2-D) matrix is a pair of `int` values returned as a `tuple`. The first item in the `tuple` is the
# number of rows; the second item in the `tuple` is the number of columns.


# noinspection PyPep8Naming,PyShadowingNames
def shape(A: Matrix) -> Tuple[int, int]:
    """Return the shape (rank) of the matrix, `A`, as a `tuple` of row count x column count.

    If `A` is an empty matrix, return the `tuple`, (0, 0).
    """
    return len(A), len(A[0]) if A else 0


# noinspection PyPep8Naming,PyShadowingNames
def row(A: Matrix, j: int) -> Vector:
    """Return the `j`th row of the matrix, `A`, as a vector.

    We sometimes think of a j x k matrix as composed of j row vectors or of k column vectors.
    """
    return A[j]


# noinspection PyPep8Naming,PyShadowingNames
def column(A: Matrix, k: int) -> Vector:
    """Return the `k`th column of the matrix, `A`, as a vector.

    We sometimes think of a j x k matrix as composed of j row vectors or of k column vectors.
    """
    return [A_j[k] for A_j in A]


def make_matrix(row_count: int,
                column_count: int,
                entry_fn: Callable[[int, int], Real]) -> Matrix:
    """Returns a `row_count` x `column_count` matrix whose (j, k) element is initialized with `entry_fn(j, k)`"""
    return [[entry_fn(j, k) for k in range(column_count)] for j in range(row_count)]


def identity_matrix(n: int) -> Matrix:
    """Returns an n x n identity matrix."""
    return make_matrix(n, n, lambda j, k: 1 if j == k else 0)


def zero_matrix(j: int, k: int) -> Matrix:
    """Return a j x k matrix all of whose elements are zero."""
    return make_matrix(j, k, lambda _j, _k: 0)


# One can use a matrix to represent "network connections" like we did with the `friendships` value previously. This
# representation is not really efficient for a "sparse" graph, but it is efficient for a dense matrix.
#            user 0  1  2  3  4  5  6  7  8  9
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

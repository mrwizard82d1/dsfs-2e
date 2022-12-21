import dsfs as scratch
import pytest


def test_canary():
    assert True


def test_add_vectors():
    assert scratch.linear_algebra.add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_subtract_vectors():
    assert scratch.linear_algebra.subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def test_vector_sum():
    assert scratch.linear_algebra.vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def test_scalar_multiply():
    assert scratch.linear_algebra.scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def test_vector_mean():
    assert scratch.linear_algebra.vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def test_dot():
    assert scratch.linear_algebra.dot([1, 2, 3], [4, 5, 6]) == 32


def test_sum_of_squares():
    assert scratch.linear_algebra.sum_of_squares([1, 2, 3]) == 14


def test_magnitude():
    assert scratch.linear_algebra.magnitude([3, 4]) == 5


def test_distance():
    assert scratch.linear_algebra.distance([4, 6], [1, 2]) == 5


# noinspection PyPep8Naming
@pytest.mark.parametrize(
    'A,expected',
    [
        ([[1, 2]], (1, 2)),
        ([[1, 2, 3],
          [4, 5, 6]], (2, 3)),
        ([[1, 2],
          [3, 4],
          [5, 6]], (3, 2)),
        ([[1]], (1, 1)),
        ([[]], (1, 0)),
        ([], (0, 0)),
    ]
)
def test_shape(A, expected):
    assert scratch.linear_algebra.shape(A) == expected

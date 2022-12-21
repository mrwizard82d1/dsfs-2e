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


def test_row():
    assert scratch.linear_algebra.row([[1, 2, 3],
                                       [4, 5, 6]], 1) == [4, 5, 6]


def test_column():
    assert scratch.linear_algebra.column([[1, 2],
                                          [3, 4],
                                          [5, 6]], 1) == [2, 4, 6]


@pytest.mark.parametrize(
    'row_count, column_count, entry_fn, expected',
    [
        (0, 0, lambda j, k: -1, []),
        (1, 0, lambda j, k: -1, [[]]),
        (1, 1, lambda j, k: 1, [[1]]),
        (1, 2, lambda j, k: (j + 1) * k, [[0, 1]]),
        (2, 3, lambda j, k: j + k, [[0, 1, 2],
                                    [1, 2, 3]]),
        (3, 2, lambda j, k: j + k, [[0, 1],
                                    [1, 2],
                                    [2, 3]]),
    ]
)
def test_make_matrix(row_count, column_count, entry_fn, expected):
    assert scratch.linear_algebra.make_matrix(row_count, column_count, entry_fn) == expected


@pytest.mark.parametrize(
    'n, expected',
    [
        (0, []),
        (1, [[1]]),
        (2, [[1, 0],
             [0, 1]]),
        (3, [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])
    ]
)
def test_identity_matrix(n, expected):
    assert scratch.linear_algebra.identity_matrix(n) == expected


@pytest.mark.parametrize(
    'j, k, expected',
    [
        (0, 0, []),
        (1, 0, [[]]),
        (1, 1, [[0]]),
        (1, 2, [[0, 0]]),
        (2, 3, [[0, 0, 0],
                [0, 0, 0]]),
        (3, 2, [[0, 0],
                [0, 0],
                [0, 0]]),
    ]
)
def test_zero_matrix(j, k, expected):
    assert scratch.linear_algebra.zero_matrix(j, k) == expected


@pytest.mark.parametrize(
    'user_1, user_2, expected',
    [
        (0, 2, True),
        (0, 8, False),
    ]
)
def test_are_friends(user_1, user_2, expected):
    assert scratch.linear_algebra.friend_matrix[user_1][user_2] == expected, f'{user_1} and {user_2} ' \
                                                                             f'are {"" if expected else "not "}'


def test_friends_of_five():
    friends_of_five = [i for i, is_friend in enumerate(scratch.linear_algebra.friend_matrix[5]) if is_friend]
    assert friends_of_five == [4, 6, 7]

import math

import pytest

import dsfs as scratch


# Copied from 05-Statistics.ipynb and renamed to `test_num_friends`
test_num_friends = [100, 49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12,
                    11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5,
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_mean():
    assert 7.333332 < float(scratch.statistics.mean(test_num_friends) )< 7.333334


@pytest.mark.parametrize(
    'data,expected',
    [
        ((1, 2, 3, 4, 5), 3),
        ((1, 2, 3, 4, 5, 6), 3.5),
        ((1, 10, 2, 9, 5), 5),
        ((9, 1, 10, 2), (2 + 9) / 2),  # because the floating point value involves 0.5, we can use an exact comparison
        (test_num_friends, 6)
    ]
)
def test_median(data, expected):
    assert scratch.statistics.median(data) == expected


@pytest.mark.parametrize(
    'p,expected',
    [
        (0.10, 1),
        (0.25, 3),
        (0.50, 6),
        (0.75, 9),
        (0.90, 13),
    ]
)
def test_quantile(p, expected):
    assert scratch.statistics.quantile(test_num_friends, p) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ([], set()),
        ([3], {3}),
        ([3, 1, 4, 1, 5, 9], {1}),
        ([2, 7, 1, 8, 2, 7, 1, 8], {2, 7, 1, 8}),
        (test_num_friends, {1, 6})
    ]
)
def test_mode(data, expected):
    assert scratch.statistics.mode(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ([3], 0),
        ([3, 1, 4, 1, 5, 9], 8),
        (test_num_friends, 99)
    ]
)
def test_data_range(data, expected):
    assert scratch.statistics.data_range(data) == expected


@pytest.mark.parametrize(
    'data, expected',
    [
        ([0] * 50 + [100] * 50, (50 ** 2 * 100) / (100 - 1)),
        ([0] + [50] * 98 + [100], (50 ** 2 * 2) / (100 - 1)),
        (test_num_friends, 81.5435)
    ]
)
def test_variance(data, expected):
    assert float(scratch.statistics.variance(data)) == pytest.approx(expected)


@pytest.mark.parametrize(
    'data, expected',
    [
        ([0] * 50 + [100] * 50, math.sqrt((50 ** 2 * 100) / (100 - 1))),
        ([0] + [50] * 98 + [100], math.sqrt((50 ** 2 * 2) / (100 - 1))),
        (test_num_friends, math.sqrt(81.5435))
    ]
)
def test_standard_deviation(data, expected):
    assert float(scratch.statistics.standard_deviation(data)) == pytest.approx(expected)


@pytest.mark.parametrize(
    'data, expected',
    [
        ([0] * 50 + [100] * 50, 100),
        ([0] + [50] * 98 + [100], 0),
        (test_num_friends, 6),
        ([200] + test_num_friends[1:], 6)
    ]
)
def test_interquartile_range(data, expected):
    assert float(scratch.statistics.interquartile_range(data)) == pytest.approx(expected)

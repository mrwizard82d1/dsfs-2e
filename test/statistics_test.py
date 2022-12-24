import math

import pytest

import dsfs as scratch


# Copied from 05-Statistics.ipynb and renamed to `test_XYZ`
test_num_friends = [100, 49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12,
                    11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5,
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


test_daily_minutes = [1, 68.77, 51.25, 52.08, 38.36, 44.54, 57.13, 51.4, 41.42, 31.22, 34.76, 54.01, 38.79, 47.59, 49.1,
                      27.66, 41.03, 36.73, 48.65, 28.12, 46.62, 35.57, 32.98, 35, 26.07, 23.77, 39.73, 40.57, 31.65,
                      31.21, 36.32, 20.45, 21.93, 26.02, 27.34, 23.49, 46.94, 30.5, 33.8, 24.23, 21.4, 27.94, 32.24,
                      40.57, 25.07, 19.42, 22.39, 18.42, 46.96, 23.72, 26.41, 26.97, 36.76, 40.32, 35.02, 29.47,
                      30.2, 31, 38.11, 38.18, 36.31, 21.03, 30.86, 36.07, 28.66, 29.08, 37.28, 15.28, 24.17, 22.31,
                      30.17, 25.53, 19.85, 35.37, 44.6, 17.23, 13.47, 26.33, 35.02, 32.09, 24.81, 19.33, 28.77,
                      24.26, 31.98, 25.73, 24.86, 16.28, 34.51, 15.23, 39.72, 40.8, 26.06, 35.76, 34.76, 16.13,
                      44.04, 18.03, 19.65, 32.62, 35.59, 39.43, 14.18, 35.24, 40.13, 41.82, 35.45, 36.07, 43.67,
                      24.61, 20.9, 21.9, 18.79, 27.61, 27.21, 26.61, 29.77, 20.59, 27.53, 13.82, 33.2, 25, 33.1,
                      36.65, 18.63, 14.87, 22.2, 36.81, 25.53, 24.62, 26.25, 18.21, 28.08, 19.42, 29.79, 32.8, 35.99,
                      28.32, 27.79, 35.88, 29.06, 36.28, 14.1, 36.63, 37.49, 26.9, 18.58, 38.48, 24.48, 18.95, 33.55,
                      14.24, 29.04, 32.51, 25.63, 22.22, 19, 32.73, 15.16, 13.9, 27.2, 32.01, 29.27, 33, 13.74,
                      20.42, 27.32, 18.23, 35.35, 28.48, 9.08, 24.62, 20.12, 35.26, 19.92, 31.02, 16.49, 12.16, 30.7,
                      31.22, 34.65, 13.13, 27.51, 33.2, 31.57, 14.1, 33.42, 17.44, 10.12, 24.42, 9.82, 23.39, 30.93,
                      15.03, 21.67, 31.09, 33.29, 22.61, 26.89, 23.48, 8.38, 27.81, 32.35, 23.84]


test_daily_hours = [tdm / 60 for tdm in test_daily_minutes]


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


# Correlation


@pytest.mark.parametrize(
    'left_data, right_data, expected, abs_tol',
    [
        ([3, 1, 4, 1, 5, 9], [3, 1, 4, 1, 5, 9], 8.966665, None),
        ([3, 1, 4, 1, 5, 9], [-3, -1, -4, -1, -5, -9], -8.966665, None),
        (test_num_friends, test_daily_minutes, 22.43, 9e-03),
        (test_num_friends, test_daily_hours, 22.43 / 60, 9e-03),
    ]
)
def test_covariance(left_data, right_data, expected, abs_tol):
    assert float(scratch.statistics.covariance(left_data, right_data)) == pytest.approx(expected, abs_tol)

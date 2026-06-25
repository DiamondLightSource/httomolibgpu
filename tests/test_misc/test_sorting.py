import pytest
import cupy as cp

from httomolibgpu.misc.sorting import argsort_with_reverse


class TestArgsort1D:
    def test_1d_sorted(self):
        data = cp.array([1, 2, 3, 4, 5])
        sort_idx, rev_idx = argsort_with_reverse(data)
        cp.testing.assert_array_equal(sort_idx, cp.array([0, 1, 2, 3, 4]))
        cp.testing.assert_array_equal(data[sort_idx], data)
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.arange(5))

    def test_1d_reverse_sorted(self):
        data = cp.array([5, 4, 3, 2, 1])
        sort_idx, rev_idx = argsort_with_reverse(data)
        cp.testing.assert_array_equal(sort_idx, cp.array([4, 3, 2, 1, 0]))
        cp.testing.assert_array_equal(data[sort_idx], cp.array([1, 2, 3, 4, 5]))
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.arange(5))

    def test_1d_random(self):
        data = cp.array([10, 3, 7, 1, 9])
        sort_idx, rev_idx = argsort_with_reverse(data)
        sorted_data = data[sort_idx]
        assert cp.all(sorted_data[:-1] <= sorted_data[1:])
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.arange(5))

    def test_1d_duplicates(self):
        data = cp.array([2, 1, 2, 1])
        sort_idx, rev_idx = argsort_with_reverse(data)
        sorted_data = data[sort_idx]
        assert cp.all(sorted_data[:-1] <= sorted_data[1:])
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.arange(4))

    def test_1d_single_element(self):
        data = cp.array([42])
        sort_idx, rev_idx = argsort_with_reverse(data)
        cp.testing.assert_array_equal(sort_idx, cp.array([0]))
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.array([0]))

    def test_1d_negative_axis(self):
        data = cp.array([3, 1, 2])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=-1)
        cp.testing.assert_array_equal(sort_idx, cp.argsort(data))
        cp.testing.assert_array_equal(rev_idx[sort_idx], cp.arange(3))


class TestArgsort2D:
    def test_2d_axis0_sorted(self):
        data = cp.array([[1, 2], [3, 4], [5, 6]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=0)
        sorted_data = data[sort_idx, cp.arange(data.shape[1])[None, :]]
        assert cp.all(sorted_data[:-1, :] <= sorted_data[1:, :])
        rows, cols = data.shape
        col_indices = cp.arange(cols)[None, :]
        cp.testing.assert_array_equal(
            rev_idx[sort_idx, col_indices], cp.tile(cp.arange(rows)[:, None], [1, 2])
        )

    def test_2d_axis0_random(self):
        data = cp.array([[5, 1], [2, 4], [3, 6]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=0)
        rows, cols = data.shape
        col_indices = cp.arange(cols)[None, :]
        cp.testing.assert_array_equal(
            rev_idx[sort_idx, col_indices], cp.tile(cp.arange(rows)[:, None], [1, 2])
        )

    def test_2d_axis0_negative_axis(self):
        data = cp.array([[1, 2], [3, 4]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=-2)
        rows, cols = data.shape
        col_indices = cp.arange(cols)[None, :]
        cp.testing.assert_array_equal(
            rev_idx[sort_idx, col_indices], cp.tile(cp.arange(rows)[:, None], [1, 2])
        )

    def test_2d_axis1_sorted(self):
        data = cp.array([[1, 2, 3], [4, 5, 6]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=1)
        rows, cols = data.shape
        row_indices = cp.arange(rows)[:, None]
        cp.testing.assert_array_equal(
            rev_idx[row_indices, sort_idx], cp.tile(cp.arange(cols), [2, 1])
        )

    def test_2d_axis1_random(self):
        data = cp.array([[3, 1, 2], [6, 4, 5]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=1)
        rows, cols = data.shape
        row_indices = cp.arange(rows)[:, None]
        cp.testing.assert_array_equal(
            rev_idx[row_indices, sort_idx], cp.tile(cp.arange(cols), [2, 1])
        )

    def test_2d_axis1_negative_axis(self):
        data = cp.array([[1, 2], [3, 4]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=-1)
        rows, cols = data.shape
        row_indices = cp.arange(rows)[:, None]
        cp.testing.assert_array_equal(
            rev_idx[row_indices, sort_idx], cp.tile(cp.arange(cols), [2, 1])
        )

    def test_2d_single_row(self):
        data = cp.array([[3, 1, 2]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=1)
        rows, cols = data.shape
        row_indices = cp.arange(rows)[:, None]
        cp.testing.assert_array_equal(
            rev_idx[row_indices, sort_idx], cp.arange(cols)[None, :]
        )

    def test_2d_single_col(self):
        data = cp.array([[3], [1], [2]])
        sort_idx, rev_idx = argsort_with_reverse(data, axis=0)
        rows, cols = data.shape
        col_indices = cp.arange(cols)[None, :]
        cp.testing.assert_array_equal(
            rev_idx[sort_idx, col_indices], cp.arange(rows)[:, None]
        )


class TestArgsortErrors:
    def test_invalid_dim_3d(self):
        data = cp.ones((2, 2, 2))
        with pytest.raises(ValueError):
            argsort_with_reverse(data)

    def test_invalid_dim_0d(self):
        data = cp.array(1)
        with pytest.raises(ValueError):
            argsort_with_reverse(data)

    def test_invalid_axis_1d(self):
        data = cp.array([1, 2, 3])
        with pytest.raises(ValueError):
            argsort_with_reverse(data, axis=1)

    def test_invalid_axis_2d(self):
        data = cp.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            argsort_with_reverse(data, axis=2)

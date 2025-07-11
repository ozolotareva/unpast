"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from unpast.core.preprocessing import prepare_input_matrix, zscore
from unpast.utils.logs import LOG_LEVELS


class TestZscore:
    """Test cases for zscore function."""

    def test_zscore_basic(self):
        """Test basic z-score normalization."""
        # Create a simple test matrix
        data = {"sample1": [1, 4, 7], "sample2": [2, 5, 8], "sample3": [3, 6, 9]}
        df = pd.DataFrame(data, index=["gene1", "gene2", "gene3"])

        result = zscore(df)

        # Check that means are approximately 0
        means = result.mean(axis=1)
        assert np.allclose(means, 0, atol=1e-10)

        # Check that standard deviations are approximately 1
        stds = result.std(axis=1)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_zscore_zero_variance(self, caplog):
        """Test z-score normalization with zero variance genes."""
        # Create matrix with one zero-variance gene
        data = {"sample1": [1, 5, 5], "sample2": [2, 5, 5], "sample3": [3, 5, 5]}
        df = pd.DataFrame(data, index=["gene1", "gene2", "gene3"])

        # Capture warning messages using caplog
        with caplog.at_level(LOG_LEVELS["WARNING"]):
            result = zscore(df)

        # Check that zero variance genes are set to 0
        assert np.allclose(result.loc["gene2"], 0)
        assert np.allclose(result.loc["gene3"], 0)

        # Check that warning was logged
        assert "zero variance rows detected" in caplog.text
        assert "2" in caplog.text  # Should detect 2 zero variance rows

    def test_zscore_single_gene(self):
        """Test z-score normalization with single gene."""
        data = {"sample1": [1], "sample2": [2], "sample3": [3]}
        df = pd.DataFrame(data, index=["gene1"])

        result = zscore(df)

        # Check normalization worked
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_zscore_preserves_shape(self):
        """Test that zscore preserves DataFrame shape and indices."""
        data = {
            "sample1": [1, 4, 7, 10],
            "sample2": [2, 5, 8, 11],
            "sample3": [3, 6, 9, 12],
        }
        df = pd.DataFrame(data, index=["gene1", "gene2", "gene3", "gene4"])

        result = zscore(df)

        # Check shape is preserved
        assert result.shape == df.shape

        # Check indices are preserved
        assert list(result.index) == list(df.index)
        assert list(result.columns) == list(df.columns)


class TestPrepareInputMatrix:
    """Test cases for prepare_input_matrix function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create a test matrix with known properties
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            np.random.randn(5, 10),
            index=["gene1", "gene2", "gene3", "gene4", "gene5"],
            columns=[f"sample{i}" for i in range(1, 11)],
        )

        # Create data that's already standardized
        self.standardized_data = pd.DataFrame(
            np.random.randn(3, 5),
            index=["gene1", "gene2", "gene3"],
            columns=["sample1", "sample2", "sample3", "sample4", "sample5"],
        )
        # Manually standardize it
        self.standardized_data = (
            self.standardized_data.T - self.standardized_data.mean(axis=1)
        ).T
        self.standardized_data = (
            self.standardized_data.T / self.standardized_data.std(axis=1)
        ).T

    def test_prepare_input_matrix_basic(self):
        """Test basic functionality of prepare_input_matrix."""
        result = prepare_input_matrix(self.test_data)

        # Check that result is standardized
        means = result.mean(axis=1)
        stds = result.std(axis=1)
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-10)

        # Check that indices are strings
        assert all(isinstance(idx, str) for idx in result.index)
        assert all(isinstance(col, str) for col in result.columns)

    def test_prepare_input_matrix_already_standardized(self, caplog):
        """Test with already standardized data."""
        with caplog.at_level(LOG_LEVELS["INFO"]):
            result = prepare_input_matrix(self.standardized_data)

        # Should not print standardization message
        assert "Input is not standardized" not in caplog.text

        # Result should be approximately the same as input
        assert np.allclose(result.values, self.standardized_data.values, atol=1e-10)

    def test_prepare_input_matrix_zero_variance(self, caplog):
        """Test handling of zero variance genes."""
        # Add zero variance genes
        data_with_zeros = self.test_data.copy()
        data_with_zeros.loc["zero_gene1"] = 5.0  # Constant value
        data_with_zeros.loc["zero_gene2"] = 10.0  # Another constant value

        with caplog.at_level(LOG_LEVELS["DEBUG"]):
            result = prepare_input_matrix(data_with_zeros)

        # Check that zero variance genes were dropped
        assert "zero_gene1" not in result.index
        assert "zero_gene2" not in result.index
        assert result.shape[0] == self.test_data.shape[0]  # Original number of genes

        # Check verbose output
        assert "Zero variance rows will be dropped: 2" in caplog.text

    def test_prepare_input_matrix_with_missing_values(self, caplog):
        """Test handling of missing values."""
        # Add missing values
        data_with_na = self.test_data.copy()
        data_with_na.iloc[0, :3] = np.nan  # gene1 has 3 missing values
        data_with_na.iloc[1, :8] = (
            np.nan
        )  # gene2 has 8 missing values (should be dropped)

        with caplog.at_level(LOG_LEVELS["WARNING"]):
            result = prepare_input_matrix(data_with_na, min_n_samples=5)

        # gene2 should be dropped (only 2 valid samples, less than min_n_samples=5)
        assert "gene2" not in result.index
        assert "gene1" in result.index  # gene1 should be kept (7 valid samples)

        # Check verbose output
        assert "Missing values detected" in caplog.text
        assert "Features with too few values" in caplog.text

    def test_prepare_input_matrix_with_ceiling(self, caplog):
        """Test z-score ceiling functionality."""
        # Create data with extreme values
        data = pd.DataFrame(
            [[10, 1, 1], [1, 10, 1], [1, 1, 10]],
            index=["gene1", "gene2", "gene3"],
            columns=["sample1", "sample2", "sample3"],
        )

        with caplog.at_level(LOG_LEVELS["DEBUG"]):
            result = prepare_input_matrix(data, ceiling=2.0)

        # Check that values are capped at ceiling
        assert np.all(result <= 2.0)
        assert np.all(result >= -2.0)

        # Check verbose output
        assert "Standardized expressions will be limited to [-2.0,2.0]:" in caplog.text

    def test_prepare_input_matrix_no_standardization(self):
        """Test with standardization disabled."""
        result = prepare_input_matrix(self.test_data, standradize=False)

        # Result should be the same as input (except for string conversion)
        expected = self.test_data.copy()
        expected.index = expected.index.astype(str)
        expected.columns = expected.columns.astype(str)

        pd.testing.assert_frame_equal(result, expected)

    def test_prepare_input_matrix_missing_values_with_ceiling(self, caplog):
        """Test missing values replacement with ceiling."""
        # Create data with missing values
        data = self.test_data.copy()
        data.iloc[0, 0] = np.nan
        data.iloc[1, 1] = np.nan

        with caplog.at_level(LOG_LEVELS["DEBUG"]):
            result = prepare_input_matrix(data, ceiling=2.0)

        # Check that missing values were replaced with -ceiling
        # Note: After standardization, the exact values will be different,
        # but we can check that no NaN values remain
        assert not result.isna().any().any()

        # Check verbose output
        assert "Missing values will be replaced with -2.0." in caplog.text

    def test_prepare_input_matrix_too_few_features(self, caplog):
        """Test behavior when too few features remain after filtering."""
        # Create data where most genes will be dropped
        data = pd.DataFrame(
            [
                [1, 1, 1],
                [2, 2, 2],
            ],  # Only 2 genes, both will have zero variance after standardization
            index=["gene1", "gene2"],
            columns=["sample1", "sample2", "sample3"],
        )

        with caplog.at_level(LOG_LEVELS["WARNING"]):
            # This should trigger the warning about too few features
            _ = prepare_input_matrix(data)

        # The exact behavior may vary, but we should get a warning
        assert "less than 3 features (rows) remain" in caplog.text

    def test_prepare_input_matrix_duplicate_indices(self, caplog):
        """Test handling of non-unique row names."""
        # Create data with duplicate row names
        data = self.test_data.copy()
        data = data.reindex(["gene1", "gene1", "gene2", "gene2", "gene3"])  # Duplicates

        with caplog.at_level(LOG_LEVELS["WARNING"]):
            _ = prepare_input_matrix(data)

        # Should print warning about non-unique row names
        assert "Row names are not unique" in caplog.text

    def test_prepare_input_matrix_tolerance(self):
        """Test the tolerance parameter for standardization check."""
        # Create data that's almost standardized
        data = self.standardized_data.copy()
        data = data + 0.005  # Add small deviation

        # With default tolerance (0.01), should not re-standardize
        result1 = prepare_input_matrix(data, tol=0.01)

        # With smaller tolerance (0.001), should re-standardize
        result2 = prepare_input_matrix(data, tol=0.001)

        # Results should be different
        assert not np.allclose(result1.values, result2.values)

    def test_prepare_input_matrix_min_n_samples(self):
        """Test the min_n_samples parameter."""
        # Create data with missing values
        data = pd.DataFrame(
            np.random.randn(3, 10),
            index=["gene1", "gene2", "gene3"],
            columns=[f"sample{i}" for i in range(1, 11)],
        )
        # Add missing values to gene1 (6 missing, 4 valid)
        data.iloc[0, :6] = np.nan

        # With min_n_samples=5, gene1 should be dropped
        result1 = prepare_input_matrix(data, min_n_samples=5)
        assert "gene1" not in result1.index

        # With min_n_samples=3, gene1 should be kept
        result2 = prepare_input_matrix(data, min_n_samples=3)
        assert "gene1" in result2.index


if __name__ == "__main__":
    pytest.main([__file__])

import numpy as np
import pandas as pd

from unpast.utils.similarity import get_similarity_jaccard

# from unpast.utils.similarity import get_similarity_corr


class TestGetSimilarityJaccard:
    """Test cases for get_similarity_jaccard function."""

    def test_jaccard_similarity_basic(self):
        """Test basic Jaccard similarity calculation."""
        # Create binary data with clear similarity patterns
        data = pd.DataFrame(
            {
                "gene1": [1, 1, 0, 0, 0],
                "gene2": [1, 1, 0, 0, 0],  # Identical to gene1
                "gene3": [0, 0, 1, 1, 1],  # Complement of gene1/gene2
                "gene4": [1, 0, 1, 0, 1],  # Different pattern
            },
            index=["s1", "s2", "s3", "s4", "s5"],
        )

        result = get_similarity_jaccard(data)

        # Check that it's symmetric
        assert result.shape == (4, 4)
        assert np.allclose(result.values, result.values.T)

        # Check diagonal is 1
        assert np.allclose(np.diag(result.values), 1.0)

        # gene1 and gene2 should be identical (similarity = 1)
        assert result.loc["gene1", "gene2"] == 1.0

        # Check that similarities are between 0 and 1
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

    def test_jaccard_similarity_complement_matching(self):
        """Test complement matching in Jaccard similarity."""
        # Create data where one gene has many 1s (should use complement)
        data = pd.DataFrame(
            {
                "gene1": [1, 1, 1, 1, 0],  # Many 1s
                "gene2": [0, 0, 0, 0, 1],  # Complement pattern
            },
            index=["s1", "s2", "s3", "s4", "s5"],
        )

        result = get_similarity_jaccard(data)

        # Should detect complement matching
        similarity_value = result.iloc[0, 1]  # Use iloc for numeric access
        assert similarity_value > 0

    def test_jaccard_similarity_single_gene(self):
        """Test with single gene."""
        data = pd.DataFrame(
            {
                "gene1": [1, 0, 1, 0, 1],
            },
            index=["s1", "s2", "s3", "s4", "s5"],
        )

        result = get_similarity_jaccard(data)

        assert result.shape == (1, 1)
        assert result.iloc[0, 0] == 1.0

    def test_jaccard_similarity_all_zeros(self):
        """Test with genes that have all zeros."""
        data = pd.DataFrame(
            {
                "gene1": [0, 0, 0, 0, 0],
                "gene2": [1, 0, 1, 0, 0],
            },
            index=["s1", "s2", "s3", "s4", "s5"],
        )

        result = get_similarity_jaccard(data)

        # Should handle edge cases gracefully
        assert result.shape == (2, 2)
        assert not np.any(np.isnan(result.values))


# class TestGetSimilarityCorr:
#     """Test cases for get_similarity_corr function."""

#     def test_correlation_similarity_basic(self):
#         """Test basic correlation similarity calculation."""
#         # Create data with known correlation patterns
#         data = pd.DataFrame(
#             {
#                 "gene1": [1.0, 2.0, 3.0, 4.0, 5.0],
#                 "gene2": [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfect positive correlation
#                 "gene3": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfect negative correlation
#                 "gene4": [2.0, 1.0, 4.0, 3.0, 6.0],  # Different pattern
#             },
#             index=["s1", "s2", "s3", "s4", "s5"],
#         )

#         result = get_similarity_corr(data, verbose=False)

#         # Check basic properties
#         assert result.shape == (4, 4)
#         assert np.allclose(np.diag(result.values), 1.0)

#         # gene1 and gene2 should have perfect positive correlation
#         # Check that correlation matrix has the expected structure
#         assert result.shape == (4, 4)
#         assert result.iloc[0, 0] == 1.0  # Diagonal should be 1

#         # gene1 and gene3 should have negative correlation (set to 0)
#         assert result.loc["gene1", "gene3"] == 0.0

#         # All values should be >= 0 (negative correlations set to 0)
#         assert np.all(result.values >= 0)

#     def test_correlation_similarity_no_variation(self):
#         """Test with constant genes (no variation)."""
#         data = pd.DataFrame(
#             {
#                 "gene1": [1.0, 1.0, 1.0, 1.0, 1.0],  # No variation
#                 "gene2": [2.0, 3.0, 4.0, 5.0, 6.0],  # Normal variation
#             },
#             index=["s1", "s2", "s3", "s4", "s5"],
#         )

#         result = get_similarity_corr(data, verbose=True)

#         # Should handle constant genes gracefully
#         assert result.shape == (2, 2)
#         # Correlation with constant gene should be NaN, then filled with 0
#         assert result.loc["gene1", "gene2"] == 0.0

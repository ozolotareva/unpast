"""Tests for feature_clustering module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from unpast.core.feature_clustering import run_Louvain
from unpast.utils.similarity import get_similarity_jaccard

class TestRunLouvain:
    """Test cases for run_Louvain function."""

    def setup_method(self):
        """Set up test data."""
        # Create a similarity matrix with clear communities
        self.similarity_matrix = pd.DataFrame(
            {
                "gene1": [1.0, 0.8, 0.1, 0.1],
                "gene2": [0.8, 1.0, 0.1, 0.1],
                "gene3": [0.1, 0.1, 1.0, 0.9],
                "gene4": [0.1, 0.1, 0.9, 1.0],
            },
            index=["gene1", "gene2", "gene3", "gene4"],
        )

    def test_louvain_basic(self):
        """Test basic Louvain clustering."""
        modules, not_clustered, best_cutoff = run_Louvain(
            self.similarity_matrix,
            similarity_cutoffs=np.array([0.5]),
            verbose=False,
            plot=False,
        )

        # Should identify some modules
        assert isinstance(modules, list)
        assert isinstance(not_clustered, list)
        assert isinstance(best_cutoff, (float, int))

        # Total genes should be conserved
        total_genes = sum(len(module) for module in modules) + len(not_clustered)
        assert total_genes == 4

    def test_louvain_empty_similarity(self):
        """Test Louvain with empty similarity matrix."""
        empty_sim = pd.DataFrame({})

        modules, not_clustered, best_cutoff = run_Louvain(
            empty_sim, similarity_cutoffs=np.array([0.5]), verbose=False, plot=False
        )

        # Should handle empty input gracefully
        assert modules == []
        assert not_clustered == []
        assert best_cutoff is None

    def test_louvain_single_cutoff(self):
        """Test Louvain with single similarity cutoff."""
        modules, not_clustered, best_cutoff = run_Louvain(
            self.similarity_matrix,
            similarity_cutoffs=np.array([0.7]),
            verbose=True,
            plot=True,
        )

        # Should use the single cutoff provided
        assert best_cutoff == 0.7

    def test_louvain_multiple_cutoffs(self):
        """Test Louvain with multiple similarity cutoffs."""
        modules, not_clustered, best_cutoff = run_Louvain(
            self.similarity_matrix,
            similarity_cutoffs=np.array([0.3, 0.5, 0.7]),
            verbose=False,
            plot=False,
        )

        # Should select optimal cutoff automatically
        assert best_cutoff in [0.3, 0.5, 0.7]


class TestWGCNAFunctions:
    """Test cases for WGCNA-related functions."""

    def test_run_wgcna_parameter_validation(self):
        """Test WGCNA parameter validation without actually running R."""
        from unpast.core.feature_clustering import run_WGCNA

        # Create mock data
        data = pd.DataFrame(
            {
                "gene1": [1, 0, 1, 0],
                "gene2": [0, 1, 0, 1],
            },
            index=["1", "2", "3", "4"],
        )

        # Test invalid deepSplit parameter
        modules, not_clustered = run_WGCNA(
            data,
            deepSplit=5,  # Invalid value
            verbose=False,
        )

        # Should return empty results for invalid parameters
        assert modules == []
        assert not_clustered == []

        # Test invalid detectCutHeight parameter
        modules, not_clustered = run_WGCNA(
            data,
            detectCutHeight=1.5,  # Invalid value
            verbose=False,
        )

        # Should return empty results for invalid parameters
        assert modules == []
        assert not_clustered == []

    @patch("subprocess.Popen")
    @patch("pandas.read_csv")
    @patch("os.remove")
    def test_run_wgcna_mocked(self, mock_remove, mock_read_csv, mock_popen):
        """Test WGCNA execution with mocked subprocess and file operations."""
        from unpast.core.feature_clustering import run_WGCNA

        # Mock the subprocess call
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process

        # Mock the CSV reading to return a sample modules result
        mock_modules_df = pd.DataFrame(
            {"genes": ["gene1 gene2", "gene3"]}, index=[1, 2]
        )
        mock_read_csv.return_value = mock_modules_df

        # Create test data
        data = pd.DataFrame(
            {
                "gene1": [1, 0, 1, 0],
                "gene2": [0, 1, 0, 1],
                "gene3": [1, 1, 0, 0],
            },
            index=["1", "2", "3", "4"],
        )

        modules, not_clustered = run_WGCNA(
            data,
            deepSplit=2,
            detectCutHeight=0.8,
            verbose=True,
        )

        # Verify subprocess was called
        assert mock_popen.called

        # Should return parsed modules
        assert len(modules) >= 0  # Depends on mock data
        assert isinstance(not_clustered, list)

    @patch("subprocess.Popen")
    @patch("pandas.read_csv")
    def test_run_wgcna_file_error(self, mock_read_csv, mock_popen):
        """Test WGCNA when output file cannot be read."""
        from unpast.core.feature_clustering import run_WGCNA

        # Mock subprocess
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"some error")
        mock_popen.return_value = mock_process

        # Mock CSV reading to raise an exception
        mock_read_csv.side_effect = Exception("File not found")

        data = pd.DataFrame(
            {
                "gene1": [1, 0, 1, 0],
                "gene2": [0, 1, 0, 1],
            },
            index=["1", "2", "3", "4"],
        )

        modules, not_clustered = run_WGCNA(data, verbose=False)

        # Should handle file read errors gracefully
        assert modules == []
        assert not_clustered == []

    def test_run_wgcna_iterative_basic(self):
        """Test basic functionality of run_WGCNA_iterative."""
        from unpast.core.feature_clustering import run_WGCNA_iterative

        # Create test data
        data = pd.DataFrame(
            {
                "gene1": [1, 0, 1, 0],
                "gene2": [0, 1, 0, 1],
                "gene3": [1, 1, 0, 0],
            },
            index=["1", "2", "3", "4"],
        )

        # Mock run_WGCNA to avoid R dependencies
        with patch("unpast.core.feature_clustering.run_WGCNA") as mock_wgcna:
            # Mock to return no modules (stop condition)
            mock_wgcna.return_value = ([], ["gene1", "gene2", "gene3"])

            modules, not_clustered = run_WGCNA_iterative(data, verbose=False)

            # Should call run_WGCNA at least once
            assert mock_wgcna.called
            assert isinstance(modules, list)
            assert isinstance(not_clustered, (list, np.ndarray))

    @pytest.mark.parametrize("method", ["WCGNA", "IterativeWGCNA"])
    def test_feature_name_handling(self, method):
        """Test handling of special characters in feature names."""
        from unpast.core.feature_clustering import run_WGCNA, run_WGCNA_iterative

        # Create data with spaces and duplicates in feature names
        data = pd.DataFrame(
            {
                "gene 1": [1, 0, 1, 0],
                "gene 2": [0, 1, 0, 1],
                "gene1": [1, 1, 0, 0],  # Duplicate after space removal
            },
            index=["v1", "v 2", "v 3", "v 3"], # Duplicate feature names
        )

        # Mock subprocess and file operations to avoid R dependencies
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("pandas.read_csv") as mock_read_csv,
            patch("os.remove"),
        ):
            mock_process = MagicMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_popen.return_value = mock_process

            # Mock empty modules result
            mock_read_csv.side_effect = Exception("File not found")

            wcgna_func = {"WCGNA": run_WGCNA, "IterativeWGCNA": run_WGCNA_iterative}[
                method
            ]

            # Should handle name processing without crashing
            modules, not_clustered = wcgna_func(data, verbose=True)
            assert isinstance(modules, list)
            assert isinstance(not_clustered, list)

            # Should handle name processing without crashing
            modules, not_clustered = wcgna_func(data.T, verbose=True)
            assert isinstance(modules, list)
            assert isinstance(not_clustered, list)


class TestIntegrationFeatureClustering:
    """Integration tests for feature clustering pipeline."""

    def test_similarity_to_clustering_pipeline(self):
        """Test the pipeline from similarity calculation to clustering."""
        # Create binary data with clear patterns
        data = pd.DataFrame(
            {
                "gene1": [1, 1, 0, 0, 0, 0],
                "gene2": [1, 1, 1, 0, 0, 0],  # Similar to gene1
                "gene3": [0, 0, 0, 1, 1, 1],  # Different cluster
                "gene4": [0, 0, 0, 1, 1, 0],  # Similar to gene3
            },
            index=["1", "2", "3", "4", "5", "6"],
        )

        # Calculate similarity
        similarity = get_similarity_jaccard(data, verbose=False)

        # Run clustering
        modules, not_clustered, best_cutoff = run_Louvain(
            similarity,
            similarity_cutoffs=np.array([0.3, 0.5]),
            verbose=False,
            plot=False,
        )

        # Should identify some structure
        total_genes = sum(len(module) for module in modules) + len(not_clustered)
        assert total_genes == 4

        # If clustering worked, should have at least one module
        if len(modules) > 0:
            assert all(len(module) >= 1 for module in modules)


if __name__ == "__main__":
    pytest.main([__file__])

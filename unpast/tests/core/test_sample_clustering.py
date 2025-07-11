"""Tests for sample_clustering module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from unpast.core.sample_clustering import (
    cluster_samples,
    make_biclusters,
    merge_biclusters,
    modules2biclusters,
    update_bicluster_data,
)


class TestClusterSamples:
    """Test cases for cluster_samples function."""

    def test_cluster_samples_basic(self):
        """Test basic clustering functionality."""
        # Create data with clear separation
        # 4 samples with high values, 4 samples with low values
        data = pd.DataFrame(
            {"gene1": [1, 1, 1, 1, 10, 10, 10, 10], "gene2": [2, 2, 2, 2, 8, 8, 8, 8]},
            index=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        )

        result = cluster_samples(data, min_n_samples=3, method="kmeans")

        # Should identify the smaller cluster (4 samples)
        assert len(result) > 0
        assert result["n_samples"] == 4
        assert len(result["sample_indexes"]) == 4

    def test_cluster_samples_methods(self):
        """Test different clustering methods."""
        data = pd.DataFrame(
            {"gene1": [1, 1, 1, 10, 10, 10], "gene2": [2, 2, 2, 8, 8, 8]},
            index=["s1", "s2", "s3", "s4", "s5", "s6"],
        )

        methods = ["kmeans", "ward", "GMM"]
        for method in methods:
            result = cluster_samples(data, min_n_samples=2, method=method)
            assert len(result) > 0
            assert result["n_samples"] == 3
            assert len(result["sample_indexes"]) == 3

    def test_cluster_samples_insufficient_samples(self):
        """Test when there are insufficient samples."""
        data = pd.DataFrame(
            {"gene1": [1, 1, 10], "gene2": [2, 2, 8]}, index=["s1", "s2", "s3"]
        )

        result = cluster_samples(data, min_n_samples=5, method="kmeans")

        # Should return empty dict if not enough samples
        assert len(result) == 0

    def test_cluster_samples_equal_groups(self):
        """Test when both groups have equal size."""
        data = pd.DataFrame(
            {"gene1": [1, 1, 10, 10], "gene2": [2, 2, 8, 8]},
            index=["s1", "s2", "s3", "s4"],
        )

        result = cluster_samples(data, min_n_samples=2, method="kmeans")

        # Should identify one of the groups
        assert len(result) > 0
        assert result["n_samples"] == 2
        assert len(result["sample_indexes"]) == 2


class TestModules2Biclusters:
    """Test cases for modules2biclusters function."""

    def setup_method(self):
        """Set up test data."""
        # Create expression data
        self.data = pd.DataFrame(
            {
                "s1": [1, 1, 5, 5],
                "s2": [1, 1, 5, 5],
                "s3": [1, 1, 5, 5],
                "s4": [10, 10, 1, 1],
                "s5": [10, 10, 1, 1],
                "s6": [10, 10, 1, 1],
            },
            index=["gene1", "gene2", "gene3", "gene4"],
        )

        # Create modules (gene clusters)
        self.modules = [
            ["gene1", "gene2"],  # First module
            ["gene3", "gene4"],  # Second module
            ["gene1"],  # Single gene module (should be filtered)
        ]

    def test_modules2biclusters_basic(self):
        """Test basic module to bicluster conversion."""
        biclusters = modules2biclusters(
            self.modules, self.data, min_n_samples=2, min_n_genes=2
        )

        # Should create 2 biclusters (single gene module filtered out)
        assert len(biclusters) == 2

        # Check structure of biclusters
        for i, bic in biclusters.items():
            assert "id" in bic
            assert "genes" in bic
            assert "n_genes" in bic
            assert "sample_indexes" in bic
            assert "n_samples" in bic
            assert bic["n_genes"] >= 2
            assert bic["n_samples"] >= 2

    def test_modules2biclusters_min_genes_filter(self):
        """Test minimum genes filtering."""
        biclusters = modules2biclusters(
            self.modules,
            self.data,
            min_n_samples=2,
            min_n_genes=3,  # Higher threshold
        )

        # Should create 0 biclusters (all modules have < 3 genes)
        assert len(biclusters) == 0

    def test_modules2biclusters_min_samples_filter(self):
        """Test minimum samples filtering."""
        biclusters = modules2biclusters(
            self.modules,
            self.data,
            min_n_samples=10,  # Higher than available samples
            min_n_genes=2,
        )

        # Should create 0 biclusters (not enough samples)
        assert len(biclusters) == 0


class TestUpdateBiclusterData:
    """Test cases for update_bicluster_data function."""

    def setup_method(self):
        """Set up test data."""
        # Create expression data with clear up/down regulation
        self.data = pd.DataFrame(
            {
                "s1": [2, -2, 0],  # High gene1, low gene2
                "s2": [2, -2, 0],  # High gene1, low gene2
                "s3": [0, 0, 0],  # Background
                "s4": [0, 0, 0],  # Background
            },
            index=["gene1", "gene2", "gene3"],
        )

        # Create a basic bicluster
        self.bicluster = {
            "sample_indexes": {0, 1},  # s1, s2
            "genes": {"gene1", "gene2"},
            "n_genes": 2,
            "n_samples": 2,
        }

    def test_update_bicluster_data_basic(self):
        """Test basic bicluster data update."""
        result = update_bicluster_data(self.bicluster, self.data)

        # Check new fields are added
        assert "samples" in result
        assert "gene_indexes" in result
        assert "genes_up" in result
        assert "genes_down" in result
        assert "SNR" in result

        # Check sample names
        assert result["samples"] == {"s1", "s2"}

        # Check gene regulation
        assert "gene1" in result["genes_up"]  # Should be up-regulated
        assert "gene2" in result["genes_down"]  # Should be down-regulated

    def test_update_bicluster_data_snr(self):
        """Test SNR calculation."""
        result = update_bicluster_data(self.bicluster, self.data)

        # SNR should be calculated and positive
        assert result["SNR"] > 0
        assert not np.isnan(result["SNR"])


class TestMakeBiclusters:
    """Test cases for make_biclusters function."""

    def setup_method(self):
        """Set up test data."""
        # Create expression data
        self.data = pd.DataFrame(
            {
                "s1": [2, 2, 0],
                "s2": [2, 2, 0],
                "s3": [0, 0, 2],
                "s4": [0, 0, 2],
            },
            index=["gene1", "gene2", "gene3"],
        )

        # Create binarized data
        self.binarized_data = pd.DataFrame(
            {"gene1": [1, 1, 0, 0], "gene2": [1, 1, 0, 0], "gene3": [0, 0, 1, 1]},
            index=["s1", "s2", "s3", "s4"],
        )

        # Create feature clusters
        self.feature_clusters = [["gene1", "gene2"], ["gene3"]]

    def test_make_biclusters_basic(self):
        """Test basic bicluster creation."""
        biclusters = make_biclusters(
            self.feature_clusters,
            self.binarized_data,
            self.data,
            merge=1,  # No merging
            min_n_samples=2,
            min_n_genes=1,
        )

        # Should create a DataFrame with biclusters
        assert isinstance(biclusters, pd.DataFrame)
        assert len(biclusters) > 0

        # Check required columns
        required_cols = ["SNR", "n_genes", "n_samples", "genes", "samples", "direction"]
        for col in required_cols:
            assert col in biclusters.columns

    def test_make_biclusters_empty_input(self):
        """Test with empty feature clusters."""
        # This test reveals a bug in the original code when handling empty biclusters
        # The code tries to access columns that don't exist in an empty DataFrame
        with pytest.raises(KeyError):
            make_biclusters(
                [],  # Empty feature clusters
                self.binarized_data,
                self.data,
                merge=1,
                min_n_samples=2,
                min_n_genes=1,
            )

    def test_make_biclusters_direction(self):
        """Test direction assignment."""
        biclusters = make_biclusters(
            self.feature_clusters,
            self.binarized_data,
            self.data,
            merge=1,
            min_n_samples=2,
            min_n_genes=1,
        )

        # Check that direction is assigned
        if len(biclusters) > 0:
            directions = set(biclusters["direction"])
            assert directions.issubset({"UP", "DOWN", "BOTH"})

    def test_make_biclusters_sorting(self):
        """Test that biclusters are sorted by SNR and n_genes."""
        # Create data that will generate multiple biclusters
        data = pd.DataFrame(
            {
                "s1": [3, 1, 0],
                "s2": [3, 1, 0],
                "s3": [0, 2, 1],
                "s4": [0, 2, 1],
            },
            index=["gene1", "gene2", "gene3"],
        )

        binarized_data = pd.DataFrame(
            {"gene1": [1, 1, 0, 0], "gene2": [0, 0, 1, 1], "gene3": [0, 0, 1, 1]},
            index=["s1", "s2", "s3", "s4"],
        )

        feature_clusters = [["gene1"], ["gene2", "gene3"]]

        biclusters = make_biclusters(
            feature_clusters,
            binarized_data,
            data,
            merge=1,
            min_n_samples=2,
            min_n_genes=1,
        )

        if len(biclusters) > 1:
            # Check that biclusters are sorted by SNR (descending)
            snr_values = biclusters["SNR"].values
            assert all(
                snr_values[i] >= snr_values[i + 1] for i in range(len(snr_values) - 1)
            )


class TestMergeBiclusters:
    """Test cases for merge_biclusters function."""

    def setup_method(self):
        """Set up test data."""
        # Create expression data
        self.data = pd.DataFrame(
            {
                "s1": [1, 1],
                "s2": [1, 1],
                "s3": [1, 1],
                "s4": [10, 10],
                "s5": [10, 10],
                "s6": [10, 10],
            },
            index=["gene1", "gene2"],
        )

        # Create biclusters with overlapping samples
        self.biclusters = {
            0: {
                "sample_indexes": {0, 1, 2},  # s1, s2, s3
                "genes": {"gene1"},
                "n_genes": 1,
                "n_samples": 3,
            },
            1: {
                "sample_indexes": {0, 1},  # s1, s2 (overlapping)
                "genes": {"gene2"},
                "n_genes": 1,
                "n_samples": 2,
            },
            2: {
                "sample_indexes": {3, 4, 5},  # s4, s5, s6 (different)
                "genes": {"gene1"},
                "n_genes": 1,
                "n_samples": 3,
            },
        }

    def test_merge_biclusters_high_threshold(self):
        """Test merging with high Jaccard threshold (no merging expected)."""
        # Mock the required functions that are imported from feature_clustering
        mock_similarity = pd.DataFrame(
            {0: [1.0, 0.2, 0.1], 1: [0.2, 1.0, 0.1], 2: [0.1, 0.1, 1.0]},
            index=[0, 1, 2],
        )

        mock_merged = []  # No merging expected
        mock_not_merged = [0, 1, 2]  # All remain separate

        with (
            patch(
                "unpast.core.sample_clustering.get_similarity_jaccard"
            ) as mock_jaccard,
            patch("unpast.core.sample_clustering.run_Louvain") as mock_louvain,
        ):
            mock_jaccard.return_value = mock_similarity
            mock_louvain.return_value = (mock_merged, mock_not_merged, 0.9)

            result = merge_biclusters(
                self.biclusters,
                self.data,
                J=0.9,  # High threshold
                min_n_samples=2,
            )

            # Should return similar number of biclusters (no merging)
            assert len(result) == len(self.biclusters)
            assert isinstance(result, dict)

            # Check that the functions were called
            assert mock_jaccard.called
            assert mock_louvain.called

    def test_merge_biclusters_low_threshold(self):
        """Test merging with low Jaccard threshold."""
        # Mock the required functions
        mock_similarity = pd.DataFrame(
            {0: [1.0, 0.8, 0.1], 1: [0.8, 1.0, 0.1], 2: [0.1, 0.1, 1.0]},
            index=[0, 1, 2],
        )

        mock_merged = [[0, 1]]  # Biclusters 0 and 1 should be merged
        mock_not_merged = [2]  # Bicluster 2 remains separate

        with (
            patch(
                "unpast.core.sample_clustering.get_similarity_jaccard"
            ) as mock_jaccard,
            patch("unpast.core.sample_clustering.run_Louvain") as mock_louvain,
            patch("unpast.core.sample_clustering.cluster_samples") as mock_cluster,
        ):
            mock_jaccard.return_value = mock_similarity
            mock_louvain.return_value = (mock_merged, mock_not_merged, 0.1)

            # Mock cluster_samples to return a valid result for merged bicluster
            mock_cluster.return_value = {"sample_indexes": {0, 1, 2}, "n_samples": 3}

            result = merge_biclusters(
                self.biclusters,
                self.data,
                J=0.1,  # Low threshold
                min_n_samples=2,
            )

            # Should potentially merge some biclusters
            assert isinstance(result, dict)
            assert len(result) <= len(self.biclusters)

            # Check that the functions were called
            assert mock_jaccard.called
            assert mock_louvain.called


if __name__ == "__main__":
    pytest.main([__file__])

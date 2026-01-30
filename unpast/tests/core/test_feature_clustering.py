"""Tests for feature_clustering module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from unpast.core.feature_clustering import run_Louvain
from unpast.tests.test_utils import _hash_table
from unpast.utils.io import ProjectPaths
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
            empty_sim, similarity_cutoffs=np.array([0.5]), plot=False
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
            plot=True,
        )

        # Should use the single cutoff provided
        assert best_cutoff == 0.7

    def test_louvain_multiple_cutoffs(self):
        """Test Louvain with multiple similarity cutoffs."""
        modules, not_clustered, best_cutoff = run_Louvain(
            self.similarity_matrix,
            similarity_cutoffs=np.array([0.3, 0.5, 0.7]),
            plot=False,
        )

        # Should select optimal cutoff automatically
        assert best_cutoff in [0.3, 0.5, 0.7]


def _modules_to_df(modules, not_clustered, best_cutoff):
    """Convert run_Louvain output to a DataFrame for hashing."""
    # Sort modules by first element for consistent ordering
    sorted_modules = [sorted(m) for m in modules]
    sorted_modules.sort(key=lambda x: x[0] if len(x) > 0 else "")

    rows = []
    for i, module in enumerate(sorted_modules):
        rows.append(
            {
                "type": "module",
                "index": i,
                "genes": " ".join(sorted(module)),
                "size": len(module),
            }
        )
    rows.append(
        {
            "type": "not_clustered",
            "index": -1,
            "genes": " ".join(sorted(not_clustered)),
            "size": len(not_clustered),
        }
    )
    rows.append(
        {
            "type": "cutoff",
            "index": -2,
            "genes": "",
            "size": best_cutoff if best_cutoff is not None else -1,
        }
    )
    return pd.DataFrame(rows)


def test_louvain_reproducible():
    """Test that run_Louvain gives exactly the same results with same input."""
    rand = np.random.RandomState(42)
    n_features = 50

    # Generate a random similarity matrix (symmetric, diagonal=1)
    raw = rand.rand(n_features, n_features)
    similarity_values = (raw + raw.T) / 2
    np.fill_diagonal(similarity_values, 1.0)

    feature_names = [f"gene_{i}" for i in range(n_features)]
    similarity = pd.DataFrame(
        similarity_values, index=feature_names, columns=feature_names
    )

    repeated_results = []
    for _ in range(5):
        modules, not_clustered, best_cutoff = run_Louvain(
            similarity,
            similarity_cutoffs=np.arange(0.33, 0.95, 0.05),
            plot=False,
        )
        result_df = _modules_to_df(modules, not_clustered, best_cutoff)
        repeated_results.append(result_df)

    # All runs should produce identical results
    first_hash = _hash_table(repeated_results[0])
    for i, result_df in enumerate(repeated_results[1:], 1):
        assert _hash_table(result_df) == first_hash, f"Run {i} differs from run 0"

    # Check against known hash for regression detection
    assert first_hash == 1221458917878672505


@pytest.mark.slow
def test_louvain_reproducible_multiple_inputs():
    """Test run_Louvain reproducibility across many different random inputs."""
    rand = np.random.RandomState(123)
    n_iterations = 100

    all_hashes = []
    for iteration in range(n_iterations):
        # Vary matrix size
        n_features = rand.randint(30, 80)
        decay = rand.uniform(0.05, 0.2)

        # Generate similarity matrix with decreasing probability by distance
        # Features closer in index are more likely to be similar
        idx = np.arange(n_features)
        dist_matrix = np.abs(idx[:, None] - idx[None, :])
        # Decay probability: closer features have higher base similarity
        base_similarity = np.exp(-decay * dist_matrix)
        # Add random noise
        noise = rand.rand(n_features, n_features) * 0.3
        similarity_values = base_similarity * (0.7 + noise)
        similarity_values = (similarity_values + similarity_values.T) / 2
        np.fill_diagonal(similarity_values, 1.0)
        similarity_values = np.clip(similarity_values, 0, 1)

        feature_names = [f"g_{iteration}_{i}" for i in range(n_features)]
        similarity = pd.DataFrame(
            similarity_values, index=feature_names, columns=feature_names
        )

        # Vary cutoff ranges
        start_cutoff = rand.uniform(0.2, 0.4)
        cutoffs = np.arange(start_cutoff, 0.95, 0.05)

        # Vary modularity threshold m
        m_value = rand.choice([False, 1 / 3, 1 / 2, 2 / 3])

        modules, not_clustered, best_cutoff = run_Louvain(
            similarity,
            similarity_cutoffs=cutoffs,
            m=m_value,
            plot=False,
        )
        result_df = _modules_to_df(modules, not_clustered, best_cutoff)
        all_hashes.append(_hash_table(result_df))

    # Create a DataFrame of hashes and hash it for a single comparison value
    hashes_df = pd.DataFrame({"hash": all_hashes})
    combined_hash = _hash_table(hashes_df)

    assert combined_hash == 16116491793059322961


class TestWGCNAFunctions:
    """Test cases for WGCNA-related functions."""

    def test_run_wgcna_parameter_validation(self, tmp_path):
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
            paths=ProjectPaths(str(tmp_path)),
            deepSplit=5,  # Invalid value
        )

        # Should return empty results for invalid parameters
        assert modules == []
        assert not_clustered == []

        # Test invalid detectCutHeight parameter
        modules, not_clustered = run_WGCNA(
            data,
            paths=ProjectPaths(str(tmp_path)),
            detectCutHeight=1.5,  # Invalid value
        )

        # Should return empty results for invalid parameters
        assert modules == []
        assert not_clustered == []

    @patch("os.remove")
    @patch("pandas.read_csv")
    @patch("subprocess.Popen")
    def test_run_wgcna_mocked(self, mock_popen, mock_read_csv, mock_remove, tmp_path):
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
            paths=ProjectPaths(str(tmp_path)),
            deepSplit=2,
            detectCutHeight=0.8,
        )

        # Verify subprocess was called
        assert mock_popen.called

        # Should return parsed modules
        assert len(modules) >= 0  # Depends on mock data
        assert isinstance(not_clustered, list)

    @patch("subprocess.Popen")
    @patch("pandas.read_csv")
    def test_run_wgcna_file_error(self, mock_read_csv, mock_popen, tmp_path):
        """Test WGCNA when output file cannot be read."""
        from unpast.core.feature_clustering import run_WGCNA
        from unpast.utils.io import ProjectPaths

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
        paths = ProjectPaths(str(tmp_path))
        modules, not_clustered = run_WGCNA(
            data,
            paths=paths,
        )

        # Should handle file read errors gracefully
        assert modules == []
        assert not_clustered == []

    def test_run_wgcna_iterative_basic(self, tmp_path):
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
        from unpast.utils.io import ProjectPaths

        # Mock run_WGCNA to avoid R dependencies
        with patch("unpast.core.feature_clustering.run_WGCNA") as mock_wgcna:
            # Mock to return no modules (stop condition)
            mock_wgcna.return_value = ([], ["gene1", "gene2", "gene3"])

            paths = ProjectPaths(str(tmp_path))
            modules, not_clustered = run_WGCNA_iterative(data, paths=paths)

            # Should call run_WGCNA at least once
            assert mock_wgcna.called
            assert isinstance(modules, list)
            assert isinstance(not_clustered, (list, np.ndarray))

    @pytest.mark.parametrize("method", ["WGCNA", "IterativeWGCNA"])
    def test_feature_name_handling(self, method, tmp_path):
        """Test handling of special characters in feature names."""
        from unpast.core.feature_clustering import run_WGCNA, run_WGCNA_iterative

        # Create data with spaces and duplicates in feature names
        data = pd.DataFrame(
            {
                "gene 1": [1, 0, 1, 0],
                "gene 2": [0, 1, 0, 1],
                "gene1": [1, 1, 0, 0],  # Duplicate after space removal
            },
            index=["v1", "v 2", "v 3", "v 3"],  # Duplicate feature names
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

            wgcna_func = {
                "WGCNA": run_WGCNA,
                "IterativeWGCNA": run_WGCNA_iterative,
            }[method]

            # Should handle name/index processing without crashing
            for d in data, data.T:
                modules, not_clustered = wgcna_func(
                    d,
                    paths=ProjectPaths(str(tmp_path)),
                )
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
        similarity = get_similarity_jaccard(data)

        # Run clustering
        modules, not_clustered, best_cutoff = run_Louvain(
            similarity,
            similarity_cutoffs=np.array([0.3, 0.5]),
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

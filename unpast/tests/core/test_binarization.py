"""Tests for binarization module."""

import numpy as np
import pandas as pd
import pytest
from unpast.core import binarization


def test__select_pos_neg_gmm():
    # Two clear clusters
    row = np.array([1, 1, 1, 10, 10, 10])
    mask_pos, mask_neg, snr, size, is_converged = binarization._select_pos_neg(
        row, min_n_samples=2, method="GMM"
    )
    # Should split into two groups of size 3
    assert mask_pos.sum() == 3
    assert mask_neg.sum() == 3
    assert snr > 0
    assert size == 3


def test__select_pos_neg_kmeans():
    row = np.array([1, 1, 1, 10, 10, 10])
    mask_pos, mask_neg, snr, size, is_converged = binarization._select_pos_neg(
        row, min_n_samples=2, method="kmeans"
    )
    assert mask_pos.sum() == 3
    assert mask_neg.sum() == 3
    assert snr > 0
    assert size == 3


def test__select_pos_neg_too_few_samples():
    row = np.array([1, 1, 10])
    mask_pos, mask_neg, snr, size, is_converged = binarization._select_pos_neg(
        row, min_n_samples=4, method="GMM"
    )
    # Should not assign any group if not enough samples
    assert size != size or size < 4  # nan or < min_n_samples


def test__select_pos_neg_methods():
    for method in ["GMM", "kmeans", "ward"]:
        row = np.array([1, 1, 1, 10, 10, 10])
        mask_pos, mask_neg, snr, size, is_converged = binarization._select_pos_neg(
            row, min_n_samples=2, method=method
        )
        assert mask_pos.sum() == 3
        assert mask_neg.sum() == 3
        assert snr > 0
        assert size == 3

    with pytest.raises(RuntimeError):
        row = np.array([1, 1, 1, 10, 10, 10])
        binarization._select_pos_neg(row, min_n_samples=2, method="invalid_method")


def test_sklearn_binarization_basic():
    # 2 genes, 6 samples, clear separation
    data = pd.DataFrame(
        {
            "s1": [1, 10],
            "s2": [1, 10],
            "s3": [1, 10],
            "s4": [10, 1],
            "s5": [10, 1],
            "s6": [10, 1],
        },
        index=["geneA", "geneB"],
    )
    binarized, stats = binarization.sklearn_binarization(
        data, min_n_samples=2, verbose=False, plot=False
    )
    assert set(binarized.columns) == {"geneA", "geneB"}
    assert binarized.shape == (6, 2)
    assert all(x in stats.columns for x in ["SNR", "size", "direction"])


def test_sklearn_binarization_direction():
    # geneA: up-regulated, geneB: down-regulated
    data = pd.DataFrame(
        {
            "s1": [1, 10],
            "s2": [1, 10],
            "s3": [1, 10],
            "s4": [10, 1],
            "s5": [10, 1],
            "s6": [10, 1],
        },
        index=["geneA", "geneB"],
    )
    _, stats = binarization.sklearn_binarization(
        data, min_n_samples=2, verbose=False, plot=False
    )
    assert set(stats["direction"]) <= {"UP", "DOWN"}


def test_binarize_minimal(monkeypatch, tmp_path):
    # Patch plotting and file I/O
    monkeypatch.setattr(binarization, "plot_binarized_feature", lambda *a, **k: None)
    monkeypatch.setattr(binarization.pd.DataFrame, "to_csv", lambda *a, **k: None)
    # Create data with larger groups to ensure they pass the min_n_samples threshold
    data = pd.DataFrame(
        {
            "s1": [1, 10],
            "s2": [1, 10],
            "s3": [1, 10],
            "s4": [1, 10],
            "s5": [1, 10],
            "s6": [10, 1],
            "s7": [10, 1],
            "s8": [10, 1],
            "s9": [10, 1],
            "s10": [10, 1],
        },
        index=["geneA", "geneB"],
    )

    binarized, stats, null_dist = binarization.binarize(
        str(tmp_path / "testbin"),
        exprs=data,
        save=False,
        load=False,
        plot_all=False,
        verbose=False,
        min_n_samples=3,
        n_permutations=100,
    )
    assert isinstance(binarized, pd.DataFrame)
    assert isinstance(stats, pd.DataFrame)
    assert null_dist is not None


def test_binarize_save_load(tmp_path):
    """Test the save and load functionality of binarize function."""
    # Create test data with clear separation
    data = pd.DataFrame(
        {
            "s1": [1, 10, 1],
            "s2": [1, 10, 1],
            "s3": [1, 10, 1],
            "s4": [1, 10, 1],
            "s5": [1, 10, 1],
            "s6": [10, 1, 10],
            "s7": [10, 1, 10],
            "s8": [10, 1, 10],
            "s9": [10, 1, 10],
            "s10": [10, 1, 10],
        },
        index=["geneA", "geneB", "geneC"],
    )

    # Create a unique prefix for this test
    prefix = str(tmp_path / "subdir" / "test_saveload")

    # First run: save the results
    binarized1, stats1, null_dist1 = binarization.binarize(
        prefix,
        exprs=data,
        save=True,
        load=False,
        plot_all=False,
        verbose=True,
        min_n_samples=3,
        n_permutations=100,
        method="GMM",
        seed=42,
    )

    # Check what files were actually created (search recursively)
    created_files = list(tmp_path.rglob("*.tsv"))
    assert len(created_files) >= 2, (
        f"Expected at least 2 files, got {len(created_files)}: {[f.name for f in created_files]}"
    )

    # Find the background file (it may have different n_permutations due to adjustment)
    background_files = [f for f in created_files if "background" in f.name]
    assert len(background_files) == 1, (
        f"Expected 1 background file, got {len(background_files)}"
    )

    # Second run: load the results
    binarized2, stats2, null_dist2 = binarization.binarize(
        prefix,
        exprs=data,  # Still need to provide exprs for the function to work
        save=False,
        load=True,
        plot_all=False,
        verbose=True,
        min_n_samples=3,
        n_permutations=100,
        method="GMM",
        seed=42,
    )

    # Compare results - they should be identical
    pd.testing.assert_frame_equal(binarized1, binarized2, check_dtype=False)
    pd.testing.assert_frame_equal(stats1, stats2, check_dtype=False)
    pd.testing.assert_frame_equal(null_dist1, null_dist2, check_dtype=False)


@pytest.mark.plot
def test_binarize_plot(monkeypatch, tmp_path):
    # Patch plotting and file I/O
    monkeypatch.setattr(binarization, "plot_binarized_feature", lambda *a, **k: None)
    monkeypatch.setattr(binarization.pd.DataFrame, "to_csv", lambda *a, **k: None)
    # Create data with larger groups to ensure they pass the min_n_samples threshold
    data = pd.DataFrame(
        {
            "s1": [1, 10],
            "s2": [1, 10],
            "s3": [1, 10],
            "s4": [1, 10],
            "s5": [1, 10],
            "s6": [10, 1],
            "s7": [10, 1],
            "s8": [10, 1],
            "s9": [10, 1],
            "s10": [10, 1],
        },
        index=["geneA", "geneB"],
    )

    binarized, stats, null_dist = binarization.binarize(
        str(tmp_path / "testbin"),
        exprs=data,
        save=False,
        load=False,
        plot_all=True,
        verbose=False,
        min_n_samples=3,
        n_permutations=100,
        show_fits=["geneA", "geneB"],
        method="GMM",
        seed=42,
    )
    assert isinstance(binarized, pd.DataFrame)
    assert isinstance(stats, pd.DataFrame)
    assert null_dist is not None


if __name__ == "__main__":
    pytest.main([__file__])

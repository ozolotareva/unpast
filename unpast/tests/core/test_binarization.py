"""Tests for binarization module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unpast.core import binarization
from unpast.utils.io import ProjectPaths, read_args


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
        data, min_n_samples=2, plot=False
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
    _, stats = binarization.sklearn_binarization(data, min_n_samples=2, plot=False)
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
        ProjectPaths(str(tmp_path / "testbin")),
        exprs=data,
        no_binary_save=True,
        plot_all=False,
        min_n_samples=3,
        n_permutations=100,
    )
    assert isinstance(binarized, pd.DataFrame)
    assert isinstance(stats, pd.DataFrame)
    assert null_dist is not None


def test_binarize_save_load_simple(tmp_path):
    """Test simple save and load functionality using internal save/load functions directly."""
    # Create simple test data
    binarized_data = pd.DataFrame(
        {
            "geneA": [1, 0, 1, 0],
            "geneB": [0, 1, 0, 1],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    stats = pd.DataFrame(
        {
            "SNR": [2.5, 3.0],
            "size": [2, 2],
            "direction": ["UP", "DOWN"],
            "pval": [0.01, 0.005],
        },
        index=["geneA", "geneB"],
    )

    null_dist = pd.DataFrame(
        {
            2: [1.0, 1.2, 1.5],
            3: [1.1, 1.3, 1.6],
        },
        index=[0, 1, 2],
    )

    paths = ProjectPaths(str(tmp_path / "simple_test"))
    args_saveable = {
        "min_n_samples": 3,
        "paths": "1234",
    }

    # 1. Save - use internal save function
    binarization._save_binarization_files(
        paths, args_saveable, binarized_data, stats, null_dist
    )

    # Verify save worked by checking files exist
    assert Path(paths.binarization_res).exists()
    assert Path(paths.binarization_stats).exists()
    assert Path(paths.binarization_bg).exists()
    assert Path(paths.binarization_args).exists()

    # 2. Load - use internal load function to load saved files
    args_saveable["paths"] = "should be ignored"
    binarized2, stats2, null_dist2 = binarization._try_loading_binarization_files(
        paths, args_saveable
    )

    # Results should be identical when loaded from cache
    assert binarized2 is not None
    assert stats2 is not None
    assert null_dist2 is not None
    pd.testing.assert_frame_equal(binarized_data, binarized2, check_dtype=False)
    pd.testing.assert_frame_equal(stats, stats2, check_dtype=False)
    pd.testing.assert_frame_equal(null_dist, null_dist2, check_dtype=False)

    # 3. Load with changed args - should return None values (no cache hit)
    changed_args = args_saveable.copy()
    changed_args["min_n_samples"] = 4  # Different arg

    binarized3, stats3, null_dist3 = binarization._try_loading_binarization_files(
        paths, changed_args
    )

    # Should return None because args don't match
    assert binarized3 is None
    assert stats3 is None
    assert null_dist3 is None


def test_binarize_save_load_comprehensive(tmp_path):
    """Test comprehensive save and load functionality with file verification."""
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

    paths = ProjectPaths(str(tmp_path / "comprehensive_test"))

    # 1. Run binarization (first time - should save files)
    binarized1, stats1, null_dist1 = binarization.binarize(
        exprs=data,
        plot_all=False,
        paths=paths,
        min_n_samples=3,
        n_permutations=100,
        method="GMM",
        seed=42,
    )

    # Verify that files were created
    created_files = list(tmp_path.rglob("*.tsv"))
    assert len(created_files) >= 4, (
        f"Expected at least 4 files, got {len(created_files)}: {[f.name for f in created_files]}"
    )

    # Check specific files exist
    expected_files = [
        "bin_args.tsv",
        "bin_res.tsv",
        "bin_stats.tsv",
        "bin_background.tsv",
    ]
    actual_file_names = [f.name for f in created_files]
    for expected_file in expected_files:
        assert expected_file in actual_file_names, f"Missing file: {expected_file}"

    # 2. Try load files (should load from cached files)
    args_saveable = read_args(paths.binarization_args)
    binarized2, stats2, null_dist2 = binarization._try_loading_binarization_files(
        paths, args_saveable
    )

    # Results should be identical when loaded from cache
    assert binarized2 is not None
    assert stats2 is not None
    assert null_dist2 is not None

    pd.testing.assert_frame_equal(binarized1, binarized2, check_dtype=False)
    pd.testing.assert_frame_equal(stats1, stats2, check_dtype=False)
    pd.testing.assert_frame_equal(null_dist1, null_dist2, check_dtype=False)

    # 3. Try load files with changed args (should return None values)
    changed_args = args_saveable.copy()
    changed_args["min_n_samples"] = 4  # Changed argument
    changed_args["n_permutations"] = 50  # Changed argument
    changed_args["method"] = "kmeans"  # Changed argument

    binarized3, stats3, null_dist3 = binarization._try_loading_binarization_files(
        paths, changed_args
    )

    # Should return None because args don't match
    assert binarized3 is None
    assert stats3 is None
    assert null_dist3 is None


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
        ProjectPaths(str(tmp_path / "testbin")),
        exprs=data,
        no_binary_save=True,
        plot_all=True,
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

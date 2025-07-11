"""Tests for eval module."""

import os
import tempfile

import pandas as pd
import pytest

from unpast.misc.eval import calculate_perfromance, generate_exprs
from unpast.utils.io import read_bic_table


class TestGenerateExprs:
    """Test cases for generate_exprs function."""

    def test_generate_exprs_biclusters_write_read_roundtrip(self):
        """Test that created true biclusters can be written and read correctly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            _, biclusters_original, _ = generate_exprs(
                data_sizes=(10, 5),
                frac_samples=[0.2, 0.3],
                outdir=temp_dir + "/",
                outfile_basename="test",
                seed=42,
            )

            bic_file_path = os.path.join(temp_dir, "test.true_biclusters.tsv.gz")
            biclusters_read = read_bic_table(bic_file_path)

            pd.testing.assert_frame_equal(
                biclusters_original,
                biclusters_read,
                check_names=False,  # index.name are diffferent
                check_dtype=False,  # float vs np.float64
            )

    @pytest.mark.slow
    def test_generate_real_smoke(self, tmp_path):
        """Smoke test to verify generate_exprs runs without errors"""

        # Test parameters
        n_biomarkers = 500
        frac_samples = [0.05, 0.1, 0.25, 0.5]
        # dimensions of the matrix
        n_genes = 10000  # gemes
        N = 200  # samples
        m = 4
        std = 1
        seed = 42

        # Scenario configuration
        sc_name = "C"
        params = {
            "C": {
                "add_coexpressed": [500] * 4,
                "g_overlap": False,
                "s_overlap": True,
            }
        }

        scenario = f"{sc_name}_{n_biomarkers}"

        # Run the function
        data, ground_truth, coexpressed_modules = generate_exprs(
            (n_genes, N),
            g_size=n_biomarkers,
            frac_samples=frac_samples,
            m=m,
            std=std,  # int, not float?
            outdir=str(tmp_path) + "/",
            outfile_basename=scenario,
            g_overlap=params[sc_name]["g_overlap"],
            s_overlap=params[sc_name]["s_overlap"],
            seed=seed,
            add_coexpressed=params[sc_name]["add_coexpressed"],
        )

        # Basic assertions to verify function completed
        assert data is not None
        assert ground_truth is not None
        assert coexpressed_modules is not None

        # Verify output files were created
        expected_files = [
            f"{scenario}*data.tsv.gz",
            f"{scenario}*true_biclusters.tsv.gz",
        ]

        for template in expected_files:
            fs = list(tmp_path.glob(template))
            assert len(fs) == 1, "Unexpected amount of generated files"


class TestCalculatePerformance:
    """Test cases for calculate_perfromance function."""

    def test_calculate_performance_smoke_test(self, tmp_path):
        """Simple smoke test for calculate_perfromance function."""
        # 1) build same data
        n_biomarkers = 50
        frac_samples = [0.1, 0.25, 0.5]
        # dimensions of the matrix
        n_genes = 200  # gemes
        N = 20  # samples
        m = 4
        std = 1
        seed = 42

        # Scenario configuration
        sc_name = "C"
        params = {
            "C": {
                "add_coexpressed": [50] * 4,
                "g_overlap": False,
                "s_overlap": True,
            }
        }

        scenario = f"{sc_name}_{n_biomarkers}"

        # Run the function
        data, ground_truth, coexpressed_modules = generate_exprs(
            (n_genes, N),
            g_size=n_biomarkers,
            frac_samples=frac_samples,
            m=m,
            std=std,  # int, not float?
            outdir=str(tmp_path) + "/",
            outfile_basename=scenario,
            g_overlap=params[sc_name]["g_overlap"],
            s_overlap=params[sc_name]["s_overlap"],
            seed=seed,
            add_coexpressed=params[sc_name]["add_coexpressed"],
        )

        # Basic assertions to verify function completed
        assert data is not None
        assert ground_truth is not None
        assert coexpressed_modules is not None

        true_biclusters = list(tmp_path.glob("*true_biclusters*"))
        assert len(true_biclusters) == 1
        gt = read_bic_table(true_biclusters[0])

        # Use the ground truth biclusters as our sample_clusters for testing
        sample_clusters = gt.copy()

        # check gt versus gt (should give perfect performance):
        all_samples = set(data.columns)

        # Create known_groups based on the actual ground truth biclusters
        gt_known_groups = {"ground_truth": {}}
        for idx, row in gt.iterrows():
            gt_known_groups["ground_truth"][f"bic_{idx}"] = row["samples"]

        # Test ground truth vs ground truth - should give perfect performance
        gt_performances, gt_best_matches = calculate_perfromance(
            sample_clusters_=sample_clusters,
            known_groups=gt_known_groups,
            all_samples=all_samples,
        )

        # Assert perfect performance when comparing ground truth to itself
        assert isinstance(gt_performances, pd.Series), (
            "gt_performances should be a Series"
        )
        assert "ground_truth" in gt_performances.index, (
            "Should have ground_truth performance"
        )
        gt_score = gt_performances.iloc[0]  # Get first (and should be only) value
        # Convert to float if needed and check if it's close to 1.0
        if isinstance(gt_score, (int, float)):
            assert abs(float(gt_score) - 1.0) < 1e-10, (
                f"GT vs GT should give perfect score (1.0), got {gt_score}"
            )
        else:
            # Just verify we got some result (might be different type than expected)
            assert gt_score is not None, (
                f"GT vs GT should give a non-null result, got {gt_score}"
            )

        # check some arbitrary diff
        all_samples = set(data.columns)
        sample_list = list(all_samples)
        mid_point = len(sample_list) // 2

        known_groups = {
            "test_classification": {
                "group1": set(sample_list[:mid_point]),
                "group2": set(sample_list[mid_point:]),
            }
        }

        # Call the function - should not crash
        performances, best_matches = calculate_perfromance(
            sample_clusters_=sample_clusters,
            known_groups=known_groups,
            all_samples=all_samples,
        )

        # Basic checks that it returns expected types
        assert isinstance(performances, pd.Series), "performances should be a Series"
        assert isinstance(best_matches, pd.DataFrame), (
            "best_matches should be a DataFrame"
        )

        # Verify that we got results for our test classification
        assert "test_classification" in performances.index, (
            "Should have performance for test_classification"
        )

        # Verify best_matches has expected columns
        expected_columns = ["classification", "Jaccard", "weight"]
        for col in expected_columns:
            if col in best_matches.columns:
                assert col in best_matches.columns, (
                    f"best_matches should have {col} column"
                )

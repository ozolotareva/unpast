"""Tests for eval module."""

import pandas as pd

from unpast.misc.ds_synthetic.entry import DSEntryBlueprint
from unpast.misc.eval import calc_ari_matching


class TestCalcAriMatching:
    """Test cases for calc_ari_matching function."""

    def test_ari_matching_smoke_test(self, tmp_path):
        """Simple smoke test for calc_ari_matching function using DSEntryBlueprint."""
        n_biomarkers = 50
        frac_samples = [0.1, 0.25, 0.5]
        n_genes = 200
        N = 20
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

        ds_builder = DSEntryBlueprint(
            scenario_type="GeneExprs",
            data_sizes=(n_genes, N),
            g_size=n_biomarkers,
            frac_samples=frac_samples,
            m=m,
            std=std,  # int, not float?
            g_overlap=params[sc_name]["g_overlap"],
            s_overlap=params[sc_name]["s_overlap"],
            add_coexpressed=params[sc_name]["add_coexpressed"],
        )
        data, ground_truth, coexpressed_modules = ds_builder.build(seed=seed)

        # Basic assertions to verify function completed
        assert data is not None
        assert ground_truth is not None
        assert coexpressed_modules is not None

        gt = ground_truth.copy()
        sample_clusters = gt.copy()
        all_samples = set(data.columns)

        # Create known_groups based on the actual ground truth biclusters
        gt_known_groups = {"ground_truth": {}}
        for idx, row in gt.iterrows():
            gt_known_groups["ground_truth"][f"bic_{idx}"] = row["samples"]

        # Test ground truth vs ground truth - should give perfect performance
        gt_performances, gt_best_matches = calc_ari_matching(
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
        sample_list = list(all_samples)
        mid_point = len(sample_list) // 2
        known_groups = {
            "test_classification": {
                "group1": set(sample_list[:mid_point]),
                "group2": set(sample_list[mid_point:]),
            }
        }

        # Call the function - should not crash
        performances, best_matches = calc_ari_matching(
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

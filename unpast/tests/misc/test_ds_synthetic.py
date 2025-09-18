"""Tests for ds_synthetic module."""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from unpast.misc.ds_synthetic import generate_exprs
from unpast.utils.io import read_bic_table, read_exprs


def hash_table(df):
    """Hash a DataFrame for reproducibility."""
    rows_hashes = pd.util.hash_pandas_object(df, index=True)
    hash = pd.util.hash_pandas_object(
        pd.DataFrame(rows_hashes).T,  # we need one value
        index=True,
    )
    return hash.sum()


class TestGenerateExprs:
    """Test cases for generate_exprs function."""

    def test_generate_exprs_biclusters_write_read_roundtrip(self):
        """Test that created true biclusters can be written and read correctly."""

        # TODO: switch to SyntheticBicluster
        with tempfile.TemporaryDirectory() as temp_dir:
            exprs, biclusters_original, _ = generate_exprs(
                data_sizes=(10, 5),
                rand=np.random.RandomState(42),
                frac_samples=[0.2, 0.3],
                outdir=temp_dir + "/",
                outfile_basename="test",
            )

            exprs_file_path = os.path.join(temp_dir, "test.data.tsv.gz")
            exprs_read = read_exprs(exprs_file_path)
            pd.testing.assert_frame_equal(
                exprs, exprs_read, check_names=True, check_dtype=True
            )

            bic_file_path = os.path.join(temp_dir, "test.true_biclusters.tsv.gz")
            biclusters_read = read_bic_table(bic_file_path)

            # pd.testing.assert_frame_equal(
            #     biclusters_original,
            #     biclusters_read,
            #     check_names=False,  # index.name are diffferent
            #     check_dtype=False,  # float vs np.float64
            # )
            for k, v in biclusters_original.items():
                v_read = biclusters_read.loc[k]
                assert set(v_read["genes"]) == set(v.genes)
                assert set(v_read["samples"]) == set(v.samples)

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
        # TODO: switch to SyntheticBicluster
        data, ground_truth, coexpressed_modules = generate_exprs(
            data_sizes=(n_genes, N),
            rand=np.random.RandomState(seed),
            g_size=n_biomarkers,
            frac_samples=frac_samples,
            m=m,
            std=std,  # int, not float?
            outdir=str(tmp_path) + "/",
            outfile_basename=scenario,
            g_overlap=params[sc_name]["g_overlap"],
            s_overlap=params[sc_name]["s_overlap"],
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

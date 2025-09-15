"""Tests for ds_synthetic module."""

import os
import tempfile
import warnings

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

    def test_generate_exprs_reproducibility(self):
        """Test that generate_exprs produces the expected output with the same seed."""
        data, biclusters, modules = generate_exprs(
            data_sizes=(100, 50),
            frac_samples=[0.2, 0.3],
            outfile_basename="",  # don't save
            seed=42,
            add_coexpressed=[10, 20],
        )

        # sets have random order
        def set_to_str(x):
            if isinstance(x, set):
                return ",".join(map(str, sorted(x)))
            return x

        assert hash_table(data.round(10)) == 13859474739268829572
        if hash_table(data) == 11546093038749057111:
            warnings.warn(
                (
                    "WARNING: Tests generation could be slightly unreproducible on some machines."
                    "\nWay to test: run the following in python and check the last symbol (1 - problem, 0 - ok)."
                    "\n---"
                    "\nimport struct"
                    "\nimport numpy as np"
                    "\nprint(np.__version__)"
                    "\nnum = float(np.random.RandomState(42).normal(loc=0, scale=1.0, size=77)[-1])"
                    "\nd = struct.unpack('>Q', struct.pack('>d', num))"
                    "\nprint(f'{d[0]:064b}')"
                    "\n---"
                )
            )
        else:
            assert hash_table(data) == 8496858187703500925

        assert (
            hash_table(biclusters.drop(columns=["frac"]).map(set_to_str))
            == 4510554005146861675
        )
        assert hash_table(biclusters.map(set_to_str)) == 17686323693856100141
        assert len(modules) > 0
        assert hash_table(pd.DataFrame(modules)) == 6483552165326287867

    def test_generate_exprs_biclusters_write_read_roundtrip(self):
        """Test that created true biclusters can be written and read correctly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            exprs, biclusters_original, _ = generate_exprs(
                data_sizes=(10, 5),
                frac_samples=[0.2, 0.3],
                outdir=temp_dir + "/",
                outfile_basename="test",
                seed=42,
            )

            exprs_file_path = os.path.join(temp_dir, "test.data.tsv.gz")
            exprs_read = read_exprs(exprs_file_path)
            pd.testing.assert_frame_equal(
                exprs, exprs_read, check_names=True, check_dtype=True
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

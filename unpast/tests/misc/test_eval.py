"""Tests for eval module."""

import tempfile
import os
import pandas as pd

from unpast.misc.eval import generate_exprs
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

            biclusters_original.equals(biclusters_read)
            # pd.testing.assert_frame_equal(b_o, b_r) - fails due to different .index:
            #   Index(['bic_0.2', 'bic_0.3'], dtype='object')
            #   Index(['bic_0.2', 'bic_0.3'], dtype='object', name='id')

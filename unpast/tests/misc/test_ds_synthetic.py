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

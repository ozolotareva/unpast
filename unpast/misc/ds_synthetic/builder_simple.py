"""Generating simpler versions of synthetic biclusters and expression data."""

from typing import Any

import numpy as np
import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import Bicluster


def build_simple_biclusters(
    data_sizes: tuple[int, int],
    bic_sizes: tuple[int, int],
    rand: np.random.RandomState,
    bic_mu: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict[str, Any]]:
    """Build simple biclusters:
        exprs = N(0, 1)
        bic = N(bic_mu, 1)

    Args:
        data_sizes (tuple[int, int]): Size of the expression data.
        bic_sizes (tuple[int, int]): Size of the biclusters.
        rand (np.random.RandomState): Random state for reproducibility.
        bic_mu (float): Mean of the biclusters.

    Returns:
        tuple[pd.DataFrame, dict[str, Bicluster], dict]:
            exprs (pd.DataFrame): Expression data.
            biclusters (dict[str, Bicluster]): Dictionary of biclusters.
            some additional data
    """
    table = pd.DataFrame(rand.normal(0, 1, size=data_sizes))
    assert bic_sizes[0] <= data_sizes[0], (
        f"bic size_0 too large ({bic_sizes[0]} > {data_sizes[0]})"
    )
    assert bic_sizes[1] <= data_sizes[1], (
        f"bic size_1 too large ({bic_sizes[1]} > {data_sizes[1]})"
    )

    bic_rows = list(range(bic_sizes[0]))
    bic_cols = list(range(bic_sizes[1]))
    table.iloc[bic_rows, bic_cols] += bic_mu

    return table, {"bic": Bicluster(bic_rows, bic_cols)}, {}

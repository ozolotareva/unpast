"""Generating synthetic biclusters with random rank-k background.

This module provides functionality to create synthetic datasets with low-rank
correlated background structure and embedded biclusters for testing purposes.
"""

from typing import Any

import numpy as np
import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import Bicluster


def _build_background_matrix(
    rand: np.random.RandomState,
    data_sizes: tuple[int, int],
    bg_rank: int = 3,
) -> pd.DataFrame:
    """Build a background matrix with correlated structure.

    Args:
        rand (np.random.RandomState): Random state for reproducibility.
        data_sizes (tuple[int, int]): Size of the expression data.
        bg_rank (int): Rank of the background matrix.

    Returns:
        pd.DataFrame: Background expression data.
    """
    assert bg_rank < min(data_sizes), "bg_rank must be less than min(data_sizes)"

    # Generate low-rank background matrix
    U = rand.normal(0, 1, size=(data_sizes[0], bg_rank))
    V = rand.normal(0, 1, size=(bg_rank, data_sizes[1]))

    background = np.dot(U, V) / np.sqrt(bg_rank)  # normalize to std=1

    # Convert to DataFrame
    exprs = pd.DataFrame(background)
    return exprs


def build_correlated_background_bicluster(
    data_sizes: tuple[int, int],
    bic_sizes: tuple[int, int],
    rand: np.random.RandomState,
    bic_mu: float = 3.0,
    bg_rank: int = 3,
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict[str, Any]]:
    """Build biclusters on a low-rank correlated background.

    Creates a synthetic dataset with a rank-k background matrix and adds a
    bicluster with elevated expression values.

    Args:
        data_sizes: Size of the expression data (n_genes, n_samples).
        bic_sizes: Size of the bicluster (n_genes, n_samples).
        rand: Random state for reproducibility.
        bic_mu: Mean elevation for bicluster values.
        bg_rank: Rank of the background correlation structure.

    Returns:
        Tuple containing:
            - table: Expression data DataFrame
            - biclusters: Dictionary with bicluster information
            - extra: Additional metadata (empty dict)
    """
    table = _build_background_matrix(rand, data_sizes, bg_rank)

    bic_rows = list(range(bic_sizes[0]))
    bic_cols = list(range(bic_sizes[1]))
    table.iloc[bic_rows, bic_cols] += bic_mu

    return table, {"bic": Bicluster(bic_rows, bic_cols)}, {}

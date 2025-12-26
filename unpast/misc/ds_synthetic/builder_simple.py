"""Generating simpler versions of synthetic biclusters and expression data."""

from typing import Any

import numpy as np
import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import Bicluster


def build_simple_biclusters(
    data_sizes: tuple[int, int],
    bic_sizes: tuple[int, int],
    rand: np.random.RandomState,
    bic_mu: float = 5.0,
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


def _parse_bic_size(size_str: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse bicluster size string of the form 'start-end x start-end'.

    Args:
        size_str (str): Bicluster size string.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: Parsed row and column size ranges.
    """
    row_part, col_part = size_str.split("x")
    row_start, row_end = map(int, row_part.strip().split("-"))
    col_start, col_end = map(int, col_part.strip().split("-"))
    return (row_start, row_end), (col_start, col_end)


def build_simple_multiple_biclusters(
    data_sizes: tuple[int, int],
    bic_codes: list[str],
    rand: np.random.RandomState,
    bic_mu: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict[str, Any]]:
    """Build multiple simple biclusters encoded in the expression data:
        exprs = N(0, 1)
        bic = N(bic_mu, 1)

    Args:
        data_sizes (tuple[int, int]): Size of the expression data.
        bic_codes (list[str]): Size of the biclusters.
            ['0-10x1-10', '5-15x5-15', ...]
        rand (np.random.RandomState): Random state for reproducibility.
        bic_mu (float): Mean of the biclusters.

    Returns:
        tuple[pd.DataFrame, dict[str, Bicluster], dict]:
            exprs (pd.DataFrame): Expression data.
            biclusters (dict[str, Bicluster]): Dictionary of biclusters.
            some additional data
    """
    table = pd.DataFrame(rand.normal(0, 1, size=data_sizes))
    biclusters: dict[str, Bicluster] = {}

    for i, bic_code in enumerate(bic_codes):
        (row_start, row_end), (col_start, col_end) = _parse_bic_size(bic_code)
        row_end += 1  # make end exclusive
        col_end += 1  # make end exclusive

        assert 0 <= row_start <= row_end <= data_sizes[0]
        assert 0 <= col_start <= col_end <= data_sizes[1]

        bic_rows = list(range(row_start, row_end))
        bic_cols = list(range(col_start, col_end))
        table.iloc[bic_rows, bic_cols] += bic_mu
        biclusters[f"bic_{i}"] = Bicluster(bic_rows, bic_cols)

    return table, biclusters, {}

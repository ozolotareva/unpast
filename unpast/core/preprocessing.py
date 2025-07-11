import sys

import numpy as np
import pandas as pd

from unpast.utils.logs import get_logger

logger = get_logger(__name__)


def zscore(df):
    """Standardize expression data by z-score normalization.

    Args:
        df (DataFrame): input expression matrix with features as rows and samples as columns

    Returns:
        DataFrame: z-score normalized expression matrix
    """
    m = df.mean(axis=1)
    df = df.T - m
    df = df.T
    s = df.std(axis=1)
    df = df.T / s
    df = df.T
    # set to 0 not variable genes
    zero_var_genes = s[s == 0].index.values
    if len(zero_var_genes) > 0:
        logger.warning(
            f"{len(zero_var_genes)} zero variance rows detected, assign zero z-scores "
        )
    df.loc[zero_var_genes, :] = 0
    return df


def prepare_input_matrix(
    input_matrix: pd.DataFrame,
    min_n_samples: int = 5,
    tol: float = 0.01,
    standradize: bool = True,
    ceiling: float = 0,  # if float>0, limit z-scores to [-x,x]
):
    """Prepare and standardize input expression matrix for biclustering analysis.

    Args:
        input_matrix (DataFrame): raw expression matrix with features as rows and samples as columns
        min_n_samples (int): minimum number of samples required for processing
        tol (float): tolerance for checking if data is already standardized
        standradize (bool): whether to perform z-score standardization
        ceiling (float): if >0, limit z-scores to [-ceiling, ceiling] range

    Returns:
        DataFrame: processed and standardized expression matrix
    """
    exprs = input_matrix.copy()
    exprs.index = [str(x) for x in exprs.index.values]
    exprs.columns = [str(x) for x in exprs.columns.values]
    m = exprs.mean(axis=1)
    std = exprs.std(axis=1)
    # find zero variance rows
    zero_var = list(std[std == 0].index.values)
    if len(zero_var) > 0:
        logger.debug(f"Zero variance rows will be dropped: {len(zero_var)}")
        exprs = exprs.loc[std > 0]
        m = m[std > 0]
        std = std[std > 0]
        if exprs.shape[0] <= 2:
            logger.warning(
                "After excluding constant features (rows), less than 3 features (rows) remain"
                f" in the input matrix. Remaining: {exprs.shape[0]}"
            )

    mean_passed = np.all(np.abs(m) < tol)
    std_passed = np.all(np.abs(std - 1) < tol)
    if not (mean_passed and std_passed):
        logger.debug("Input is not standardized.")
        if standradize:
            exprs = zscore(exprs)
            if not mean_passed:
                logger.debug("- centering mean to 0")
            if not std_passed:
                logger.debug("- scaling std to 1")
    if len(set(exprs.index.values)) < exprs.shape[0]:
        logger.warning("Row names are not unique.")
    missing_values = exprs.isna().sum(axis=1)
    n_na = missing_values[missing_values > 0].shape[0]
    if n_na > 0:
        logger.warning(
            f"Missing values detected in {missing_values[missing_values > 0].shape[0]} rows"
        )
        keep_features = missing_values[
            missing_values <= exprs.shape[1] - min_n_samples
        ].index.values
        logger.warning(
            f"Features with too few values (<{min_n_samples}) dropped: {exprs.shape[0] - len(keep_features)}"
        )
        exprs = exprs.loc[keep_features, :]

    if standradize:
        if ceiling > 0:
            logger.debug(
                f"Standardized expressions will be limited to [-{ceiling},{ceiling}]:"
            )
            exprs[exprs > ceiling] = ceiling
            exprs[exprs < -ceiling] = -ceiling
            if n_na > 0:
                exprs.fillna(-ceiling, inplace=True)
                logger.debug(f"Missing values will be replaced with -{ceiling}.")
    return exprs

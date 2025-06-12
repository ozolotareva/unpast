import sys
import pandas as pd
import numpy as np


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
        print(
            len(zero_var_genes),
            "zero variance rows detected, assign zero z-scores ",
            file=sys.stderr,
        )
    df.loc[zero_var_genes, :] = 0
    return df


def prepare_input_matrix(
    input_matrix: pd.DataFrame,
    min_n_samples: int = 5,
    tol: float = 0.01,
    standradize: bool = True,
    ceiling: float = 0,  # if float>0, limit z-scores to [-x,x]
    verbose: bool = False,
):
    """Prepare and standardize input expression matrix for biclustering analysis.

    Args:
        input_matrix (DataFrame): raw expression matrix with features as rows and samples as columns
        min_n_samples (int): minimum number of samples required for processing
        tol (float): tolerance for checking if data is already standardized
        standradize (bool): whether to perform z-score standardization
        ceiling (float): if >0, limit z-scores to [-ceiling, ceiling] range
        verbose (bool): whether to print processing information

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
        if verbose:
            print(
                "\tZero variance rows will be dropped: %s" % len(zero_var),
                file=sys.stdout,
            )
        exprs = exprs.loc[std > 0]
        m = m[std > 0]
        std = std[std > 0]
        if exprs.shape[0] <= 2:
            print(
                "After excluding constant features (rows) , less than 3 features (rows) remain in the input matrix."
                % exprs.shape[0],
                file=sys.stderr,
            )

    mean_passed = np.all(np.abs(m) < tol)
    std_passed = np.all(np.abs(std - 1) < tol)
    if not (mean_passed and std_passed):
        if verbose:
            print("\tInput is not standardized.", file=sys.stdout)
        if standradize:
            exprs = zscore(exprs)
            if not mean_passed:
                if verbose:
                    print("\tCentering mean to 0", file=sys.stdout)
            if not std_passed:
                if verbose:
                    print("\tScaling std to 1", file=sys.stdout)
    if len(set(exprs.index.values)) < exprs.shape[0]:
        print("\tRow names are not unique.", file=sys.stderr)
    missing_values = exprs.isna().sum(axis=1)
    n_na = missing_values[missing_values > 0].shape[0]
    if n_na > 0:
        if verbose:
            print(
                "\tMissing values detected in %s rows"
                % missing_values[missing_values > 0].shape[0],
                file=sys.stdout,
            )
        keep_features = missing_values[
            missing_values <= exprs.shape[1] - min_n_samples
        ].index.values
        if verbose:
            print(
                "\tFeatures with too few values (<%s) dropped: %s"
                % (min_n_samples, exprs.shape[0] - len(keep_features)),
                file=sys.stdout,
            )
        exprs = exprs.loc[keep_features, :]

    if standradize:
        if ceiling > 0:
            if verbose:
                print(
                    "\tStandardized expressions will be limited to [-%s,%s]:"
                    % (ceiling, ceiling),
                    file=sys.stdout,
                )
            exprs[exprs > ceiling] = ceiling
            exprs[exprs < -ceiling] = -ceiling
            if n_na > 0:
                exprs.fillna(-ceiling, inplace=True)
                if verbose:
                    print(
                        "\tMissing values will be replaced with -%s." % ceiling,
                        file=sys.stdout,
                    )
    return exprs

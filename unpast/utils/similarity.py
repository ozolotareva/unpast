"""Cluster binarized genes"""

import numpy as np
import pandas as pd

from unpast.utils.logs import get_logger, log_function_duration

logger = get_logger(__name__)


@log_function_duration(name="Similarity ARI")
def get_similarity_ari(binarized_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Adjusted Rand Index between features based on binary patterns.

    Treats each feature column as a binary clustering of samples, then computes
    the ARI between all pairs of features using vectorized operations.

    Args:
        binarized_data: Binary matrix with samples as rows, features as columns.
            Values should be 0 or 1.

    Returns:
        Symmetric similarity matrix with ARI values in [-1, 1].
        Diagonal entries are 1.0.
    """
    X = binarized_data.to_numpy(dtype=int)
    n_samples = X.shape[0]

    # Contingency table components for all feature pairs
    n11 = X.T @ X
    n1 = X.sum(axis=0)
    n10 = n1[:, None] - n11
    n01 = n10.T
    n00 = n_samples - n01 - n10 - n11

    def comb2(k):
        return k * (k - 1) / 2

    index = comb2(n00) + comb2(n01) + comb2(n10) + comb2(n11)
    self_comb = comb2(n1) + comb2(n_samples - n1)
    si = self_comb[:, None]
    sj = self_comb[None, :]
    expected_index = si * sj / comb2(n_samples)
    max_index = (si + sj) / 2

    numerator = index - expected_index
    denominator = max_index - expected_index

    with np.errstate(divide="ignore", invalid="ignore"):
        # if denominator is 0, set ARI to 1.0 (perfect match)
        # (not expected to happen from the binarization step)
        # using 0.1 threshold to avoid floating point issues
        ari = np.where(np.abs(denominator) > 0.1, numerator / denominator, 1.0)

    ari = ari.clip(-1.0, 1.0)  # Ensure valid range in case of floating point errors
    ari_df = pd.DataFrame(
        ari, index=binarized_data.columns, columns=binarized_data.columns
    )

    logger.debug(
        f"ARI similarities for binarized data with shape {binarized_data.shape} computed."
    )
    return ari_df


@log_function_duration(name="Jaccard Similarity")
def get_similarity_jaccard(binarized_data):  # ,J=0.5
    """Calculate Jaccard similarity matrix between features based on binary expression patterns.

    Args:
        binarized_data (DataFrame): binary expression matrix with samples as rows and features as columns

    Returns:
        DataFrame: symmetric similarity matrix with Jaccard coefficients between all feature pairs
    """
    genes = binarized_data.columns.values
    n_samples = binarized_data.shape[0]
    size_threshold = int(min(0.45 * n_samples, (n_samples) / 2 - 10))
    # print("size threshold",size_threshold)
    n_genes = binarized_data.shape[1]
    df = np.array(binarized_data.T, dtype=bool)
    results = np.zeros((n_genes, n_genes))
    for i in range(0, n_genes):
        results[i, i] = 1
        g1 = df[i]

        for j in range(i + 1, n_genes):
            g2 = df[j]
            o = g1 * g2
            u = g1 + g2
            jaccard = o.sum() / u.sum()
            # try matching complements
            if g1.sum() > size_threshold:
                g1_complement = ~g1
                o = g1_complement * g2
                u = g1_complement + g2
                jaccard_c = o.sum() / u.sum()
            elif g2.sum() > size_threshold:
                g2 = ~g2
                o = g1 * g2
                u = g1 + g2
                jaccard_c = o.sum() / u.sum()
            else:
                jaccard_c = 0
            jaccard = max(jaccard, jaccard_c)
            results[i, j] = jaccard
            results[j, i] = jaccard

    results = pd.DataFrame(data=results, index=genes, columns=genes)
    logger.debug(
        f"Jaccard similarities for {binarized_data.shape[1]} features computed."
    )
    return results


# @log_function_duration(name="Pearson Similarity")

# def get_similarity_corr(df, verbose=True):
#     """Calculate correlation-based similarity matrix between features.

#     Args:
#         df (DataFrame): expression matrix with features as columns
#         verbose (bool): whether to print progress information

#     Returns:
#         DataFrame: correlation similarity matrix with positive correlations only
#     """
#     corr = df.corr()  # .applymap(abs)
#     corr = corr[corr > 0]  # to consider only direct correlations
#     corr = corr.fillna(0)
#     return corr

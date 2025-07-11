"""Cluster binarized genes"""

import sys
from time import time

import numpy as np
import pandas as pd

from unpast.utils.logs import get_logger, log_function_duration

logger = get_logger(__name__)


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

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import interp1d

from .logs import get_logger, log_function_duration

logger = get_logger(__name__)


def calc_snr_per_row(s, N, exprs, exprs_sums, exprs_sq_sums):
    """Calculate SNR per row for given bicluster size.

    Args:
        s (int): bicluster size (number of samples)
        N (int): total number of samples
        exprs (array): expression matrix
        exprs_sums (array): precomputed row sums
        exprs_sq_sums (array): precomputed squared row sums

    Returns:
        array: SNR values per row
    """
    bic = exprs[:, :s]
    bic_sums = bic.sum(axis=1)
    bic_sq_sums = np.square(bic).sum(axis=1)

    bg_counts = N - s
    bg_sums = exprs_sums - bic_sums
    bg_sq_sums = exprs_sq_sums - bic_sq_sums

    bic_mean, bic_std = calc_mean_std_by_powers((s, bic_sums, bic_sq_sums))
    bg_mean, bg_std = calc_mean_std_by_powers((bg_counts, bg_sums, bg_sq_sums))

    snr_dist = (bic_mean - bg_mean) / (bic_std + bg_std)

    return snr_dist


def calc_mean_std_by_powers(powers):
    """Calculate mean and standard deviation from power statistics.

    Args:
        powers (tuple): tuple containing (count, sum, sum_of_squares)

    Returns:
        tuple: (mean, std) calculated from the power statistics
    """
    count, val_sum, sum_sq = powers

    mean = val_sum / count  # what if count == 0?
    std = np.sqrt((sum_sq / count) - mean * mean)
    return mean, std


def calc_SNR(ar1, ar2, pd_mode=False):
    """Calculate Signal-to-Noise Ratio (SNR) for two arrays.

    Args:
        ar1 (array): first array
        ar2 (array): second array
        pd_mode (bool): if True, use pandas-like mean/std methods
            i.e. n-1 for std, ignore nans

    Returns:
        float: SNR value
    """

    if pd_mode:
        std = lambda x: np.nanstd(x, ddof=1.0)
        mean = np.nanmean
    else:
        std = np.nanstd
        mean = np.mean

    mean_diff = mean(ar1) - mean(ar2)
    std_sum = std(ar1) + std(ar2)

    if std_sum == 0:
        return np.inf * mean_diff

    return mean_diff / std_sum


@log_function_duration(name="Background distribution generation")
def generate_null_dist(N, sizes, n_permutations=10000, pval=0.001, seed=42):
    """Generate null distribution of SNR values for statistical testing.

    Args:
        N (int): total number of samples
        sizes (array): bicluster sizes to generate null distributions for
        n_permutations (int): number of permutations for empirical distribution
        pval (float): p-value threshold for background distribution
        seed (int): random seed for reproducibility

    Returns:
        DataFrame: null distribution matrix with sizes as rows and permutations as columns
    """
    # samples 'N' values from standard normal distribution, and split them into bicluster and background groups
    # 'sizes' defines bicluster sizes to test
    # returns a dataframe with the distribution of SNR for each bicluster size (sizes x n_permutations )
    logger.debug(
        "Generate background distribution of SNR depending on the bicluster size:"
    )
    logger.debug(f"- total samples: {N}")
    logger.debug(f"- samples in a bicluster: {min(sizes)}-{max(sizes)}")
    logger.debug(f"- tn_permutations: {n_permutations}")
    logger.debug(f"- snr pval threshold: {pval}")

    exprs = np.zeros((n_permutations, N))  # generate random expressions from st.normal
    # values = exprs.values.reshape(-1) # random samples from expression matrix
    # exprs = np.random.choice(values,size=exprs.shape[1])
    np.random.seed(seed=seed)
    for i in range(n_permutations):
        exprs[i,] = sorted(np.random.normal(size=N))

    exprs_sums = exprs.sum(axis=1)
    exprs_sq_sums = np.square(exprs).sum(axis=1)

    null_distribution = pd.DataFrame(
        np.zeros((sizes.shape[0], n_permutations)),
        index=sizes,
        columns=range(n_permutations),
    )

    for s in sizes:
        null_distribution.loc[s, :] = -1 * calc_snr_per_row(
            s, N, exprs, exprs_sums, exprs_sq_sums
        )

    return null_distribution


def get_trend(sizes, thresholds, plot=True):
    """Smoothen the trend and return a function min_SNR(size; p-val cutoff).

    Given a set of points (x_i, y_i), returns a function f(x) that interpolates
    the data with LOWESS+linear interpolation.

    Args:
        sizes (array): values of x_i (bicluster sizes)
        thresholds (array): values of y_i (SNR thresholds)
        plot (bool): if True, plots the trend

    Returns:
        function: get_min_snr function that returns the minimal SNR for a given size
    """
    assert len(sizes) >= 0
    if len(sizes) == 1:
        return lambda x: thresholds[0]

    lowess = sm.nonparametric.lowess
    frac = max(1, min(math.floor(int(0.1 * len(sizes))), 15) / len(sizes))
    lowess_curve = lowess(
        thresholds, sizes, frac=frac, return_sorted=True, is_sorted=False
    )
    get_min_snr = interp1d(
        lowess_curve[:, 0], lowess_curve[:, 1]
    )  # ,kind="nearest-up",fill_value="extrapolate")
    if plot:
        plt.plot(sizes, thresholds, "b--", lw=2)
        plt.plot(sizes, get_min_snr(sizes), "r-", lw=2)
        plt.xlabel("n_samples")
        plt.ylabel("SNR")
        plt.ylim((0, 5))
        plt.show()
    return get_min_snr


def calc_e_pval(snr, size, null_distribution):
    """Calculate empirical p-value from null distribution.

    Args:
        snr (float): observed SNR value
        size (int): bicluster size
        null_distribution (DataFrame): precomputed null distribution

    Returns:
        float: empirical p-value
    """
    e_dist = null_distribution.loc[int(size), :]
    return (len(e_dist[e_dist >= abs(snr)]) + 1.0) / (null_distribution.shape[1] + 1.0)

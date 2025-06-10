import numpy as np
import pandas as pd


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


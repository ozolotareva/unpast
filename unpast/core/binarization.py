"""Binarization module for gene expression data."""

import sys
import warnings
import pandas as pd
import numpy as np
from time import time

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering

from statsmodels.stats.multitest import fdrcorrection

from unpast.utils.statistics import calc_SNR, generate_null_dist, get_trend, calc_e_pval
from unpast.utils.visualization import plot_binarized_feature, plot_binarization_results
from unpast.utils.io import ProjectPaths, write_args
from unpast.utils.logs import get_logger, log_function_duration

logger = get_logger(__name__)


def _fit_gmm_model(row, seed=42, prob_cutoff=0.5):
    """Fit a Gaussian Mixture Model to identify two groups in expression data.

    Args:
        row (array): expression values for a single feature across samples
        seed (int): random seed for reproducibility
        prob_cutoff (float): probability cutoff for group assignment

    Returns:
        tuple: (labels, is_converged) where
            - labels: boolean array indicating group membership
            - is_converged: whether the GMM fitting converged
    """
    with warnings.catch_warnings():  # this is to ignore convergence warnings
        warnings.simplefilter("ignore")
        row2d = row[:, np.newaxis]  # adding mock axis for GM interface
        model = GaussianMixture(
            n_components=2,
            init_params="kmeans",
            max_iter=len(row),
            n_init=1,
            covariance_type="spherical",
            random_state=seed,
        ).fit(row2d)

        labels = model.predict_proba(row2d)[:, 0] > prob_cutoff
        is_converged = model.converged_
        return labels, is_converged


def _select_pos_neg(row, min_n_samples, seed=42, prob_cutoff=0.5, method="GMM"):
    """Identify positive and negative signal groups using GMM or other method of binarization.

    Args:
        row (array): expression values for a single feature across samples
        min_n_samples (int): minimum number of samples required for each group
        seed (int): random seed for reproducible clustering
        prob_cutoff (float): probability threshold for group assignment
        method (str): binarization method to use ("GMM")

    Returns:
        tuple: (mask_pos, mask_neg, snr, size, is_converged) where
            - mask_pos: boolean mask for positive/high expression samples
            - mask_neg: boolean mask for negative/low expression samples
            - snr: signal-to-noise ratio between groups
            - size: effective sample size
            - is_converged: whether GMM fitting converged
    """
    is_converged = None
    if method == "GMM":
        labels, is_converged = _fit_gmm_model(row, seed=seed, prob_cutoff=prob_cutoff)

    elif method in ["kmeans", "ward"]:
        row2d = row[:, np.newaxis]  # adding mock axis
        if method == "kmeans":
            model = KMeans(n_clusters=2, max_iter=len(row), n_init=1, random_state=seed)
        else:
            assert method == "ward"
            model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        # elif method == "HC_ward":
        #    model = Ward(n_clusters=2)

        labels = model.fit_predict(row2d) == 1

    else:
        raise NotImplementedError(
            f"wrong binarization method name {method},"
            " must be ['GMM', 'kmeans', 'ward']",
        )

    assert labels.dtype is np.dtype(bool)
    assert len(labels) == len(row)

    # let labels == True be always a smaller sample set
    if labels.sum() >= (~labels).sum():
        labels = ~labels

    # special treatment for cases when bic distribution is too wide and overlaps bg distribution
    # remove from bicluster samples with the sign different from its median sign
    # TODO: consider removing/using more robust method
    if labels.sum() > 0:
        if np.median(row[labels]) >= 0:
            labels[row < 0] = False
        else:
            labels[row > 0] = False

    if labels.sum() >= min_n_samples:
        size = labels.sum()  # the smaller group size
        snr = calc_SNR(row[labels], row[~labels])

        # return first the positive group
        if snr <= 0:
            labels = ~labels
        return labels, ~labels, abs(snr), size, is_converged

    else:
        # not enough samples in the group, return empty masks
        return (
            np.zeros_like(labels),
            np.zeros_like(labels),
            np.nan,
            np.nan,
            is_converged,
        )


@log_function_duration(name="Binarization")
def sklearn_binarization(
    exprs,
    min_n_samples,
    verbose=True,
    plot=True,
    plot_SNR_thr=2.0,
    show_fits=[],
    seed=1,
    prob_cutoff=0.5,
    method="GMM",
):
    """Perform binarization of gene expression data using GMM or other method.

    Args:
        exprs (DataFrame): expression matrix with genes as rows and samples as columns
        min_n_samples (int): minimum number of samples required for each group
        verbose (bool): whether to print progress information
        plot (bool): whether to generate plots for binarization
        plot_SNR_thr (float): SNR threshold above which to generate plots
        show_fits (list): specific gene names for which to show fitting plots
        seed (int): random seed for reproducible results
        prob_cutoff (float): probability threshold for group assignment
        method (str): binarization method to use ("GMM")

    Returns:
        tuple: (binarized_expressions, stats) where
            - binarized_expressions: dict mapping gene names to binary sample groups
            - stats: dict containing SNR and size statistics for each gene
    """
    binarized_expressions = {}
    stats = {}
    for i, (gene, row_) in enumerate(exprs.iterrows()):
        row = row_.values

        pos_mask, neg_mask, snr, size, is_converged = _select_pos_neg(
            row, min_n_samples, seed=seed, prob_cutoff=prob_cutoff, method=method
        )

        # bicluster is the smaller part
        if pos_mask.sum() <= neg_mask.sum():
            mask, direction = pos_mask, "UP"
        else:
            mask, direction = neg_mask, "DOWN"

        binarized_expressions[gene] = mask.astype(int)
        stats[gene] = {
            "pval": 0,
            "SNR": snr,
            "size": size,  # size of positive group
            "direction": direction,
            "convergence": is_converged,
        }

        # TODO: tqdm
        if i % 1000 == 0:
            logger.debug(f"processed {i}/{exprs.shape[0]} samples")

        if (plot and abs(snr) > plot_SNR_thr) or (gene in show_fits):
            plot_binarized_feature(row, direction, pos_mask, neg_mask, snr)

    stats = pd.DataFrame.from_dict(stats).T
    binarized_expressions = pd.DataFrame.from_dict(binarized_expressions)
    return binarized_expressions, stats


def _try_loading_binarization_files(paths, verbose=True):
    """Try to load existing binarization files.

    Args:
        paths (ProjectPaths): Object containing file paths for binarization files
        verbose (bool): Whether to print progress information

    Returns:
        tuple: (binarized_data, stats, null_distribution) where
            - binarized_data: DataFrame with binary expression profiles or None if failed
            - stats: DataFrame with binarization statistics or None if failed
            - null_distribution: DataFrame with background distribution or None if failed
    """
    binarized_data, stats, null_distribution = None, None, None
    logger.debug("Loading binarization files ...")

    # Try to load binarized data
    try:
        logger.debug(f"Loading binarized features from {paths.binarization_res}")
        binarized_data = pd.read_csv(paths.binarization_res, sep="\t", index_col=0)
    except Exception as e:
        logger.error(f"Failed to load {paths.binarization_res}, {e}")

    # Try to load stats
    try:
        logger.debug(f"Loading binarization statistics from {paths.binarization_stats}")
        stats = pd.read_csv(paths.binarization_stats, sep="\t", index_col=0)
    except Exception as e:
        logger.error(f"Failed to load {paths.binarization_stats}, {e}")

    # Try to load background distribution
    try:
        logger.debug(f"Loading background distribution from {paths.binarization_bg}")
        null_distribution = pd.read_csv(paths.binarization_bg, sep="\t", index_col=0)
        null_distribution.columns = [int(x) for x in null_distribution.columns.values]
    except Exception as e:
        logger.error(f"Failed to load {paths.binarization_bg}, {e}")
    return binarized_data, stats, null_distribution


def _save_binarization_files(
    paths, binarized_data, stats, null_distribution, verbose=True
):
    """Save binarization results to files.

    Args:
        paths (ProjectPaths): Object containing file paths for binarization files
        binarized_data (DataFrame): DataFrame with binary expression profiles
        stats (DataFrame): DataFrame with binarization statistics
        null_distribution (DataFrame): DataFrame with background distribution
        verbose (bool): Whether to print progress information

    Returns:
        None: Saves files to specified paths
    """
    binarized_data.to_csv(paths.binarization_res, sep="\t")
    logger.debug(f"Binarized data saved to {paths.binarization_res}")

    stats.to_csv(paths.binarization_stats, sep="\t")
    logger.debug(f"Statistics saved to {paths.binarization_stats}")

    null_distribution.to_csv(paths.binarization_bg, sep="\t")
    logger.debug(f"Background distribution saved to {paths.binarization_bg}")


def _generate_missing_null_distribution_sizes(
    null_distribution, N, sizes, pval, n_permutations, seed, verbose
):
    """Ensure that the null distribution contains all required sizes."""
    if null_distribution is None:
        precomputed_sizes = set()
    else:
        precomputed_sizes = set(null_distribution.index.values)

    add_sizes = np.array(sorted(set(sizes) - (set(precomputed_sizes))))
    if len(add_sizes) > 0:
        null_distribution2 = generate_null_dist(
            N,
            add_sizes,
            pval=pval,
            n_permutations=n_permutations,
            seed=seed,
            verbose=verbose,
        )
        null_distribution2.columns = [int(x) for x in null_distribution2.columns.values]

        if null_distribution is None:
            null_distribution = null_distribution2
        else:
            # join
            null_distribution = pd.concat(
                [
                    null_distribution,
                    null_distribution2.loc[:, null_distribution.columns.values],
                ],
                axis=0,
            )
    return null_distribution


def _add_snrs(stats, null_distribution, sizes, pval, verbose):
    """Calculate SNR p-values and thresholds based on null distribution.

    Args:
        stats (DataFrame): DataFrame with statistics including 'SNR' and 'size'
        null_distribution (DataFrame): DataFrame with empirical null distribution
        verbose (bool): Whether to print progress information
    """
    stats = stats.dropna(subset=["size"])
    stats["pval"] = stats.apply(
        lambda row: calc_e_pval(row["SNR"], row["size"], null_distribution), axis=1
    )
    _accepted, pval_adj = fdrcorrection(stats["pval"])
    stats["pval_BH"] = pval_adj

    # find SNR threshold
    thresholds = np.quantile(null_distribution.loc[sizes, :].values, q=1 - pval, axis=1)
    size_snr_trend = get_trend(sizes, thresholds, plot=False, verbose=verbose)
    stats["SNR_threshold"] = stats["size"].apply(lambda x: size_snr_trend(x))
    return stats, size_snr_trend


@log_function_duration(name="Binarization")
def binarize(
    out_dir,
    exprs,
    method="GMM",
    save=True,
    load=False,
    min_n_samples=5,
    pval=0.001,
    plot_all=True,
    plot_SNR_thr=np.inf,
    show_fits=[],
    verbose=True,
    seed=42,
    prob_cutoff=0.5,
    n_permutations=10000,
):
    """Main binarization function that creates binary expression profiles with significance testing.

    Args:
        out_dir (str | ProjectPaths): output directory or ProjectPaths object for saving results
        exprs (DataFrame): normalized expression matrix with features as rows and samples as columns
        method (str): binarization method to use ("GMM")
        save (bool): whether to save binarized data to files
        load (bool): whether to try loading existing binarized data
        min_n_samples (int): minimum number of samples required for each group
        pval (float): p-value threshold for significance testing
        plot_all (bool): whether to generate plots for all features
        plot_SNR_thr (float): SNR threshold above which to generate plots
        show_fits (list): specific feature names for which to show fitting plots
        verbose (bool): whether to print progress information
        seed (int): random seed for reproducible results
        prob_cutoff (float): probability threshold for group assignment
        n_permutations (int): number of permutations for null distribution generation

    Returns:
        tuple: (binarized_data, stats, null_distribution) where
            - binarized_data: DataFrame with binary expression profiles
            - stats: DataFrame with binarization statistics (SNR, size, direction)
            - null_distribution: DataFrame containing empirical null distribution for significance testing
    """
    logger.debug("Binarization started ...")
    if isinstance(out_dir, ProjectPaths):
        paths = out_dir
    else:
        paths = ProjectPaths(out_dir)
    binarization_args = locals()  # for saving later

    binarized_data, stats, null_distribution = None, None, None
    if load:
        # TODO: add _warn_diff_args(locals(), paths)
        binarized_data, stats, null_distribution = _try_loading_binarization_files(
            paths, verbose
        )

    # Calculate binarization data and statistics if not loaded
    if (binarized_data is None) or (stats is None):
        if exprs is None:
            raise RuntimeError("Provide either raw or binarized data.")

        # binarize features
        binarized_data, stats = sklearn_binarization(
            exprs,
            min_n_samples,
            plot=plot_all,
            plot_SNR_thr=plot_SNR_thr,
            prob_cutoff=prob_cutoff,
            show_fits=show_fits,
            verbose=verbose,
            seed=seed,
            method=method,
        )

    # bicluster sizes for null distribution
    assert exprs is not None, "exprs must be defined"
    N = exprs.shape[1]

    def calc_size():
        assert stats is not None, "stats must be defined"
        sizes1 = set([x for x in stats["size"].values if not np.isnan(x)])
        # no more than 100 of bicluster sizes are computed
        # step = max(int((N - min_n_samples) / 100), 1)
        step = max(int((int(N / 2) - min_n_samples) / 100), 1)
        # sizes2 = set(map(int, np.arange(min_n_samples, int(N / 2), step)))
        sizes2 = set(map(int, np.arange(min_n_samples, int(N / 2) + 1, step)))
        sizes = np.array(sorted(sizes1 | sizes2))
        return sizes

    sizes = calc_size()

    # If null distribution was loaded, ensure all required sizes
    null_distribution = _generate_missing_null_distribution_sizes(
        null_distribution, N, sizes, pval, n_permutations, seed, verbose
    )

    # remove data for processing
    null_distribution_full = null_distribution
    null_distribution = null_distribution.loc[sizes, :]

    stats, size_snr_trend = _add_snrs(stats, null_distribution, sizes, pval, verbose)

    if save:
        paths.create_binarization_paths()
        write_args(
            binarization_args,
            paths.binarization_args,
            args_label="Binarization arguments",
        )

        null_distribution_full = null_distribution_full.sort_index()
        _save_binarization_files(
            paths,
            binarized_data,
            stats,
            null_distribution_full,  # save even not-used distributions
            verbose=verbose,
        )

    ### keep features passed binarization
    passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
    # passed = stats.loc[stats["pval_BH"]<pval,:]

    logger.debug(
        f"UP-regulated features:\t{passed.loc[passed['direction'] == 'UP', :].shape[0]}"
    )
    logger.debug(
        f"DOWN-regulated features:\t{passed.loc[passed['direction'] == 'DOWN', :].shape[0]}"
    )
    # print("ambiguous features:\t%s"%(passed.loc[passed["direction"]=="UP,DOWN",:].shape[0]),file = sys.stdout)

    # keep only binarized features
    binarized_data = binarized_data.loc[:, list(passed.index.values)]
    # add sample names
    binarized_data.index = exprs.columns.values

    if plot_all:
        plot_binarization_results(stats, size_snr_trend, sizes, pval)

    return binarized_data, stats, null_distribution

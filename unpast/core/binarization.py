"""Binarization module for gene expression data."""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from time import time

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering

import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

from unpast.utils.statistics import calc_SNR, generate_null_dist, get_trend, calc_e_pval
from unpast.utils.visualization import plot_binarized_feature
from unpast.utils.io import ProjectPaths


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


def select_pos_neg(row, min_n_samples, seed=42, prob_cutoff=0.5, method="GMM"):
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
        mask_pos, is_converged = _fit_gmm_model(row, seed=seed, prob_cutoff=prob_cutoff)

    elif method in ["kmeans", "ward"]:
        row2d = row[:, np.newaxis]  # adding mock axis
        if method == "kmeans":
            model = KMeans(n_clusters=2, max_iter=len(row), n_init=1, random_state=seed)
        else:
            assert method == "ward"
            model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        # elif method == "HC_ward":
        #    model = Ward(n_clusters=2)

        mask_pos = model.fit_predict(row2d) == 1

    else:
        print(
            f"wrong binarization method name {method},"
            " must be ['GMM', 'kmeans', 'ward']",
            file=sys.stderr,
        )
        raise NotImplementedError(f"method {method} is not implemented")

    assert mask_pos.dtype is np.dtype(bool)
    assert len(mask_pos) == len(row)

    # let mask_pos == True be always a smaller sample set
    if mask_pos.sum() >= (~mask_pos).sum():
        mask_pos = ~mask_pos

    # special treatment for cases when bic distribution is too wide and overlaps bg distribution
    # remove from bicluster samples with the sign different from its median sign
    # TODO: consider removing/using more robust method
    if mask_pos.sum() > 0:
        if np.median(row[mask_pos]) >= 0:
            mask_pos[row < 0] = False
        else:
            mask_pos[row > 0] = False

    # if SNR < 0, then switch groups
    # TODO: consider how it works with the previous two steps
    snr = np.nan
    size = np.nan
    if mask_pos.sum() >= min_n_samples:
        snr = calc_SNR(row[mask_pos], row[~mask_pos])
        if snr <= 0:
            mask_pos = ~mask_pos

        size = mask_pos.sum()  # always the pos group size or nan

    return mask_pos, ~mask_pos, abs(snr), size, is_converged


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
    t0 = time()

    binarized_expressions = {}
    stats = {}
    for i, (gene, row_) in enumerate(exprs.iterrows()):
        row = row_.values

        pos_mask, neg_mask, snr, size, is_converged = select_pos_neg(
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

        # logging
        # TODO: tqdm
        if verbose:
            if i % 1000 == 0:
                print("\t\tgenes processed:", i)

        if (plot and abs(snr) > plot_SNR_thr) or (gene in show_fits):
            plot_binarized_feature(row, direction, pos_mask, neg_mask, snr)

    stats = pd.DataFrame.from_dict(stats).T
    binarized_expressions = pd.DataFrame.from_dict(binarized_expressions)

    # logging
    if verbose:
        print(
            "\tBinarization for {} features completed in {:.2f} s".format(
                len(exprs), time() - t0
            )
        )

    return binarized_expressions, stats


def try_loading_binarization_files(paths, verbose=True):
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
    if verbose:
        print("Trying to load binarization files ...", file=sys.stdout)

    # Try to load binarized data
    try:
        if verbose:
            print(
                "Load binarized features from",
                paths.bin_exprs_fname,
                "\n",
                file=sys.stdout,
            )
        binarized_data = pd.read_csv(paths.bin_exprs_fname, sep="\t", index_col=0)
    except Exception:
        print("Failed to load", paths.bin_exprs_fname, file=sys.stderr)

    # Try to load stats
    try:
        if verbose:
            print(
                "Loading statistics from", paths.bin_stats_fname, "\n", file=sys.stdout
            )
        stats = pd.read_csv(paths.bin_stats_fname, sep="\t", index_col=0)
    except Exception:
        print("Failed to load", paths.bin_stats_fname, file=sys.stderr)

    # Try to load background distribution
    try:
        if verbose:
            print(
                "Loading background distribution from",
                paths.bin_bg_fname,
                "\n",
                file=sys.stdout,
            )
        null_distribution = pd.read_csv(paths.bin_bg_fname, sep="\t", index_col=0)
        null_distribution.columns = [int(x) for x in null_distribution.columns.values]
    except Exception:
        print(
            "Failed to load",
            paths.bin_bg_fname,
            file=sys.stderr,
        )

    return binarized_data, stats, null_distribution


def generate_missing_null_distribution_sizes(
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


def binarize(
    fname_prefix,
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
        fname_prefix (str): basename for output binarized data files
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
        tuple: (binarized_data, null_distribution) where
            - binarized_data: DataFrame with binary expression profiles
            - null_distribution: DataFrame containing empirical null distribution for significance testing
    """
    t0 = time()

    paths = ProjectPaths(
        fname_prefix=fname_prefix,
        seed=seed,
        method=method,
        min_n_samples=min_n_samples,
        n_permutations=n_permutations,
        pval=pval,
    )

    binarized_data, stats, null_distribution = None, None, None
    if load:
        binarized_data, stats, null_distribution = try_loading_binarization_files(
            paths, verbose
        )

    # Run binarization for data and statistics if not loaded
    if (binarized_data is None) or (stats is None):
        if exprs is None:
            print("Provide either raw or binarized data.", file=sys.stderr)
            return None  # maybe exception?

        # binarize features
        if verbose:
            print("\nBinarization started ....\n")

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
    null_distribution = generate_missing_null_distribution_sizes(
        null_distribution, N, sizes, pval, n_permutations, seed, verbose
    )
    # may remove some sizes before saving
    null_distribution_full = null_distribution
    null_distribution = null_distribution.loc[sizes, :]

    # add SNR p-val depends on bicluster size, get trend
    def add_snrs(stats):
        stats = stats.dropna(subset=["size"])
        stats["pval"] = stats.apply(
            lambda row: calc_e_pval(row["SNR"], row["size"], null_distribution), axis=1
        )
        _accepted, pval_adj = fdrcorrection(stats["pval"])
        stats["pval_BH"] = pval_adj

        # find SNR threshold
        thresholds = np.quantile(
            null_distribution.loc[sizes, :].values, q=1 - pval, axis=1
        )
        size_snr_trend = get_trend(sizes, thresholds, plot=False, verbose=verbose)
        stats["SNR_threshold"] = stats["size"].apply(lambda x: size_snr_trend(x))
        return stats, size_snr_trend

    stats, size_snr_trend = add_snrs(stats)

    if save:

        def _save(data, filepath, name):
            """Helper function to save DataFrame to a file."""
            data.to_csv(filepath, sep="\t")
            if verbose:
                print(f"{name} is saved to {filepath}", file=sys.stdout)

        # save binarized data
        paths.makedirs_if_missing()
        _save(
            binarized_data,
            paths.bin_exprs_fname,
            "Binarized gene expressions",
        )
        _save(stats, paths.bin_stats_fname, "Statistics")
        _save(
            null_distribution_full,  # saving extra rows, as it may be useful later
            paths.bin_bg_fname,
            "Background distribution",
        )

    ### keep features passed binarization
    passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
    # passed = stats.loc[stats["pval_BH"]<pval,:]

    if verbose:
        print(
            "\t\tUP-regulated features:\t%s"
            % (passed.loc[passed["direction"] == "UP", :].shape[0]),
            file=sys.stdout,
        )
        print(
            "\t\tDOWN-regulated features:\t%s"
            % (passed.loc[passed["direction"] == "DOWN", :].shape[0]),
            file=sys.stdout,
        )
        # print("\t\tambiguous features:\t%s"%(passed.loc[passed["direction"]=="UP,DOWN",:].shape[0]),file = sys.stdout)

    # keep only binarized features
    binarized_data = binarized_data.loc[:, list(passed.index.values)]
    # add sample names
    binarized_data.index = exprs.columns.values

    if plot_all:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # plot binarization results for real genes
        passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
        failed = stats.loc[stats["SNR"] <= stats["SNR_threshold"], :]
        tmp = ax.scatter(
            failed["size"], failed["SNR"], alpha=1, color="black", label="not passed"
        )
        for i, txt in enumerate(failed["size"].index.values):
            ax.annotate(txt, (failed["size"][i], failed["SNR"][i] + 0.1), fontsize=18)
        tmp = ax.scatter(
            passed["size"], passed["SNR"], alpha=1, color="red", label="passed"
        )
        for i, txt in enumerate(passed["size"].index.values):
            ax.annotate(txt, (passed["size"][i], passed["SNR"][i] + 0.1), fontsize=18)

        # plot cutoff
        tmp = ax.plot(
            sizes,
            [size_snr_trend(x) for x in sizes],
            color="darkred",
            lw=2,
            ls="--",
            label="e.pval<" + str(pval),
        )
        plt.gca().legend(("not passed", "passed", "e.pval<" + str(pval)), fontsize=18)
        tmp = ax.set_xlabel("n_samples", fontsize=18)
        tmp = ax.yaxis.tick_right()
        tmp = ax.set_ylabel("SNR", fontsize=18)
        tmp = ax.set_ylim((0, 4))
        # tmp = plt.savefig("figs_binarization/dist.svg", bbox_inches='tight', transparent=True)
        tmp = plt.show()

    return binarized_data, stats, null_distribution

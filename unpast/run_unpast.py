#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd

from unpast import __version__
from unpast.core.binarization import binarize
from unpast.core.feature_clustering import run_Louvain, run_WGCNA, run_WGCNA_iterative
from unpast.core.preprocessing import prepare_input_matrix
from unpast.core.sample_clustering import make_biclusters
from unpast.utils.io import ProjectPaths, write_args, write_bic_table
from unpast.utils.logs import (
    LOG_LEVELS,
    get_logger,
    log_function_duration,
    setup_logging,
)
from unpast.utils.similarity import get_similarity_jaccard

logger = get_logger(__name__)


def check_input_shape(exprs_shape, min_n_samples):
    """Check if the input expression matrix has the required shape.

    Args:
        exprs_shape (tuple): shape of the input expression matrix (rows, columns)
        min_n_samples (int): minimum number of samples required

    Raises:
        AssertionError: if the input matrix does not meet the requirements
    """
    errs = []
    if exprs_shape[1] < 5:
        errs.append(
            "Input matrix must contain at least 5 samples (columns)"
            f" and 2 features (rows), but only {exprs_shape[1]}"
            " columns are found."
        )

    if exprs_shape[0] < 2:
        errs.append(
            "Input matrix must contain at least 2 features (rows)"
            f", but only {exprs_shape[0]} rows are found."
        )

    if min_n_samples < 2:
        errs.append(
            "The minimal number of samples in a bicluster `min_n_samples`"
            f" must be >= 2, found {min_n_samples}."
        )

    if min_n_samples > exprs_shape[1] // 2:
        errs.append(
            "The minimal number of samples in a bicluster `min_n_samples`"
            " must be not greater than half of the cohort size"
            f", found {min_n_samples} > {exprs_shape[1] // 2}."
        )

    if errs:
        for msg in errs:
            logger.error(msg)
        raise ValueError(errs[0])

    if min_n_samples < 5:
        logger.warning(
            f"min_n_samples is recommended to be >= 5, found {min_n_samples}"
        )


@log_function_duration(name="UnPaSt", indent="")
def unpast(
    exprs_file: str,
    out_dir: str = "runs/run_<timestamp>",
    binary_dir: str = "<out_dir>/binarization",
    no_binary_save: bool = False,
    ceiling: float = 3,
    bin_method: str = "kmeans",
    clust_method: str = "Louvain",
    min_n_samples: int = 5,
    show_fits: list = [],
    pval: float = 0.01,
    directions: list = ["DOWN", "UP"],
    modularity: float = 1 / 3,
    similarity_cutoffs=-1,  # for Louvain
    ds: int = 3,
    dch: float = 0.995,
    max_power: int = 10,
    precluster: bool = True,
    rpath: str = "",  # for WGCNA
    # cluster_binary: bool = False,
    merge: float = 1,
    seed: int = 42,
    verbose: bool = False,
    plot_all: bool = False,
    e_dist_size: int = 10000,
    standradize: bool = True,
):
    """Main UnPaSt biclustering algorithm for identifying differentially expressed biclusters.

    Args:
        exprs_file (str): expression matrix with features as rows and samples as columns
        out_dir (str): output directory path (default: "runs/run_<timestamp>")
        binary_dir (str): directory to save/load binarization results (default: "<out_dir>/binarization")
        no_binary_save (bool): if True, do not save binarization results (default: False)
        ceiling (float): absolute threshold for z-scores (default: 3)
        bin_method (str): binarization method - "kmeans", "ward", "GMM" (default: "kmeans")
        clust_method (str): clustering method - "WGCNA", "iWGCNA", "Louvain" (default: "Louvain")
        min_n_samples (int): minimal number of samples in bicluster (default: 5)
        show_fits (list): features to show binarization plots for (default: [])
        pval (float): binarization p-value threshold (default: 0.01)
        directions (list): clustering directions - ["DOWN","UP"] or ["BOTH"] (default: ["DOWN","UP"])
        modularity (float): modularity parameter for Louvain clustering (default: 1/3)
        similarity_cutoffs (float or list): similarity cutoffs for Louvain clustering (default: -1 for auto)
        ds (int): deepSplit parameter for WGCNA (default: 3)
        dch (float): detectCutHeight parameter for WGCNA (default: 0.995)
        max_power (int): maximum power for WGCNA (default: 10)
        precluster (bool): whether to precluster for WGCNA (default: True)
        rpath (str): path to Rscript executable for WGCNA (default: "")
        merge (float): similarity threshold for merging biclusters (default: 1)
        seed (int): random seed for reproducibility (default: 42)
        verbose (bool): whether to print progress messages (default: False)
        plot_all (bool): whether to show all plots (default: False)
        e_dist_size (int): size of empirical SNR distribution (default: 10000)
        standardize (bool): whether to standardize input matrix (default: True)

    Returns:
        pd.DataFrame: biclusters table with columns for genes, samples, SNR, etc.
    """
    np.random.seed(seed)  # todo: check if this is needed
    paths = ProjectPaths(out_dir=out_dir, binary_dir=binary_dir)

    # prepare logging and save args
    if verbose:
        setup_logging(log_file=paths.log, log_level=LOG_LEVELS["DEBUG"])
    else:
        setup_logging(log_file=paths.log, log_level=LOG_LEVELS["INFO"])
    version = __version__
    logger.debug(f"Running UnPaSt v. {version}")
    write_args(locals(), paths.args, args_label="Run arguments")

    # read inputs
    exprs = pd.read_csv(exprs_file, sep="\t", index_col=0)
    logger.info(f"Loaded input from {exprs_file}")
    logger.info(f"- input shape = {exprs.shape[0]} features x {exprs.shape[1]} samples")
    check_input_shape(exprs.shape, min_n_samples)

    e_dist_size = max(e_dist_size, int(1.0 / pval * 10))
    logger.debug(f"The size of empirical SNR distribution: {e_dist_size}")

    # check if input is standardized (between-sample)
    # if necessary, standardize and limit values to [-ceiling,ceiling]
    exprs = prepare_input_matrix(
        exprs,
        min_n_samples=min_n_samples,
        standradize=standradize,
        ceiling=ceiling,
    )

    ######### binarization #########

    binarized_features, stats, null_distribution = binarize(
        paths=paths,
        exprs=exprs,
        method=bin_method,
        no_binary_save=no_binary_save,
        min_n_samples=min_n_samples,
        pval=pval,
        plot_all=plot_all,
        show_fits=show_fits,
        seed=seed,
        prob_cutoff=0.5,
        n_permutations=e_dist_size,
    )

    bin_data_dict = {}
    stats = stats.loc[stats["pval"] <= pval, :]
    features_up = set(stats.loc[stats["direction"] == "UP", :].index.values)
    features_up = sorted(
        set(binarized_features.columns.values).intersection(set(features_up))
    )
    features_down = stats.loc[stats["direction"] == "DOWN", :].index.values
    features_down = sorted(
        set(binarized_features.columns.values).intersection(set(features_down))
    )

    df_up = binarized_features.loc[:, features_up]
    df_down = binarized_features.loc[:, features_down]
    if directions[0] == "BOTH":
        bin_data_dict["BOTH"] = pd.concat([df_up, df_down], axis=1)
    else:
        bin_data_dict["UP"] = df_up
        bin_data_dict["DOWN"] = df_down

    ######### gene clustering #########

    feature_clusters, not_clustered, used_similarity_cutoffs = [], [], []

    if clust_method == "Louvain":
        for d in directions:
            logger.debug(f"Clustering {d}-regulated features")
            df = bin_data_dict[d]
            if df.shape[0] > 1:
                similarity = get_similarity_jaccard(df)
                # similarity = get_similarity_corr(df,verbose = verbose)

                if similarity_cutoffs == -1:  # guess from the data
                    similarity_cutoffs = np.arange(0.3, 0.9, 0.01)
                # if similarity cuttofs is a single value turns it to a list
                try:
                    similarity_cutoffs = [elem for elem in similarity_cutoffs]
                except:
                    similarity_cutoffs = [similarity_cutoffs]

                # if modularity m is defined, choses a similarity cutoff corresponding to this modularity
                # and rund Louvain clustering
                modules, single_features, similarity_cutoff = run_Louvain(
                    similarity,
                    similarity_cutoffs=similarity_cutoffs,
                    m=modularity,
                )
                used_similarity_cutoffs.append(similarity_cutoff)
                feature_clusters += modules
                not_clustered += single_features
            elif df.shape[0] == 1:
                not_clustered += list(df.index.values)
                used_similarity_cutoffs.append(None)
        used_similarity_cutoffs = ",".join(map(str, used_similarity_cutoffs))

    elif "WGCNA" in clust_method:
        if clust_method == "iWGCNA":
            WGCNA_func = run_WGCNA_iterative
        else:
            WGCNA_func = run_WGCNA

        for d in directions:
            # WGCNA tmp file prefix
            logger.debug(f"Clustering {d}-regulated features")
            df = bin_data_dict[d]
            if df.shape[0] > 1:
                modules, single_features = WGCNA_func(
                    df,
                    paths=paths,
                    deepSplit=ds,
                    detectCutHeight=dch,
                    nt="signed_hybrid",
                    max_power=max_power,
                    precluster=precluster,
                    rpath=rpath,
                )
                feature_clusters += modules
                not_clustered += single_features

    else:
        raise NotImplementedError(
            f"'clust_method' must be 'WGCNA', 'iWGCNA', or 'Louvain', found {clust_method}"
        )

    ######### making biclusters #########
    if len(feature_clusters) == 0:
        logger.warning("No biclusters found")
        return pd.DataFrame()

    biclusters = make_biclusters(
        feature_clusters,
        binarized_features,
        exprs,
        method=bin_method,
        merge=merge,
        min_n_samples=min_n_samples,
        min_n_genes=2,
        seed=seed,
        cluster_binary=False,
    )

    ######### save biclusters #########
    if "WGCNA" in clust_method:
        modularity, similarity_cutoff = None, None
    elif clust_method == "Louvain":
        ds, dhs = None, None
    write_bic_table(
        biclusters,
        results_file_name=paths.res,
        to_str=True,
        add_metadata=True,
        seed=seed,
        min_n_samples=min_n_samples,
        pval=pval,
        bin_method=bin_method,
        clust_method=clust_method,
        directions=directions,
        similarity_cutoff=used_similarity_cutoffs,
        m=modularity,
        ds=ds,
        dch=dch,
        max_power=max_power,
        precluster=precluster,
        merge=merge,
    )

    # log results
    logger.info(f"Biclusters saved to {paths.res}")
    shapes = [
        (len(b["gene_indexes"]), len(b["sample_indexes"]))
        for _, b in biclusters.iterrows()
    ]
    shapes = sorted(shapes, key=lambda x: (x[0] * x[1]), reverse=True)
    log_str = ", ".join(f"{x}x{y}" for (x, y) in shapes[:10])
    if len(shapes) > 10:
        log_str += f"... and {len(shapes) - 10} more"
    logger.info(f"- found {len(biclusters)} biclusters with shapes: {log_str}")

    return biclusters


def parse_args():
    """Parse command line arguments for UnPaSt.

    Returns:
        Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "UnPaSt identifies differentially expressed biclusters"
        f" in a 2-dimensional matrix. (version {__version__})"
    )
    parser.add_argument("--seed", metavar=42, default=42, type=int, help="random seed")
    parser.add_argument(
        "--exprs",
        metavar="exprs.z.tsv",
        required=True,
        help=".tsv file with between-sample normalized input data matrix. The first column and row must contain unique feature and sample ids, respectively. At least 5 samples (columns) and at least 2 features (rows) are required.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        metavar="runs/run_<timestamp>",
        default="runs/run_<timestamp>",
        help="output folder",
    )
    parser.add_argument(
        "-bd",
        "--binary_dir",
        metavar="<out_dir>/binarization",
        default="<out_dir>/binarization",
        help="folder to save/load binarization results.",
    )
    parser.add_argument(
        "--ceiling",
        default=3,
        metavar="3",
        type=float,
        required=False,
        help="Absolute threshold for z-scores. For example, when set to 3, z-scores greater than 3 are set to 3 and z-scores less than -3 are set to -3. No ceiling if set to 0.",
    )
    parser.add_argument(
        "-s",
        "--min_n_samples",
        metavar=5,
        default=5,
        type=int,
        help="The minimal number of samples in a bicluster `min_n_samples` must be >= 2 and not greater than half of the cohort size.",
    )
    parser.add_argument(
        "-b",
        "--binarization",
        metavar="kmeans",
        default="kmeans",
        type=str,
        choices=["kmeans", "ward", "GMM"],
        help="binarization method",
    )
    parser.add_argument(
        "-p",
        "--pval",
        metavar=0.01,
        default=0.01,
        type=float,
        help="binarization p-value",
    )
    parser.add_argument(
        "-c",
        "--clustering",
        metavar="Louvain",
        default="Louvain",
        type=str,
        choices=["Louvain", "WGCNA"],
        help="feature clustering method",
    )
    # Louvain parameters
    parser.add_argument(
        "-m",
        "--modularity",
        default=1 / 3,
        metavar="1/3",
        type=float,
        help="Modularity corresponding to a cutoff for similarity matrix (Louvain clustering)",
    )
    parser.add_argument(
        "-r",
        "--similarity_cutoffs",
        default=-1,
        metavar="-1",
        type=float,
        help="A cutoff or a list of cuttofs for similarity matrix (Louvain clustering). If set to -1, will be chosen authomatically from [1/5,4/5] using elbow method.",
    )
    # WGCNA parameters
    parser.add_argument(
        "--ds",
        default=3,
        metavar="3",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="deepSplit parameter, see WGCNA documentation",
    )
    parser.add_argument(
        "--dch",
        default=0.995,
        metavar="0.995",
        type=float,
        help="dynamicTreeCut parameter, see WGCNA documentation",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Whether to cluster up- and down-regulated features together.",
    )
    parser.add_argument(
        "--rpath", default="", metavar="", type=str, help="Full path to Rscript."
    )
    # parser.add_argument('--merge', default=1, metavar="1", type=float,help = "Whether to merge biclustres similar in samples with Jaccard index not less than the specified.")
    parser.add_argument(
        "--no_binary_save",
        action="store_true",
        help="don't save binarization results (load only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    # parser.add_argument('--plot', action='store_true', help = "show plots")

    return parser.parse_args()


def main():
    """Main entry point for UnPaSt command line interface.

    Parses command line arguments and runs the UnPaSt biclustering algorithm.
    """
    args = parse_args()

    directions = ["DOWN", "UP"]
    if args.bidirectional:
        directions = ["BOTH"]

    try:
        biclusters = unpast(
            args.exprs,
            out_dir=args.out_dir,
            binary_dir=args.binary_dir,
            no_binary_save=args.no_binary_save,
            ceiling=args.ceiling,
            bin_method=args.binarization,
            clust_method=args.clustering,
            pval=args.pval,
            directions=directions,
            min_n_samples=args.min_n_samples,
            show_fits=[],
            modularity=args.modularity,
            similarity_cutoffs=args.similarity_cutoffs,  # for Louvain
            ds=args.ds,
            dch=args.dch,
            rpath=args.rpath,
            precluster=True,  # for WGCNA
            # cluster_binary = False,
            # merge = args.merge,
            seed=args.seed,
            # plot_all = args.plot,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(e)
        raise


if __name__ == "__main__":
    main()

"""Cluster binarized genes"""

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sknetwork
from scipy.sparse import csr_matrix
from sknetwork.clustering import get_modularity

from unpast.utils.logs import get_logger, log_function_duration

logger = get_logger(__name__)

RSCRIPTS_DIR = (Path(__file__).parent.parent / "rscripts").resolve()


@log_function_duration(name="WGCNA Iterative feature clustering")
def run_WGCNA_iterative(
    binarized_expressions,
    paths,
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    rscr_path=False,
    rpath="",
):
    """Run WGCNA clustering iteratively until all features are clustered or stopping condition is met.

    Args:
        binarized_expressions (DataFrame): binary expression matrix with features as rows, samples as columns
        paths (ProjectPaths): project paths object containing paths for temporary files
        deepSplit (int): WGCNA parameter controlling module splitting sensitivity (0-4)
        detectCutHeight (float): WGCNA height cutoff for merging modules (0-1)
        nt (str): WGCNA network type ("signed_hybrid", "signed", "unsigned")
        max_power (int): maximum soft thresholding power to test in WGCNA
        precluster (bool): whether to perform pre-clustering before WGCNA
        rscr_path (bool): whether to use custom R script path
        rpath (str): path to R installation

    Returns:
        tuple: (modules, not_clustered) where
            - modules: list of feature modules/clusters found
            - not_clustered: list of features that could not be clustered
    """

    not_clustered = binarized_expressions.columns.values
    binarized_expressions_ = binarized_expressions.loc[:, :].copy()
    stop_condition = False

    modules = []
    i = 0
    while len(not_clustered) >= 3 and not stop_condition:
        binarized_expressions_ = binarized_expressions_.loc[:, not_clustered]

        m, not_clustered = run_WGCNA(
            binarized_expressions_,
            paths=paths,
            deepSplit=deepSplit,
            detectCutHeight=detectCutHeight,
            nt=nt,
            max_power=max_power,
            precluster=precluster,
            rscr_path=rscr_path,
            rpath=rpath,
        )
        logger.debug(
            f"WGCNA iteration {i}, modules:{len(m)}, not clustered:{len(not_clustered)}"
        )
        modules += m
        # stop when the number of not clustred samples does not change
        if len(m) == 0:
            stop_condition = True
            logger.debug(f"WGCNA iterations terminated at step {i}")

        i += 1
    return (modules, not_clustered)


@log_function_duration(name="WGCNA feature clustering")
def run_WGCNA(
    binarized_expressions,
    paths,
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    rscr_path=False,
    rpath="",
):
    """Run WGCNA (Weighted Gene Co-expression Network Analysis) clustering on binarized expression data.

    Args:
        binarized_expressions (DataFrame): binary expression matrix with features as rows, samples as columns
        paths (ProjectPaths): project paths object containing paths for temporary files
        deepSplit (int): WGCNA parameter controlling module splitting sensitivity (0-4)
        detectCutHeight (float): WGCNA height cutoff for merging modules (0-1)
        nt (str): WGCNA network type ("signed_hybrid", "signed", "unsigned")
        max_power (int): maximum soft thresholding power to test in WGCNA
        precluster (bool): whether to perform pre-clustering before WGCNA
        rscr_path (bool): whether to use custom R script path
        rpath (str): path to R installation

    Returns:
        tuple: (modules, not_clustered) where
            - modules: list of feature modules/clusters found by WGCNA
            - not_clustered: list of features that could not be clustered
    """
    # create unique suffix for tmp files

    fname = paths.get_wgcna_tmp_file()

    logger.debug(f"WGCNA pre-clustering: {precluster}")
    if precluster:
        precluster = "T"
    else:
        precluster = "F"

    deepSplit = int(deepSplit)
    if deepSplit not in [0, 1, 2, 3, 4]:
        logger.error("deepSplit must be 1,2,3 or 4. See WGCNA documentation.")
        return ([], [])
    if not 0 < detectCutHeight < 1:
        logger.error(
            "detectCutHeight must be between 0 and 1. See WGCNA documentation."
        )
        return ([], [])
    logger.debug(f"Running WGCNA for {fname} ...")
    if not rscr_path:
        rscr_path = str(RSCRIPTS_DIR / "run_WGCNA.R")

    binarized_expressions_ = binarized_expressions.loc[:, :].copy()

    # add suffixes to duplicated feature names
    feature_names = binarized_expressions.columns.values
    duplicated_feature_ndxs = np.arange(binarized_expressions.shape[1])[
        binarized_expressions.columns.duplicated()
    ]

    if len(duplicated_feature_ndxs) > 0:
        new_feature_names = []
        for i in range(len(feature_names)):
            fn = feature_names[i]
            if i in duplicated_feature_ndxs:
                fn = str(fn) + "*" + str(i)
            new_feature_names.append(fn)
        logger.info(
            f"{len(duplicated_feature_ndxs)} duplicated feature names detected."
        )
        dup_fn_mapping = dict(zip(new_feature_names, feature_names))
        binarized_expressions_.columns = new_feature_names

    # replace spaces in feature names
    # otherwise won't parse R output
    feature_names = binarized_expressions.columns.values
    feature_names_with_space = [x for x in feature_names if " " in x]
    if len(feature_names_with_space) > 0:
        logger.debug(
            f"feature names containing spaces (will be replaced):{len(feature_names_with_space)}"
        )
        fn_mapping = {}
        fn_mapping_back = {}
        for fn in feature_names:
            if " " in fn:
                fn_ = fn.replace(" ", "_")
                fn_mapping[fn] = fn_
                fn_mapping_back[fn_] = fn
        binarized_expressions_ = binarized_expressions.rename(
            fn_mapping, axis="columns"
        )

    # save binarized expression to a file
    binarized_expressions_.to_csv(fname, sep="\t")

    # run Rscript
    if len(rpath) > 0:
        rpath = rpath + "/"

    r_cmd_args = [
        rpath + "Rscript",
        rscr_path,
        fname,
        str(deepSplit),
        str(detectCutHeight),
        nt,
        str(max_power),
        precluster,
    ]
    logger.debug("R command line: '" + " ".join(r_cmd_args) + "'")

    process = subprocess.Popen(
        r_cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()

    # log stdout and stderr
    if len(stdout) > 0:
        if len(stdout) > 100:
            stdout = str(stdout[:100] + b"...") + "(truncated)"
        else:
            stdout = str(stdout)
        logger.debug(f"WGCNA stdout: {stdout}")
    if len(stderr) > 0:
        logger.warning(f"WGCNA stderr: {stderr}")

    module_file = fname.replace(".tsv", ".modules.tsv")
    try:
        modules_df = pd.read_csv(module_file, sep="\t", index_col=0)
    except Exception as e:
        logger.warning(f"Failed to read output WGCNA file {module_file}, error: {e}")
        logger.warning(
            "- that may be ok on some inputs. But if you don't have R and WGCNA installed,"
            " please use the default -c Louvain method instead of -c WGCNA."
        )
        modules_df = pd.DataFrame.from_dict({})
        # TODO: avoid error in case of small error

    # read WGCNA output
    modules = []
    not_clustered = []
    module_dict = modules_df.T.to_dict()
    for i in module_dict.keys():
        genes = module_dict[i]["genes"].strip().split()
        # change feature names if they were modified

        # return spaces in feature names back if necessary
        if len(feature_names_with_space) > 0:
            for j in range(len(genes)):
                if genes[j] in fn_mapping_back.keys():
                    genes[j] = fn_mapping_back[genes[j]]
        # remove suffixes from duplicated feature names
        if len(duplicated_feature_ndxs) > 0:
            for j in range(len(genes)):
                if genes[j] in dup_fn_mapping.keys():
                    genes[j] = dup_fn_mapping[genes[j]]

        if i == 0:
            not_clustered = genes
        else:
            modules.append(genes)

    # remove WGCNA input and output files
    try:
        os.remove(module_file)
        paths.clear_wgcna_tmp_files()
    except Exception as e:
        logger.debug(f"Failed to remove WGCNA temporary files: {e}")

    logger.debug(
        f"Detected modules: {len(modules)}, not clustered features {len(not_clustered)} "
    )

    return (modules, not_clustered)


def run_Louvain(*args, **kwargs):
    """Run Louvain community detection clustering on similarity matrix.
    Wrapper for run_sknetwork_clustering.
    """
    return run_sknetwork_clustering("Louvain", *args, **kwargs)


def run_Leiden(*args, **kwargs):
    """Run Leiden community detection clustering on similarity matrix.
    Wrapper for run_sknetwork_clustering.
    """
    return run_sknetwork_clustering("Leiden", *args, **kwargs)


def _load_algo_cls(clust_method):
    """Load clustering algorithm class from sknetwork."""

    if clust_method == "Leiden":
        try:
            from sknetwork.clustering import Leiden

            return Leiden
        except ImportError:
            logger.warning(
                "Leiden requires sknetwork >= 0.32.1."
                f" Found {sknetwork.__version__}."
                " Falling back to Louvain."
            )
            from sknetwork.clustering import Louvain

            return Louvain
    elif clust_method == "Louvain":
        from sknetwork.clustering import Louvain

        return Louvain
    else:
        raise ValueError(f"Unknown clustering method: {clust_method}")


def _cluster_at_cutoff(algo_cls, similarity, cutoff, modularity_measure):
    """Binarize similarity matrix and run clustering at given cutoff.

    Returns:
        tuple: (labels, modularity_score, feature_names)
    """
    sim_binary = similarity.copy()
    sim_binary[sim_binary < cutoff] = 0
    sim_binary[sim_binary != 0] = 1

    # Keep only features with at least one connection
    row_sums = sim_binary.sum()
    connected = row_sums[row_sums > 0].index
    sim_binary = sim_binary.loc[connected, connected]

    feature_names = sim_binary.index.values
    sparse_matrix = csr_matrix(sim_binary).astype("bool")

    labels = algo_cls(modularity=modularity_measure).fit_predict(sparse_matrix)
    modularity_score = get_modularity(sparse_matrix, labels)

    # Bugfix for Louvain: if all similarities are 1, put everything in one cluster
    if sim_binary.min().min() == 1:
        labels = np.zeros(len(labels), dtype=int)

    return labels, modularity_score, feature_names


def _find_knee(cutoffs, modularities):
    """Find optimal cutoff using knee detection. Returns (cutoff, Q) or (None, None)."""
    from kneed import KneeLocator

    direction = "increasing" if modularities[0] < modularities[-1] else "decreasing"
    logger.debug(f"curve type: {direction}")

    try:
        kn = KneeLocator(
            cutoffs, modularities, curve="concave", direction=direction, online=True
        )
        return kn.knee, kn.knee_y
    except Exception as e:
        logger.error(f"Failed to identify similarity cutoff, error: {e}")
        logger.info(f"Similarity cutoff: set to {cutoffs[0]}")
        logger.info(f"Modularity: {modularities}")
        return None, None


def _find_first_above_threshold(cutoffs, modularities, threshold):
    """Find first cutoff where modularity >= threshold. Returns (cutoff, Q) or (None, None)."""
    for i, mod in enumerate(modularities):
        if mod >= threshold:
            return cutoffs[i], mod
    return None, None


@log_function_duration(name="Sknetwork feature clustering")
def run_sknetwork_clustering(
    clust_method,
    similarity,
    similarity_cutoffs=np.arange(0.33, 0.95, 0.05),
    m=False,
    plot=False,
    modularity_measure="newman",
    legacy_m_labels=True,
):
    """Run Louvain/Leiden clustering on similarity matrix.

    Scans similarity thresholds, binarizes the matrix at each threshold,
    runs clustering, and selects optimal threshold via knee detection.

    Args:
        clust_method: "Louvain" or "Leiden"
        similarity: feature similarity matrix (DataFrame)
        similarity_cutoffs: thresholds to test
        m: optional modularity threshold (uses lowest cutoff achieving it)
        plot: whether to plot modularity curve
        modularity_measure: "newman" or "dugue"
        legacy_m_labels: if True (default), when m parameter triggers a lower cutoff,
            clustering labels are taken from the knee-detected cutoff (legacy behavior).
            If False, labels are taken from the threshold cutoff (correct behavior).

    Returns:
        (modules, not_clustered, best_cutoff)
    """
    if similarity.shape[0] == 0:
        logger.error("no features to cluster")
        return [], [], None

    logger.debug(f"modularity: {modularity_measure}")
    algo_cls = _load_algo_cls(clust_method)

    # Phase 1: Scan cutoffs to find modularities
    modularities = []
    for cutoff in similarity_cutoffs:
        _, Q, _ = _cluster_at_cutoff(algo_cls, similarity, cutoff, modularity_measure)
        modularities.append(Q)

    # Phase 2: Select optimal cutoff
    if len(similarity_cutoffs) == 1:
        best_cutoff, best_Q = similarity_cutoffs[0], modularities[0]
        label_cutoff = best_cutoff

    elif len(set(modularities)) == 1:
        best_cutoff, best_Q = similarity_cutoffs[-1], modularities[-1]
        label_cutoff = best_cutoff

    else:
        knee_cutoff, knee_Q = _find_knee(similarity_cutoffs, modularities)

        if knee_cutoff is not None:
            best_cutoff, best_Q = knee_cutoff, knee_Q
        else:
            best_cutoff, best_Q = similarity_cutoffs[0], np.nan
            if plot:
                plt.plot(similarity_cutoffs, modularities, "bx-")
                plt.xlabel("similarity cutoff")
                plt.ylabel("modularity")
                plt.show()

        label_cutoff = best_cutoff

        # Check if modularity threshold gives a lower cutoff
        if m:
            thresh_cutoff, thresh_Q = _find_first_above_threshold(
                similarity_cutoffs, modularities, m
            )
            if thresh_cutoff is not None and thresh_cutoff < best_cutoff:
                best_cutoff, best_Q = thresh_cutoff, thresh_Q
                if not legacy_m_labels:
                    label_cutoff = thresh_cutoff

    # Phase 3: Run clustering at selected cutoff
    labels, _, feature_names = _cluster_at_cutoff(
        algo_cls, similarity, label_cutoff, modularity_measure
    )

    # Plot if requested
    if plot and len(similarity_cutoffs) > 1:
        plt.plot(similarity_cutoffs, modularities, "bx-")
        plt.vlines(
            label_cutoff, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", color="red"
        )
        plt.xlabel("similarity cutoff")
        plt.ylabel("modularity")
        plt.show()

    # Convert labels to modules
    modules, not_clustered = [], []
    for label in set(labels):
        features = feature_names[labels == label]
        if len(features) > 1:
            modules.append(features)
        else:
            not_clustered.append(features[0])

    logger.debug(
        f"Detected modules: {len(modules)}, not clustered features {len(not_clustered)} "
    )
    logger.debug(f"- similarity cutoff: {label_cutoff:.2f}")
    logger.debug(f"- modularity: {best_Q:.3f}")
    return modules, not_clustered, label_cutoff

"""Cluster binarized genes"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from time import time
from pathlib import Path

import matplotlib.pyplot as plt

RSCRIPTS_DIR = (Path(__file__).parent.parent / "rscripts").resolve()


def run_WGCNA_iterative(
    binarized_expressions,
    tmp_prefix="",
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    verbose=False,
    rscr_path=False,
    rpath="",
):
    """Run WGCNA clustering iteratively until all features are clustered or stopping condition is met.

    Args:
        binarized_expressions (DataFrame): binary expression matrix with features as rows, samples as columns
        tmp_prefix (str): prefix for temporary files during WGCNA execution
        deepSplit (int): WGCNA parameter controlling module splitting sensitivity (0-4)
        detectCutHeight (float): WGCNA height cutoff for merging modules (0-1)
        nt (str): WGCNA network type ("signed_hybrid", "signed", "unsigned")
        max_power (int): maximum soft thresholding power to test in WGCNA
        precluster (bool): whether to perform pre-clustering before WGCNA
        verbose (bool): whether to print progress information
        rscr_path (bool): whether to use custom R script path
        rpath (str): path to R installation

    Returns:
        tuple: (modules, not_clustered) where
            - modules: list of feature modules/clusters found
            - not_clustered: list of features that could not be clustered
    """

    t0 = time()

    not_clustered = binarized_expressions.columns.values
    binarized_expressions_ = binarized_expressions.loc[:, :].copy()
    stop_condition = False

    modules = []
    i = 0
    while len(not_clustered) >= 3 and not stop_condition:
        binarized_expressions_ = binarized_expressions_.loc[:, not_clustered]

        m, not_clustered = run_WGCNA(
            binarized_expressions_,
            tmp_prefix=tmp_prefix,
            deepSplit=deepSplit,
            detectCutHeight=detectCutHeight,
            nt=nt,
            max_power=max_power,
            precluster=precluster,
            verbose=verbose,
            rscr_path=rscr_path,
            rpath=rpath,
        )
        if verbose:
            print(
                "\t\t\tWGCNA iteration %s, modules:%s, not clustered:%s"
                % (i, len(m), len(not_clustered)),
                file=sys.stdout,
            )
        modules += m
        # stop when the number of not clustred samples does not change
        if len(m) == 0:
            stop_condition = True
            if verbose:
                print("\t\t\tWGCNA iterations terminated at step ", i, file=sys.stdout)

        i += 1
    return (modules, not_clustered)


def run_WGCNA(
    binarized_expressions,
    tmp_prefix="",
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    verbose=False,
    rscr_path=False,
    rpath="",
):
    """Run WGCNA (Weighted Gene Co-expression Network Analysis) clustering on binarized expression data.

    Args:
        binarized_expressions (DataFrame): binary expression matrix with features as rows, samples as columns
        tmp_prefix (str): prefix for temporary files during WGCNA execution
        deepSplit (int): WGCNA parameter controlling module splitting sensitivity (0-4)
        detectCutHeight (float): WGCNA height cutoff for merging modules (0-1)
        nt (str): WGCNA network type ("signed_hybrid", "signed", "unsigned")
        max_power (int): maximum soft thresholding power to test in WGCNA
        precluster (bool): whether to perform pre-clustering before WGCNA
        verbose (bool): whether to print progress information
        rscr_path (bool): whether to use custom R script path
        rpath (str): path to R installation

    Returns:
        tuple: (modules, not_clustered) where
            - modules: list of feature modules/clusters found by WGCNA
            - not_clustered: list of features that could not be clustered
    """
    t0 = time()
    # create unique suffix for tmp files
    from datetime import datetime

    now = datetime.now()
    fname = "tmpWGCNA_" + now.strftime("%y.%m.%d_%H:%M:%S") + ".tsv"
    if len(tmp_prefix) > 0:
        fname = tmp_prefix + "." + fname

    # create dir if required
    if os.path.dirname(fname) != "":
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
            print(
                "\t\tCreated directory for WGCNA tmp files:",
                os.path.dirname(fname),
                file=sys.stdout,
            )

    if verbose:
        print("\t\tWGCNA pre-clustering:", precluster, file=sys.stdout)
    if precluster:
        precluster = "T"
    else:
        precluster = "F"

    deepSplit = int(deepSplit)
    if not deepSplit in [0, 1, 2, 3, 4]:
        print("deepSplit must be 1,2,3 or 4. See WGCNA documentation.", file=sys.stderr)
        return ([], [])
    if not 0 < detectCutHeight < 1:
        print(
            "detectCutHeight must be between 0 and 1. See WGCNA documentation.",
            file=sys.stderr,
        )
        return ([], [])
    if verbose:
        print("\tRunning WGCNA for", fname, "...", file=sys.stdout)
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
        print(
            "\t\t%s duplicated feature names detected." % len(duplicated_feature_ndxs),
            file=sys.stdout,
        )
        dup_fn_mapping = dict(zip(new_feature_names, feature_names))
        binarized_expressions_.columns = new_feature_names

    # replace spaces in feature names
    # otherwise won't parse R output
    feature_names = binarized_expressions.columns.values
    feature_names_with_space = [x for x in feature_names if " " in x]
    if len(feature_names_with_space) > 0:
        if verbose:
            print(
                "\t\tfeature names containing spaces (will be replaced):%s"
                % len(feature_names_with_space),
                file=sys.stdout,
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

    if verbose:
        print("\tR command line:", file=sys.stdout)
        print(
            "\t"
            + " ".join(
                [
                    rpath + "Rscript",
                    rscr_path,
                    fname,
                    str(deepSplit),
                    str(detectCutHeight),
                    nt,
                    str(max_power),
                    precluster,
                ]
            ),
            file=sys.stdout,
        )

    process = subprocess.Popen(
        [
            rpath + "Rscript",
            rscr_path,
            fname,
            str(deepSplit),
            str(detectCutHeight),
            nt,
            str(max_power),
            precluster,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    # stdout = stdout.decode('utf-8')
    module_file = fname.replace(".tsv", ".modules.tsv")  # stdout.rstrip()
    try:
        modules_df = pd.read_csv(module_file, sep="\t", index_col=0)
    except:
        # print("WGCNA output:", stdout, file = sys.stdout)
        stderr = stderr.decode("utf-8")
        if verbose:
            print("WGCNA error:", stderr, file=sys.stdout)
        modules_df = pd.DataFrame.from_dict({})
    if verbose:
        print(
            "\tWGCNA runtime: modules detected in {:.2f} s.".format(time() - t0),
            file=sys.stdout,
        )

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
        os.remove(fname)
        os.remove(module_file)
    except:
        pass

    if verbose:
        print(
            "\tmodules: {}, not clustered features {} ".format(
                len(modules), len(not_clustered)
            ),
            file=sys.stdout,
        )
        # print(stdout,file = sys.stdout)
        if len(stderr) > 0:
            print(stderr, file=sys.stderr)

    return (modules, not_clustered)


def run_Louvain(
    similarity,
    similarity_cutoffs=np.arange(0.33, 0.95, 0.05),
    m=False,
    verbose=True,
    plot=False,
    modularity_measure="newman",
):
    """Run Louvain community detection clustering on similarity matrix.

    Args:
        similarity (DataFrame): feature similarity matrix
        similarity_cutoffs (array): range of similarity thresholds to test for clustering
        m (bool): whether to return additional modularity information
        verbose (bool): whether to print progress information
        plot (bool): whether to generate plots of modularity vs cutoffs
        modularity_measure (str): modularity measure to use ("newman", "dugue")

    Returns:
        tuple: (modules, not_clustered, best_Q) where
            - modules: list of feature modules/clusters found
            - not_clustered: list of features that could not be clustered
            - best_Q: best modularity score achieved
    """
    t0 = time()
    if similarity.shape[0] == 0:
        print("no features to cluster", file=sys.stderr)
        return [], [], None

    if verbose:
        print("\tRunning Louvain ...", file=sys.stdout)
        print("\t\tmodularity:", modularity_measure, file=sys.stdout)

    from sknetwork.clustering import Louvain
    import sknetwork

    try:
        from sknetwork.clustering import modularity

        old_sknetwork_version = True
    except:
        from sknetwork.clustering import get_modularity

        print("sknetwork version used:", sknetwork.__version__, file=sys.stderr)
        old_sknetwork_version = False
    try:
        from scipy.sparse.csr import csr_matrix
    except:
        from scipy.sparse import csr_matrix

    modularities = []
    feature_clusters = {}
    best_Q = np.nan
    for cutoff in similarity_cutoffs:
        # scan the whole range of similarity cutoffs
        # e.g. [1/4;9/10] with step 0.5
        sim_binary = similarity.copy()
        sim_binary[sim_binary < cutoff] = 0
        sim_binary[sim_binary != 0] = 1
        rsums = sim_binary.sum()
        non_zero_features = rsums[rsums > 0].index
        sim_binary = sim_binary.loc[non_zero_features, non_zero_features]
        gene_names = sim_binary.index.values
        sparse_matrix = csr_matrix(sim_binary)

        if old_sknetwork_version:
            labels = Louvain(modularity=modularity_measure).fit_transform(sparse_matrix)
            Q = modularity(sparse_matrix, labels)
        else:
            sparse_matrix = sparse_matrix.astype("bool")
            labels = Louvain(modularity=modularity_measure).fit_predict(sparse_matrix)
            Q = get_modularity(sparse_matrix, labels)

        modularities.append(Q)
        # if binary similarity matrix contains no zeroes
        # bugfix for Louvain()
        if sim_binary.min().min() == 1:
            labels = np.zeros(len(labels))
        feature_clusters[cutoff] = labels

    # if similarity_cutoffs contains only one value, choose it as best_cutoff
    if len(similarity_cutoffs) == 1:
        best_cutoff = similarity_cutoffs[0]
        best_Q = Q

    # find best_cutoff automatically
    else:
        # check if modularity(cutoff)=const
        if len(set(modularities)) == 1:
            best_cutoff = similarity_cutoffs[-1]
            best_Q = modularities[-1]
            labels = feature_clusters[best_cutoff]

        #  if modularity!= const, scan the whole range of similarity cutoffs
        #  e.g. [1/4;9/10] with step 0.05
        else:
            # find the knee point in the dependency modularity(similarity curoff)
            from kneed import KneeLocator

            # define the type of the curve
            curve_type = "increasing"
            if modularities[0] >= modularities[-1]:
                curve_type = "decreasing"
            if verbose:
                print("\tcurve type:", curve_type, file=sys.stdout)
            # detect knee and choose the one with the highest modularity
            try:
                kn = KneeLocator(
                    similarity_cutoffs,
                    modularities,
                    curve="concave",
                    direction=curve_type,
                    online=True,
                )
                best_cutoff = kn.knee
                best_Q = kn.knee_y
                labels = feature_clusters[best_cutoff]
            except:
                print("Failed to identify similarity cutoff", file=sys.stderr)
                print(
                    "Similarity cutoff: set to ", similarity_cutoffs[0], file=sys.stdout
                )
                best_cutoff = similarity_cutoffs[0]
                best_Q = np.nan
                print("Modularity:", modularities, file=sys.stdout)
                if plot:
                    plt.plot(similarity_cutoffs, modularities, "bx-")
                    plt.xlabel("similarity cutoff")
                    plt.ylabel("modularity")
                    plt.show()
                    # return [], [], None
            if m:
                # if upper threshold for modularity m is specified
                # chose the lowest similarity cutoff at which modularity reaches >= m
                best_cutoff_m = 1
                for i in range(len(modularities)):
                    if modularities[i] >= m:
                        best_cutoff_m = similarity_cutoffs[i]
                        best_Q_m = modularities[i]
                        labels_m = feature_clusters[best_cutoff]
                        break
                if best_cutoff_m < best_cutoff:
                    best_cutoff = best_cutoff_m
                    best_Q = best_Q_m
                    labels = labels_m

    if plot and len(similarity_cutoffs) > 1:
        plt.plot(similarity_cutoffs, modularities, "bx-")
        plt.vlines(
            best_cutoff, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", color="red"
        )
        plt.xlabel("similarity cutoff")
        plt.ylabel("modularity")
        plt.show()

    modules = []
    not_clustered = []

    for label in set(labels):
        ndx = np.argwhere(labels == label).reshape(1, -1)[0]
        genes = gene_names[ndx]
        if len(genes) > 1:
            modules.append(genes)
        else:
            not_clustered.append(genes[0])
    if verbose:
        print(
            "\tLouvain runtime: modules detected in {:.2f} s.".format(time() - t0),
            file=sys.stdout,
        )
        print(
            "\tmodules: {}, not clustered features {} ".format(
                len(modules), len(not_clustered)
            ),
            file=sys.stdout,
        )
        print(
            "\t\tsimilarity cutoff: {:.2f} modularity: {:.3f}".format(
                best_cutoff, best_Q
            ),
            file=sys.stdout,
        )
    return modules, not_clustered, best_cutoff

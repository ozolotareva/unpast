######## Make biclusters #########
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from unpast.core.feature_clustering import run_Louvain
from unpast.utils.logs import get_logger, log_function_duration
from unpast.utils.similarity import get_similarity_jaccard
from unpast.utils.statistics import calc_SNR

logger = get_logger(__name__)


def cluster_samples(data, min_n_samples=5, seed=0, method="kmeans"):
    """Cluster samples into bicluster and background groups using unsupervised methods.

    Args:
        data (DataFrame): expression data with samples as rows and features as columns
        min_n_samples (int): minimum number of samples required for each cluster
        seed (int): random seed for reproducible clustering
        method (str): clustering method to use ("kmeans", "Jenks", "ward")

    Returns:
        dict: bicluster information containing sample indices and cluster size
    """
    # identify identify bicluster and backgound groups using 2-means
    max_n_iter = max(max(data.shape), 500)
    if method == "kmeans" or method == "Jenks":
        labels = (
            KMeans(
                n_clusters=2,
                random_state=seed,
                init="random",
                n_init=10,
                max_iter=max_n_iter,
            )
            .fit(data)
            .labels_
        )
    elif method == "ward":
        labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(data).labels_
    # elif method == "HC_ward":
    #        model = Ward(n_clusters=2).fit(data).labels_
    elif method == "GMM":
        labels = GaussianMixture(
            n_components=2,
            init_params="kmeans",
            max_iter=max_n_iter,
            n_init=5,
            covariance_type="spherical",
            random_state=seed,
        ).fit_predict(data)
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1), len(ndx0)) < min_n_samples:
        return {}
    if len(ndx1) > len(ndx0):
        samples = ndx0
    else:
        samples = ndx1

    bicluster = {"sample_indexes": set(samples), "n_samples": len(samples)}
    return bicluster


@log_function_duration(name="Building biclusters from modules")
def modules2biclusters(
    modules,
    data_to_cluster,
    method="kmeans",
    min_n_samples=5,
    min_n_genes=2,
    seed=0,
):
    """Convert feature modules to biclusters by identifying optimal sample sets for each module.

    Args:
        modules (list): list of feature modules/clusters from clustering algorithms
        data_to_cluster (DataFrame): expression data with samples as rows and features as columns
        method (str): sample clustering method to use ("kmeans", "Jenks", "ward", "GMM")
        min_n_samples (int): minimum number of samples required for each bicluster
        min_n_genes (int): minimum number of genes required for each bicluster
        seed (int): random seed for reproducible clustering

    Returns:
        dict: biclusters dictionary with bicluster IDs as keys and bicluster information as values
    """
    biclusters = {}
    i = 0

    for mid in range(0, len(modules)):
        genes = modules[mid]
        if len(genes) >= min_n_genes:
            # cluster samples in a space of selected genes
            data = data_to_cluster.loc[genes, :].T
            bicluster = cluster_samples(
                data, min_n_samples=min_n_samples, seed=seed, method=method
            )
            if len(bicluster) > 0:
                bicluster["id"] = i
                bicluster["genes"] = set(genes)
                bicluster["n_genes"] = len(bicluster["genes"])
                biclusters[i] = bicluster
                i += 1

    logger.debug(f"Identified optimal sample sets for {len(modules)} modules.")
    logger.debug(
        f"Passed biclusters (>={min_n_genes} genes, >= {min_n_samples} samples): {i - 1}"
    )

    return biclusters


def update_bicluster_data(bicluster, data):
    """Update bicluster with additional metadata including up/down-regulated genes and z-scores.

    Args:
        bicluster (dict): bicluster dictionary containing "genes" and "samples" or "sample_indexes"
        data (DataFrame): complete expression data with all features (not just binarized)

    Returns:
        dict: updated bicluster with additional fields:
            - "samples": sample names
            - "gene_indexes": gene indices
            - "genes_up": up-regulated genes
            - "genes_down": down-regulated genes
            - "SNR": SNR for absolute average z-scores of all bicluster genes
    """
    sample_names = data.columns.values
    gene_names = data.index.values

    # ensure "sample_indexes" is present
    if "sample_indexes" not in bicluster.keys():
        assert "samples" in bicluster.keys(), (
            '"samples" or "sample_indexes" of a bicluster not specified'
        )
        sample_mask = np.isin(sample_names, list(bicluster["samples"]))
        bicluster["sample_indexes"] = set(np.where(sample_mask)[0])

    # ensure "samples" is present
    if "samples" not in bicluster.keys():
        inds = list(bicluster["sample_indexes"])
        bicluster["samples"] = set(sample_names[inds])

    sample_mask = np.isin(
        np.arange(len(sample_names)), list(bicluster["sample_indexes"])
    )
    bic_samples = sample_names[sample_mask]
    bg_samples = sample_names[~sample_mask]

    gene_mask = np.isin(gene_names, list(bicluster["genes"]))
    bic_genes = gene_names[gene_mask]
    bicluster["gene_indexes"] = set(np.where(gene_mask)[0])

    # distinguish up- and down-regulated features
    m_bic = data.loc[bic_genes, bic_samples].mean(axis=1)
    m_bg = data.loc[bic_genes, bg_samples].mean(axis=1)
    genes_up = m_bic[m_bic >= m_bg].index.values
    genes_down = m_bic[m_bic < m_bg].index.values
    bicluster["genes_up"] = set(genes_up)
    bicluster["genes_down"] = set(genes_down)

    genes_up = m_bic[m_bic >= m_bg].index.values
    genes_down = m_bic[m_bic < m_bg].index.values

    # calculate average z-score for each sample
    if min(len(genes_up), len(genes_down)) > 0:  # take sign into account
        avg_zscore = (
            data.loc[genes_up, :].sum() - data.loc[genes_down, :].sum()
        ) / bicluster["n_genes"]
    else:
        avg_zscore = data.loc[sorted(bicluster["genes"]), :].mean()

    # compute SNR for average z-score for this bicluster
    bicluster["SNR"] = np.abs(
        calc_SNR(avg_zscore[bic_samples], avg_zscore[bg_samples], pd_mode=True)
    )
    return bicluster


@log_function_duration(name="Bicluster merging")
def merge_biclusters(
    biclusters, data, J=0.8, min_n_samples=5, seed=42, method="kmeans"
):
    """Merge biclusters with similar sample sets based on Jaccard similarity.

    Args:
        biclusters (dict): dictionary of biclusters to be merged
        data (DataFrame): expression data with samples as columns
        J (float): Jaccard similarity threshold for merging biclusters
        min_n_samples (int): minimum number of samples required for merged biclusters
        seed (int): random seed for reproducible clustering
        method (str): clustering method for sample grouping

    Returns:
        dict: merged biclusters dictionary with reduced redundancy
    """
    #  bicluaters -> binary -> jaccard sim
    binary_representation = {}
    N = data.shape[1]
    for i in biclusters.keys():
        b = np.zeros(N)
        s_ndx = list(biclusters[i]["sample_indexes"])
        b[s_ndx] = 1
        binary_representation[i] = b
    binary_representation = pd.DataFrame.from_dict(binary_representation)
    binary_representation.index = data.columns.values
    bic_similarity = get_similarity_jaccard(binary_representation)
    # bic_similarity[bic_similarity >= J] = 1
    # bic_similarity[bic_similarity < J] = 0
    # find groups of biclusters including the same sample sets
    merged, not_merged, similarity_cutoff = run_Louvain(
        bic_similarity, plot=False, similarity_cutoffs=[J]
    )
    if len(merged) == 0:
        logger.debug("No biclusters to merge")
        return biclusters

    merged_biclusters = {}
    # add biclusters with no changes
    for bic_id in not_merged:
        merged_biclusters[bic_id] = biclusters[bic_id]

    # merge biclusters with overlapping sample sets
    for bic_group in merged:
        bic_group = sorted(bic_group)
        logger.debug(f"merged biclusters {bic_group}")
        new_bicluster = biclusters[bic_group[0]]
        # update genes
        for bic_id in bic_group[1:]:
            bic2 = biclusters[bic_id]
            new_bicluster["genes"] = new_bicluster["genes"] | bic2["genes"]
            new_bicluster["n_genes"] = len(new_bicluster["genes"])
        # update sample set for new bicluster
        # cluster samples in a space of selected genes
        new_bicluster.update(
            cluster_samples(
                data.loc[list(new_bicluster["genes"]), :].T,
                min_n_samples=min_n_samples,
                seed=seed,
                method=method,
            )
        )
        new_bicluster["n_samples"] = len(new_bicluster["sample_indexes"])
        merged_biclusters[bic_group[0]] = new_bicluster
    return merged_biclusters


@log_function_duration(name="Creating final biclusters")
def make_biclusters(
    feature_clusters,
    binarized_data,
    data,
    merge=1,
    min_n_samples=10,
    min_n_genes=2,
    method="kmeans",
    seed=42,
    cluster_binary=False,
):
    """Create final biclusters from feature clusters by adding sample information and performing merging.

    Args:
        feature_clusters (list): list of feature modules/clusters from clustering algorithms
        binarized_data (DataFrame): binary expression matrix
        data (DataFrame): original z-score normalized expression data
        merge (float): Jaccard similarity threshold for merging biclusters (0-1)
        min_n_samples (int): minimum number of samples required for each bicluster
        min_n_genes (int): minimum number of genes required for each bicluster
        method (str): sample clustering method ("kmeans", "ward", "GMM")
        seed (int): random seed for reproducible clustering
        cluster_binary (bool): whether to cluster on binary data (True) or z-scores (False)

    Returns:
        DataFrame: final biclusters with metadata including genes, samples, SNR, and z-scores
    """

    biclusters = []

    if cluster_binary:
        data_to_cluster = binarized_data.loc[:, :].T  # binarized expressions
    else:
        data_to_cluster = data.loc[binarized_data.columns.values, :]  # z-scores

    if len(feature_clusters) == 0:
        logger.warning("No biclusters found.")
    else:
        biclusters = modules2biclusters(
            feature_clusters,
            data_to_cluster,
            method=method,
            min_n_samples=min_n_samples,
            min_n_genes=min_n_genes,
            seed=seed,
        )

        ### merge biclusters with highly similar sample sets
        if merge < 1.0:
            biclusters = merge_biclusters(
                biclusters,
                data,
                method=method,
                J=merge,
                min_n_samples=min_n_samples,
                seed=seed,
            )

        for i in list(biclusters.keys()):
            biclusters[i] = update_bicluster_data(biclusters[i], data)

    biclusters = pd.DataFrame.from_dict(biclusters).T
    # add direction
    biclusters["direction"] = "BOTH"
    biclusters.loc[
        biclusters["n_genes"] == biclusters["genes_up"].apply(len), "direction"
    ] = "UP"
    biclusters.loc[
        biclusters["n_genes"] == biclusters["genes_down"].apply(len), "direction"
    ] = "DOWN"

    # add p-value for bicluster SNR (computed for avg. zscores)
    # use the same distribution as for single features
    # biclusters["e_pval"] = biclusters.apply(lambda row: calc_e_pval(row["SNR"], row["n_samples"], null_distribution),axis=1)

    # sort and reindex
    biclusters = biclusters.sort_values(by=["SNR", "n_genes"], ascending=[False, False])
    biclusters.index = range(0, biclusters.shape[0])

    biclusters = biclusters.loc[
        :,
        [
            "SNR",
            "n_genes",
            "n_samples",
            "genes",
            "samples",
            "direction",
            "genes_up",
            "genes_down",
            "gene_indexes",
            "sample_indexes",
        ],
    ]

    return biclusters

import random
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fisher import pvalue
from scipy.stats import chi2_contingency

from unpast.core.feature_clustering import run_Louvain
from unpast.core.sample_clustering import cluster_samples, update_bicluster_data

def make_consensus_biclusters(
    biclusters_list,
    exprs,
    similarity="samples",  # can be 'both','genes','samples'
    p=0.05,  # bicluster overlap significance threshold
    min_similarity=1 / 3,  # minimal Jaccard similarity
    max_similarity=0.9,  # maximal Jaccard similarity
    frac_runs=1 / 3,
    min_n_genes=2,
    min_n_samples=5,
    min_n_times_detected=2,
    modularity_measure="newman",
    method="kmeans",  # sample clustering method
    seed=42,  # fixed by default, can be set to -1 to use random seed
    plot=False,
    figsize=(7, 7),
    labels=False,
    colorbar_off=True,
    verbose=True,
):
    # considers only significantly overlapping and best matching bicluster pairs
    # list of biclusters from several runs
    n_runs = len(biclusters_list)
    if n_runs < min_n_times_detected:
        print(
            "The number of biclusterins results should be not less than min_n_times_detected=%s"
            % (min_n_times_detected),
            file=sys.stderr,
        )
        return

    t0 = time()
    N = exprs.shape[0]

    if seed == -1:
        seed = random.randint(0, 1000000)
        print("Seed for sample clustering: %s" % (seed), file=sys.stderr)

    for i in range(n_runs):
        biclusters_list[i]["run"] = i
    biclusters = pd.concat(biclusters_list)
    bic_ids = biclusters.index.values
    biclusters["detected_n_times"] = 1

    n_bics = biclusters.shape[0]
    J_heatmap = pd.DataFrame(np.zeros((n_bics, n_bics)), index=bic_ids, columns=bic_ids)

    # add only best matches to Jaccard similarity matrix
    avg_J_sim = {}
    for i in range(n_runs):
        # bicluster is always the best match of itself,
        # similarity matrix block for output of a biclustering method w. itself is an identity matrix
        bics1 = biclusters.loc[biclusters["run"] == i, :]
        J_heatmap.loc[bics1.index.values, bics1.index.values] = np.identity(
            bics1.shape[0]
        )
        avg_J_sim[i] = {i: 1}
        for j in range(n_runs):
            if i != j:
                bics2 = biclusters.loc[biclusters["run"] == j, :]
                # find best matches between bics1 and bics2
                bm = _find_best_matching_biclusters(
                    bics1,
                    bics2,
                    exprs.shape,
                    by=similarity,
                    min_g=min_n_genes,
                    adj_pval_thr=p,
                )
                bm = bm.dropna()
                if bm.shape[0] > 0:
                    avg_J_sim[i][j] = np.mean(bm["J"])
                    df = bm.loc[:, ["bm_id", "J"]]
                    for row in df.iterrows():
                        J_heatmap.loc[row[0], row[1]["bm_id"]] += row[1]["J"] / 2
                        J_heatmap.loc[row[1]["bm_id"], row[0]] += row[1]["J"] / 2

    # if all biclusters are exactly the same
    if J_heatmap.min().min() == 1:
        # return the first bicluster
        consensus_biclusters = biclusters.iloc[[0], :].copy()
        consensus_biclusters.index = [0]
        consensus_biclusters.loc[0, "detected_n_times"] = biclusters.shape[0]
        print("all biclusters are exactly the same", file=sys.stderr)
        return consensus_biclusters

    # plot bicluster similarity heatmaps
    if plot:
        import seaborn as sns

        avg_J_sim = pd.DataFrame.from_dict(avg_J_sim)
        avg_J_sim = avg_J_sim.fillna(0)
        if avg_J_sim.shape[0] > 0:
            g = sns.clustermap(
                avg_J_sim,
                linewidths=0,
                vmin=0,
                vmax=1,
                figsize=(3, 3),
                center=0,
                annot=True,
            )
            g.ax_cbar.set_visible(False)
            plt.show()

        labels = True
        if len(bic_ids) > 20:
            labels = False
        g = sns.clustermap(
            J_heatmap,
            yticklabels=labels,
            xticklabels=labels,
            linewidths=0,
            vmin=0,
            vmax=1,
            figsize=figsize,
            center=0,
            annot=labels,
        )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        if colorbar_off:
            g.cax.set_visible(False)
        plt.show()

    t1 = time()
    if verbose:
        print("%s s for similarity matrix" % round(t1 - t0))

    # cluster biclusters by similarity
    matched, not_matched, cutoff = run_Louvain(
        J_heatmap,
        similarity_cutoffs=np.arange(min_similarity, max_similarity, 0.05),
        m=False,
        plot=plot,
        modularity_measure=modularity_measure,
    )
    t2 = time()

    # make consensus biclusters
    # for each group of matched biclusters, select consensus gene set
    # keep genes occuring at least 'min_n_times_detected' times
    min_n_times_detected = max(min_n_times_detected, int(frac_runs * n_runs))
    if verbose:
        print(
            "keep genes included in at least %s merged biclusters"
            % round(min_n_times_detected)
        )

    consensus_biclusters = []
    # for each group of matched biclusters
    for i in range(len(matched)):
        gsets = biclusters.loc[matched[i], "genes"].values
        bic_ids = biclusters.loc[matched[i], :].index.values
        detected_n_times = biclusters.loc[matched[i], "detected_n_times"].sum()
        # count gene occurencies
        gene_occurencies = {}
        for gene in set().union(*gsets):
            gene_occurencies[gene] = 0
            for gset in gsets:
                if gene in gset:
                    gene_occurencies[gene] += 1

        gene_occurencies = pd.Series(gene_occurencies).sort_values()
        passed_genes = sorted(
            gene_occurencies[
                gene_occurencies >= min(n_runs, len(matched)) * frac_runs
            ].index.values
        )
        not_passed_genes = sorted(
            gene_occurencies[
                gene_occurencies < min(n_runs, len(matched)) * frac_runs
            ].index.values
        )

        if len(passed_genes) < min_n_genes:
            # move all biclusters to not matched
            not_matched += list(matched[i])
        else:
            # cluster samples again in a subspace of a new gene set
            bicluster = cluster_samples(
                exprs.loc[passed_genes, :].T,
                min_n_samples=min_n_samples,
                seed=seed,
                method=method,
            )
            # if bicluster is not empty, add it to consenesus
            if "sample_indexes" in bicluster.keys():
                bicluster["genes"] = set(passed_genes)
                bicluster["n_genes"] = len(bicluster["genes"])
                bicluster = update_bicluster_data(bicluster, exprs)
                bicluster["detected_n_times"] = detected_n_times
                bicluster["ids"] = set(
                    bic_ids
                )  # ids of biclusters merged to the consensus biclusters
                consensus_biclusters.append(bicluster)
    consensus_biclusters = pd.DataFrame.from_records(consensus_biclusters)

    # add not matched
    not_changed_biclusters = biclusters.loc[not_matched, :]
    # not_changed_biclusters["detected_n_times"] = 1
    consensus_biclusters = pd.concat([consensus_biclusters, not_changed_biclusters])

    # add direction
    consensus_biclusters["direction"] = "BOTH"
    consensus_biclusters.loc[
        consensus_biclusters["n_genes"] == consensus_biclusters["genes_up"].apply(len),
        "direction",
    ] = "UP"
    consensus_biclusters.loc[
        consensus_biclusters["n_genes"]
        == consensus_biclusters["genes_down"].apply(len),
        "direction",
    ] = "DOWN"

    # sort
    col_order = [
        "SNR",
        "n_genes",
        "n_samples",
        "genes",
        "samples",
        "genes_up",
        "genes_down",
        "gene_indexes",
        "sample_indexes",
        "direction",
        "detected_n_times",
        "ids",
    ]
    consensus_biclusters = consensus_biclusters.sort_values(
        by=["SNR", "n_samples"], ascending=[False, True]
    )
    consensus_biclusters = consensus_biclusters.loc[:, col_order]
    consensus_biclusters = consensus_biclusters.loc[
        consensus_biclusters["n_samples"] >= min_n_samples, :
    ]
    if verbose:
        print("all consensus biclusters:", consensus_biclusters.shape[0])

    consensus_biclusters = consensus_biclusters.loc[
        consensus_biclusters["detected_n_times"] >= min_n_times_detected, :
    ]

    if verbose:
        print(
            "detected %s+ times:%s"
            % (min_n_times_detected, consensus_biclusters.shape[0])
        )

    consensus_biclusters.index = range(consensus_biclusters.shape[0])

    if verbose:
        print(
            round(time() - t2),
            "s for making consensus biclusters from consensus gene sets",
        )

    return consensus_biclusters


def calc_signif_bicluster_similarities(
    biclusters_dict,
    exprs,
    similarity="both",
    adj_pval_thr=0.05,
    plot=True,
    figsize=(10, 10),
    labels=False,
    colorbar_off=True,
):
    """WILL REPLACE _find_best_matching_biclusters(), currently is not used.
    Calculates Jaccard similarities of significanly overlapping bicluster paires based on gene and/or sample overlaps, and optionally visualize the results.

        Parameters
        ----------
        biclusters_dict : dict
            A dictionary of bicluster DataFrames. A bicluster DataFrame where each row represents a bicluster, and columns contain sets of 'genes' and 'samples' associated with the bicluster.
            Keys of the dictionary may correspond to run names.

        exprs : pd.DataFrame
            Gene expression data with rows corresponding to genes and columns to samples. Used for determining overlaps between biclusters.

        similarity : str, optional, default="both"
            The type of similarity to compute:
            - "genes": only compute similarities based on gene overlaps.
            - "samples": only compute similarities based on sample overlaps.
            - "both": compute similarities based on both gene and sample overlaps.

        adj_pval_thr : float, optional, default=0.05
            Adjusted p-value threshold for significance testing. Similarity values will be stored only for bicluster pairs which significantly overlap. Statistical significance of the overlaps is tested using the chi-squared test.

        plot : bool, optional, default=True
            If True, generate a heatmap visualization of the similarity matrix.

        figsize : tuple, optional, default=(10, 10)
            Size of the heatmap plot, used when `plot=True`.

        labels : bool, optional, default=False
            If True, display labels for the biclusters on the axes of the heatmap.

        colorbar_off : bool, optional, default=True
            If True, hide the colorbar in the heatmap visualization.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing pairwise Jaccard similarity values between biclusters, adjusted for statistical significance.
    """

    if similarity not in ["genes", "samples", "both"]:
        print(
            "Similarity must be 'genes','samples','both'. Set to 'both'.",
            file=sys.stderr,
        )
        similarity = "both"

    J_heatmap = {}
    s = set(exprs.columns.values)
    g = set(exprs.index.values)
    N_bics = len(biclusters_dict.keys())
    ### plot pairwise comparisons:
    for bic_n1 in biclusters_dict.keys():
        b1 = biclusters_dict[bic_n1]
        g1 = b1["genes"]
        s1 = b1["samples"]
        J_heatmap[bic_n1] = {}
        for bic_n2 in biclusters_dict.keys():
            b2 = biclusters_dict[bic_n2]
            g2 = b2["genes"]
            s2 = b2["samples"]
            g_overlap = g1.intersection(g2)
            g_union = g1.union(g2)
            s_overlap = s1.intersection(s2)
            s_union = s1.union(s2)
            J_g = len(g_overlap) / len(g_union)
            J_s = len(s_overlap) / len(s_union)
            J_heatmap[bic_n1][bic_n2] = 0

            # significance of gene overlap
            if similarity != "samples":
                if len(g_overlap) == 0:
                    p_g = 1
                else:
                    chi2, p_g, dof, expected = chi2_contingency(
                        [
                            [len(g_overlap), len(g1.difference(g2))],
                            [len(g2.difference(g1)), len(g.difference(g1 | g2))],
                        ]
                    )

            # significance of sample overlap
            if similarity != "genes":
                # skip if similarity==both and gene overlap is empty
                if similarity == "both" and len(g) == 0:
                    p_s = 1
                else:
                    chi2, p_s, dof, expected = chi2_contingency(
                        [
                            [len(s_overlap), len(s1.difference(s2))],
                            [len(s2.difference(s1)), len(s.difference(s1 | s2))],
                        ]
                    )

            if similarity == "genes":
                if p_g * (N_bics - 1) * N_bics / 2 < adj_pval_thr:
                    J_heatmap[bic_n1][bic_n2] = J_g
            elif similarity == "samples":
                if p_s * (N_bics - 1) * N_bics / 2 < adj_pval_thr:
                    J_heatmap[bic_n1][bic_n2] = J_s
            elif similarity == "both":  # both genes and samples considered
                # consider significant overlaps in g and s and save max. J
                if (
                    p_g * (N_bics - 1) * N_bics / 2 < adj_pval_thr
                    and len(s_overlap) > 0
                ) or (
                    p_s * (N_bics - 1) * N_bics / 2 < adj_pval_thr
                    and len(g_overlap) > 0
                ):
                    J_heatmap[bic_n1][bic_n2] = max(J_s, J_g)

    J_heatmap = pd.DataFrame.from_dict(J_heatmap)

    if plot:
        import seaborn as sns

        g = sns.clustermap(
            J_heatmap,
            yticklabels=labels,
            xticklabels=labels,
            linewidths=0,
            vmin=0,
            vmax=1,
            figsize=figsize,
            center=0,
            annot=labels,
        )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        if colorbar_off:
            g.cax.set_visible(False)
        plt.show()

    return J_heatmap

def _calc_overlap_pval(overlap, group1_only, group2_only, background, max_N=5000):
    # if sample size < max_N), use Fisher's exact
    # otherwise replacing exact Fisher's with chi2
    if overlap + group1_only + group2_only + background < max_N:
        pval = pvalue(overlap, group1_only, group2_only, background).right_tail
    else:
        chi2, pval, dof, expected = chi2_contingency(
            [[overlap, group1_only], [group2_only, background]]
        )
    return pval


def _find_best_matching_biclusters(
    bics1, bics2, sizes, by="genes", adj_pval_thr=0.05, min_g=2
):
    # takes two biluster dafaframes from read_bic_table
    # by = "genes" or "samples" or "both"
    # sizes - dimensions of input matrix (n_genes,n_samples)
    # finds best matches of bics1 biclusters among bics2 biclusters

    N_g, N_s = sizes
    n_bics1 = bics1.shape[0]
    n_bics2 = bics2.shape[0]

    best_matches = {}  # OrderedDict({})
    for row1 in bics1.iterrows():
        bic1 = row1[1]
        i1 = row1[0]

        best_matches[i1] = {}
        bm_J = 0
        bm_o = 0
        bm_adj_pval = 1
        bm_id = None

        for row2 in bics2.iterrows():
            bic2 = row2[1]
            i2 = row2[0]

            g1 = bic1["genes"]
            s1 = bic1["samples"]
            g2 = bic2["genes"]
            s2 = bic2["samples"]
            o_g = len(g1.intersection(g2))
            o_s = len(s1.intersection(s2))
            s1 = len(s1)
            s2 = len(s2)
            g1 = len(g1)
            g2 = len(g2)
            J = 0
            adj_pval = 1
            # if not by="samples", ignore overlaps with gene < min_g
            if (by != "samples" and o_g >= min_g) or by == "samples":
                if by == "genes" or by == "both":
                    g2_ = g2 - o_g  # genes exclusively in bicluster 2
                    g1_ = g1 - o_g
                    u_g = g1_ + g2_ + o_g
                    bg_g = N_g - u_g
                    J_g = o_g * 1.0 / u_g
                    if not by == "both":
                        pval_g = _calc_overlap_pval(o_g, g1_, g2_, bg_g)
                elif by == "samples" or by == "both":
                    s2_ = s2 - o_s  # samples exclusively in bicluster 2
                    s1_ = s1 - o_s
                    u_s = s1_ + s2_ + o_s
                    bg_s = N_s - u_s
                    pval_s = _calc_overlap_pval(o_s, s1_, s2_, bg_s)
                    # if p-val is high but one of the biclusters is large,
                    # try flipping the largest bicluster if it is close to 50% of the cohort
                    if pval_s > adj_pval_thr and max(s1, s2) > 0.4 * N_s:
                        if s1 > s2:  # flip s1
                            s1 = N_s - s1
                            u_s = bg_s + s2
                            o_s = s2_
                            s2_ = s2 - o_s
                            bg_s = s1_
                            s1_ = s1 - o_s
                        else:  # flip s2
                            s2 = N_s - s2
                            u_s = bg_s + s1
                            o_s = s1_
                            s1_ = s1 - o_s
                            bg_s = s2_
                            s2_ = s2 - o_s
                        assert bg_s == N_s - u_s, (
                            "i1=%s; i2=%s: bg=%s, N_s=%s, u_s=%s"
                            % (
                                i1,
                                i2,
                                bg_s,
                                N_s,
                                u_s,
                            )
                        )
                        assert u_s == o_s + s1_ + s2_, (
                            "i1=%s; i2=%s: u_s=%s, o_s=%s, s1_=%s, s2_=%s"
                            % (
                                i1,
                                i2,
                                u_s,
                                o_s,
                                s1_,
                                s2_,
                            )
                        )
                        if not by == "both":
                            # compute p-value again
                            pval_s = _calc_overlap_pval(o_s, s1_, s2_, bg_s)
                    J_s = o_s * 1.0 / u_s

                if by == "genes":
                    J = J_g
                    pval = pval_g
                    o = o_g
                elif by == "samples":
                    J = J_s
                    pval = pval_s
                    o = o_s
                else:
                    o = o_s * o_g  # bicluster overlap
                    b1_ = s1 * g1 - o  # exclusive bicluster 1 area
                    b2_ = s2 * g2 - o
                    u = o + b1_ + b2_
                    bg = N_s * N_g - u
                    J = o * 1.0 / u
                    pval = _calc_overlap_pval(o, b1_, b2_, bg)

                adj_pval = pval * n_bics2 * n_bics1
                if adj_pval < adj_pval_thr and J > 0:
                    if J > bm_J or (J == bm_J and adj_pval < bm_adj_pval):
                        bm_J = J
                        bm_adj_pval = adj_pval
                        bm_id = i2
                        bm_o = o
        best_matches[i1]["bm_id"] = bm_id
        best_matches[i1]["J"] = bm_J
        best_matches[i1]["adj_pval"] = bm_adj_pval
        if "genes" in bics1.columns and "genes" in bics2.columns:
            if bm_id:
                best_matches[i1]["shared_genes"] = bics2.loc[
                    bm_id, "genes"
                ].intersection(bics1.loc[i1, "genes"])
                best_matches[i1]["n_shared_genes"] = len(
                    best_matches[i1]["shared_genes"]
                )
                best_matches[i1]["bm_genes"] = bics2.loc[bm_id, "genes"]
                best_matches[i1]["bm_n_genes"] = bics2.loc[bm_id, "n_genes"]
            best_matches[i1]["genes"] = bics1.loc[i1, "genes"]
            best_matches[i1]["n_genes"] = bics1.loc[i1, "n_genes"]
        if "samples" in bics1.columns and "samples" in bics2.columns:
            best_matches[i1]["n_samples"] = bics1.loc[i1, "n_samples"]
            best_matches[i1]["samples"] = bics1.loc[i1, "samples"]
            if bm_id:
                best_matches[i1]["bm_n_samples"] = bics2.loc[bm_id, "n_samples"]
                best_matches[i1]["bm_samples"] = bics2.loc[bm_id, "samples"]
                best_matches[i1]["shared_samples"] = bics2.loc[
                    bm_id, "samples"
                ].intersection(bics1.loc[i1, "samples"])
                best_matches[i1]["n_shared_samples"] = len(
                    best_matches[i1]["shared_samples"]
                )

    best_matches = pd.DataFrame.from_dict(best_matches).T
    return best_matches

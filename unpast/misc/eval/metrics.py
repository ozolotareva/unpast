import sys

import numpy as np
import pandas as pd
from fisher import pvalue
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import fdrcorrection


from sklearn.metrics import adjusted_rand_score


def calculate_perfromance(
    sample_clusters_,  # data.Frame with "samples" column
    known_groups,  # dict={"classification1":{"group1":{"s1","s2",...},"group2":{...}, ...}}
    all_samples,  # set of all samples in input; needed for overlap p-value computations
    performance_measure="Jaccard",  # must be "ARI" or "Jaccard"
    adjust_pvals="B",  # ["B", "BH", False] # correction for multiple testing
    pval_cutoff=0.05,  # cutoff for p-values to select significant matches
    min_SNR=0,
    min_n_genes=False,
    min_n_samples=1,
    verbose=False,
):
    # select the sample set best matching the subtype based on p-value
    # adj. overlap p-value should be:
    # below pval_cutoff, e.g. < 0.05
    # the lowest among overlap p-values computed for this sample set vs all subtypes
    # cluster with the highest J is chosen as the best match
    if sample_clusters_ is None or sample_clusters_.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    if adjust_pvals not in ["B", "BH", False]:
        print(
            adjust_pvals,
            "is not recognized, Bonferroni method will be used,",
            file=sys.stderr,
        )
    sample_clusters = sample_clusters_[
        sample_clusters_["samples"].apply(lambda x: len(x)) >= min_n_samples
    ]
    sample_clusters["n_samples"] = sample_clusters["samples"].apply(lambda x: len(x))
    if min_SNR:
        sample_clusters = sample_clusters[sample_clusters["SNR"] >= min_SNR]
    if min_n_genes:
        sample_clusters = sample_clusters[
            sample_clusters["genes"].apply(lambda x: len(x)) >= min_n_genes
        ]

    if sample_clusters.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    best_matches = []
    performances = {}
    for cl in known_groups.keys():
        # denominator for weights
        N = 0
        for subt in known_groups[cl].keys():
            N += len(known_groups[cl][subt])
        if performance_measure == "ARI":
            pvals, is_enriched, performance = evaluate_overlaps_ARI(
                sample_clusters, known_groups[cl], all_samples
            )
        if performance_measure == "Jaccard":
            pvals, is_enriched, performance = evaluate_overlaps(
                sample_clusters, known_groups[cl], all_samples
            )
        if adjust_pvals:
            if adjust_pvals == "B":
                pvals = pvals * pvals.shape[0]
                pvals = pvals.applymap(lambda x: min(x, 1))
            elif adjust_pvals == "BH":
                pvals = apply_bh(pvals, a=pval_cutoff)
        best_match_stats = {}
        best_pval = pvals.min(axis=1)
        for subt in known_groups[cl].keys():
            w = len(known_groups[cl][subt]) / N  # weight
            subt_pval = pvals[subt]
            passed_pvals = subt_pval[subt_pval == best_pval]
            passed_pvals = subt_pval[subt_pval < pval_cutoff].index.values
            d = performance.loc[passed_pvals, subt].sort_values(ascending=False)
            if d.shape[0] == 0:
                best_match_stats[subt] = {
                    "bm_id": np.nan,
                    performance_measure: 0,
                    "weight": w,
                    "adj_pval": np.nan,
                    "is_enriched": np.nan,
                    "samples": set([]),
                    "n_samples": 0,
                }
            else:
                bm_j = d.values[0]
                d = d[d == bm_j]
                bm_id = sorted(
                    pvals.loc[d.index.values, :]
                    .sort_values(by=subt, ascending=True)
                    .index.values
                )[0]
                bm_pval = pvals.loc[bm_id, subt]
                bm_j = performance.loc[bm_id, subt]
                bm_is_enrich = is_enriched.loc[bm_id, subt]
                bm_samples = sample_clusters.loc[bm_id, "samples"]
                best_match_stats[subt] = {
                    "bm_id": bm_id,
                    performance_measure: bm_j,
                    "weight": w,
                    "adj_pval": bm_pval,
                    "is_enriched": bm_is_enrich,
                    "samples": bm_samples,
                    "n_samples": len(bm_samples),
                }

        best_match_stats = pd.DataFrame.from_dict(best_match_stats).T
        best_match_stats["classification"] = cl
        best_matches.append(best_match_stats)
        performances[cl] = sum(
            best_match_stats[performance_measure] * best_match_stats["weight"]
        )

    performances = pd.Series(performances)
    best_matches = pd.concat(best_matches, axis=0)
    return performances, best_matches


def evaluate_overlaps_ARI(biclusters, known_groups, all_elements):
    # compute exact Fisher's p-values and Jaccard overlaps for samples
    pvals = {}
    is_enriched = {}
    ARI = {}
    N = len(all_elements)
    all_elements_list = sorted(all_elements)
    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, "samples"]
        if not bic_members.intersection(all_elements) == bic_members:
            print(
                "bicluster {} elements {} are not in 'all_elements'".format(
                    i, " ".join(bic_members.difference(all_elements))
                ),
                file=sys.stderr,
            )
            bic_members = bic_members.intersection(all_elements)
    # sanity check and sorting
    group_names = list(known_groups.keys())
    sorted_group_names = [group_names[0]]  # group names ordered by group size
    for group in group_names:
        group_members = known_groups[group]
        if not group_members.intersection(all_elements) == group_members:
            print(group, "elements are not in 'all_elements'", file=sys.stderr)
            return

        if group != group_names[0]:
            for gn in range(len(sorted_group_names)):
                if len(group_members) < len(known_groups[sorted_group_names[gn]]):
                    sorted_group_names = (
                        sorted_group_names[:gn] + [group] + sorted_group_names[gn:]
                    )
                    break
                elif gn == len(sorted_group_names) - 1:
                    sorted_group_names = [group] + sorted_group_names
    # print(sorted_group_names)
    for group in sorted_group_names:
        group_members = known_groups[group]
        pvals[group] = {}
        is_enriched[group] = {}
        ARI[group] = {}
        # binary vector for target cluster
        group_binary = np.zeros(len(all_elements_list))
        for j in range(len(all_elements_list)):
            if all_elements_list[j] in group_members:
                group_binary[j] = 1
        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic["samples"]
            bic_binary = np.zeros(len(all_elements_list))
            for j in range(len(all_elements_list)):
                if all_elements_list[j] in bic_members:
                    bic_binary[j] = 1
            # calculate ARI for 2 binary vectors:
            ARI[group][i] = adjusted_rand_score(group_binary, bic_binary)

            # Fisher's exact test
            shared = len(bic_members.intersection(group_members))
            bic_only = len(bic_members.difference(group_members))
            group_only = len(group_members.difference(bic_members))
            union = shared + bic_only + group_only
            pval = pvalue(shared, bic_only, group_only, N - union)
            pvals[group][i] = pval.two_tail
            is_enriched[group][i] = False
            if pval.right_tail < pval.left_tail:
                is_enriched[group][i] = True

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    ARI = pd.DataFrame.from_dict(ARI).loc[:, sorted_group_names]
    return pvals, is_enriched, ARI


def apply_bh(df_pval, a=0.05):
    # applies BH procedure to each column of p-value table
    df_adj = {}
    for group in df_pval.columns.values:
        bh_res, adj_pval = fdrcorrection(df_pval[group].fillna(1).values, alpha=a)
        df_adj[group] = adj_pval
    df_adj = pd.DataFrame.from_dict(df_adj)
    df_adj.index = df_pval.index
    return df_adj


def evaluate_overlaps(biclusters, known_groups, all_elements, dimension="samples"):
    # compute exact Fisher's p-values and Jaccard overlaps for samples
    pvals = {}
    is_enriched = {}
    jaccards = {}
    N = len(all_elements)
    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, dimension]
        if not bic_members.intersection(all_elements) == bic_members:
            print(
                "bicluster {} elements {} are not in 'all_elements'".format(
                    i, " ".join(bic_members.difference(all_elements))
                ),
                file=sys.stderr,
            )
            bic_members = bic_members.intersection(all_elements)
    # sanity check and sorting
    group_names = list(known_groups.keys())
    sorted_group_names = [group_names[0]]  # group names ordered by group size
    for group in group_names:
        group_members = known_groups[group]
        if not group_members.intersection(all_elements) == group_members:
            print(group, "elements are not in 'all_elements'", file=sys.stderr)
            return

        if group != group_names[0]:
            for gn in range(len(sorted_group_names)):
                if len(group_members) < len(known_groups[sorted_group_names[gn]]):
                    sorted_group_names = (
                        sorted_group_names[:gn] + [group] + sorted_group_names[gn:]
                    )
                    break
                elif gn == len(sorted_group_names) - 1:
                    sorted_group_names = [group] + sorted_group_names
    # print(sorted_group_names)
    for group in sorted_group_names:
        group_members = known_groups[group]
        pvals[group] = {}
        is_enriched[group] = {}
        jaccards[group] = {}
        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic[dimension]
            shared = len(bic_members.intersection(group_members))
            bic_only = len(bic_members.difference(group_members))
            group_only = len(group_members.difference(bic_members))
            union = shared + bic_only + group_only
            pval = pvalue(shared, bic_only, group_only, N - union)
            """pvals[group][i] = 1 
            if pval.right_tail < pval.left_tail:
                pvals[group][i] = pval.right_tail
                is_enriched[group][i] = True
            # if under-representation, flip query set
            else:
                pvals[group][i] = pval.left_tail # save left-tail p-value and record that this is not enrichment
                is_enriched[group][i] = False
                bic_members = all_elements.difference(bic_members)
                shared = len(bic_members.intersection(group_members))
                union = len(bic_members.union(group_members))"""
            pvals[group][i] = pval.two_tail
            is_enriched[group][i] = False
            if pval.right_tail < pval.left_tail:
                is_enriched[group][i] = True

            jaccards[group][i] = shared / union

        # print(group,jaccards[group])

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    jaccards = pd.DataFrame.from_dict(jaccards).loc[:, sorted_group_names]
    return pvals, is_enriched, jaccards


def calc_overlap_pval(overlap, group1_only, group2_only, background, max_N=5000):
    # if sample size < max_N), use Fisher's exact
    # otherwise replacing exact Fisher's with chi2
    if overlap + group1_only + group2_only + background < max_N:
        pval = pvalue(overlap, group1_only, group2_only, background).right_tail
    else:
        chi2, pval, dof, expected = chi2_contingency(
            [[overlap, group1_only], [group2_only, background]]
        )
    return pval


def find_best_matching_biclusters(
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
                        pval_g = calc_overlap_pval(o_g, g1_, g2_, bg_g)
                elif by == "samples" or by == "both":
                    s2_ = s2 - o_s  # samples exclusively in bicluster 2
                    s1_ = s1 - o_s
                    u_s = s1_ + s2_ + o_s
                    bg_s = N_s - u_s
                    pval_s = calc_overlap_pval(o_s, s1_, s2_, bg_s)
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
                            pval_s = calc_overlap_pval(o_s, s1_, s2_, bg_s)
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
                    pval = calc_overlap_pval(o, b1_, b2_, bg)

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

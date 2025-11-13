import sys

import numpy as np
import pandas as pd
from fisher import pvalue
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import fdrcorrection


def calculate_performance(
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

    sample_clusters = _filter_sample_clusters(
        sample_clusters_,
        min_n_samples=min_n_samples,
        min_SNR=min_SNR,
        min_n_genes=min_n_genes,
    )

    if sample_clusters.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    best_matches = []
    performances = {}
    for cl in known_groups.keys():
        perf_val, best_match_stats = _process_class(
            cl,
            sample_clusters,
            known_groups[cl],
            all_samples,
            performance_measure,
            adjust_pvals,
            pval_cutoff,
        )
        best_matches.append(best_match_stats)
        performances[cl] = perf_val

    performances = pd.Series(performances)
    best_matches = pd.concat(best_matches, axis=0)
    return performances, best_matches


def _process_class(
    cl,
    sample_clusters,
    known_groups_cl,
    all_samples,
    performance_measure,
    adjust_pvals,
    pval_cutoff,
):
    """Process a single classification 'cl' and return (performance_value, best_match_stats_df).

    This extracts the logic previously inside the 'for cl in known_groups.keys()' loop.
    """
    # denominator for weights
    N = 0
    for subt in known_groups_cl.keys():
        N += len(known_groups_cl[subt])

    pvals, is_enriched, performance = _evaluate_overlaps(
        sample_clusters, known_groups_cl, all_samples, method=performance_measure
    )

    pvals = _adjust_pvals_df(pvals, adjust_pvals=adjust_pvals, pval_cutoff=pval_cutoff)
    best_match_stats = {}
    best_pval = pvals.min(axis=1)
    for subt in known_groups_cl.keys():
        w = len(known_groups_cl[subt]) / N  # weight
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

    perf_value = sum(best_match_stats[performance_measure] * best_match_stats["weight"])

    return perf_value, best_match_stats


def _filter_sample_clusters(sample_clusters_, min_n_samples, min_SNR, min_n_genes):
    """Filter sample_clusters DataFrame according to provided thresholds.

    Returns a DataFrame (possibly empty) with an added "n_samples" column.
    If input is None or empty, returns an empty DataFrame.
    """
    if sample_clusters_.shape[0] == 0:
        return pd.DataFrame()

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

    return sample_clusters


def _apply_bh(df_pval, a=0.05):
    # applies BH procedure to each column of p-value table
    df_adj = {}
    for group in df_pval.columns.values:
        bh_res, adj_pval = fdrcorrection(df_pval[group].fillna(1).values, alpha=a)
        df_adj[group] = adj_pval
    df_adj = pd.DataFrame.from_dict(df_adj)
    df_adj.index = df_pval.index
    return df_adj


def _adjust_pvals_df(pvals, adjust_pvals, pval_cutoff=0.05):
    """Adjust p-values DataFrame according to requested method.

    - If adjust_pvals is 'B', apply Bonferroni (multiply by number of tests and cap at 1).
    - If adjust_pvals is 'BH', apply Benjamini-Hochberg using _apply_bh.
    - If adjust_pvals is False or None, return pvals unchanged.
    """
    if adjust_pvals == "B":
        # Bonferroni across each row/entry: multiply p-values by number of tests (columns)
        pvals = pvals * pvals.shape[0]
        pvals = pvals.applymap(lambda x: min(x, 1))
        return pvals
    elif adjust_pvals == "BH":
        return _apply_bh(pvals, a=pval_cutoff)
    elif adjust_pvals == False:
        return pvals
    else:
        raise ValueError(
            f"Unrecognized adjust_pvals value: {adjust_pvals}, expected 'B', 'BH', or False."
        )


def _evaluate_overlaps(biclusters, known_groups, all_elements, method, dimension="samples"):
    # compute exact Fisher's p-values and Jaccard/ARI overlaps for samples
    
    assert method in ["Jaccard", "ARI"], f"Unsupported method: {method}, expected 'Jaccard' or 'ARI'."
    pvals = {}
    is_enriched = {}
    metric_vals = {}
    N = len(all_elements)

    if method == "ARI":
        all_elements_list = sorted(all_elements)

    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, dimension]
        if not bic_members.intersection(all_elements) == bic_members:
            _diff_str = " ".join(bic_members.difference(all_elements))
            print(
                f"bicluster {i} elements {_diff_str} are not in 'all_elements'",
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
        metric_vals[group] = {}

        if method == "ARI":
            # binary vector for target cluster
            group_binary = np.zeros(len(all_elements_list))
            for j in range(len(all_elements_list)):
                if all_elements_list[j] in group_members:
                    group_binary[j] = 1

        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic[dimension]
           
            if method == "ARI":
                bic_binary = np.zeros(len(all_elements_list))
                for j in range(len(all_elements_list)):
                    if all_elements_list[j] in bic_members:
                        bic_binary[j] = 1
                # calculate ARI for 2 binary vectors:
                metric_vals[group][i] = adjusted_rand_score(group_binary, bic_binary)

            # Fisher's exact test
            shared = len(bic_members.intersection(group_members))   
            bic_only = len(bic_members.difference(group_members))
            group_only = len(group_members.difference(bic_members))
            union = shared + bic_only + group_only
            pval = pvalue(shared, bic_only, group_only, N - union)


            # some commented out code in the Jaccard version above for under-representation handling
            """
            pvals[group][i] = 1 
            if pval.right_tail < pval.left_tail:
                pvals[group][i] = pval.right_tail
                is_enriched[group][i] = True
            # if under-representation, flip query set
            else:
                pvals[group][i] = pval.left_tail # save left-tail p-value and record that this is not enrichment
                is_enriched[group][i] = False
                bic_members = all_elements.difference(bic_members)
                shared = len(bic_members.intersection(group_members))
                union = len(bic_members.union(group_members))
            """
            
            pvals[group][i] = pval.two_tail
            is_enriched[group][i] = False
            if pval.right_tail < pval.left_tail:
                is_enriched[group][i] = True

            if method == "Jaccard":
                metric_vals[group][i] = shared / union

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    metric_vals = pd.DataFrame.from_dict(metric_vals).loc[:, sorted_group_names]
    return pvals, is_enriched, metric_vals

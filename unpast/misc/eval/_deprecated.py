"""
# Disabling Unused Code: make_ref_groups, make_known_groups, and compare_gene_clusters

def make_ref_groups(subtypes, annotation, exprs):
    import copy
    from collections import OrderedDict

    # prepared a dict of subtype classifications {"class1":{"subt1":[],"subt2":[]},"class2":{"subtA":[],"subtB":[]}}
    all_samples = set(exprs.columns.values).intersection(set(subtypes.index.values))
    s = sorted(all_samples)

    pam50 = make_known_groups(
        subtypes.loc[s, :], exprs.loc[:, s], target_col="PAM50", verbose=False
    )
    lum = {}
    lum["Luminal"] = pam50["LumA"].union(pam50["LumB"]).intersection(all_samples)
    scmod2 = make_known_groups(
        subtypes.loc[s, :], exprs.loc[:, s], target_col="SCMOD2", verbose=False
    )
    claudin = {}
    claudin["Claudin-low"] = set(
        subtypes.loc[subtypes["claudin_low"] == 1, :].index.values
    ).intersection(all_samples)

    ihc = {}
    for x in ["IHC_HER2", "IHC_ER", "IHC_PR"]:
        ihc[x] = set(
            annotation.loc[annotation[x] == "Positive", :].index.values
        )  # .intersection(all_samples)
    ihc["IHC_TNBC"] = set(
        annotation.loc[annotation["IHC_TNBC"] == 1, :].index.values
    )  # .intersection(all_samples)

    pam50_lum = copy.copy(pam50)
    del pam50_lum["LumA"]
    del pam50_lum["LumB"]
    pam50_lum["Luminal"] = lum["Luminal"]
    intrinsic = copy.copy(pam50_lum)
    intrinsic["Claudin-low"] = claudin["Claudin-low"]
    known_groups = OrderedDict(
        {
            "PAM50": pam50_lum,
            "Intrinsic": intrinsic,
            "PAM50_AB": pam50,
            "SCMOD2": scmod2,
            "IHC": ihc,
        }
    )
    known_groups["Luminal"] = {"Luminal": pam50_lum["Luminal"]}
    known_groups["Basal"] = {"Basal": pam50["Basal"]}
    known_groups["Her2"] = {"Her2": pam50["Her2"]}
    known_groups["LumA"] = {"LumA": pam50["LumA"]}
    known_groups["LumB"] = {"LumB": pam50["LumB"]}
    known_groups["Normal"] = {"Normal": pam50["Normal"]}
    known_groups["Claudin-low"] = {"Claudin-low": claudin["Claudin-low"]}
    known_groups["IHC_HER2"] = {"IHC_HER2": ihc["IHC_HER2"]}
    known_groups["IHC_ER"] = {"IHC_ER": ihc["IHC_ER"]}
    known_groups["IHC_PR"] = {"IHC_PR": ihc["IHC_PR"]}
    known_groups["IHC_TNBC"] = {"IHC_TNBC": ihc["IHC_TNBC"]}
    known_groups["NET_kmeans"] = {
        "NET": set(subtypes.loc[subtypes["NET_km"] == 1, :].index.values)
    }
    known_groups["NET_ward"] = {
        "NET": set(subtypes.loc[subtypes["NET_w"] == 1, :].index.values)
    }
    return known_groups, all_samples


def make_known_groups(annot, exprs, target_col="genefu_z", verbose=False):
    samples = set(exprs.columns.values).intersection(set(annot.index.values))
    if verbose:
        print("Total samples:", len(samples), file=sys.stdout)
    annot = annot.loc[list(samples), :]
    groups = set(annot.loc[:, target_col].values)

    known_groups = {}
    for group in groups:
        if group == group:
            group_samples = set(annot.loc[annot[target_col] == group, :].index.values)
            group_samples = group_samples.intersection(samples)
            if len(group_samples) > int(len(samples) / 2):
                print("take complement of ", group, file=sys.stderr)
                group_samples = samples.difference(group_samples)
            known_groups[group] = (
                group_samples  # {"set":group_samples,"complement": samples.difference(group_samples)}
            )
            if verbose:
                print(
                    group,
                    round(len(group_samples) / len(samples), 2),
                    len(group_samples),
                    len(samples.difference(group_samples)),
                )
    return known_groups

def compare_gene_clusters(bics1, bics2, N):
    # N - total number of genes
    # finds best matched B1 -> B2 and B2 -> B1
    # calculates % of matched clusters, number of genes in matched clusters,
    # and the average J index for best matches
    bm = find_best_matching_biclusters(bics1, bics2, (N, 0), by="genes")
    bm = bm.dropna()
    bm2 = find_best_matching_biclusters(bics2, bics1, (N, 0), by="genes")
    bm2 = bm2.dropna()

    if "n_shared_genes" in bm.columns:
        bm = bm.loc[bm["n_shared_genes"] > 1, :].sort_values(
            by="n_shared_genes", ascending=False
        )
    else:
        # no match -> remove all rows
        bm = bm.head(0)
    if "n_shared_genes" in bm2.columns:
        bm2 = bm2.loc[bm2["n_shared_genes"] > 1, :].sort_values(
            by="n_shared_genes", ascending=False
        )
    else:
        bm2 = bm.head(0)

    clust_similarity = {}
    # number of biclusters
    clust_similarity["n_1"] = bics1.shape[0]
    clust_similarity["n_2"] = bics2.shape[0]
    # print("% matched biclusters:",bm.shape[0]/tcga_result.shape[0],bm2.shape[0]/metabric_result.shape[0])
    clust_similarity["percent_matched_1"] = bm.shape[0] / bics1.shape[0]
    clust_similarity["percent_matched_2"] = bm2.shape[0] / bics2.shape[0]

    # print("n matched genes:",bm.loc[:,"n_shared"].sum(),bm2.loc[:,"n_shared"].sum())
    if "n_shared_genes" in bm.columns:
        clust_similarity["n_shared_genes_1"] = bm.loc[:, "n_shared_genes"].sum()
        clust_similarity["avg_bm_J_1"] = bm.loc[:, "J"].mean()
    if "n_shared_genes" in bm2.columns:
        clust_similarity["n_shared_genes_2"] = bm2.loc[:, "n_shared_genes"].sum()
        # print("avg. J:",bm.loc[:,"J"].mean(),bm2.loc[:,"J"].mean())
        clust_similarity["avg_bm_J_2"] = bm2.loc[:, "J"].mean()

    return clust_similarity, bm, bm2
"""
"""
# Disabling Unused Code Functions
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

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

def test_sample_overlap(row, sample_set, N):
    # usage:
    # biclusters_df.apply(lambda row: test_sample_overlap(row, sample_set, N),axis=1)
    # N - total number of samples in dataset
    bic_samples = row["samples"]
    o = len(sample_set.intersection(bic_samples))
    bic_only = len(bic_samples) - o
    sample_set_only = len(sample_set) - o
    bg = N - o - bic_only - sample_set_only
    p = pvalue(o, bic_only, sample_set_only, bg).right_tail
    # if p<0.001:
    #    print(p,(o,bic_only,sample_set_only,bg),row["genes"])
    return pd.Series({"pval": p, "counts": (o, bic_only, sample_set_only, bg)})

def add_sex(biclusters, males=[], females=[]):
    sample_sets = {}
    # if len(males)>0:
    sample_sets["male"] = set(males)
    # if len(females)>0:
    sample_sets["female"] = set(females)

    N = len(males) + len(females)
    dfs = []
    for sex in sample_sets.keys():
        sample_set = sample_sets[sex]
        df = biclusters.apply(
            lambda row: test_sample_overlap(row, sample_set, N), axis=1
        )
        df.columns = [sex + "." + x for x in df.columns]
        bh_res, adj_pval = fdrcorrection(df[sex + ".pval"].values, alpha=0.05)
        df[sex + ".pval_BH"] = adj_pval
        dfs.append(df)
    dfs = pd.concat(dfs, axis=1)
    dfs["sex.pval_BH"] = dfs.loc[:, ["male.pval_BH", "female.pval_BH"]].min(axis=1)
    dfs["sex"] = ""
    try:
        dfs.loc[dfs["male.pval_BH"] < 0.05, "sex"] = "male"
    except:
        pass
    try:
        dfs.loc[dfs["female.pval_BH"] < 0.05, "sex"] = "female"
    except:
        pass
    return pd.concat([biclusters, dfs], axis=1)

def bic_survival(surv_anno, samples, event="OS", surv_time="", lr=True, verbose=True):
    # surival annotation - annotation matrix with time,event, and covariates
    # samples - samples in a group, e.g. biclsuter samples
    # check  complete separation
    # if all events are either inside or outside sample group
    if not surv_time:
        surv_time = event + ".time"
    surv_data = surv_anno.copy()
    surv_data = surv_data.dropna(axis=0)

    # check zero variance columns:
    v = surv_data.var()
    for col in v.index:
        if v[col] == 0:
            if verbose:
                print(col, "with zero variance excluded", file=sys.stderr)
            surv_data = surv_data.drop(col, axis=1)

    surv_data.loc[:, "x"] = 0
    surv_data.loc[list(set(samples).intersection(set(surv_data.index.values))), "x"] = 1

    pval = np.nan
    hr, upper_95CI, lower_95CI = np.nan, np.nan, np.nan
    results = {}

    events = surv_data[event].astype(bool)

    v1 = surv_data.loc[events, "x"].var()
    v2 = surv_data.loc[~events, "x"].var()

    v3 = surv_data.loc[surv_data["x"] == 1, event].var()
    v4 = surv_data.loc[surv_data["x"] == 0, event].var()

    if v1 == 0 or v2 == 0:
        if verbose:
            in_bic = surv_data.loc[surv_data["x"] == 1, :].shape[0]
            in_bg = surv_data.loc[surv_data["x"] == 0, :].shape[0]
            print(
                "perfect separation for biclsuter of  %s/%s samples" % (in_bic, in_bg),
                "variances: {:.2f} {:.2f}".format(v1, v2),
                file=sys.stderr,
            )
    if v3 == 0:
        if verbose:
            print(
                "zero variance for events in group; all events are ",
                set(surv_data.loc[surv_data["x"] == 1, event].values),
            )
    if v4 == 0:
        if verbose:
            print(
                "zero variance for events in background; all events are ",
                set(surv_data.loc[surv_data["x"] == 0, event].values),
            )

    # check variance of covariates in event groups
    exclude_covars = []
    for c in [x for x in surv_data.columns.values if not x in ["x", event, surv_time]]:
        if surv_data.loc[events, c].var() == 0:
            exclude_covars.append(c)
            print("\t", c, "variance is 0 in event group", file=sys.stdout)
        if surv_data.loc[~events, c].var() == 0:
            exclude_covars.append(c)
            print("\t", c, "variance is 0 in no-event group", file=sys.stdout)
    # if len(exclude_covars)>0:
    #    cols = surv_data.columns.values
    #    cols = [x for x in cols if not x in exclude_covars]
    #    surv_data = surv_data.loc[:,cols]

    else:
        try:
            cph = CoxPHFitter()
            res = cph.fit(
                surv_data, duration_col=surv_time, event_col=event, show_progress=False
            )
            res_table = res.summary
            res_table = res_table  # .sort_values("p")
            pval = res_table.loc["x", "p"]
            hr = res_table.loc["x", "exp(coef)"]
            upper_95CI = res_table.loc["x", "exp(coef) upper 95%"]
            lower_95CI = res_table.loc["x", "exp(coef) lower 95%"]
        except:
            pass

    results = {
        "p_value": pval,
        "HR": hr,
        "upper_95CI": upper_95CI,
        "lower_95CI": lower_95CI,
    }
    # Log-rank test
    if lr:
        bic = surv_data.loc[surv_data["x"] == 1, :]
        bg = surv_data.loc[surv_data["x"] == 0, :]

        lr_result = logrank_test(
            bic.loc[:, surv_time],
            bg.loc[:, surv_time],
            event_observed_A=bic.loc[:, event],
            event_observed_B=bg.loc[:, event],
        )
        results["LogR_p_value"] = lr_result.p_value

    return results


def add_survival(
    biclusters,  # dataframes with biclustes
    sample_data,  # sample annotation
    event="OS",
    surv_time="",  # event and time column names
    covariates=[],
    min_n_events=5,
    verbose=True,
):
    # if too few events, add na columns
    if sample_data[event].sum() < min_n_events:
        df = biclusters.copy()
        for col in [
            ".p_value",
            ".p_value_BH",
            ".HR",
            ".upper_95CI",
            ".lower_95CI",
            ".LogR_p_value",
            ".LogR_p_value_BH",
        ]:
            df[event + col] = np.nan
        return df
    if not surv_time:
        surv_time = event + ".time"
    surv_results = {}
    for bic in biclusters.iterrows():
        sample_set = bic[1]["samples"]
        surv_data = sample_data.loc[:, covariates + [event, surv_time]]

        surv_results[bic[0]] = bic_survival(
            surv_data, sample_set, event=event, surv_time=surv_time, verbose=verbose
        )
        if "pval" in surv_results[bic[0]].keys():
            if np.isnan(surv_results[bic[0]]["pval"]):
                print(
                    "failed to fit CPH model for %s ~ bicluster %s" % (event, bic[0]),
                    file=sys.stderr,
                )

    surv_results = pd.DataFrame.from_dict(surv_results).T
    surv_results.columns = [event + "." + x for x in surv_results.columns]

    pvals = surv_results.loc[
        ~surv_results[event + ".p_value"].isna(), event + ".p_value"
    ].values
    bh_res, adj_pval = fdrcorrection(pvals, alpha=0.05)
    surv_results.loc[
        ~surv_results[event + ".p_value"].isna(), event + ".p_value_BH"
    ] = adj_pval

    pvals = surv_results.loc[
        ~surv_results[event + ".LogR_p_value"].isna(), event + ".LogR_p_value"
    ].values
    bh_res, adj_pval = fdrcorrection(pvals, alpha=0.05)
    surv_results.loc[
        ~surv_results[event + ".LogR_p_value"].isna(), event + ".LogR_p_value_BH"
    ] = adj_pval

    return pd.concat([biclusters, surv_results], axis=1)




"""

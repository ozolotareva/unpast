from typing import Any
import numpy as np
import pandas as pd

from unpast.misc.eval.metrics import calc_ari_matching


def _add_performance_cols(
    best_matches_, true_biclusters, pred_biclusters, target="genes", all_samples={None}
):
    best_matches = best_matches_.loc[:, :].copy()

    best_matches.loc[best_matches["bm_id"].dropna().index, "pred_" + target] = (
        best_matches["bm_id"].dropna().apply(lambda x: pred_biclusters.loc[x, target])
    )
    best_matches.loc[:, "pred_" + target] = best_matches.loc[
        :, "pred_" + target
    ].fillna("")
    best_matches.loc[best_matches["pred_" + target] == "", "pred_" + target] = (
        best_matches.loc[best_matches["pred_" + target] == "", "pred_" + target].apply(
            lambda x: set([])
        )
    )

    if target == "samples":
        best_matches.loc[best_matches["is_enriched"] == False, "pred_" + target] = (
            best_matches.loc[
                best_matches["is_enriched"] == False, "pred_" + target
            ].apply(lambda x: all_samples.difference(x))
        )
    best_matches["bic_id"] = best_matches.index
    best_matches["true_" + target] = best_matches["bic_id"].apply(
        lambda x: true_biclusters.loc[x, target]
    )

    best_matches["TP_" + target] = best_matches.apply(
        lambda row: row["true_" + target].intersection(row["pred_" + target]),
        axis=1,
        result_type="reduce",
    )
    # true positive rate == Recall
    best_matches["TPR_" + target] = best_matches.apply(
        lambda row: len(row["TP_" + target]) / len(row["true_" + target]),
        axis=1,
        result_type="reduce",
    )

    # precision
    best_matches["Prec_" + target] = 0.0
    non_zero = best_matches["pred_" + target].apply(len) > 0
    best_matches.loc[non_zero, "Prec_" + target] = best_matches.loc[non_zero, :].apply(
        lambda row: len(row["TP_" + target]) / len(row["pred_" + target]),
        axis=1,
        result_type="reduce",
    )

    # keep only biclusters matching anything
    best_matches["F1_" + target] = 0.0
    mbic_ids = best_matches["Prec_" + target] + best_matches["TPR_" + target]
    mbic_ids = mbic_ids[mbic_ids > 0].index.values

    prec = best_matches.loc[mbic_ids, "Prec_" + target] * 1.0
    rec = best_matches.loc[mbic_ids, "TPR_" + target] * 1.0
    pr = prec + rec
    best_matches.loc[mbic_ids, "F1_" + target] = 2.0 * np.multiply(prec, rec) / pr

    return best_matches


def calc_performance_measures(best_matches_, true_biclusters, pred_biclusters, exprs):
    best_matches = _add_performance_cols(
        best_matches_,
        true_biclusters,
        pred_biclusters,
        target="samples",
        all_samples=set(exprs.columns.values),
    )

    best_matches = _add_performance_cols(
        best_matches, true_biclusters, pred_biclusters, target="genes"
    )

    F1_f_avg = best_matches["F1_genes"].sum() / true_biclusters.shape[0]
    F1_s_avg = best_matches["F1_samples"].sum() / true_biclusters.shape[0]

    # FP in matched biclusters
    # number of elements in matched biclusters not in true biclusters
    df = best_matches
    pred_P_matched = df["pred_genes"].apply(len) * df["pred_samples"].apply(len)
    TP = df["TP_genes"].apply(len) * df["TP_samples"].apply(len)
    FP1 = pred_P_matched - TP

    #  number of elements in non-matched biclusters
    bm_ids = list(set(best_matches["bm_id"].values))
    not_mbic_ids = [x for x in pred_biclusters.index if not x in bm_ids]
    df2 = pred_biclusters.loc[not_mbic_ids, :]
    FP2 = df2["n_genes"] * df2["n_samples"]

    FP = FP2.sum() + FP1.sum()

    # ratio of predicted bicluster elements (FP) not matching true biclusters, to all biclusters predicted elements (TP+FP)
    P = true_biclusters["n_genes"] * true_biclusters["n_samples"]
    P = P.sum()
    # FDR
    FDR_bic = FP / (TP.sum() + FP)

    # Recall
    Recall_bic = TP.sum() / P

    return best_matches, F1_f_avg, F1_s_avg, FDR_bic, Recall_bic


def calculate_metrics(
    true_biclusters: pd.DataFrame,
    pred_biclusters: pd.DataFrame,
    _exprs: pd.DataFrame,
    matching_measure="ARI",
    **calc_matching_args: Any,
):
    # 1. Find best matches and estimate performance
    _all_samples = set(_exprs.columns.values)  # all samples in the dataset
    _known_groups = true_biclusters.loc[:, ["samples"]].to_dict()["samples"]
    _known_groups = {
        "true_biclusters": _known_groups
    }  # can be more than one classification => it is a dict of dicts
    res, best_matches = calc_ari_matching(
        pred_biclusters,
        _known_groups,
        _all_samples,
        matching_measure=matching_measure,
        **calc_matching_args,
    )

    # 2. Calc other metrics
    wARIs = res["true_biclusters"]
    _, F1_f_avg, F1_s_avg, FDR_bic, Recall_bic = calc_performance_measures(
        best_matches.dropna(), true_biclusters, pred_biclusters, _exprs
    )

    return {
        "wARIs": wARIs,
        "F1_f_avg": F1_f_avg,
        "F1_s_avg": F1_s_avg,
        "FDR_bic": FDR_bic,
        "Recall_bic": Recall_bic,
    }

    # best_matches = calc_best_matches(
    #     true_bics, pred_bics, _exprs
    # )

    # metrics = {}

    # # best_matches_metrics
    # precision_recall = calc_precision_recall(best_matches, true_bics, pred_bics)
    # metrics['FDR_bic'] = precision_recall['FDR_bic']
    # metrics['Recall_bic'] = precision_recall['Recall_bic']

    # metrics.update({
    #     "wARIs": _calc_wARIs(true_bics, pred_bics, matches),
    #     "F1_f_avg": _calc_F1_f_avg(true_bics, pred_bics, matches),
    #     "F1_s_avg": F1_s_avg(true_bics, pred_bics, matches),
    #     "FDR_bic": FDR_bic(true_bics, pred_bics, matches),
    #     "Recall_bic": Recall_bic(true_bics, pred_bics, matches),
    # })

    # })

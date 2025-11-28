"""Performance metrics for bicluster evaluation.

This module provides comprehensive evaluation metrics for comparing predicted biclusters
against ground truth, including precision, recall, F1 scores, and ARI-based matching.
"""

from typing import Any

import numpy as np
import pandas as pd

from unpast.misc.eval.calc_ari_matching import calc_ari_matching
from unpast.misc.eval.calc_average_precision import calc_average_precision_at_thresh


def _add_performance_cols(
    best_matches_: pd.DataFrame,
    true_biclusters: pd.DataFrame,
    pred_biclusters: pd.DataFrame,
    target: str = "genes",
    all_samples: set = {None},
) -> pd.DataFrame:
    """Add performance columns (TPR, Precision, F1) to best matches DataFrame.

    Args:
        best_matches_: DataFrame with best matches between predicted and true biclusters.
        true_biclusters: DataFrame containing ground truth biclusters.
        pred_biclusters: DataFrame containing predicted biclusters.
        target: Target dimension to evaluate ("genes" or "samples").
        all_samples: Set of all samples in the dataset (used for sample evaluation).

    Returns:
        DataFrame with added performance columns for the specified target dimension.
    """
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


def calc_performance_measures(
    best_matches_: pd.DataFrame,
    true_biclusters: pd.DataFrame,
    pred_biclusters: pd.DataFrame,
    exprs: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float, float]:
    """Calculate performance measures for bicluster predictions.

    Args:
        best_matches_: DataFrame with best matches between predicted and true biclusters.
        true_biclusters: DataFrame containing ground truth biclusters.
        pred_biclusters: DataFrame containing predicted biclusters.
        exprs: Expression data DataFrame.

    Returns:
        Tuple containing:
            - best_matches: DataFrame with added performance metrics
            - F1_f_avg: Average F1 score for genes
            - F1_s_avg: Average F1 score for samples
            - FDR_bic: False discovery rate for biclusters
            - Recall_bic: Recall for biclusters
    """
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
    not_mbic_ids = [x for x in pred_biclusters.index if x not in bm_ids]
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


def _handle_empty_bicluster_case(
    true_biclusters: pd.DataFrame,
    pred_biclusters: pd.DataFrame,
) -> dict[str, float]:
    """Handle edge cases where there are no predicted or true biclusters.

    Args:
        true_biclusters: DataFrame containing ground truth biclusters.
        pred_biclusters: DataFrame containing predicted biclusters.

    Returns:
        Dictionary containing evaluation metrics for edge cases.
    """
    if len(true_biclusters) == 0:
        raise ValueError("True biclusters cannot be empty for metric calculation.")

    assert len(pred_biclusters) == 0, (
        "Unexpected non-empty predicted and true biclusters."
    )
    return {
        "wARIs": 0.0,
        "F1_f_avg": 0.0,
        "F1_s_avg": 0.0,
        "FDR_bic": 1.0,
        "Recall_bic": 0.0,
        "AP_ARI": 0.0,
    }


def calc_metrics(
    true_biclusters: pd.DataFrame,
    pred_biclusters: pd.DataFrame,
    exprs: pd.DataFrame,
    matching_measure: str = "ARI",
    **calc_matching_args: Any,
) -> dict[str, float]:
    """Calculate comprehensive evaluation metrics for bicluster predictions.

    Args:
        true_biclusters: DataFrame containing ground truth biclusters.
        pred_biclusters: DataFrame containing predicted biclusters.
        exprs: Expression data DataFrame.
        matching_measure: Measure to use for matching ("ARI" or "Jaccard").
        **calc_matching_args: Additional arguments for calc_ari_matching.

    Returns:
        Dictionary containing evaluation metrics:
            - wARIs: Weighted adjusted Rand index
            - F1_f_avg: Average F1 score for genes
            - F1_s_avg: Average F1 score for samples
            - FDR_bic: False discovery rate for biclusters
            - Recall_bic: Recall for biclusters
            - AP_ARI: Average precision at IoU thresholds 0.5-0.95, using ARI
    """
    # 0. Handle empty cases
    if len(pred_biclusters) == 0 or len(true_biclusters) == 0:
        return _handle_empty_bicluster_case(
            true_biclusters=true_biclusters, pred_biclusters=pred_biclusters
        )

    # 1. Find best matches and estimate performance
    _all_samples = set(exprs.columns.values)  # all samples in the dataset
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
        best_matches.dropna(), true_biclusters, pred_biclusters, exprs
    )

    # 3. Calc average precision metric
    AP_ARI = calc_average_precision_at_thresh(
        true_biclusters, pred_biclusters, method="ARI", exprs=exprs
    )
    return {
        "wARIs": wARIs,
        "F1_f_avg": F1_f_avg,
        "F1_s_avg": F1_s_avg,
        "FDR_bic": FDR_bic,
        "Recall_bic": Recall_bic,
        "AP_ARI": AP_ARI,
    }

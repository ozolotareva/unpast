from typing import Iterable

import numpy as np
import pandas as pd

THRESHS = tuple(x / 100 for x in range(50, 100, 5))
USE_MAX_PRECISION = True


def _validate_cols(bics: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    """Validate and standardize a bicluster DataFrame.

    Args:
        bics: DataFrame containing biclusters. Expected columns are ``genes``
            and ``samples``; ``score_col`` is optional but required when
            score_col is not None.
        score_col: Column name in `bics` DataFrame to use for ranking.

    Returns:
        A validated copy of the input DataFrame containing only the required
        columns.
    """
    if score_col is not None:
        df = bics[["genes", "samples", score_col]].copy()
        df["score"] = pd.to_numeric(df[score_col]).astype(
            "float64"
        )  # validate SNR is numeric
        df.drop(columns=score_col, inplace=True)

    else:
        df = bics[["genes", "samples"]].copy()

    # validate genes/samples columns
    for col in ["genes", "samples"]:
        for obj in df[col]:
            assert isinstance(obj, set), (
                f"Bicluster column '{col}' must contain sets, found {type(obj)}"
            )
            for el in obj:
                assert isinstance(el, str), (
                    f"Bicluster cols must contain sets of strings, found element of type {type(el)}"
                )

    return df


def _calc_jaccard(bic1: pd.Series, bic2: pd.Series) -> float:
    """Calculate Jaccard similarity between two biclusters.

    Args:
        bic1: First bicluster as a row from a Series.
        bic2: Second bicluster as a row from a Series.

    Returns:
        Jaccard similarity as a float in [0.0, 1.0].
    """
    genes1, samples1 = bic1["genes"], bic1["samples"]
    genes2, samples2 = bic2["genes"], bic2["samples"]

    def _area(genes: set[str], samples: set[str]) -> int:
        return len(genes) * len(samples)

    area1 = _area(genes1, samples1)
    area2 = _area(genes2, samples2)
    area12 = _area(
        genes1.intersection(genes2),
        samples1.intersection(samples2),
    )

    jaccard_sim = area12 / (area1 + area2 - area12)
    return jaccard_sim


def _calc_mat_iou(
    bics_pred: pd.DataFrame,
    bics_true: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate IoU / Jaccard similarity matrix between predicted and true biclusters.

    Args:
        bics_pred: Predicted biclusters as a DataFrame.
        bics_true: Ground-truth

    Returns:
        A 2D numpy array representing the IoU / Jaccard similarity matrix.
            Rows correspond to predicted biclusters, columns to true biclusters.
    """
    # mat_iou = np.zeros((bics_pred.shape[0], bics_true.shape[0]), dtype=float)
    mat_iou = pd.DataFrame(
        index=bics_pred.index,
        columns=bics_true.index,
        data=0.0,
    )

    for pred_ind, pred_bic in bics_pred.iterrows():
        for true_ind, true_bic in bics_true.iterrows():
            mat_iou.loc[pred_ind, true_ind] = _calc_jaccard(pred_bic, true_bic)

    return mat_iou


def _calc_average_precision_by_matrix(
    mat_iou_pred_to_true: pd.DataFrame,
    thresh: float,
) -> float:
    """Calculate Average Precision (AP) for predicted biclusters vs. ground truth
    based on a precomputed IoU / Jaccard similarity matrix.

    Args:
        mat_iou: DataFrame representing the IoU / Jaccard similarity matrix.

    Returns:
        Average Precision as a float in ``[0.0, 1.0]``.
    """

    assert 0.0 <= thresh <= 1.0, (
        f"IoU / Jaccard thresholds must be in [0.0, 1.0], found {thresh}"
    )
    mat = mat_iou_pred_to_true.values.copy()

    pred_count = mat.shape[0]
    gt_count = mat.shape[1]
    assert gt_count >= 0

    TP = 0

    precision_recall_points = [(1.0, 0.0)]

    for pred_ind in range(pred_count):
        # find best matching true bicluster
        best_fit_ind = np.argmax(mat[pred_ind])
        if mat[pred_ind][best_fit_ind] >= thresh:
            # mark as matched
            mat[:, best_fit_ind] = -1  # invalidate this true bicluster
            TP += 1

        # update precision and recall, and integrate AP
        precision = TP / (pred_ind + 1)
        recall = TP / gt_count
        prev_recall = precision_recall_points[-1][1]

        if recall > prev_recall:
            precision_recall_points.append((precision, recall))

    # integrate AP using precision-recall points
    pr_ar = np.array(precision_recall_points)
    if USE_MAX_PRECISION:
        # making the pr[i] = max(pr[i:]), as in Pascal VOC challenge
        # Questionable, whether this is useful for biclustering AP
        for i in range(len(pr_ar) - 2, -1, -1):
            pr_ar[i][0] = max(pr_ar[i][0], pr_ar[i + 1][0])

    ap_sum = 0.0
    for (p1, r1), (p2, r2) in zip(precision_recall_points, precision_recall_points[1:]):
        ap_sum += (p1 + p2) * (r2 - r1) / 2.0

    return ap_sum


def calc_average_precision_at_thresh(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    threshs: Iterable[float] = THRESHS,
    score_col: str = "SNR",
) -> float:
    """Compute Average Precision at thresholds (by default AP@[0.5,0.95]) for predicted biclusters vs. ground truth.
        but we still need to average over IoU / Jaccard thresholds
        (there is only one class, so it's not mean AP)

    Args:
        bics_true: Ground-truth biclusters as a DataFrame.
        bics_pred: Predicted biclusters as a DataFrame.
        threshs: Iterable of precision thresholds to evaluate
            default metric is mAP@0.5-0.95 (with step 0.05)
        score_col: Column name in `bics_pred` DataFrame to use for ranking predicted biclusters.
            default is "SNR".

    Returns:
        Mean Average Precision as a float in ``[0.0, 1.0]``.

    Note:
        Implementation intentionally omitted in this draft.
    """
    if len(bics_true) == 0:
        print(
            "No ground-truth biclusters provided, returning AP=0.0 or 1.0 based on predictions."
        )
        return 0.0 if len(bics_pred) > 0 else 1.0

    elif len(bics_pred) == 0:
        return 0.0

    bics_true = _validate_cols(bics_true)
    bics_pred = _validate_cols(bics_pred, score_col)

    bics_pred = bics_pred.sort_values(by="score", ascending=False)
    mat_iou_pred_to_true = _calc_mat_iou(bics_pred, bics_true)

    ap_scores = []
    for thr in threshs:
        # todo: double-checking that don't mixing cols and rows
        ap = _calc_average_precision_by_matrix(mat_iou_pred_to_true, thr)
        ap_scores.append(ap)

    # Return mean AP across requested thresholds
    return float(sum(ap_scores) / len(ap_scores))

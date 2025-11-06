import pandas as pd
from typing import Tuple, Set, List, Optional, Iterable

PROBS = tuple(x / 100 for x in range(50, 100, 5))

def _validate_cols(bics: pd.DataFrame, use_SNR: bool) -> pd.DataFrame:
    """Validate and standardize a bicluster DataFrame.

    Args:
        bics: DataFrame containing biclusters. Expected columns are ``genes``
            and ``samples``; ``SNR`` is optional but required when ``use_SNR``
            is True.
        use_SNR: Whether to require and validate the ``SNR`` column.

    Returns:
        A validated copy of the input DataFrame containing only the required
        columns.
    """
    if use_SNR:
        df = bics[["genes", "samples", "SNR"]].copy()
        df['SNR'] = pd.to_numeric(df['SNR']).astype('float64')  # validate SNR is numeric

    else: 
        df = bics[["genes", "samples"]].copy()
    
    # validate genes/samples columns
    for col in ["genes", "samples"]:
        for obj in df[col]:
            assert isinstance(obj, set), f"Bicluster column '{col}' must contain sets, found {type(obj)}"
            for el in obj:
                assert isinstance(el, str), f"Bicluster cols must contain sets of strings, found element of type {type(el)}"

    return df


def _precision_recall_for_pred(pred_pairs: Set[Tuple[str, str]], true_pairs: Set[Tuple[str, str]]) -> Tuple[float, float]:
    """Compute precision and recall for a single predicted bicluster.

    Args:
        pred_pairs: Set of predicted (gene, sample) pairs for one bicluster.
        true_pairs: Set of ground-truth (gene, sample) pairs.

    Returns:
        A tuple ``(precision, recall)`` where precision = TP / (TP + FP) and
        recall = TP / (TP + FN). Values are floats in ``[0.0, 1.0]``.

    Note:
        Implementation intentionally omitted in this draft.
    """
    raise NotImplementedError


def calc_mean_average_precision(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    probs: Iterable[float] = PROBS,
) -> float:
    """Compute mean Average Precision (mAP) for predicted biclusters vs. ground truth.

        Args:
        bics_true: Ground-truth biclusters as a DataFrame.
        bics_pred: Predicted biclusters as a DataFrame.
        probs: Iterable of precision thresholds to evaluate 
            default metric is mAP@0.5-0.95 (with step 0.05)
    Returns:
        Mean Average Precision as a float in ``[0.0, 1.0]``.

    Note:
        Implementation intentionally omitted in this draft.
    """
    bics_true = _validate_cols(bics_true, use_SNR=False)
    bics_pred = _validate_cols(bics_pred, use_SNR=True)


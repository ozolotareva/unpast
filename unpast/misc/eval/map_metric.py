"""File for metrics calculation for synthetic and real datasets.

There are two types of metrics:
1) performance (for synthetic datasets with known biclusters)
2) quality of found biclusters (bics_true may include only some of the real biclusters)

The mAP metrics are general metrics used in object detection evaluation in computer vision. 
They require detected objects to be ranked by confidence score (here SNR).

"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from unpast.utils.logs import get_logger

logger = get_logger(__name__)

def _validate_cols(bics: pd.DataFrame, use_SNR: bool) -> pd.DataFrame:
    """Validate bicluster DataFrame structure and keep only necessary columns.
    
    Args:
        bics: DataFrame with biclusters
        use_SNR: Whether to use SNR column
    
    Returns:
        Standardized bicluster DataFrame (without other cols)
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


def calculate_jaccard_index(set1: Set, set2: Set) -> float:
    """Calculate Jaccard index between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard index (intersection over union)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def calculate_overlap_stats(set1: Set, set2: Set) -> Dict[str, int]:
    """Calculate overlap statistics between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Dictionary with overlap statistics
    """
    intersection = set1.intersection(set2)
    return {
        "intersection": len(intersection),
        "set1_only": len(set1 - set2),
        "set2_only": len(set2 - set1),
        "union": len(set1.union(set2))
    }


def calculate_bicluster_similarity(
    bic1: pd.Series, 
    bic2: pd.Series, 
    dimension: str = "both"
) -> Dict[str, float]:
    """Calculate similarity metrics between two biclusters.
    
    Args:
        bic1: First bicluster (pandas Series with 'genes' and 'samples' columns)
        bic2: Second bicluster (pandas Series with 'genes' and 'samples' columns)
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Dictionary with similarity metrics
    """
    similarities = {}
    
    if dimension in ["genes", "both"]:
        gene_jaccard = calculate_jaccard_index(bic1["genes"], bic2["genes"])
        similarities["gene_jaccard"] = gene_jaccard
        
    if dimension in ["samples", "both"]:
        sample_jaccard = calculate_jaccard_index(bic1["samples"], bic2["samples"])
        similarities["sample_jaccard"] = sample_jaccard
        
    if dimension == "both":
        # Combined similarity (geometric mean)
        similarities["combined_jaccard"] = np.sqrt(
            similarities["gene_jaccard"] * similarities["sample_jaccard"]
        )
        
    return similarities


def find_best_matches(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    dimension: str = "both",
    min_similarity: float = 0.0
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Find best matching biclusters between true and predicted sets.
    
    Args:
        bics_true: True biclusters DataFrame
        bics_pred: Predicted biclusters DataFrame  
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (true_to_pred_matches, pred_to_true_matches)
    """
    true_to_pred = {}
    pred_to_true = {}
    
    # Find best match for each true bicluster
    for true_idx, true_bic in bics_true.iterrows():
        best_match = None
        best_similarity = 0.0
        
        for pred_idx, pred_bic in bics_pred.iterrows():
            similarities = calculate_bicluster_similarity(true_bic, pred_bic, dimension)
            
            if dimension == "both":
                similarity = similarities["combined_jaccard"]
            elif dimension == "genes":
                similarity = similarities["gene_jaccard"]
            else:  # samples
                similarity = similarities["sample_jaccard"]
                
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_match = {
                    "match_idx": pred_idx,
                    "similarity": similarity,
                    **similarities
                }
        
        if best_match:
            true_to_pred[true_idx] = best_match
    
    # Find best match for each predicted bicluster
    for pred_idx, pred_bic in bics_pred.iterrows():
        best_match = None
        best_similarity = 0.0
        
        for true_idx, true_bic in bics_true.iterrows():
            similarities = calculate_bicluster_similarity(pred_bic, true_bic, dimension)
            
            if dimension == "both":
                similarity = similarities["combined_jaccard"]
            elif dimension == "genes":
                similarity = similarities["gene_jaccard"]
            else:  # samples
                similarity = similarities["sample_jaccard"]
                
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_match = {
                    "match_idx": true_idx,
                    "similarity": similarity,
                    **similarities
                }
        
        if best_match:
            pred_to_true[pred_idx] = best_match
            
    return true_to_pred, pred_to_true


def calculate_average_precision(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    iou_threshold: float = 0.5,
    dimension: str = "both"
) -> float:
    """Calculate Average Precision at a specific IoU threshold.
    
    Args:
        bics_true: True biclusters DataFrame
        bics_pred: Predicted biclusters DataFrame
        iou_threshold: IoU threshold for considering a match
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Average Precision score
    """
    if bics_pred.empty:
        return 0.0
    
    if bics_true.empty:
        return 0.0
    
    # Find matches above threshold
    _, pred_to_true = find_best_matches(
        bics_true, bics_pred, dimension, min_similarity=iou_threshold
    )
    
    # Sort predictions by confidence (using SNR if available, otherwise by index)
    if "SNR" in bics_pred.columns:
        sorted_pred_indices = bics_pred.sort_values("SNR", ascending=False).index
    else:
        sorted_pred_indices = bics_pred.index
    
    # Calculate precision at each recall level
    true_positives = 0
    false_positives = 0
    precision_scores = []
    
    matched_true_indices = set()
    
    for pred_idx in sorted_pred_indices:
        if pred_idx in pred_to_true:
            # This is a true positive if we haven't matched this true bicluster yet
            true_match_idx = pred_to_true[pred_idx]["match_idx"]
            if true_match_idx not in matched_true_indices:
                true_positives += 1
                matched_true_indices.add(true_match_idx)
            else:
                false_positives += 1
        else:
            false_positives += 1
        
        # Calculate precision at this point
        total_predictions = true_positives + false_positives
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0
        precision_scores.append(precision)
    
    # Average precision is the mean of precision scores
    return float(np.mean(precision_scores)) if precision_scores else 0.0


def calculate_map_at_iou_range(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    iou_range: Tuple[float, float] = (0.5, 0.95),
    step: float = 0.05,
    dimension: str = "both"
) -> Dict[str, float]:
    """Calculate mean Average Precision across IoU thresholds 0.5-0.95.
    
    Args:
        bics_true: True biclusters DataFrame
        bics_pred: Predicted biclusters DataFrame
        iou_range: Range of IoU thresholds (start, end)
        step: Step size for IoU thresholds
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Dictionary with mAP scores at different IoU ranges
    """
    start_iou, end_iou = iou_range
    iou_thresholds = np.arange(start_iou, end_iou + step, step)
    
    ap_scores = []
    ap_by_threshold = {}
    
    for iou_threshold in iou_thresholds:
        ap = calculate_average_precision(
            bics_true, bics_pred, iou_threshold, dimension
        )
        ap_scores.append(ap)
        ap_by_threshold[f"AP@{iou_threshold:.2f}"] = ap
    
    # Calculate mean AP across all thresholds
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    
    result = {
        f"mAP@{start_iou:.1f}-{end_iou:.2f}": map_score,
        **ap_by_threshold
    }
    
    return result


def calculate_precision_recall_f1(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    iou_threshold: float = 0.5,
    dimension: str = "both"
) -> Dict[str, float]:
    """Calculate precision, recall, and F1-score.
    
    This function handles extra outputs (false positives) gracefully by only
    counting unique matches.
    
    Args:
        bics_true: True biclusters DataFrame
        bics_pred: Predicted biclusters DataFrame
        iou_threshold: IoU threshold for considering a match
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if bics_pred.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(bics_true) if not bics_true.empty else 0
        }
    
    if bics_true.empty:
        return {
            "precision": 0.0,
            "recall": 1.0 if bics_pred.empty else 0.0,
            "f1_score": 0.0,
            "true_positives": 0,
            "false_positives": len(bics_pred),
            "false_negatives": 0
        }
    
    # Find matches
    true_to_pred, pred_to_true = find_best_matches(
        bics_true, bics_pred, dimension, min_similarity=iou_threshold
    )
    
    # Count unique matches
    matched_true_indices = set()
    matched_pred_indices = set()
    
    # Count true positives based on unique matches
    for true_idx, match_info in true_to_pred.items():
        pred_idx = match_info["match_idx"]
        if pred_idx in pred_to_true and pred_to_true[pred_idx]["match_idx"] == true_idx:
            # Mutual best match
            matched_true_indices.add(true_idx)
            matched_pred_indices.add(pred_idx)
    
    true_positives = len(matched_true_indices)
    false_positives = len(bics_pred) - len(matched_pred_indices)
    false_negatives = len(bics_true) - len(matched_true_indices)
    
    # Calculate metrics
    precision = true_positives / len(bics_pred) if len(bics_pred) > 0 else 0.0
    recall = true_positives / len(bics_true) if len(bics_true) > 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def calculate_recovery_relevance(
    bics_true: pd.DataFrame,
    bics_pred: pd.DataFrame,
    iou_threshold: float = 0.5,
    dimension: str = "both"
) -> Dict[str, float]:
    """Calculate recovery and relevance metrics.
    
    Recovery = fraction of true biclusters that are recovered
    Relevance = fraction of predicted biclusters that are relevant
    
    Args:
        bics_true: True biclusters DataFrame
        bics_pred: Predicted biclusters DataFrame
        iou_threshold: IoU threshold for considering a match
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Dictionary with recovery and relevance scores
    """
    pr_metrics = calculate_precision_recall_f1(
        bics_true, bics_pred, iou_threshold, dimension
    )
    
    return {
        "recovery": pr_metrics["recall"],
        "relevance": pr_metrics["precision"],
        "recovery_relevance_f1": pr_metrics["f1_score"]
    }


def calculate_size_metrics(bics_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate size-related metrics for biclusters.
    
    Args:
        bics_df: Biclusters DataFrame
        
    Returns:
        Dictionary with size metrics
    """
    if bics_df.empty:
        return {
            "avg_n_genes": 0.0,
            "avg_n_samples": 0.0,
            "avg_area": 0.0,
            "total_genes": 0,
            "total_samples": 0,
            "total_area": 0,
            "n_biclusters": 0
        }
    
    n_genes = bics_df["n_genes"] if "n_genes" in bics_df.columns else bics_df["genes"].apply(len)
    n_samples = bics_df["n_samples"] if "n_samples" in bics_df.columns else bics_df["samples"].apply(len)
    
    areas = n_genes * n_samples
    
    return {
        "avg_n_genes": float(n_genes.mean()),
        "avg_n_samples": float(n_samples.mean()),
        "avg_area": float(areas.mean()),
        "total_genes": int(n_genes.sum()),
        "total_samples": int(n_samples.sum()),
        "total_area": int(areas.sum()),
        "n_biclusters": len(bics_df)
    }


def calculate_quality_metrics(
    bics_df: pd.DataFrame,
    data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Calculate quality metrics for biclusters.
    
    Args:
        bics_df: Biclusters DataFrame
        data: Optional expression data for MSR/variance calculations
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}
    
    if bics_df.empty:
        return {
            "avg_snr": 0.0,
            "median_snr": 0.0,
            "max_snr": 0.0,
            "min_snr": 0.0
        }
    
    # SNR-based metrics
    if "SNR" in bics_df.columns:
        snr_values = bics_df["SNR"].dropna()
        if not snr_values.empty:
            quality_metrics.update({
                "avg_snr": float(snr_values.mean()),
                "median_snr": float(snr_values.median()),
                "max_snr": float(snr_values.max()),
                "min_snr": float(snr_values.min())
            })
    
    # TODO: Add MSR and variance calculations if expression data is provided
    if data is not None:
        logger.info("Expression data provided but MSR/variance calculations not implemented yet")
    
    return quality_metrics




def calculate_metrics(
    bics_true: pd.DataFrame, 
    bics_pred: pd.DataFrame, 
    data_shape: Optional[Tuple[int, int]] = None,
    expression_data: Optional[pd.DataFrame] = None,
    iou_thresholds: List[float] = [0.5, 0.75],
    dimension: str = "both"
) -> Dict[str, Union[float, int]]:
    """Calculate comprehensive metrics for bicluster prediction performance.
    
    Args:
        bics_true: True biclusters DataFrame with 'genes' and 'samples' columns
        bics_pred: Predicted biclusters DataFrame with 'genes' and 'samples' columns
        data_shape: Optional tuple of (n_genes, n_samples) for the original data
        expression_data: Optional expression data for quality metrics
        iou_thresholds: List of IoU thresholds to calculate metrics for
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics = {}
    
    # Size metrics
    metrics.update({
        "true_" + k: v for k, v in calculate_size_metrics(bics_true).items()
    })
    metrics.update({
        "pred_" + k: v for k, v in calculate_size_metrics(bics_pred).items()
    })
    
    # Quality metrics for predicted biclusters
    quality_pred = calculate_quality_metrics(bics_pred, expression_data)
    metrics.update({
        "pred_" + k: v for k, v in quality_pred.items()
    })
    
    # Performance metrics at different IoU thresholds
    for iou_threshold in iou_thresholds:
        # Precision, Recall, F1
        pr_metrics = calculate_precision_recall_f1(
            bics_true, bics_pred, iou_threshold, dimension
        )
        for key, value in pr_metrics.items():
            metrics[f"{key}@{iou_threshold:.2f}"] = value
        
        # Recovery and Relevance
        rr_metrics = calculate_recovery_relevance(
            bics_true, bics_pred, iou_threshold, dimension
        )
        for key, value in rr_metrics.items():
            metrics[f"{key}@{iou_threshold:.2f}"] = value
    
    # mAP metrics
    map_metrics = calculate_map_at_iou_range(
        bics_true, bics_pred, (0.5, 0.95), 0.05, dimension
    )
    metrics.update(map_metrics)
    
    # Overall Jaccard similarity (average of best matches)
    if not bics_true.empty and not bics_pred.empty:
        true_to_pred, _ = find_best_matches(bics_true, bics_pred, dimension, 0.0)
        if true_to_pred:
            avg_jaccard = np.mean([match["similarity"] for match in true_to_pred.values()])
            metrics["avg_jaccard_similarity"] = float(avg_jaccard)
        else:
            metrics["avg_jaccard_similarity"] = 0.0
    else:
        metrics["avg_jaccard_similarity"] = 0.0
    
    return metrics


def calculate_metrics_on_dataset(
    dataset_results: Dict[str, Dict[str, pd.DataFrame]],
    expression_data: Optional[Dict[str, pd.DataFrame]] = None,
    iou_thresholds: List[float] = [0.5, 0.75],
    dimension: str = "both"
) -> pd.DataFrame:
    """Calculate metrics for multiple datasets.
    
    Args:
        dataset_results: Dictionary with dataset names as keys and 
                        dictionaries containing 'bics_true' and 'bics_pred' DataFrames
        expression_data: Optional dictionary with expression data for each dataset
        iou_thresholds: List of IoU thresholds to calculate metrics for
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        DataFrame with metrics for each dataset
    """
    results = {}
    
    for dataset_name, data in dataset_results.items():
        bics_true = data.get("bics_true", pd.DataFrame())
        bics_pred = data.get("bics_pred", pd.DataFrame())
        
        expr_data = None
        if expression_data and dataset_name in expression_data:
            expr_data = expression_data[dataset_name]
        
        try:
            metrics = calculate_metrics(
                bics_true, 
                bics_pred, 
                expression_data=expr_data,
                iou_thresholds=iou_thresholds,
                dimension=dimension
            )
            results[dataset_name] = metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for dataset {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
    
    return pd.DataFrame.from_dict(results, orient="index")


def calculate_metrics_on_df(
    df: pd.DataFrame,
    true_col: str = "bics_true",
    pred_col: str = "bics_pred",
    expr_col: Optional[str] = None,
    iou_thresholds: List[float] = [0.5, 0.75],
    dimension: str = "both"
) -> pd.DataFrame:
    """Calculate metrics on a DataFrame where each row represents a dataset.
    
    Args:
        df: DataFrame where each row contains bicluster data
        true_col: Column name containing true biclusters
        pred_col: Column name containing predicted biclusters  
        expr_col: Optional column name containing expression data
        iou_thresholds: List of IoU thresholds to calculate metrics for
        dimension: Which dimension to compare ('genes', 'samples', or 'both')
        
    Returns:
        DataFrame with metrics for each row/dataset
    """
    results = {}
    
    for idx, row in df.iterrows():
        bics_true = row[true_col] if true_col in row and not pd.isna(row[true_col]) else pd.DataFrame()
        bics_pred = row[pred_col] if pred_col in row and not pd.isna(row[pred_col]) else pd.DataFrame()
        
        expr_data = None
        if expr_col and expr_col in row and not pd.isna(row[expr_col]):
            expr_data = row[expr_col]
        
        try:
            metrics = calculate_metrics(
                bics_true,
                bics_pred,
                expression_data=expr_data,
                iou_thresholds=iou_thresholds,
                dimension=dimension
            )
            results[idx] = metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for row {idx}: {e}")
            results[idx] = {"error": str(e)}
    
    return pd.DataFrame.from_dict(results, orient="index")


# Example usage functions
def example_calculate_metrics():
    """Example of how to use the metrics calculation functions."""
    # Create example data
    bics_true = pd.DataFrame({
        "genes": [{"gene1", "gene2"}, {"gene3", "gene4", "gene5"}],
        "samples": [{"s1", "s2", "s3"}, {"s1", "s4"}],
        "n_genes": [2, 3],
        "n_samples": [3, 2]
    })
    
    bics_pred = pd.DataFrame({
        "genes": [{"gene1", "gene2", "gene6"}, {"gene3", "gene4"}],
        "samples": [{"s1", "s2"}, {"s1", "s4", "s5"}],
        "n_genes": [3, 2],
        "n_samples": [2, 3],
        "SNR": [2.5, 1.8]
    })
    
    # Calculate metrics
    metrics = calculate_metrics(bics_true, bics_pred)
    
    print("Calculated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return metrics



    return metrics


# metrics to calculate:
# * Jaccard index ✓
# * recovery and relevance (precision and recall) ✓
# * F1 score ✓
# * Clustering Error (CE) - TODO
# * Mean Squared Residue (MSR) - TODO  
# * Variance (VAR) - TODO
# * AUC of the ROC curve - TODO
# * size of biclusters (genes, samples, area) ✓
# * number of biclusters found ✓
# * mAP@[0.5-0.95] ✓


"""
Available metrics calculated by this module:

Performance Metrics (for synthetic datasets with known biclusters):
- precision@{threshold}: Precision at specific IoU threshold
- recall@{threshold}: Recall at specific IoU threshold  
- f1_score@{threshold}: F1-score at specific IoU threshold
- recovery@{threshold}: Recovery (same as recall)
- relevance@{threshold}: Relevance (same as precision)
- mAP@0.5-0.95: Mean Average Precision across IoU thresholds 0.5-0.95
- AP@{threshold}: Average Precision at specific IoU threshold
- avg_jaccard_similarity: Average Jaccard similarity of best matches
- true_positives@{threshold}: Number of true positive matches
- false_positives@{threshold}: Number of false positive matches
- false_negatives@{threshold}: Number of false negative matches

Size/Quality Metrics:
- true_n_biclusters: Number of true biclusters
- pred_n_biclusters: Number of predicted biclusters  
- true_avg_n_genes: Average number of genes in true biclusters
- pred_avg_n_genes: Average number of genes in predicted biclusters
- true_avg_n_samples: Average number of samples in true biclusters
- pred_avg_n_samples: Average number of samples in predicted biclusters
- true_avg_area: Average area (genes × samples) of true biclusters
- pred_avg_area: Average area of predicted biclusters
- pred_avg_snr: Average SNR of predicted biclusters (if available)
- pred_median_snr: Median SNR of predicted biclusters (if available)
- pred_max_snr: Maximum SNR of predicted biclusters (if available)
- pred_min_snr: Minimum SNR of predicted biclusters (if available)

Notes:
- Metrics handle extra outputs (false positives) gracefully
- mAP calculation follows object detection conventions
- IoU thresholds can be customized
- Supports comparison by genes, samples, or both dimensions
- Missing or empty DataFrames are handled appropriately
""" 
import numpy as np
import pandas as pd
import pytest

from unpast.misc.eval.calc_average_precision import (
    _calc_mat_iou,
    calc_average_precision_at_thresh,
)
from unpast.tests.misc.eval.test_metrics import _gen_random_biclusters


def _build_bics_example():
    # schema of the data
    # 1,2,3,4 - true bics. 1,2,5 - predicted bics
    # gene \ sample      s1      s2     s3    s4
    #       -------------------------------------------------
    #        g1 |       12___   123__  1_3__ __3__
    #
    #        g2 |       1__45   1_3__  1_3_5 __3__
    #
    #        g3 |       ____5   __3__  __3_5 __3__

    bics = pd.DataFrame(
        {
            "genes": [
                {"g1", "g2"},
                {"g1"},
                {"g1", "g2", "g3"},
                {"g2"},
                {"g2", "g3"},
            ],
            "samples": [
                {"s1", "s2", "s3"},
                {"s1", "s2"},
                {"s2", "s3", "s4"},
                {"s1"},
                {"s1", "s3"},
            ],
        },
        index=["bic1", "bic2", "bic3", "bic4", "bic5"],
    )
    exprs = pd.DataFrame(
        np.random.random([10, 10]),
        index=[f"g{i}" for i in range(10)],
        columns=[f"s{i}" for i in range(10)],
    )
    true_bics = bics.iloc[[0, 1, 2, 3]]
    pred_bics = bics.iloc[[0, 1, 4]]
    return true_bics, pred_bics, exprs


def test__calc_mat_iou():
    """Test IoU / Jaccard similarity matrix calculation."""
    ###
    bics_true, bics_pred, exprs = _build_bics_example()

    mat_iou = _calc_mat_iou(bics_pred, bics_true, exprs=exprs, method="Jaccard")
    expected_mat_iou = pd.DataFrame(
        data=np.array(
            [
                [6 / 6, 2 / 6, 4 / 11, 1 / 6],
                [2 / 6, 6 / 6, 1 / 10, 0 / 3],
                [2 / 8, 0 / 8, 2 / 11, 1 / 4],
            ]
        ),
        index=bics_pred.index,
        columns=bics_true.index,
    )

    pd.testing.assert_frame_equal(mat_iou, expected_mat_iou)

    # no exact vals for ARI
    mat_iou = _calc_mat_iou(bics_pred, bics_true, exprs=exprs, method="ARI")
    # it is different from Jaccard in non-monotonic way
    assert mat_iou.loc["bic1", "bic3"] < mat_iou.loc["bic1", "bic2"]
    assert expected_mat_iou.loc["bic1", "bic3"] > expected_mat_iou.loc["bic1", "bic2"]

    assert mat_iou.loc["bic2", "bic1"] == mat_iou.loc["bic1", "bic2"]
    assert mat_iou.loc["bic1", "bic1"] == 1.0

    # expected_mat_iou = pd.DataFrame(
    #     data=np.array(
    #         [
    #             [6 / 6, 2 / 6, 4 / 11, 1 / 6],
    #             [2 / 6, 6 / 6, 1 / 10, 0 / 3],
    #             [2 / 8, 0 / 8, 2 / 11, 1 / 4],
    #         ]
    #     ),
    #     index=bics_pred.index,
    #     columns=bics_true.index,
    # )

    # pd.testing.assert_frame_equal(mat_iou, expected_mat_iou)


def test_calc_average_precision_at_thresh_smoke():
    rand = np.random.RandomState(42)
    n_genes = 100
    n_samples = 100
    row_names = [f"g_{i}" for i in range(n_genes)]
    col_names = [f"s_{i}" for i in range(n_samples)]

    true_bics = _gen_random_biclusters(rand, col_names, row_names, n_bics=10)
    pred_bics = _gen_random_biclusters(rand, col_names, row_names, n_bics=15)
    pred_bics["SNR"] = rand.rand(pred_bics.shape[0]) * 10.0
    exprs = pd.DataFrame(
        rand.rand(n_genes, n_samples), columns=col_names, index=row_names
    )

    map_score = calc_average_precision_at_thresh(true_bics, pred_bics, method="Jaccard")
    assert 0.0 <= map_score <= 1.0

    map_score_ari = calc_average_precision_at_thresh(
        true_bics, pred_bics, exprs=exprs, method="ARI"
    )
    assert 0.0 <= map_score_ari <= 1.0


@pytest.mark.parametrize("method", ["Jaccard", "ARI"])
def test_map_metric(method):
    pred_bics_dict = {}
    for i in range(1, 10):
        size = i
        genes = samples = [str(100 * i + j) for j in range(size)]
        pred_bics_dict[f"bic_{i}"] = {
            "genes": set(genes),
            "samples": set(samples),
            "SNR": float(i),
        }

    pred_bics = pd.DataFrame.from_dict(pred_bics_dict).T

    def _shift(vals, i):
        return set([str(int(v) + i) for v in vals])

    gt_bics = pred_bics.copy()
    del gt_bics["SNR"]
    for name, bic in gt_bics.iterrows():
        gt_bics.at[name, "genes"] = _shift(bic["genes"], 1)
        gt_bics.at[name, "samples"] = _shift(bic["samples"], 1)

    exprs = pd.DataFrame(
        np.random.rand(1100, 1100),
        index=[str(i) for i in range(1100)],
        columns=[str(i) for i in range(1100)],
    )

    map_score = calc_average_precision_at_thresh(
        gt_bics, pred_bics, exprs=exprs, method=method
    )

    pred_bics_no_good_bic = pred_bics.copy()
    # bic5 has IoU < 0.5, bic7 has IoU >= 0.5
    # pred_bics_no_good_bic.drop(index='bic_5', inplace=True)
    pred_bics_no_good_bic.drop(index="bic_7", inplace=True)
    map_score_no_good_bic = calc_average_precision_at_thresh(
        gt_bics, pred_bics_no_good_bic, exprs=exprs, method=method
    )
    assert map_score > map_score_no_good_bic

    pred_bics_no_bad_bic = pred_bics.copy()
    pred_bics_no_bad_bic.drop(index="bic_1", inplace=True)
    map_score_no_bad_bic = calc_average_precision_at_thresh(
        gt_bics, pred_bics_no_bad_bic, exprs=exprs, method=method
    )
    assert map_score == map_score_no_bad_bic

    pred_bics_bad_order = pred_bics.copy()
    pred_bics_bad_order["SNR"] = pred_bics_bad_order["SNR"].values[::-1]
    map_score_bad_order = calc_average_precision_at_thresh(
        gt_bics, pred_bics_bad_order, exprs=exprs, method=method
    )
    assert map_score > map_score_bad_order

    pred_bics_bad_order_no_bad_bic = pred_bics_bad_order.copy()
    pred_bics_bad_order_no_bad_bic.drop(index="bic_1", inplace=True)
    map_score_bad_order_no_bad_bic = calc_average_precision_at_thresh(
        gt_bics, pred_bics_bad_order_no_bad_bic, exprs=exprs, method=method
    )
    assert map_score_bad_order_no_bad_bic > map_score_bad_order

    pred_bics_two_9_detections = pred_bics.copy()
    pred_bics_two_9_detections = pd.concat(
        [pred_bics_two_9_detections, pred_bics.loc[["bic_9"]]]
    )
    map_score_two_9_detections = calc_average_precision_at_thresh(
        gt_bics, pred_bics_two_9_detections, exprs=exprs, method=method
    )
    assert map_score > map_score_two_9_detections


def test_map_edge_cases():
    # No ground truth biclusters
    pred_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g4"}],
            "samples": [{"s1", "s2"}, {"s3", "s4"}],
            "SNR": [2.0, 1.5],
        }
    )
    map_score = calc_average_precision_at_thresh(
        pd.DataFrame(columns=["genes", "samples"]), pred_bics
    )
    assert map_score == 0.0

    # No predicted biclusters
    true_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g4"}],
            "samples": [{"s1", "s2"}, {"s3", "s4"}],
        }
    )
    map_score = calc_average_precision_at_thresh(
        true_bics, pd.DataFrame(columns=["genes", "samples", "SNR"])
    )
    assert map_score == 0.0

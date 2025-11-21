import numpy as np
import pandas as pd
import pytest

from unpast.misc.eval.calc_average_precision import (
    _calc_mat_iou,
    calc_average_precision_at_thresh,
)
from unpast.misc.eval.metrics import calc_metrics
from unpast.tests.test_utils import _hash_table


@pytest.mark.parametrize("n_genes_samples", [(10, 10), (8, 10)])
def test_reproducible(n_genes_samples):
    """Test metrics calculations give exactly the same results as now."""
    true_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2", "g6"}, {"g1", "g5", "g7"}, {"g1", "g4"}],
            "samples": [
                {"s1", "s2", "s6"},
                {"s2", "s4", "s5"},
                {"s3", "s4", "s5", "s7"},
            ],
        }
    )

    pred_bics = pd.DataFrame(
        {
            "genes": [
                {"g1", "g2", "g6"},
                {"g1", "g2", "g3"},
                {"g1", "g5"},
                {"g6", "g7"},
            ],
            "samples": [
                {"s1", "s2", "s6"},
                {"s3", "s4", "s5"},
                {"s2", "s3", "s7"},
                {"s1", "s2"},
            ],
            "SNR": [3.5, 2.5, 1.5, 0.5],
        }
    )

    n_genes, n_samples = n_genes_samples
    data = pd.DataFrame(
        np.ones((n_genes, n_samples)),
        columns=[f"s{i}" for i in range(n_samples)],
        index=[f"g{i}" for i in range(n_genes)],
    )

    pred_bics["n_genes"] = pred_bics["genes"].apply(len)
    pred_bics["n_samples"] = pred_bics["samples"].apply(len)
    true_bics["n_genes"] = true_bics["genes"].apply(len)
    true_bics["n_samples"] = true_bics["samples"].apply(len)

    metrics = calc_metrics(true_bics, pred_bics, data)
    assert metrics["wARIs"] == 0.3

    # smoke test
    metrics = calc_metrics(true_bics, pred_bics, data.iloc[:8, :8])
    assert metrics["wARIs"] == 0.0  # correspondence removed by pval thresholding


def _gen_random_biclusters(rand, cols, inds, n_bics):
    gene_weights = np.power(0.9, np.arange(len(inds)))
    gene_weights /= gene_weights.sum()
    sample_weights = np.power(0.9, np.arange(len(cols)))
    sample_weights /= sample_weights.sum()

    bics = {}
    for i in range(n_bics):
        n_g = rand.randint(5, 15)
        n_s = rand.randint(5, 15)

        genes = set(rand.choice(inds, size=n_g, replace=False, p=gene_weights))
        samples = set(rand.choice(cols, size=n_s, replace=False, p=sample_weights))
        bics[f"bic_{i}"] = {"genes": genes, "samples": samples}

    return pd.DataFrame.from_dict(bics).T


def test_reproducible_big_random():
    rand = np.random.RandomState(42)
    n_genes = 50
    n_samples = 50
    data = pd.DataFrame(
        rand.rand(n_genes, n_samples),
        columns=[f"s_{i}" for i in range(n_samples)],
        index=[f"g_{i}" for i in range(n_genes)],
    )

    n_true_bics = 5
    n_pred_bics = 20

    repeated_metrics = []

    for i in range(10):
        # use weights to force bics to be often the same
        true_bics = _gen_random_biclusters(
            rand, data.columns.values, data.index.values, n_true_bics
        )
        pred_bics = _gen_random_biclusters(
            rand, data.columns.values, data.index.values, n_pred_bics
        )

        # todo: avoid requiring extra columns for the metrics calculations
        pred_bics["n_genes"] = pred_bics["genes"].apply(len)
        pred_bics["n_samples"] = pred_bics["samples"].apply(len)
        pred_bics["SNR"] = range(len(pred_bics))
        true_bics["n_genes"] = true_bics["genes"].apply(len)
        true_bics["n_samples"] = true_bics["samples"].apply(len)

        metrics = calc_metrics(true_bics, pred_bics, data)
        repeated_metrics.append(metrics)

    metrics_df = pd.DataFrame(repeated_metrics)
    assert _hash_table(metrics_df.drop(columns=["AP_50_95", "AP_ARI_50_95"])) == 18424040714768897732
    assert _hash_table(metrics_df.drop(columns=["AP_ARI_50_95"])) == 8765606674938893851
    assert _hash_table(metrics_df) == 4212380896713124175


    # different extra calculations:
    params_metrics = {}
    for measure in ["ARI", "Jaccard"]:
        for adjust_pvals in ["B", "BH", False]:
            params_metrics[f"{measure}_{adjust_pvals}"] = calc_metrics(
                true_bics,
                pred_bics,
                data,
                matching_measure=measure,
                adjust_pvals=adjust_pvals,
            )
    metrics_df_params = pd.DataFrame(params_metrics)
    assert (
        _hash_table(metrics_df_params.drop(index=["AP_50_95", "AP_ARI_50_95"])) == 9152769111870831455
    )
    assert _hash_table(metrics_df_params.drop(index=["AP_ARI_50_95"])) == 6034197052908192337
    assert _hash_table(metrics_df_params) == 9234669327890087507


def test_calc_metrics_empty_data():
    exprs = pd.DataFrame(
        np.random.rand(10, 10),
        index=[f"g{i}" for i in range(10)],
        columns=[f"s{i}" for i in range(10)],
    )
    true_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g4"}],
            "samples": [{"s1", "s2"}, {"s3", "s4"}],
        }
    )
    pred_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g9"}],
            "samples": [{"s1", "s2"}, {"s3", "s9"}],
            "SNR": [2.0, 1.5],
        }
    )

    # todo: avoid requiring extra columns for the metrics calculations
    pred_bics["n_genes"] = pred_bics["genes"].apply(len)
    pred_bics["n_samples"] = pred_bics["samples"].apply(len)
    pred_bics["SNR"] = range(len(pred_bics))
    true_bics["n_genes"] = true_bics["genes"].apply(len)
    true_bics["n_samples"] = true_bics["samples"].apply(len)

    # empty data
    empty_true_bics = true_bics.copy()[:0]
    empty_pred_bics = pred_bics.copy()[:0]

    metrics_11 = calc_metrics(true_bics, pred_bics, exprs)

    metrics_10 = calc_metrics(true_bics, empty_pred_bics, exprs)
    assert metrics_10["FDR_bic"] == 1.0
    assert all(v == 0.0 for (k, v) in metrics_10.items() if k != "FDR_bic")
    assert metrics_10.keys() == metrics_11.keys(), "Metrics keys differ in edge case"

    with pytest.raises(ValueError):
        calc_metrics(empty_true_bics, pred_bics, exprs)

    with pytest.raises(ValueError):
        calc_metrics(empty_true_bics, empty_pred_bics, exprs)

    # metrics_01 = calc_metrics(empty_true_bics, pred_bics, exprs)
    # assert metrics_01["wARIs"] == 1.0

    # metrics_00 = calc_metrics(empty_true_bics, empty_pred_bics, exprs)
    # assert metrics_00["wARIs"] == 0.0

    # # has same keys as full metrics
    # for m in metrics_01, metrics_10, metrics_00:
    #     assert m.keys() == metrics_11.keys()


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
    assert mat_iou.loc['bic1', 'bic3'] < mat_iou.loc['bic1', 'bic2']
    assert expected_mat_iou.loc['bic1', 'bic3'] > expected_mat_iou.loc['bic1', 'bic2']
    
    assert mat_iou.loc['bic2', 'bic1'] == mat_iou.loc['bic1', 'bic2']
    assert mat_iou.loc['bic1', 'bic1'] == 1.0


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

    map_score = calc_average_precision_at_thresh(true_bics, pred_bics, method='Jaccard')
    assert 0.0 <= map_score <= 1.0

    map_score_ari = calc_average_precision_at_thresh(true_bics, pred_bics, exprs=exprs, method='ARI')
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

    map_score = calc_average_precision_at_thresh(gt_bics, pred_bics, exprs=exprs, method=method)

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
    map_score_bad_order = calc_average_precision_at_thresh(gt_bics, pred_bics_bad_order, exprs=exprs, method=method)
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

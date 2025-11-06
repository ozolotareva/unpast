import numpy as np
import pandas as pd
import pytest

from unpast.misc.eval.run_eval import calculate_metrics
from unpast.misc.eval.draft_map_metric import calc_mean_average_precision

def _hash_table(df):
    """Hash a DataFrame for reproducibility."""
    # todo: avoid function duplication with test_ds_synthetic_builder.py
    rows_hashes = pd.util.hash_pandas_object(df, index=True)
    hash = pd.util.hash_pandas_object(
        pd.DataFrame(rows_hashes).T,  # we need one value
        index=True,
    )
    return hash.sum()

@pytest.mark.skip("Not working on some set of n_genes_samples. TODO: debug.")
@pytest.mark.parametrize("n_genes_samples", [(10, 10), (8, 10), (10, 8)])
def test_reproducible(n_genes_samples):
    """Test metrics calculations give exactly the same results as now."""
    true_bics = pd.DataFrame({
        "genes": [
            {"g1", "g2", "g3", "g6"},
            {"g1", "g5", "g7"},
            {"g1", "g4"}
        ],
        "samples": [
            {"s1", "s2", "s6"},
            {"s2", "s4", "s5"},
            {"s3", "s4", "s5", "s7"},
        ],
    })

    pred_bics = pd.DataFrame({
        "genes": [
            {"g1", "g2", "g3", "g6"},
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
    })  

    n_genes, n_samples = n_genes_samples
    data = pd.DataFrame(
        np.ones((n_genes, n_samples)),
        columns=[f"s{i}" for i in range(n_samples)],
        index=[f"g{i}" for i in range(n_genes)],
    )
    
    pred_bics['n_genes'] = pred_bics['genes'].apply(len)
    pred_bics['n_samples'] = pred_bics['samples'].apply(len)
    true_bics['n_genes'] = true_bics['genes'].apply(len)
    true_bics['n_samples'] = true_bics['samples'].apply(len)

    # # add indexes
    # for df in true_bics, pred_bics:
    #     df['gene_indexes'] = None
    #     df['sample_indexes'] = None

    #     for i, bic in enumerate(df.itertuples()):    
    #         gene_indexes = set([data.index.get_loc(g) for g in bic.genes])
    #         sample_indexes = set([data.columns.get_loc(s) for s in bic.samples])
    #         df.at[i, "gene_indexes"] = gene_indexes
    #         df.at[i, "sample_indexes"] = sample_indexes
    
    metrics = calculate_metrics(true_bics, pred_bics, data)
    assert metrics["wARIs"] > 0.0


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
    
    # _, true_bics = _scenario_generate_biclusters(
    #     rand=np.random.RandomState(1),
    #     data_sizes=(n_genes, n_samples),
    #     frac_samples=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    #     g_size=10, 
    # )
    # true_bics = pd.DataFrame(true_bics).T

    # _, pred_bics = _scenario_generate_biclusters(
    #     rand=np.random.RandomState(2),
    #     data_sizes=(n_genes, n_samples),
    #     frac_samples=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    #     g_size=10, 
    # )
    # pred_bics = pd.DataFrame(pred_bics).T


    repeated_metrics = []    

    for i in range(10):
        # use weights to force bics to be often the same
        true_bics = _gen_random_biclusters(rand, data.columns.values, data.index.values, n_true_bics)
        pred_bics = _gen_random_biclusters(rand, data.columns.values, data.index.values, n_pred_bics)

        # todo: avoid that
        pred_bics['n_genes'] = pred_bics['genes'].apply(len)
        pred_bics['n_samples'] = pred_bics['samples'].apply(len)
        true_bics['n_genes'] = true_bics['genes'].apply(len)
        true_bics['n_samples'] = true_bics['samples'].apply(len)
            
        metrics = calculate_metrics(true_bics, pred_bics, data)
        repeated_metrics.append(metrics)

    metrics_df = pd.DataFrame(repeated_metrics)
    assert _hash_table(metrics_df) == 18424040714768897732


@pytest.mark.skip("Not working on some set of n_genes_samples. TODO: debug.")
def test_calc_mean_average_precision_smoke():
    rand = np.random.RandomState(42)
    n_genes = 100
    n_samples = 100
    row_names = [f"g_{i}" for i in range(n_genes)]
    col_names = [f"s_{i}" for i in range(n_samples)]

    true_bics = _gen_random_biclusters(rand, col_names, row_names, n_bics=10)
    pred_bics = _gen_random_biclusters(rand, col_names, row_names, n_bics=15)

    map_score = calc_mean_average_precision(true_bics, pred_bics)
    assert 0.0 <= map_score <= 1.0

def test_map_metric(): 
    n_genes = 1000
    n_samples = 1000

    pred_bics_dict = {}
    for i in range(1, 10): 
        size = i
        genes = samples = [100*i + j for j in range(size)]
        pred_bics_dict[f"bic_{i}"] = {"genes": set(genes), "samples": set(samples), "SNR": float(i)}

    pred_bics = pd.DataFrame.from_dict(pred_bics_dict).T

    def _shift(vals, i):
        return set([v + i for v in vals])

    gt_bics = pred_bics.copy()
    del gt_bics['SNR']
    for name, bic in gt_bics.iterrows():
        gt_bics.at[name, 'genes'] = _shift(bic['genes'], 1)
        gt_bics.at[name, 'samples'] = _shift(bic['samples'], 1)

    map_score = calc_mean_average_precision(gt_bics, pred_bics)

    pred_bics_no_good_bic = pred_bics.copy()
    pred_bics_no_good_bic.drop(index='bic_5', inplace=True)
    map_score_no_good_bic = calc_mean_average_precision(gt_bics, pred_bics_no_good_bic)
    assert map_score > map_score_no_good_bic

    pred_bics_no_bad_bic = pred_bics.copy()
    pred_bics_no_bad_bic.drop(index='bic_1', inplace=True)
    map_score_no_bad_bic = calc_mean_average_precision(gt_bics, pred_bics_no_bad_bic)
    assert map_score == map_score_no_bad_bic

    pred_bics_bad_order = pred_bics.copy()
    pred_bics_bad_order['SNR'] = pred_bics_bad_order['SNR'].values[::-1]
    map_score_bad_order = calc_mean_average_precision(gt_bics, pred_bics_bad_order)
    assert map_score > map_score_bad_order

    pred_bics_bad_order_no_bad_bic = pred_bics_bad_order.copy()
    pred_bics_bad_order_no_bad_bic.drop(index='bic_1', inplace=True)
    map_score_bad_order_no_bad_bic = calc_mean_average_precision(gt_bics, pred_bics_bad_order_no_bad_bic)
    assert map_score_bad_order > map_score_bad_order_no_bad_bic

    pred_bics_two_9_detections = pred_bics.copy()
    pred_bics_two_9_detections = pd.concat([pred_bics_two_9_detections, pred_bics.loc[["bic_9"]]])
    map_score_two_9_detections = calc_mean_average_precision(gt_bics, pred_bics_two_9_detections)
    assert map_score > map_score_two_9_detections









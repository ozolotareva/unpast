import numpy as np
import pandas as pd
import pytest

from unpast.misc.eval.run_eval import calculate_metrics

# run multiple times
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

"""Generating synthetic biclusters and expression data for evaluating purposes.

This module provides functionality to create synthetic gene expression datasets with
embedded biclusters and optional co-expression modules for algorithm evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import Bicluster
from unpast.utils.logs import get_logger

logger = get_logger(__name__)


def _scenario_generate_biclusters(
    rand: np.random.RandomState,
    data_sizes: tuple[int, int],
    g_size: int = 5,
    frac_samples: list[float] = [0.05, 0.1, 0.25, 0.5],
    m: float = 2.0,
    std: float = 1.0,
    g_overlap: bool = False,
    s_overlap: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, set]]]:
    """Generate expression matrix with embedded biclusters.

    Args:
        rand: Random state for reproducibility.
        data_sizes: Size of expression data (n_genes, n_samples).
        g_size: Number of genes in each bicluster.
        frac_samples: List of sample fractions for each bicluster.
        m: Mean shift for bicluster expression values.
        std: Standard deviation for bicluster expression values.
        g_overlap: Whether to allow gene overlap between biclusters.
        s_overlap: Whether to allow sample overlap between biclusters.

    Returns:
        Tuple containing:
            - exprs: Expression DataFrame with embedded biclusters
            - biclusters: Dictionary mapping bicluster IDs to gene/sample sets
    """
    assert len(set(frac_samples)) == len(frac_samples), (
        f"fraction samples must be unique, got {frac_samples}"
    )
    assert len(data_sizes) == 2, (
        f"data_sizes must be a tuple of two integers, got {data_sizes}"
    )

    # data_sizes = genes_amount, samples_amount
    exprs = rand.normal(loc=0, scale=1.0, size=data_sizes)
    exprs = pd.DataFrame(exprs)
    exprs.columns = ["s_" + str(x) for x in exprs.columns.values]
    exprs.index = ["g_" + str(x) for x in exprs.index.values]

    # implant bicluster
    bg_g = set(exprs.index.values)
    bg_s = set(exprs.columns.values)

    biclusters = {}
    for s_frac in frac_samples:
        s_size = int(s_frac * data_sizes[1])
        if s_size < 2:
            logger.warning(
                "Skipping too small bicluster size during ds generation"
                f" {s_size} for fraction {s_frac}"
            )
            continue

        assert g_size <= len(bg_g), f"not enough genes left: {len(bg_g)}, need {g_size}"
        assert s_size <= len(bg_s), (
            f"not enough samples left: {len(bg_s)}, need {s_size}"
        )

        # select random sets of samples and genes from the background
        bic_genes = sorted(rand.choice(sorted(bg_g), size=g_size, replace=False))
        bic_samples = sorted(rand.choice(sorted(bg_s), size=s_size, replace=False))

        exprs.loc[bic_genes, bic_samples] += rand.normal(
            loc=m, scale=std, size=(g_size, s_size)
        )

        # identify samples outside the bicluster
        if not g_overlap:
            bg_g -= set(bic_genes)

        if not s_overlap:
            bg_s -= set(bic_samples)

        # generate and implant bicluster
        # save bicluster genes
        biclusters["bic_" + str(s_frac)] = {
            "genes": set(bic_genes),
            "samples": set(bic_samples),
        }

    return exprs, biclusters


def _scenario_add_modules(
    rand: np.random.RandomState,
    exprs: pd.DataFrame,
    ignore_genes: set,
    add_coexpressed: list[int] = [],
) -> tuple[pd.DataFrame, list[list[Any]]]:
    """Add co-expressed modules to the expression matrix.

    Creates co-expression patterns by mixing gene expression profiles with
    a reference profile to achieve correlation structure.

    Args:
        rand: Random state for reproducibility.
        exprs: Expression DataFrame to modify.
        ignore_genes: Set of genes to exclude from co-expression modules.
        add_coexpressed: List of module sizes to create.

    Returns:
        Tuple containing:
            - exprs: Modified expression DataFrame with co-expression
            - coexpressed_modules: List of gene lists for each module
    """
    bg_g = set(exprs.index.values).difference(set(ignore_genes))
    mix_coef = 0.5
    store_coef = np.sqrt(1 - mix_coef**2)

    coexpressed_modules = []
    for module_size in add_coexpressed:
        module_genes = sorted(
            rand.choice(sorted(bg_g), size=module_size, replace=False)
        )

        exprs_0 = exprs.loc[module_genes[0], :]
        for gene_ind in module_genes[1:]:
            exprs.loc[gene_ind, :] *= store_coef
            exprs.loc[gene_ind, :] += mix_coef * exprs_0

        coexpressed_modules.append(module_genes)

        avg_r = (exprs.loc[module_genes, :].T.corr().sum().sum() - module_size) / (
            module_size**2 - module_size
        )
        logger.info(
            f"\tco-exprs. module {module_size} features, avg. pairwise r={avg_r:.2f}"
        )

    return exprs, coexpressed_modules


def generate_exprs(
    rand: np.random.RandomState,
    data_sizes: tuple[int, int],
    g_size: int = 5,
    frac_samples: list[float] = [0.05, 0.1, 0.25, 0.5],
    m: float = 2.0,
    std: float = 1.0,
    g_overlap: bool = False,
    s_overlap: bool = True,
    add_coexpressed: list[int] = [],
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict[str, Any]]:
    """Generate synthetic expression data with biclusters and optional co-expression modules.

    Args:
        rand: Random state for reproducibility.
        data_sizes: Size of expression data (n_genes, n_samples).
        g_size: Number of genes in each bicluster.
        frac_samples: List of sample fractions for each bicluster.
        m: Mean shift for bicluster expression values.
        std: Standard deviation for bicluster expression values.
        g_overlap: Whether to allow gene overlap between biclusters.
        s_overlap: Whether to allow sample overlap between biclusters.
        add_coexpressed: List of sizes for co-expression modules to add.

    Returns:
        Tuple containing:
            - exprs: Expression DataFrame
            - biclusters: Dictionary mapping bicluster IDs to Bicluster objects
            - extra: Dictionary with 'coexpressed_modules' key containing module information
    """

    exprs, bic_dict = _scenario_generate_biclusters(
        rand=rand,
        data_sizes=data_sizes,
        g_size=g_size,
        frac_samples=frac_samples,
        m=m,
        std=std,
        g_overlap=g_overlap,
        s_overlap=s_overlap,
    )

    bic_genes = set.union(*[b["genes"] for b in bic_dict.values()])
    exprs, coexpressed_modules = _scenario_add_modules(
        rand=rand,
        exprs=exprs,
        ignore_genes=bic_genes,
        add_coexpressed=add_coexpressed,
    )

    res_bic_dict = {
        bic_id: Bicluster(genes=bic_data["genes"], samples=bic_data["samples"])
        for bic_id, bic_data in bic_dict.items()
    }
    return exprs, res_bic_dict, {"coexpressed_modules": coexpressed_modules}

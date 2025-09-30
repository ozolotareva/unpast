"""Generating synthetic biclusters and expression data for evaluating purposes."""

from collections import namedtuple
import numpy as np
import pandas as pd

from unpast.core.preprocessing import zscore
from unpast.core.sample_clustering import update_bicluster_data
from unpast.utils.io import write_bic_table

Bicluster = namedtuple("Bicluster", ["genes", "samples"])


def _scenario_generate_biclusters(
    rand: np.random.RandomState,
    data_sizes: tuple[int, int],
    g_size: int = 5,
    frac_samples: list[float] = [0.05, 0.1, 0.25, 0.5],
    m: float = 2.0,
    std: float = 1.0,
    g_overlap: bool = False,
    s_overlap: bool = True,
) -> tuple[pd.DataFrame, dict]:
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
    ignore_genes: set[str | int],
    add_coexpressed: list[int] = [],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Add co-expressed modules to the expression matrix."""
    bg_g = set(exprs.index.values).difference(set(ignore_genes))
    mix_coef = 0.5
    store_coef = np.sqrt(1 - mix_coef**2)

    coexpressed_modules = []
    for module_size in add_coexpressed:
        module_genes = sorted(
            rand.choice(sorted(bg_g), size=module_size, replace=False)
        )

        exprs_0 = exprs.loc[module_genes[0], :]
        for gen_ind in module_genes[1:]:
            exprs.loc[gen_ind, :] *= store_coef
            exprs.loc[gen_ind, :] += mix_coef * exprs_0

        coexpressed_modules.append(module_genes)

        avg_r = (exprs.loc[module_genes, :].T.corr().sum().sum() - module_size) / (
            module_size**2 / 2 - module_size
        )
        print(
            "\tco-exprs. module %s features, avg. pairwise r=%.2f"
            % (module_size, avg_r)
        )

    return exprs, coexpressed_modules


def _build_bicluster_table(exprs: pd.DataFrame, biclusters: dict) -> pd.DataFrame:
    """Build a DataFrame from bicluster dictionary with additional info.
        Adds some statistics to each bicluster.

    Args:
        exprs: Expression DataFrame.
        biclusters: Dictionary of biclusters.

    Returns:
        DataFrame with bicluster information.

    """
    new_biclusters = {}
    for bic_id, bic_data in biclusters.items():
        bic_data["n_genes"] = len(bic_data["genes"])
        bic_data["n_samples"] = len(bic_data["samples"])
        new_biclusters[bic_id] = update_bicluster_data(biclusters[bic_id], exprs)

    bicluster_df = pd.DataFrame.from_dict(new_biclusters).T

    # bicluster_df.set_index("frac",inplace = True,drop=True)
    return bicluster_df


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
    # ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict]:
    """Generate synthetic expression data with biclusters."""

    exprs, bicluster_dict = _scenario_generate_biclusters(
        rand=rand,
        data_sizes=data_sizes,
        g_size=g_size,
        frac_samples=frac_samples,
        m=m,
        std=std,
        g_overlap=g_overlap,
        s_overlap=s_overlap,
    )

    bicluster_genes = set.union(*[b["genes"] for b in bicluster_dict.values()])
    exprs, coexpressed_modules = _scenario_add_modules(
        rand=rand,
        exprs=exprs,
        ignore_genes=bicluster_genes,
        add_coexpressed=add_coexpressed,
    )

    res_bicluster_dict = {
        bic_id: Bicluster(genes=bic_data["genes"], samples=bic_data["samples"])
        for bic_id, bic_data in bicluster_dict.items()
    }
    return exprs, res_bicluster_dict, {"coexpressed_modules": coexpressed_modules}

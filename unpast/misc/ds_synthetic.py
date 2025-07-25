"""Generating synthetic biclusters and expression data for evaluating purposes."""

import numpy as np
import pandas as pd

from unpast.core.preprocessing import zscore
from unpast.core.sample_clustering import update_bicluster_data
from unpast.utils.io import write_bic_table


def generate_biclusters_common(
    data_sizes,
    g_size=5,
    frac_samples=[0.05, 0.1, 0.25, 0.5],
    m=2.0,
    std=1,
    g_overlap=False,
    s_overlap=True,
    seed=42,
) -> tuple[pd.DataFrame, dict]:
    assert len(set(frac_samples)) == len(frac_samples), (
        f"fraction samples must be unique, got {frac_samples}"
    )
    assert len(data_sizes) == 2, (
        f"data_sizes must be a tuple of two integers, got {data_sizes}"
    )

    # data_sizes = genes_amount, samples_amount
    np.random.seed(seed)
    exprs = np.random.normal(loc=0, scale=1.0, size=data_sizes)
    exprs = pd.DataFrame(exprs)
    exprs.columns = ["s_" + str(x) for x in exprs.columns.values]
    exprs.index = ["g_" + str(x) for x in exprs.index.values]

    # implant bicluster
    bg_g = set(exprs.index.values)
    bg_s = set(exprs.columns.values)

    np.random.seed(seed)
    seeds = np.random.choice(range(0, 1000000), size=len(frac_samples), replace=False)
    biclusters = {}
    for s_frac, cur_seed in zip(frac_samples, seeds):
        s_size = int(s_frac * data_sizes[1])

        # select random sets of samples and genes from the background
        np.random.seed(cur_seed)
        bic_genes = sorted(np.random.choice(sorted(bg_g), size=g_size, replace=False))

        np.random.seed(cur_seed)
        bic_samples = sorted(np.random.choice(sorted(bg_s), size=s_size, replace=False))

        np.random.seed(cur_seed)
        exprs.loc[bic_genes, bic_samples] += np.random.normal(
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
            "frac": s_frac,
        }

    return exprs, biclusters


def _standard_add_modules(
    exprs,
    ignore_genes,
    seed,
    add_coexpressed=[],
):
    """Add co-expressed modules to the expression matrix."""
    bg_g = set(exprs.index.values).difference(set(ignore_genes))
    mix_coef = 0.5
    store_coef = np.sqrt(1 - mix_coef**2)

    coexpressed_modules = []
    np.random.seed(seed + 1)
    seeds = np.random.choice(
        range(0, 1000000), size=len(add_coexpressed), replace=False
    )

    for module_size, cur_seed in zip(add_coexpressed, seeds):
        np.random.seed(cur_seed)
        module_genes = sorted(
            np.random.choice(sorted(bg_g), size=module_size, replace=False)
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


def generate_exprs(
    data_sizes,
    g_size=5,
    frac_samples=[0.05, 0.1, 0.25, 0.5],
    m=2.0,
    std=1,
    z=True,
    outdir="./",
    outfile_basename="",
    g_overlap=False,
    s_overlap=True,
    seed=42,
    add_coexpressed=[],
):
    exprs, biclusters = generate_biclusters_common(
        data_sizes,
        g_size=g_size,
        frac_samples=frac_samples,
        m=m,
        std=std,
        g_overlap=g_overlap,
        s_overlap=s_overlap,
        seed=seed,
    )

    bicluster_genes = set.union(*[b["genes"] for b in biclusters.values()])
    exprs, coexpressed_modules = _standard_add_modules(
        exprs,
        ignore_genes=bicluster_genes,
        seed=seed,
        add_coexpressed=add_coexpressed,
    )

    if z:
        # center to 0 and scale std to 1
        exprs = zscore(exprs)

    # calculate bicluster SNR
    # distinguish up- and down-regulated genes
    for b in biclusters.values():
        b["n_genes"] = len(b["genes"])
        b["n_samples"] = len(b["samples"])
    for bic_id in biclusters.keys():
        biclusters[bic_id] = update_bicluster_data(biclusters[bic_id], exprs)

    biclusters = pd.DataFrame.from_dict(biclusters).T
    # biclusters.set_index("frac",inplace = True,drop=True)

    if outfile_basename:
        exprs_file = outdir + outfile_basename + ".data.tsv.gz"
        print("input data:", exprs_file)
        exprs.to_csv(exprs_file, sep="\t")

        # save ground truth
        bic_file_name = outdir + outfile_basename + ".true_biclusters.tsv.gz"
        print("true biclusters:", bic_file_name)
        write_bic_table(biclusters, bic_file_name)

    return exprs, biclusters, coexpressed_modules

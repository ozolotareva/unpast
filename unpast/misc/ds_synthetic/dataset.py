"""Dataset generation and configuration for synthetic biclusters.

This module provides predefined dataset schemas and generation functions for
creating modular and scenario-based synthetic datasets with embedded biclusters.
"""

import random
from pathlib import Path
from typing import Mapping

import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import save_dataset_entry
from unpast.misc.ds_synthetic.entry import DSEntryBlueprint


def get_modular_dataset_blueprint() -> dict[str, DSEntryBlueprint]:
    """Get a modular dataset schema for basic bicluster testing.

    Returns:
        Dictionary mapping dataset names to DSEntryBlueprint configurations
        for simple biclusters with various parameter settings.
    """
    ds_schema: dict[str, DSEntryBlueprint] = {
        # "GeneExprs": DSEntryBlueprint(
        #     scenario_type="GeneExprs",
        #     data_sizes=(10, 10),
        #     frac_samples=[0.2, 0.3],
        # ),
        "simple_3": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "simple_5": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 5)
        ),
        "over_half": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(15, 15)
        ),
        "over_half_rows": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(15, 5)
        ),
        "over_half_cols": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 15)
        ),
        **{
            f"more_rows_{i:02d}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(10 * i, 10), bic_sizes=(5, 5)
            )
            for i in (3, 10, 30)
        },
        **{
            f"more_cols_{i:02d}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(10, 10 * i), bic_sizes=(5, 5)
            )
            for i in (3, 10, 30)
        },
        **{
            f"small_mu_{f:.1f}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 5), bic_mu=f
            )
            for f in [1.0, 2.0, 3.0, 5.0, 7.0]
        },
        **{
            f"big_mu_{f:.1f}": DSEntryBlueprint(
                scenario_type="Simple",
                data_sizes=(200, 200),
                bic_sizes=(50, 50),
                bic_mu=f,
            )
            for f in [1.0, 2.0, 3.0, 5.0, 7.0]
        },
        **{
            f"scaled_{i:02d}": DSEntryBlueprint(
                scenario_type="Simple",
                data_sizes=(20 * i, 20 * i),
                bic_sizes=(5 * i, 5 * i),
            )
            for i in (1, 3, 10, 30)
        },
        **{
            f"scaled_data_{i:02d}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(20 * i, 20 * i), bic_sizes=(5, 5)
            )
            for i in (1, 3, 10)
        },
        # should be ignored, because
        # 1) alternative clastering is not checked and
        # 2) biclusters here are defined ROW-wise
        # e.g. biclustering A and B:
        #     A: (1-10x1-10, 11-20x1-5) and
        #     B: (1-10x6-10, 1-20x1-5)
        # Current implementation does not calculate metrics correctly for such cases.
        # "same_rows": DSEntryBlueprint(
        #     scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-9x0-9", "0-5x10-19"]
        # ),
        # "same_cols": DSEntryBlueprint(
        #     scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-5x0-9", "10-19x0-9"]
        # ),
        # "common_corner": DSEntryBlueprint(
        #     scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-9x0-9", "7-16x7-16"]
        # ),
        # some_rows_overlap_{i]} is not supported with row-wise
        **{
            f"cols_overlap_{i}": DSEntryBlueprint(
                scenario_type="SimpleMult",
                data_sizes=(30, 30),
                bic_codes=["0-9x0-9", f"10-19x{10 - i}-{19 - i}"],
            )
            for i in range(5)
        },
        **{
            f"diff_size_{i:02d}": DSEntryBlueprint(
                scenario_type="SimpleMult",
                data_sizes=(100, 100),
                bic_codes=["0-4x0-4", f"{5}-{4 + 5 * i}x{5}-{4 + 5 * i}"],
            )
            for i in [1, 3, 10]
        },
        **{
            f"many_small_{n:02d}": DSEntryBlueprint(
                scenario_type="SimpleMult",
                data_sizes=(100, 100),
                bic_codes=[
                    f"{i * 5}-{i * 5 + 4}x{i * 5}-{i * 5 + 4}" for i in range(n)
                ],
            )
            for n in [3, 5, 10, 15]
        },
        **{
            f"complex_bg_r{n:02d}": DSEntryBlueprint(
                scenario_type="CorrelatedBG",
                data_sizes=(20, 20),
                bic_sizes=(5, 5),
                bg_rank=n,
            )
            for n in [2, 3, 5, 10]
        },
    }
    return ds_schema


def get_scenario_dataset_blueprint(
    scale: float = 1.0,
) -> dict[str, DSEntryBlueprint]:
    """Get a dataset schema for realistic gene expression scenarios.

    Creates configurations for testing different scenarios:
    - Scenario A: No gene/sample overlap, no co-expression
    - Scenario C: Sample overlap allowed, with co-expression modules

    Args:
        scale: Scale factor for the dataset size (useful for debugging on smaller datasets).

    Returns:
        Dictionary mapping dataset names to DSEntryBlueprint configurations
        for gene expression scenarios with varying bicluster sizes.
    """
    common_args = {
        "m": 4,
        "std": 1,
        # fractions of samples included to each subtype
        "frac_samples": [0.05, 0.1, 0.25, 0.5],
    }

    scenario_args = {
        "A": {"add_coexpressed": [], "g_overlap": False, "s_overlap": False},
        "B": {"add_coexpressed": [], "g_overlap": False, "s_overlap": True},
        "C": {
            "add_coexpressed": [int(500 * scale)]
            * 4,  # add 4 co-expression modules of 500 genes each, with avg. r=0.5
            "g_overlap": False,
            "s_overlap": True,
        },
    }

    data_sizes = (
        int(10000 * scale),  # number of genes
        int(200 * scale),  # number of samples
    )

    ds_schema: dict[str, DSEntryBlueprint] = {}
    for letter in ["A", "C"]:
        for bic_genes in [5, 50, 500]:
            bic_genes = max(1, int(bic_genes * scale))
            ds_schema[f"{letter}_{bic_genes}"] = DSEntryBlueprint(
                scenario_type="GeneExprs",
                data_sizes=data_sizes,
                g_size=bic_genes,
                **common_args,
                **scenario_args[letter],
            )
    return ds_schema


def build_dataset(
    entry_builders: Mapping[str, DSEntryBlueprint],
    seed_prefix: str = "",
    output_dir: str = "./synthetic_ds",
    show_images: bool = True,
) -> pd.DataFrame:
    """Build a dataset from the given generators.

    Generates multiple dataset entries according to the provided blueprints,
    saves expression data, bicluster information, and heatmap visualizations.

    Args:
        entry_builders: A mapping of dataset entry names to their corresponding bicluster generators.
        seed_prefix: Prefix for random seed generation to ensure reproducibility.
        output_dir: Directory to save the generated dataset entries.
        show_images: Whether to display images during generation.

    Returns:
        pd.DataFrame: DataFrame summarizing the built dataset entries.
    """
    output_path = Path(output_dir)
    build_info = {}
    for name, builder in entry_builders.items():
        exprs, bic_df, extra_info = builder.build(
            random.Random(seed_prefix + name).randint(0, 10**6)
        )

        save_path = output_path / name
        build_info[name] = save_dataset_entry(
            name, exprs, bic_df, extra_info, save_path, show_images
        )
    return pd.DataFrame.from_dict(build_info, orient="index")

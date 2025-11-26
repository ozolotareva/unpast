"""Dataset generation and configuration for synthetic biclusters.

This module provides predefined dataset schemas and generation functions for
creating standard and scenario-based synthetic datasets with embedded biclusters.
"""

import random
from pathlib import Path
from typing import Mapping

import pandas as pd

from unpast.misc.ds_synthetic.ds_utils import save_dataset_entry
from unpast.misc.ds_synthetic.entry import DSEntryBlueprint


def get_standard_dataset_blueprint() -> dict[str, DSEntryBlueprint]:
    """Get a standard dataset schema for basic bicluster testing.

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
        "more_rows": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(100, 10), bic_sizes=(5, 5)
        ),
        "more_cols": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 100), bic_sizes=(5, 5)
        ),
        **{
            f"test_mu_{f}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(5, 5), bic_mu=f
            )
            for f in [1.0, 2.0, 3.0, 5.0, 7.0]
        },
        "same_rows": DSEntryBlueprint(
            scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-9x0-9", "0-9x10-19"]
        ),
        "same_cols": DSEntryBlueprint(
            scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-9x0-9", "10-19x0-9"]
        ),
        "common_angle": DSEntryBlueprint(
            scenario_type="SimpleMult", data_sizes=(30, 30), bic_codes=["0-9x0-9", "7-16x7-16"]
        ),
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

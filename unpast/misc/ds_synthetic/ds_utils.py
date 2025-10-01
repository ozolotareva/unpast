from collections import namedtuple
from pathlib import Path

import pandas as pd
from unpast.utils.io import write_bic_table
from unpast.utils.visualization import plot_biclusters_heatmap

Bicluster = namedtuple("Bicluster", ["genes", "samples"])


def save_dataset_entry(
    name: str,
    exprs: pd.DataFrame,
    bicluster_df: pd.DataFrame,
    extra_info: dict,
    ds_path: Path,
    show_images: bool = True,
) -> dict:
    """Save dataset entry files and plot heatmap.

    Args:
        name (str): Name of the dataset entry.
        exprs (pd.DataFrame): Expression data.
        bicluster_df (pd.DataFrame): Bicluster information.
        extra_info (dict): Additional information.
        ds_path (Path): Output directory path.
        show_images (bool): Whether to show images.

    Returns:
        dict: Paths to saved files and extra information.
    """
    ds_path.mkdir(parents=True, exist_ok=True)
    exprs_file = ds_path / "data.tsv"
    bicluster_file = ds_path / "true_biclusters.tsv"

    exprs.to_csv(exprs_file, sep="\t")
    write_bic_table(bicluster_df, str(bicluster_file))

    plot_biclusters_heatmap(
        exprs=exprs,
        biclusters=bicluster_df,
        coexpressed_modules=extra_info.get("coexpressed_modules", []),
        fig_title=name,
        fig_path=ds_path / "heatmap.png",
        visualize=show_images,
    )

    return {
        "exprs_file": str(exprs_file),
        "bic_file": str(bicluster_file),
        "extra_info": extra_info,
    }

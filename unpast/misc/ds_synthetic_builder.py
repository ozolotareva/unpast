import os
import random
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from unpast.core.preprocessing import zscore
from unpast.misc.ds_synthetic import _build_bicluster_table, generate_exprs, Bicluster
from unpast.utils.io import write_bic_table
from unpast.utils.visualization import plot_biclusters_heatmap


class SyntheticBiclusterGeneratorABC:
    """Class to generate synthetic biclusters."""

    def build(self, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Build synthetic biclusters and expression data."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_args(self) -> dict:
        """Describe the scenario."""
        raise NotImplementedError("This method should be implemented in subclasses.")


def build_simple_biclusters(
    data_sizes: tuple[int, int],
    bic_sizes: tuple[int, int],
    rand: np.random.RandomState,
    bic_mu: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict]:
    """Build simple biclusters:
        exprs = N(0, 1)
        bic = N(bic_mu, 1)

    Args:
        data_sizes (tuple[int, int]): Size of the expression data.
        bic_sizes (tuple[int, int]): Size of the biclusters.
        seed (int): Random seed.
        bic_mu (float): Mean of the biclusters.

    Returns:
        tuple[pd.DataFrame, list[tuple[list[int], list[int]]], dict]:
            exprs (pd.DataFrame): Expression data.
            biclusters (list[tuple[list[int], list[int]]]): List of biclusters.
            some additional data
    """
    table = pd.DataFrame(rand.normal(0, 1, size=data_sizes))

    bic_rows = list(range(bic_sizes[0]))
    bic_cols = list(range(bic_sizes[1]))
    table.iloc[bic_rows, bic_cols] += bic_mu

    return table, {"bic": Bicluster(bic_rows, bic_cols)}, {}


def _shuffle_exprs(exprs: pd.DataFrame, rand: np.random.RandomState) -> pd.DataFrame:
    """Shuffle the expression data.
        preserves index-value correspondence
        i.e. changes only iloc, not loc

    Args:
        exprs (pd.DataFrame): Expression data.
        rand (np.random.RandomState): Random state for shuffling.

    Returns:
        pd.DataFrame: Shuffled expression data.
    """
    new_index = rand.permutation(exprs.index)
    new_columns = rand.permutation(exprs.columns)
    return pd.DataFrame(
        exprs.loc[new_index, new_columns], index=new_index, columns=new_columns
    )


def _rename_rows_cols(
    exprs: pd.DataFrame, biclusters: dict, extra_info: dict = {}
) -> tuple[pd.DataFrame, dict, dict]:
    """Rename rows and columns of the exprs.
        preserves exprs values positions
        i.e. changes only loc, not iloc

    Args:
        exprs (pd.DataFrame): Expression data.
        biclusters (dict): Biclusters information.
        extra_info (dict): Additional information.

    Returns:
        tuple[pd.DataFrame, dict, dict]: Renamed expression data, biclusters, and extra information.
    """
    # exprs
    renaming_rows = {name: f"g_{ind}" for (ind, name) in enumerate(exprs.index.values)}
    renaming_cols = {
        name: f"s_{ind}" for (ind, name) in enumerate(exprs.columns.values)
    }
    exprs.rename(index=renaming_rows, columns=renaming_cols, inplace=True)

    new_biclusters = {}
    for bic_id, bic_data in biclusters.items():
        # new_biclusters[bic_id] = {
        #     "genes": {renaming_rows[g] for g in bic_data["genes"]},
        #     "samples": {renaming_cols[s] for s in bic_data["samples"]},
        #     "frac": bic_data["frac"],
        # }
        new_biclusters[bic_id] = Bicluster(
            genes={renaming_rows[g] for g in bic_data.genes},
            samples={renaming_cols[s] for s in bic_data.samples},
        )

    new_extra_info = {}
    if "coexpressed_modules" in extra_info:
        coexpressed_modules = extra_info["coexpressed_modules"]

        # Update coexpressed_modules with new gene names
        new_coexpressed_modules = []
        if coexpressed_modules:
            for module in coexpressed_modules:
                new_module = [renaming_rows[gene] for gene in module]
                new_coexpressed_modules.append(sorted(new_module))

        new_extra_info["coexpressed_modules"] = new_coexpressed_modules

    assert extra_info.keys() == new_extra_info.keys(), (
        "Missing logic for renaming logic for some keys: "
        f"{extra_info.keys() - new_extra_info.keys()}."
    )
    return exprs, new_biclusters, new_extra_info


# class ScenarioBiclusters(SyntheticBiclusterGeneratorABC):
#     """Class to generate synthetic biclusters.

#     TODO: remove, tmp class during changing the logic
#     """

#     def __init__(
#         self,
#         data_sizes: tuple[int, int],
#         g_size: int = 5,
#         frac_samples: list[float] = [0.05, 0.1, 0.25, 0.5],
#         m: float = 2.0,
#         std: float = 1.0,
#         z: bool = True,
#         outdir: str = "./",
#         outfile_basename: str = "",
#         g_overlap: bool = False,
#         s_overlap: bool = True,
#         add_coexpressed: list[int] = [],
#     ):
#         assert outfile_basename is "", "Output directory must be not specified."

#         self.kwargs = {
#             "data_sizes": data_sizes,
#             "g_size": g_size,
#             "frac_samples": frac_samples,
#             "m": m,
#             "std": std,
#             "z": z,
#             "outdir": outdir,
#             "outfile_basename": outfile_basename,
#             "g_overlap": g_overlap,
#             "s_overlap": s_overlap,
#             "add_coexpressed": add_coexpressed,
#         }

#     def build(self, seed: int):
#         """Generate synthetic biclusters."""
#         rand = np.random.RandomState(seed)
#         exprs, bic_dict, modules = generate_exprs(rand=rand, **self.kwargs)

#         return exprs, bic_dict, {"coexpressed_modules": modules}

#     def get_args(self) -> dict:
#         return self.kwargs


class SyntheticBicluster(SyntheticBiclusterGeneratorABC):
    """Class to generate synthetic biclusters."""

    SCENARIO_TYPES = {
        "GeneExprs": generate_exprs,
        "Simple": build_simple_biclusters,
    }

    def __init__(
        self,
        scenario_type: str,
        z_score: bool = True,
        shuffle: bool = True,
        rename_cols: bool = True,
        **kwargs,
    ):
        self.scenario_type = scenario_type
        assert scenario_type in self.SCENARIO_TYPES, (
            f"Invalid scenario type: {scenario_type}"
        )

        self.z_score = z_score
        self.shuffle = shuffle
        self.rename_cols = rename_cols

        self.scenario_args = kwargs
        assert "seed" not in self.scenario_args, (
            "Seed should not be in scenario_args, use the build method to set it."
        )

    def build(self, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Build synthetic biclusters."""
        build_func = self.SCENARIO_TYPES[self.scenario_type]

        rand = np.random.RandomState(seed)
        exprs, bic_dict, extra = build_func(rand=rand, **self.scenario_args)

        if self.z_score:
            exprs = zscore(exprs)

        if self.shuffle:
            # shuffle col&row indexes
            # without changing the index-value correspondence
            exprs = _shuffle_exprs(exprs, rand=rand)

        if self.rename_cols:
            # rename rows and columns to same s_ and g_ format
            exprs, bic_dict, extra = _rename_rows_cols(exprs, bic_dict, extra)

        bic_dict = {
            k: {"genes": v.genes, "samples": v.samples} for k, v in bic_dict.items()
        }  # todo: remove
        bic_df = _build_bicluster_table(exprs, bic_dict)

        return exprs, bic_df, extra

    def get_args(self) -> dict:
        """Describe the scenario."""
        return {
            "scenario_type": self.scenario_type,
            "z_score": self.z_score,
            "shuffle": self.shuffle,
            "rename_cols": self.rename_cols,
            **self.scenario_args,
        }


class ScenarioBiclusters(SyntheticBicluster):
    """Class to generate synthetic biclusters.

    TODO: remove, tmp class during changing the logic
    """

    def __init__(
        self,
        data_sizes: tuple[int, int],
        g_size: int = 5,
        frac_samples: list[float] = [0.05, 0.1, 0.25, 0.5],
        m: float = 2.0,
        std: float = 1.0,
        z: bool = True,
        outdir: str = "./",
        outfile_basename: str = "",
        g_overlap: bool = False,
        s_overlap: bool = True,
        add_coexpressed: list[int] = [],
    ):
        assert outfile_basename is "", "Output directory must be not specified."

        kwargs = {
            "data_sizes": data_sizes,
            "g_size": g_size,
            "frac_samples": frac_samples,
            "m": m,
            "std": std,
            "z": z,
            "outdir": outdir,
            "outfile_basename": outfile_basename,
            "g_overlap": g_overlap,
            "s_overlap": s_overlap,
            "add_coexpressed": add_coexpressed,
        }
        super().__init__(
            scenario_type="GeneExprs",
            **kwargs,
            z_score=False,
            shuffle=False,
            rename_cols=False,
        )

    def build(self, seed: int):
        """Generate synthetic biclusters."""
        exprs, bic_df, modules = super().build(seed)

        frac = [float(s.split("_")[1]) for s in bic_df.index]
        bic_df["frac"] = frac

        # make frac the 3-d one
        cols = list(bic_df.columns)
        cols.insert(2, cols.pop(cols.index("frac")))
        bic_df = bic_df[cols]

        return exprs, bic_df, modules


def get_standard_dataset_schema() -> dict[str, SyntheticBiclusterGeneratorABC]:
    ds_schema: dict[str, SyntheticBiclusterGeneratorABC] = {
        # "GeneExprs": SyntheticBicluster(
        #     scenario_type="GeneExprs",
        #     data_sizes=(10, 10),
        #     frac_samples=[0.2, 0.3],
        # ),
        "simple_3": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "simple_5": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 5)
        ),
        "more_rows": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(100, 10), bic_sizes=(3, 3)
        ),
        "more_cols": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 100), bic_sizes=(3, 3)
        ),
        **{
            f"test_mu_{i}": SyntheticBicluster(
                scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3), bic_mu=i
            )
            for i in [0.1, 0.5, 1.0, 2.0, 5.0]
        },
    }
    return ds_schema


def get_scenario_dataset_schema(
    scale: float = 1.0,
) -> dict[str, SyntheticBiclusterGeneratorABC]:
    """Get a dataset schema for scenarios.

    Args:
        scale (float): Scale factor for the dataset size,
            useful for debugging on smaller datasets.
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

    size = (
        int(10000 * scale),  # number of genes
        int(200 * scale),  # number of samples
    )

    ds_schema: dict[str, SyntheticBiclusterGeneratorABC] = {}
    for letter in ["A", "C"]:
        for n_genes in [5, 50, 500]:
            n_genes = max(1, int(n_genes * scale))
            ds_schema[f"{letter}_{n_genes}"] = ScenarioBiclusters(
                size, g_size=n_genes, **common_args, **scenario_args[letter]
            )
    return ds_schema


def _save_dataset_entry(
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


def build_dataset(
    entry_builders: Mapping[str, SyntheticBiclusterGeneratorABC],
    seed_prefix="",
    output_dir="./synthetic_ds",
    show_images=True,
) -> pd.DataFrame:
    """Build a dataset from the given generators.

    Args:
        entry_builders (Mapping[str, SyntheticBiclusterGeneratorABC]):
            A mapping of dataset entry names to their corresponding bicluster generators.
        seed_prefix (str): Prefix for random seed generation to ensure reproducibility.
        output_dir (str): Directory to save the generated dataset entries.
        show_images (bool): Whether to display images during generation.

    Returns:
        pd.DataFrame: DataFrame summarizing the built dataset entries.
    """
    output_path = Path(output_dir)
    build_info = {}
    for name, builder in entry_builders.items():
        exprs, bicluster_df, extra_info = builder.build(
            random.Random(seed_prefix + name).randint(0, 10**6)
        )

        save_path = output_path / name
        build_info[name] = _save_dataset_entry(
            name, exprs, bicluster_df, extra_info, save_path, show_images
        )
    return pd.DataFrame.from_dict(build_info, orient="index")

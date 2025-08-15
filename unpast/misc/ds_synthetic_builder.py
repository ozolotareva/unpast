import os
import random
from collections import namedtuple
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from unpast.core.preprocessing import zscore
from unpast.misc.ds_synthetic import _build_bicluster_table, generate_exprs
from unpast.utils.io import write_bic_table

Bicluster = namedtuple("Bicluster", ["genes", "samples"])


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


def shuffle_exprs(exprs: pd.DataFrame, rand: np.random.RandomState) -> pd.DataFrame:
    """Shuffle the expression data."""
    new_index = rand.permutation(exprs.index)
    new_columns = rand.permutation(exprs.columns)
    return pd.DataFrame(
        exprs.loc[new_index, new_columns], index=new_index, columns=new_columns
    )


def _rename_rows_cols(
    exprs: pd.DataFrame, biclusters: dict, coexpressed_modules: list | None = None
) -> tuple[pd.DataFrame, dict, list]:
    """Rename rows and columns of the exprs."""
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

    # Update coexpressed_modules with new gene names
    new_coexpressed_modules = []
    if coexpressed_modules:
        for module in coexpressed_modules:
            new_module = [renaming_rows[gene] for gene in module]
            new_coexpressed_modules.append(sorted(new_module))

    return exprs, new_biclusters, new_coexpressed_modules


class SyntheticBicluster(SyntheticBiclusterGeneratorABC):
    """Class to generate synthetic biclusters."""

    SCENARIO_TYPES = {
        "GeneExprs": generate_exprs,
        "Simple": build_simple_biclusters,
    }

    def __init__(
        self,
        scenario_type,
        not_z_score: bool = False,
        not_shuffle: bool = False,
        **kwargs,
    ):
        self.scenario_type = scenario_type
        assert scenario_type in self.SCENARIO_TYPES, (
            f"Invalid scenario type: {scenario_type}"
        )

        self.scenario_args = kwargs
        assert "seed" not in self.scenario_args, (
            "Seed should not be in scenario_args, use the build method to set it."
        )

        self.not_z_score = not_z_score
        self.not_shuffle = not_shuffle

    def build(self, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Build synthetic biclusters."""
        build_func = self.SCENARIO_TYPES[self.scenario_type]

        rand = np.random.RandomState(seed)
        if self.scenario_type == "GeneExprs":
            # TODO: use the same interface in gene_exprs
            exprs, bic_dict, extra = build_func(**self.scenario_args)
        else:
            exprs, bic_dict, extra = build_func(rand=rand, **self.scenario_args)

        if not self.not_z_score:
            exprs = zscore(exprs)

        if not self.not_shuffle:
            exprs = shuffle_exprs(exprs, rand=rand)

        if self.scenario_type != "GeneExprs":
            # TODO: use the same interface in gene_exprs
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
            "not_z_score": self.not_z_score,
            "not_shuffle": self.not_shuffle,
            **self.scenario_args,
        }


# dataset = {
#     "A_5": SynthBic("A", size=5),
#     "A_50": SynthBic("A", size=50),
#     "A_500": SynthBic("A", size=500),
#     "C_5": SynthBic("C", size=5),
#     "C_50": SynthBic("C", size=50),
#     "C_500": SynthBic("C", size=500),

# }

# for c in ['A', 'C']:
#     for size in [5, 50, 500]:
#         dataset[f"{c}_{size}"] = SyntheticBicluster(c, size=size)

# for size in [10]:
#     dataset[f"Simple_{size}"] = SyntheticBicluster("Simple", size=size)


def build_dataset(
    entry_builders: Mapping[str, SyntheticBiclusterGeneratorABC],
    seed_prefix="",
    output_dir="./synthetic_ds",
) -> pd.DataFrame:
    """Build a dataset from the given generators.

    Args:
        entry_builders: A mapping of dataset names to their corresponding SyntheticBicluster instances.
        seed_prefix: A prefix to use for seeding the random number generator, "" by default
        output_dir: The directory to save the generated dataset files.

    Returns:
        DataFrame, containing info about generated tuples:
            * saved expressions paths
            * saved bicluster paths
            * some additional info

    """
    output_path = Path(output_dir)

    build_info = {}
    for name, builder in entry_builders.items():
        random.seed(seed_prefix + name)
        seed = random.randint(0, 10**6)

        exprs, bicluster_df, extra_info = builder.build(seed)

        # save
        (output_path / name).mkdir(parents=True, exist_ok=True)
        exprs_file = output_path / name / "data.tsv"
        bicluster_file = output_path / name / "biclusters.tsv"

        exprs.to_csv(exprs_file, sep="\t")
        write_bic_table(bicluster_df, str(bicluster_file))

        # save build info
        build_info[name] = {
            "exprs_file": str(exprs_file),
            "bic_file": str(bicluster_file),
            "extra_info": extra_info,
        }

    return pd.DataFrame.from_dict(build_info, orient="index")

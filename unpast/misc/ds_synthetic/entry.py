"""Entry point for synthetic dataset generation with configurable blueprints.

This module provides the DSEntryBlueprint class for configuring and building
synthetic datasets with various scenarios and preprocessing options.
"""

from typing import Any

import numpy as np
import pandas as pd

from unpast.core.preprocessing import zscore
from unpast.core.sample_clustering import update_bicluster_data
from unpast.misc.ds_synthetic.builder_correlated import (
    build_correlated_background_bicluster,
)
from unpast.misc.ds_synthetic.builder_gene_expr import generate_exprs
from unpast.misc.ds_synthetic.builder_simple import (
    build_simple_biclusters,
    build_simple_multiple_biclusters,
)
from unpast.misc.ds_synthetic.ds_utils import Bicluster


def _shuffle_exprs(exprs: pd.DataFrame, rand: np.random.RandomState) -> pd.DataFrame:
    """Shuffle the expression data while preserving index-value correspondence.

    Changes only iloc (integer-based indexing), not loc (label-based indexing).
    This randomizes the order of genes and samples without changing their identities.

    Args:
        exprs: Expression data.
        rand: Random state for shuffling.

    Returns:
        Shuffled expression data.
    """
    new_index = rand.permutation(exprs.index)
    new_columns = rand.permutation(exprs.columns)
    return exprs.loc[new_index, new_columns]


def _rename_rows_cols(
    exprs: pd.DataFrame,
    bics: dict[str, Bicluster],
    extra: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Bicluster], dict[str, Any]]:
    """Rename rows and columns to standardized format (g_* and s_*).

    Preserves expression values at their positions (changes loc, not iloc).
    Renames genes to g_0, g_1, ... and samples to s_0, s_1, ...

    Args:
        exprs: Expression data.
        bics: Bicluster information.
        extra: Additional information, e.g. co-expressed modules.

    Returns:
        Tuple containing renamed expression data, biclusters, and extra information.
    """
    renaming_rows = {name: f"g_{ind}" for (ind, name) in enumerate(exprs.index.values)}
    renaming_cols = {
        name: f"s_{ind}" for (ind, name) in enumerate(exprs.columns.values)
    }
    exprs.rename(index=renaming_rows, columns=renaming_cols, inplace=True)

    new_bics = {}
    for bic_id, bic_data in bics.items():
        new_bics[bic_id] = Bicluster(
            genes={renaming_rows[g] for g in bic_data.genes},
            samples={renaming_cols[s] for s in bic_data.samples},
        )

    new_extra = {}
    if "coexpressed_modules" in extra:
        coexpressed_modules = extra["coexpressed_modules"]

        new_coexpressed_modules = []
        if coexpressed_modules:
            for module in coexpressed_modules:
                new_module = [renaming_rows[gene] for gene in module]
                new_coexpressed_modules.append(sorted(new_module))

        new_extra["coexpressed_modules"] = new_coexpressed_modules

    assert extra.keys() == new_extra.keys(), (
        f"Missing logic for some keys renaming: {extra.keys() - new_extra.keys()}."
    )
    return exprs, new_bics, new_extra


def _build_bicluster_table(
    exprs: pd.DataFrame, biclusters: dict[str, Bicluster]
) -> pd.DataFrame:
    """Build a DataFrame from bicluster dictionary with additional statistics.

    Adds statistics such as size and expression metrics to each bicluster.

    Args:
        exprs: Expression DataFrame.
        biclusters: Dictionary of biclusters.

    Returns:
        DataFrame with bicluster information and computed statistics.
    """
    new_biclusters = {}
    for bic_id, bic in biclusters.items():
        bic_data = {
            "genes": bic.genes,
            "samples": bic.samples,
            "n_genes": len(bic.genes),
            "n_samples": len(bic.samples),
        }
        new_biclusters[bic_id] = update_bicluster_data(bic_data, exprs)

    bicluster_df = pd.DataFrame.from_dict(new_biclusters).T
    return bicluster_df


class DSEntryBlueprint:
    """Class to generate synthetic biclusters."""

    SCENARIO_TYPES = {
        "GeneExprs": generate_exprs,
        "Simple": build_simple_biclusters,
        "SimpleMult": build_simple_multiple_biclusters,
        "CorrelatedBG": build_correlated_background_bicluster,
    }

    def __init__(
        self,
        scenario_type: str,
        z_score: bool = True,
        shuffle: bool = True,
        rename_rows_cols: bool = True,
        **kwargs: Any,
    ) -> None:
        self.scenario_type = scenario_type
        assert scenario_type in self.SCENARIO_TYPES, (
            f"Invalid scenario type: {scenario_type}"
        )

        self.z_score = z_score
        self.shuffle = shuffle
        self.rename_rows_cols = rename_rows_cols

        self.scenario_args = kwargs
        assert "seed" not in self.scenario_args, (
            "Seed should not be in scenario_args, use the build method to set it."
        )

    def build(self, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Build synthetic biclusters.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tuple containing expression data, bicluster DataFrame, and extra information.
        """
        build_func = self.SCENARIO_TYPES[self.scenario_type]

        rand = np.random.RandomState(seed)
        exprs, bic_dict, extra = build_func(rand=rand, **self.scenario_args)

        if self.z_score:
            exprs = zscore(exprs)

        if self.shuffle:
            # shuffle col&row indexes
            # without changing the index-value correspondence
            exprs = _shuffle_exprs(exprs, rand=rand)

        if self.rename_rows_cols:
            # rename rows and columns to same s_ and g_ format
            exprs, bic_dict, extra = _rename_rows_cols(exprs, bic_dict, extra)

        bic_df = _build_bicluster_table(exprs, bic_dict)

        return exprs, bic_df, extra

    def get_args(self) -> dict[str, Any]:
        """Describe the scenario.

        Returns:
            Dictionary containing scenario configuration.
        """
        return {
            "scenario_type": self.scenario_type,
            "z_score": self.z_score,
            "shuffle": self.shuffle,
            "rename_rows_cols": self.rename_rows_cols,
            **self.scenario_args,
        }

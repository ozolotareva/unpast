
import numpy as np
import pandas as pd
from unpast.core.preprocessing import zscore
from unpast.core.sample_clustering import update_bicluster_data
from unpast.misc.ds_synthetic.builder_gene_expr import generate_exprs
from unpast.misc.ds_synthetic.builder_simple import build_simple_biclusters
from unpast.misc.ds_synthetic.ds_utils import Bicluster


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


def _build_bicluster_table(
    exprs: pd.DataFrame, biclusters: dict[str, Bicluster]
) -> pd.DataFrame:
    """Build a DataFrame from bicluster dictionary with additional info.
        Adds some statistics to each bicluster.

    Args:
        exprs: Expression DataFrame.
        biclusters: Dictionary of biclusters.

    Returns:
        DataFrame with bicluster information.

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
        # "CorrelatedBackground": build_correlated_background_bicluster,
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

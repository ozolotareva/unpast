"""Tests for ds_synthetic module."""

import warnings
import pandas as pd

from test_ds_synthetic import hash_table
from unpast.misc.ds_synthetic_builder import (
    SyntheticBicluster,
    ScenarioBiclusters,
    build_dataset,
    get_scenario_dataset_schema,
    get_standard_dataset_schema,
)
from unpast.utils.io import read_bic_table


class TestScenarioBiclusters:
    """Test cases for ScenarioBiclusters class."""

    def test_gene_exprs_reproducibility(self):
        builder = ScenarioBiclusters(
            data_sizes=(100, 50),
            frac_samples=[0.2, 0.3],
            outfile_basename="",  # don't save
            seed=42,
            add_coexpressed=[10, 20],
        )

        data, biclusters, extra = builder.build(seed = 42)
        assert extra.keys() == {"coexpressed_modules"}
        modules = extra["coexpressed_modules"]

        # sets have random order
        def set_to_str(x):
            if isinstance(x, set):
                return ",".join(map(str, sorted(x)))
            return x

        assert hash_table(data.round(10)) == 13859474739268829572
        if hash_table(data) == 11546093038749057111:
            warnings.warn(
                (
                    "WARNING: Tests generation could be slightly unreproducible on some machines."
                    "\nWay to test: run the following in python and check the last symbol (1 - problem, 0 - ok)."
                    "\n---"
                    "\nimport struct"
                    "\nimport numpy as np"
                    "\nprint(np.__version__)"
                    "\nnum = float(np.random.RandomState(42).normal(loc=0, scale=1.0, size=77)[-1])"
                    "\nd = struct.unpack('>Q', struct.pack('>d', num))"
                    "\nprint(f'{d[0]:064b}')"
                    "\n---"
                )
            )
        else:
            assert hash_table(data) == 8496858187703500925

        assert (
            hash_table(biclusters.drop(columns=["frac"]).map(set_to_str))
            == 4510554005146861675
        )
        assert hash_table(biclusters.map(set_to_str)) == 17686323693856100141
        assert len(modules) > 0
        assert hash_table(pd.DataFrame(modules)) == 6483552165326287867


class TestSyntheticBicluster:
    def test_build(self):
        """Test the SyntheticBicluster class."""
        generator = SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        )
        exprs, biclusters, modules = generator.build(seed=42)

        assert isinstance(exprs, pd.DataFrame)
        assert isinstance(biclusters, pd.DataFrame)
        assert isinstance(modules, dict)

        # Check if the generated expression data has the expected shape
        assert exprs.shape[0] > 0 and exprs.shape[1] > 0

        exprs_2, biclusters_2, modules_2 = generator.build(seed=42)
        assert exprs.equals(exprs_2)
        assert biclusters.equals(biclusters_2)
        assert modules == modules_2

        exprs_3, biclusters_3, modules_3 = generator.build(seed=1)
        assert not exprs.equals(exprs_3)
        assert not biclusters.equals(biclusters_3)
        # assert modules != modules_3  # both are empty here, so equal


def test_build_dataset(tmp_path):
    """Test the build method of SyntheticBicluster."""
    dataset = {
        # "GeneExprs": SyntheticBicluster(
        #     scenario_type="GeneExprs",
        #     data_sizes=(10, 10),
        #     frac_samples=[0.2, 0.3],
        # ),
        "name1": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "name2": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "name3": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 5)
        ),
        "test_many_rows": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(100, 10), bic_sizes=(3, 3)
        ),
        "test_many_cols": SyntheticBicluster(
            scenario_type="Simple", data_sizes=(10, 100), bic_sizes=(3, 3)
        ),
        **{
            f"test_mu_{i}": SyntheticBicluster(
                scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3), bic_mu=i
            )
            for i in [0.1, 0.5, 1.0, 2.0, 5.0]
        },
    }

    # Call build_dataset and capture the returned DataFrame
    result_df = build_dataset(dataset, output_dir=tmp_path, show_images=False)

    # Check that the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check that the DataFrame has the expected number of rows (one per dataset)
    expected_count = len(dataset)
    assert len(result_df) == expected_count

    # Check that the DataFrame has the expected columns
    expected_columns = {"exprs_file", "bic_file", "extra_info"}
    assert set(result_df.columns) == expected_columns

    # Check that all dataset names are in the index
    assert set(result_df.index) == set(dataset.keys())

    # Check that all files were created and exist
    for dataset_name in dataset.keys():
        exprs_file = str(result_df.loc[dataset_name, "exprs_file"])
        bic_file = str(result_df.loc[dataset_name, "bic_file"])

        # Check that file paths are strings
        assert isinstance(result_df.loc[dataset_name, "exprs_file"], str)
        assert isinstance(result_df.loc[dataset_name, "bic_file"], str)

        # Check that files actually exist
        assert pd.read_csv(exprs_file, sep="\t", index_col=0).shape[0] > 0
        assert pd.read_csv(bic_file, sep="\t").shape[0] > 0

        # Check file naming convention
        assert exprs_file.endswith(f"{dataset_name}/data.tsv")
        assert bic_file.endswith(f"{dataset_name}/true_biclusters.tsv")

    # Check specific dataset properties
    # Test that different data_sizes produce different file sizes
    exprs_name1 = pd.read_csv(
        str(result_df.loc["name1", "exprs_file"]), sep="\t", index_col=0
    )
    exprs_name3 = pd.read_csv(
        str(result_df.loc["name3", "exprs_file"]), sep="\t", index_col=0
    )

    assert exprs_name1.shape == (10, 10)  # data_sizes=(10, 10)
    assert exprs_name3.shape == (20, 20)  # data_sizes=(20, 20)

    # Test that many_rows and many_cols have expected shapes
    exprs_many_rows = pd.read_csv(
        str(result_df.loc["test_many_rows", "exprs_file"]), sep="\t", index_col=0
    )
    exprs_many_cols = pd.read_csv(
        str(result_df.loc["test_many_cols", "exprs_file"]), sep="\t", index_col=0
    )

    assert exprs_many_rows.shape == (100, 10)
    assert exprs_many_cols.shape == (10, 100)

    # Check that bicluster files have proper structure
    bic_df = read_bic_table(str(result_df.loc["name1", "bic_file"]))
    assert len(bic_df) > 0  # Should have at least one bicluster entry


def test_get_scenario_dataset_schema(tmp_path):
    ds_schema = get_scenario_dataset_schema(scale=0.1)
    build_dataset(ds_schema, output_dir=tmp_path, show_images=False)

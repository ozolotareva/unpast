"""Tests for ds_synthetic module."""

import pandas as pd

from unpast.misc.ds_synthetic_builder import (
    SyntheticBicluster,
    build_dataset,
    get_scenario_dataset_schema,
    get_standard_dataset_schema,
)
from unpast.utils.io import read_bic_table


def test_synthetic_bicluster():
    """Test the SyntheticBicluster class."""
    generator = SyntheticBicluster(
        scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
    )
    exprs, biclusters, modules = generator.build(seed=42)

    assert isinstance(exprs, pd.DataFrame)
    assert isinstance(biclusters, pd.DataFrame)
    assert isinstance(modules, list)

    # Check if the generated expression data has the expected shape
    assert exprs.shape[0] > 0 and exprs.shape[1] > 0

    exprs_2, biclusters_2, modules_2 = generator.build(seed=42)
    assert exprs.equals(exprs_2)
    assert biclusters.equals(biclusters_2)
    assert modules == modules_2

    exprs_3, biclusters_3, modules_3 = generator.build(seed=1)
    assert not exprs.equals(exprs_3)
    assert not biclusters.equals(biclusters_3)
    # assert modules == modules_3  # both are empty here


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
    result_df = build_dataset(dataset, output_dir=tmp_path)

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
        assert bic_file.endswith(f"{dataset_name}/biclusters.tsv")

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
    build_dataset(ds_schema, output_dir=tmp_path)

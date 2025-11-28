"""Tests for ds_synthetic module."""

import random
import warnings

import pandas as pd
import pytest

from unpast.misc.ds_synthetic.dataset import (
    build_dataset,
    get_modular_dataset_blueprint,
    get_scenario_dataset_blueprint,
)
from unpast.misc.ds_synthetic.entry import DSEntryBlueprint
from unpast.tests.test_utils import _hash_table
from unpast.utils.io import read_bic_table, read_exprs


def _set_to_str(x):
    # fixing, as sets have random order
    if isinstance(x, set):
        return ",".join(map(str, sorted(x)))
    return x


class TestDSEntryBlueprint:
    def test_build(self):
        """Test the DSEntryBlueprint class."""
        generator = DSEntryBlueprint(
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

    def test_gene_exprs_reproducibility(self):
        builder = DSEntryBlueprint(
            scenario_type="GeneExprs",
            data_sizes=(100, 50),
            frac_samples=[0.2, 0.3],
            add_coexpressed=[10, 20],
        )

        data, biclusters, extra = builder.build(seed=42)
        assert extra.keys() == {"coexpressed_modules"}
        modules = extra["coexpressed_modules"]

        assert _hash_table(data.round(10)) == 8598621594420307458
        if _hash_table(data) == 10459612308049334423:
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
            assert _hash_table(data) == 16708327152901014055

        assert "frac" not in biclusters.columns
        assert _hash_table(biclusters.map(_set_to_str)) == 12863702880069835519
        assert len(modules) > 0
        assert _hash_table(pd.DataFrame(modules)) == 10671481854807148724


def test_synthetic_bicluster_write_read_roundtrip(tmp_path):
    """Test that created true biclusters can be written and read correctly using DSEntryBlueprint."""
    # Setup
    builder = DSEntryBlueprint(
        scenario_type="Simple",
        data_sizes=(10, 5),
        bic_sizes=(3, 2),
        bic_mu=2.0,
    )
    df = build_dataset(
        {"test": builder},
        output_dir=tmp_path,
        show_images=False,
    )

    exprs, bics, _extra = builder.build(seed=random.Random("test").randint(0, 10**6))

    # Read and compare
    exprs_read = read_exprs(df.loc["test", "exprs_file"])
    pd.testing.assert_frame_equal(exprs, exprs_read, check_names=True, check_dtype=True)

    biclusters_read = read_bic_table(df.loc["test", "bic_file"])
    for k in bics.index:
        v = bics.loc[k]
        v_read = biclusters_read.loc[k]
        assert set(v_read["genes"]) == set(v["genes"])
        assert set(v_read["samples"]) == set(v["samples"])


def test_build_dataset(tmp_path):
    """Test the build method of DSEntryBlueprint."""
    dataset = {
        "GeneExprs": DSEntryBlueprint(
            scenario_type="GeneExprs",
            data_sizes=(10, 10),
            frac_samples=[0.2, 0.3],
        ),
        "name1": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "name2": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3)
        ),
        "name3": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(20, 20), bic_sizes=(5, 5)
        ),
        "test_many_rows": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(100, 10), bic_sizes=(3, 3)
        ),
        "test_many_cols": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 100), bic_sizes=(3, 3)
        ),
        **{
            f"test_mu_{i}": DSEntryBlueprint(
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


def test_build_dataset_reproducibility(tmp_path):
    """Check that results has exactly the same results (by hashes)"""
    dataset = {
        "name1": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3), bic_mu=3.0
        ),
        "name2": DSEntryBlueprint(
            scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3), bic_mu=3.0
        ),
        **{
            f"test_mu_{i}": DSEntryBlueprint(
                scenario_type="Simple", data_sizes=(10, 10), bic_sizes=(3, 3), bic_mu=i
            )
            for i in [0.1, 0.5, 1.0, 2.0, 5.0]
        },
    }
    df = build_dataset(dataset, output_dir=tmp_path, show_images=False)

    expected_hashes = {
        "name1": (720770725035090210, 7405225186472012081),
        "name2": (6485718250229255822, 618799685746951062),
    }

    for name, (exprs_hash, bic_hash) in expected_hashes.items():
        exprs = pd.read_csv(
            str(df.loc[name, "exprs_file"]), sep="\t", index_col=0
        ).round(10)  # round to avoid small fp differences
        assert _hash_table(exprs) == exprs_hash

        biclusters = read_bic_table(str(df.loc[name, "bic_file"]))
        assert _hash_table(biclusters.map(_set_to_str)) == bic_hash


def test_get_scenario_dataset_blueprint(tmp_path):
    """Smoke test for scenario dataset schema generation."""
    ds_schema = get_scenario_dataset_blueprint(scale=0.02)
    build_dataset(ds_schema, output_dir=tmp_path, show_images=False)


@pytest.mark.slow
def test_get_scenario_dataset_blueprint_no_scale(tmp_path):
    """Smoke test for scenario dataset schema generation without scaling."""
    ds_schema = get_scenario_dataset_blueprint()
    df = build_dataset(ds_schema, output_dir=tmp_path, show_images=False)
    for _name, row in df.iterrows():
        exprs = pd.read_csv(row["exprs_file"], sep="\t", index_col=0)
        assert len(exprs) > 0

        bic = read_bic_table(row["bic_file"])
        assert len(bic) > 0


def test_generate_modular_dataset_blueprint_part(tmp_path):
    """Smoke test for modular dataset schema generation."""
    ds_schema = get_modular_dataset_blueprint()
    rand = random.Random(42)
    filtered = {
        name: entry
        for name, entry in ds_schema.items()
        if entry.get_args().get("data_sizes", (0, 0))[0] <= 20
        and entry.get_args().get("data_sizes", (0, 0))[1] <= 20
        and rand.random() < 0.5
    }
    build_dataset(filtered, output_dir=tmp_path, show_images=False)


@pytest.mark.slow
def test_generate_modular_dataset_blueprint(tmp_path):
    """Smoke test for modular dataset schema generation."""
    ds_schema = get_modular_dataset_blueprint()
    build_dataset(ds_schema, output_dir=tmp_path, show_images=False)

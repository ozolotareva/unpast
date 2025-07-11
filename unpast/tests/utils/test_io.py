import os
import tempfile

import pandas as pd
import pytest

from unpast.utils import io


# Helper to create a minimal bicluster DataFrame
def make_bicluster_df():
    return pd.DataFrame(
        {
            "genes": [set(["g1", "g2"])],
            "genes_up": [set(["g1"])],
            "genes_down": [set(["g2"])],
            "samples": [set(["s1", "s2"])],
            "gene_indexes": [set([0, 1])],
            "sample_indexes": [set([0, 1])],
        }
    )


def test_read_write_bic_table(tmp_path):
    df = make_bicluster_df()
    file_path = tmp_path / "biclusters.tsv"
    io.write_bic_table(df, str(file_path))
    result = io.read_bic_table(str(file_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["genes"] == set(["g1", "g2"])
    assert result.iloc[0]["genes_up"] == set(["g1"])
    assert result.iloc[0]["genes_down"] == set(["g2"])
    assert result.iloc[0]["samples"] == set(["s1", "s2"])
    assert result.iloc[0]["gene_indexes"] == set([0, 1])
    assert result.iloc[0]["sample_indexes"] == set([0, 1])


def test_read_bic_table_missing_file():
    with pytest.raises(FileNotFoundError):
        io.read_bic_table("nonexistent_file.tsv")


def test_read_bic_table_empty(tmp_path):
    empty_df = make_bicluster_df().iloc[0:0]
    file_path = tmp_path / "empty.tsv"

    io.write_bic_table(empty_df, str(file_path))
    result = io.read_bic_table(str(file_path))

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_write_bic_table_metadata(tmp_path):
    df = make_bicluster_df()
    file_path = tmp_path / "biclusters_meta.tsv"
    io.write_bic_table(
        df,
        str(file_path),
        add_metadata=True,
        seed=123,
        pval=0.05,
        min_n_samples=2,
        bin_method="kmeans",
        clust_method="Louvain",
        directions=["UP", "DOWN"],
        similarity_cutoff=0.9,
        m=0.33,
        merge=True,
    )
    with open(file_path) as f:
        first_line = f.readline()
    assert first_line.startswith("#")


def test_project_paths(tmp_path):
    paths = io.ProjectPaths(str(tmp_path))
    assert os.path.exists(paths.root)
    assert paths.res.endswith(".tsv")

    root = paths.get_root_dir()
    new_paths = io.ProjectPaths(root)
    assert new_paths.get_root_dir() == root

    assert not os.path.exists(paths.bin_dir)
    paths.create_binarization_paths()
    assert os.path.exists(paths.bin_dir)


def test_write_and_read_args(tmp_path):
    args = {
        "float_value": 0.1,
        "str_value": "test",
        "ing_value": 42,
        "bool_value": True,
        "list_value": [1, 2, 3],
    }
    file_path = tmp_path / "args.tsv"
    io.write_args(args, str(file_path))
    loaded_args = io.read_args(str(file_path))
    # All values are read as strings from the file
    expected_args = {k: str(v) for k, v in args.items()}
    assert loaded_args == expected_args

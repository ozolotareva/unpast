"""Tests for run_unpast, and hence all the core code. Usage: python -m pytest test/test_run_unpast.py"""

import os
import sys

import pandas as pd
import pytest

TEST_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(TEST_DIR, "results")
if not os.access(RESULTS_DIR, os.W_OK):
    # repo dir is currently read-only during the testing stage in github-action
    RESULTS_DIR = "/tmp/unpast/results"
REFERENCE_OUTPUT_DIR = os.path.join(TEST_DIR, "test_reference_output")

from unpast.run_unpast import unpast

### Helper functions ###


def run_unpast_on_file(filename, basename, *args, **kwargs):
    out_dir = os.path.join(RESULTS_DIR, f"runs/run_{basename}")
    unpast(
        os.path.join(TEST_DIR, filename),
        out_dir=out_dir,
        verbose=True,  # use pytest -s ... to see the output
        *args,
        **kwargs,
    )
    return parse_answer(out_dir)


def parse_answer(answer_dir, startswith=""):
    files = os.listdir(answer_dir)
    answer_files = [
        f for f in files if f.startswith(startswith) and f.endswith("biclusters.tsv")
    ]
    assert len(answer_files) == 1, f"There are {len(answer_files)} files instead of 1"
    return pd.read_csv(os.path.join(answer_dir, answer_files[0]), sep="\t", comment="#")


def parse_to_features_samples_ids(answer):
    def to_set_of_nums(s):
        return set(map(int, s.strip().split()))

    return (
        to_set_of_nums(answer["gene_indexes"]),
        to_set_of_nums(answer["sample_indexes"]),
    )


### Tests ###


@pytest.mark.slow
def test_smoke():
    """Smoke test - check that the program runs on some input without failure."""
    run_unpast_on_file(
        filename="test_input/synthetic_clear_biclusters.tsv",
        basename="test_smoke",
        clust_method="WGCNA",
    )


def test_simple():
    """Check that clear biclusters are found."""
    res = run_unpast_on_file(
        filename="test_input/synthetic_small_example.tsv",
        basename="test_simple",
        min_n_samples=2,
        clust_method="Louvain",
    )
    assert len(res) == 1, "Too many clusters found"
    features, samples = parse_to_features_samples_ids(res.iloc[0])
    assert features == {0, 1}
    assert samples == {0, 1}


@pytest.mark.slow
def test_clear_biclusters():
    """Check that clear biclusters are found."""
    res = run_unpast_on_file(
        filename="test_input/synthetic_clear_biclusters.tsv",
        basename="test_clear_biclusters",
        clust_method="WGCNA",
    )

    found_correct_bicluster = False
    for _, row in res.iterrows():
        features, samples = parse_to_features_samples_ids(row)
        if features == set(range(1, 22, 2)) and samples == set(range(1, 22, 2)):
            found_correct_bicluster = True

    assert found_correct_bicluster


@pytest.mark.slow
def test_reproducible_wgcna():
    """Check that the same data is found on a complicated input with no clear answer."""
    res = run_unpast_on_file(
        filename="test_input/synthetic_noise.tsv",
        basename="test_reproducible_wgcna",
        clust_method="WGCNA",
        dch=0.999,  # no biclusters found in noise otherwise
    )
    reference = parse_answer(
        answer_dir=REFERENCE_OUTPUT_DIR,
        startswith="test_reproducible_wgcna",
    )
    assert res.equals(reference), "The results are not reproducible"


def test_reproducible_louvain():
    """Check that the same data is found on a complicated input with no clear answer."""
    res = run_unpast_on_file(
        filename="test_input/synthetic_noise.tsv",
        basename="test_reproducible_louvain",
        clust_method="Louvain",
        similarity_cutoffs=[0.5],  # no biclusters found in noise otherwise
    )
    reference = parse_answer(
        answer_dir=REFERENCE_OUTPUT_DIR,
        startswith="test_reproducible_louvain",
    )
    assert res.equals(reference), "The results are not reproducible"

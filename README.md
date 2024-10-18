# UnPaSt
[![Python Versions](https://img.shields.io/pypi/pyversions/unpast.svg)](https://pypi.org/project/unpast/)
![Tests status](https://github.com/ozolotareva/unpast/actions/workflows/run_tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/unpast.svg)](https://badge.fury.io/py/unpast)
![Docker Build Status](https://github.com/ozolotareva/unpast/actions/workflows/docker-publish.yml/badge.svg)
![Docker Image Pulls](https://img.shields.io/docker/pulls/freddsle/unpast)
[![License](https://img.shields.io/pypi/l/unpast.svg)](https://github.com/ozolotareva/unpast/blob/main/LICENSE)


UnPaSt is a novel method for identification of differentially expressed biclusters.

<img src="https://apps.cosy.bio/unpast/assets/DESMOND2_steps2.png"  height="350">

## Cite
UnPaSt preprint [https://arxiv.org/abs/2408.00200](https://arxiv.org/abs/2408.00200).

Code: [https://github.com/ozolotareva/unpast_paper/](https://github.com/ozolotareva/unpast_paper/)

## Web server
[Run UnPaSt at CoSy.Bio server](https://apps.cosy.bio/unpast/)

## Install

### Install via pip

UnPaSt is available on PyPI and can be installed using pip:

```bash
pip install unpast
```
Do not forget to install necessary R packages (see below).

You can run UnPaSt from the command line using the `unpast` command.

```bash
unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500
```

### Docker Environment

UnPaSt is also available as a Docker image. To pull the Docker Image:

```bash
docker pull freddsle/unpast:latest
```

Replace `latest` with a specific version tag if desired (for version before 10.2024 - v0.1.8).

#### Run UnPaSt using Docker

```bash
# Clone the repository to get example data
git clone https://github.com/ozolotareva/unpast.git
cd unpast
mkdir -p results

# Define the command to run UnPaSt
command="unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500 --verbose"

# Run UnPaSt using Docker
docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/data --entrypoint bash freddsle/unpast -c "cd /data && PYTHONPATH=/data $command"
```

### Requirements

UnPaSt requires Python 3.8 or higher (<3.11) and certain Python and R packages.

#### Python Dependencies

The Python dependencies are installed automatically when installing via pip (or you can use requirements.txt). They include (with recommended versions):

```
fisher = ">=0.1.9,<=0.1.14"
pandas = "1.3.5"
python-louvain = "0.15"
matplotlib = "3.7.1"
seaborn = "0.11.1"
numba = ">=0.51.2,<=0.55.2"
numpy = "1.22.3"
scikit-learn = "1.2.2"
scikit-network = ">=0.24.0,<0.26.0"
scipy = ">=1.7.1,<=1.7.3"
statsmodels = "0.13.2"
kneed = "0.8.1"
```

#### R Dependencies

UnPaSt utilizes R packages for certain analyses. Ensure that you have R installed with the following packages:

- `WGCNA` (version 1.70-3 or higher)
- `limma` (version 3.42.2 or higher)

### Installation Tips

#### Installing R Dependencies

It is recommended to use `BiocManager` for installing R packages:

```R
install.packages("BiocManager")
BiocManager::install("WGCNA")
BiocManager::install("limma")
```

#### Installing R

Ensure that R (version 4.3.1 or higher) is installed on your system. You can download R from [CRAN](https://cran.r-project.org/).


## Input
UnPaSt requires a tab-separated file with features (e.g. genes) in rows, and samples in columns.
* Feature and sample names must be unique.
* At least 2 features and 5 samples are required.
* Data must be between-sample normalized.

### Recommendations: 
* It is recommended that UnPaSt be applied to datasets with 20+ samples.
* If the cohort is not large (<20 samples), reducing the minimal number of samples in a bicluster (`min_n_samples`) to 2 is recommended. 
* If the number of features is small, using Louvain method for feature clustering instead of WGCNA and/or disabling feature selection by setting the binarization p-value (`p-val`) to 1 might be helpful.

## Examples
* Simulated data example. Biclustering of a matrix with 10000 rows (features) and 200 columns (samples) with four implanted biclusters consisting of 500 features and 10-100 samples each. For more details, see figure 3 and Methods [here](https://arxiv.org/abs/2408.00200).
  
```bash
mkdir -p results;

# running UnPaSt with default parameters and example data
python -m unpast.run_unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500

# with different binarization and clustering methods
python -m unpast.run_unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500 --binarization ward --clustering Louvain

# help
python run_unpast.py -h
```
* Real data example. Analysis of a subset of 200 samples randomly chosen from TCGA-BRCA dataset, including consensus biclustering and visualization:
  [jupyter-notebook](https://github.com/ozolotareva/unpast/blob/main/notebooks/UnPaSt_examples.ipynb).
  
## Outputs
`<basename>.[parameters].biclusters.tsv` - A `.tsv` file containing the identified biclusters with the following structure:

- * the first line starts with `#`, storing the parameters of UnPaSt
- * the second line contains the column headers.
- * each subsequent line represents a bicluster with the following columns:
  - **SNR**: Signal-to-noise ratio of the bicluster, calculated as the average SNR of its features.
  - **n_genes**: Number of genes in the bicluster.
  - **n_samples**: Number of samples in the bicluster.
  - **genes**: Space-separated list of gene names.
  - **samples**: Space-separated list of sample names.
  - **direction**: Indicates whether the bicluster consists of up-regulated ("UP"), down-regulated ("DOWN"), or both types of genes ("BOTH").
  - **genes_up**, **genes_down**: Space-separated lists of up- and down-resulated genes respectively.
  - **gene_indexes**: 0-based index of the genes in the input matrix.
  - **sample_indexes**: 0-based index of the samples in the input matrix.

Along with the biclustering result, UnPaSt creates three files with intermediate results in the output folder `out_dir`:
  - `<basename>.[parameters].binarized.tsv` with binarized input data.
  - `<basename>.[parameters].binarization_stats.tsv` provides binarization statistics for each processed feature.
  - `<basename>.[parameters].background.tsv` stores background distributions of SNR values for each evaluated bicluster size.
These files can be used to restart UnPaSt with the same input and seed from the feature clustering step and skip time-consuming feature binarization. 

## Versions
UnPaSt version used in PathoPlex paper: [UnPaSt_PathoPlex.zip](https://github.com/ozolotareva/unpast_paper/blob/main/paper/UnPaSt_PathoPlex.zip)

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .logs import get_logger

logger = get_logger(__name__)
DEFAULT_SAVE_DIR = "."


def read_bic_table(file_name: str, parse_metadata: bool = False) -> pd.DataFrame:
    """
    Reads a bicluster table from a tab-separated file and processes the data into a pandas DataFrame.
    Optionally, it can also parses metadata from the first line of the file if available.

    Args:
        file_name (str): The path to the tab-separated bicluster file.
        parse_metadata (bool, optional): If True, parses metadata from the first line of the file.
                                         The metadata line must start with a "#" and contain key-value
                                         pairs separated by "; ". Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the bicluster data with columns such as:
                          - "genes": set of all genes in the bicluster.
                          - "genes_up": set of upregulated genes.
                          - "genes_down": set of downregulated genes.
                          - "samples": set of all samples in the bicluster.
                          - "gene_indexes": set of gene index numbers (as integers).
                          - "sample_indexes": set of sample index numbers (as integers).

        If `parse_metadata` is True and metadata exists, returns a tuple:
        (pandas.DataFrame, dict): The bicluster DataFrame and a dictionary of metadata.

    Notes:
        - If the file does not exist returns None, if the file exists but is empty, an empty DataFrame is returned.
        - Missing values in "genes_up" and "genes_down" columns are filled with empty strings.
        - The "genes", "genes_up", "genes_down", "samples", "gene_indexes", and "sample_indexes"
          fields are processed as sets, splitting the original space-separated string values.

    Example:
        biclusters, metadata = read_bic_table('biclusters.tsv', parse_metadata=True)
    """

    biclusters = pd.read_csv(
        file_name,
        sep="\t",
        index_col=0,
        comment="#",
        dtype={
            "sample_indexes": str,
            "gene_indexes": str,
        },
    )
    if len(biclusters) == 0:
        return pd.DataFrame()
    else:
        biclusters["genes"] = biclusters["genes"].apply(
            lambda x: set([g for g in x.split(" ") if not g == ""])
        )
        biclusters["genes_up"] = (
            biclusters["genes_up"]
            .fillna("")
            .apply(lambda x: set([g for g in x.split(" ") if not g == ""]))
        )
        biclusters["genes_down"] = (
            biclusters["genes_down"]
            .fillna("")
            .apply(lambda x: set([g for g in x.split(" ") if not g == ""]))
        )
        biclusters["samples"] = biclusters["samples"].apply(lambda x: set(x.split(" ")))
        if "gene_indexes" in biclusters.columns:
            biclusters["gene_indexes"] = biclusters["gene_indexes"].apply(
                lambda x: set(map(int, set(x.split(" "))))
            )
        if "sample_indexes" in biclusters.columns:
            biclusters["sample_indexes"] = biclusters["sample_indexes"].apply(
                lambda x: set(map(int, set(x.split(" "))))
            )

    if parse_metadata:
        f = open(file_name, "r")
        metadata = f.readline()
        f.close()
        if metadata.startswith("#"):
            metadata = metadata.replace("#", "").rstrip()
            metadata = metadata.split("; ")
            metadata = dict([x.split("=") for x in metadata])
            return biclusters, metadata
    return biclusters


def write_bic_table(
    biclusters: pd.DataFrame,
    results_file_name: str,
    to_str: bool = True,
    add_metadata: bool = False,
    seed: int = None,
    min_n_samples: int = None,
    bin_method: str = None,
    clust_method: str = None,
    pval: float = None,
    directions: list = [],
    similarity_cutoff: float = None,
    ds: float = None,
    dch: float = None,
    m: float = None,
    max_power: int = None,
    precluster: str = None,
    merge: bool = None,
) -> None:
    """
    Writes a bicluster table to a tab-separated file and optionally adds metadata to the file.

    Args:
        biclusters (pd.DataFrame): A DataFrame containing biclusters.
        results_file_name (str): The path where the results should be written.


        to_str (bool, optional): If True, converts sets of values (genes, samples, etc.)
                                 into space-separated strings before writing. Defaults to True.
        add_metadata (bool, optional): Defaults to False. If True, writes metadata to the file's first line.
                                       For example, if `add_metadata` is True, metadata is written to the file
                                        in a format like "#seed=123; pval=0.05; min_n_samples=5; ...".
        seed (int, optional): Seed used for generating biclusters, added to metadata if provided.
        min_n_samples (int, optional): Minimum number of samples, included in metadata.
        bin_method (str, optional): Binning method used, included in metadata.
        clust_method (str, optional): Clustering method used (e.g., "Louvain" or "WGCNA"),
                                      included in metadata.
        pval (float, optional): P-value threshold used for significance, included in metadata.
        directions (list, optional): Directions used in biclustering (e.g., "up", "down"),
                                     included in metadata.
        similarity_cutoff (float, optional): Similarity cutoff used in the Louvain method,
                                             included in metadata.
        ds (float, optional): Soft-thresholding power used in WGCNA, included in metadata.
        dch (float, optional): Dynamic cut height used in WGCNA, included in metadata.
        m (float, optional): Modularity value used in Louvain method, included in metadata.
        max_power (int, optional): Maximum power used in WGCNA, included in metadata.
        precluster (str, optional): Preclustering strategy used, included in metadata.
        merge (bool, optional): Whether merging of biclusters was performed, included in metadata.

    Returns:
        None: This function does not return any value, it writes the results to a file.
    """
    bics = biclusters.copy()
    if add_metadata:
        metadata = (
            "#seed="
            + str(seed)
            + "; "
            + "pval="
            + str(pval)
            + "; "
            + "min_n_samples="
            + str(min_n_samples)
            + "; "
        )
        metadata = metadata + "b=" + bin_method + "; "
        metadata = metadata + "c=" + clust_method + "; "
        if len(directions):
            metadata = metadata + "directions=" + "-".join(directions) + "; "
        if clust_method == "Louvain":
            metadata = (
                metadata
                + "similarity_cutoff="
                + str(similarity_cutoff)
                + "; modularity="
                + str(m)
            )
        elif "WGCNA" in clust_method:
            metadata = (
                metadata
                + "ds="
                + str(ds)
                + "; dch="
                + str(dch)
                + "; max_power="
                + str(max_power)
                + "; precluster="
                + str(precluster)
            )

        else:
            logger.error(f"Unknown 'clust_method': {clust_method}")
        metadata = metadata + "; merge=" + str(merge)
        with open(results_file_name, "w") as f:
            f.write(metadata + "\n")
        write_mode = "a"
    else:
        write_mode = "w"

    if len(bics) == 0:
        logger.warning("No biclusters found")
    else:
        if to_str:
            bics["genes"] = bics["genes"].apply(lambda x: " ".join(map(str, sorted(x))))
            bics["genes_up"] = bics["genes_up"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["genes_down"] = bics["genes_down"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["samples"] = bics["samples"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            if "gene_indexes" in bics.columns:
                bics["gene_indexes"] = bics["gene_indexes"].apply(
                    lambda x: " ".join(map(str, sorted(x)))
                )
            if "sample_indexes" in bics.columns:
                bics["sample_indexes"] = bics["sample_indexes"].apply(
                    lambda x: " ".join(map(str, sorted(x)))
                )
            if "ids" in bics.columns:
                bics["ids"] = bics["ids"].apply(lambda x: " ".join(map(str, sorted(x))))
        bics.index.name = "id"
    bics.to_csv(results_file_name, sep="\t", mode=write_mode)


class ProjectPaths:
    """
    A class to manage project paths for input and output files.
    """

    def _init_roots(self, out_dir: str, binary_dir: str) -> tuple[Path, Path, bool]:
        """
        Initialize root and bin_root directories

        Args:
            out_dir (str, optional): The base directory for runs. Defaults to "runs/run_<timestamp>".
            binary_dir (str, optional): The directory for binarization results. Defaults to "<out_dir>/binarization".

        Returns:
            Path: The calculated root directory path.
            Path: The calculated binary directory path.
            created = False if the directory already exists, True otherwise.
        """
        now_str = f"{datetime.now():%Y%m%dT%H%M%S}"
        out_dir = out_dir.replace("<timestamp>", now_str)
        binary_dir = binary_dir.replace("<timestamp>", now_str)
        binary_dir = binary_dir.replace("<out_dir>", out_dir)

        root = Path(out_dir).resolve()
        bin_dir = Path(binary_dir).resolve()

        # Create output directory if it doesn't exist
        created = False
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            created = True

        return root, bin_dir, created

    def __init__(
        self,
        out_dir: str = "runs/run_<timestamp>",
        binary_dir: str = "<out_dir>/binarization",
    ) -> None:
        """
        Initializes the ProjectPaths with the given directories.

        Args:
            out_dir (str): Output directory path.
            binary_dir (str): Directory to save/load binarization results.
        """
        # replace placeholders
        root, bin_dir, created = self._init_roots(out_dir, binary_dir)
        self.root = root
        self.bin_dir = bin_dir
        using = "Created new" if created else "Using existing"
        logger.debug(f"{using} directory for outputs: {self.root}")

        # Main output files
        self.args = str(self.root / "args.tsv")
        self.res = str(self.root / "biclusters.tsv")
        self.log = str(self.root / "unpast.log")

        # Set default binary_dir if not provided
        self.binarization_args = str(self.bin_dir / "bin_args.tsv")
        self.binarization_res = str(self.bin_dir / "bin_res.tsv")
        self.binarization_stats = str(self.bin_dir / "bin_stats.tsv")
        self.binarization_bg = str(self.bin_dir / "bin_background.tsv")

        self.tmp_wgcna = self.root / "tmp_wgcna"

    def create_binarization_paths(self) -> None:
        """
        Creates the binarization directory and ensures all binarization paths exist.
        This method is called before saving binarization files.
        """
        Path(self.bin_dir).mkdir(parents=True, exist_ok=True)

    def get_root_dir(self) -> str:
        """
        Returns the root directory of outputs.
        """
        return str(self.root)

    def get_wgcna_tmp_file(self) -> str:
        """
        Returns a temporary file path for WGCNA results.
        The file name includes a timestamp to ensure uniqueness.
        """
        self.tmp_wgcna.mkdir(exist_ok=True)
        self.tmp_wgcna_file = str(
            self.tmp_wgcna / f"wgcna_{datetime.now():%Y%m%dT%H%M%S}.tsv"
        )
        return self.tmp_wgcna_file

    def clear_wgcna_tmp_files(self) -> None:
        """
        Clears the temporary WGCNA file and folder by removing it if it exists.
        """
        try:
            os.remove(self.tmp_wgcna_file)
        except Exception as e:
            logger.warning(
                f"Failed to remove temporary WGCNA file: {self.tmp_wgcna_file}."
            )
            logger.warning(f"  Error: {e}")

        try:
            self.tmp_wgcna.rmdir()
        except Exception as e:
            logger.warning(
                f"Failed to remove temporary WGCNA directory: {self.tmp_wgcna}"
            )
            logger.warning(f"  Error: {e}")

    def __str__(self) -> str:
        """
        Returns a string representation of the ProjectPaths object.
        """
        return f"ProjectPaths(root='{self.root}')"


def write_args(
    args: dict, file_path: str, args_label: Optional[str] = "Arguments"
) -> None:
    """
    Writes the arguments to a file in a tab-separated format.

    Args:
        args (dict): A dictionary of arguments to save.
        file_path (str): The path to the file where the arguments will be saved.
        args_label (str, optional): The name of args, use None to skip logging.
    """
    assert file_path.endswith(".tsv"), "File for args must end with '.tsv'. "
    df = pd.DataFrame(list(args.items()), columns=["arg", "value"])
    df.to_csv(file_path, sep="\t", index=False)
    if args_label:
        logger.debug(f"{args_label} saved to {file_path}")


def read_args(file_path: str) -> dict:
    """
    Reads arguments from a file in a tab-separated format.

    Args:
        file_path (str): The path to the file from which to load the arguments.

    Returns:
        dict: A dictionary of loaded arguments.
    """
    assert file_path.endswith(".tsv"), "File for args must end with '.tsv'. "
    df = pd.read_csv(file_path, sep="\t")
    args = dict(zip(df["arg"], df["value"]))
    return args

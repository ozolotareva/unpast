import matplotlib.pyplot as plt


def plot_binarized_feature_impl(
    feature_name, down_group, up_group, colors, hist_range, snr
):
    """Plot histogram of binarized feature values showing up and down groups.

    Args:
        feature_name (str): name of the feature being plotted
        down_group (array): values in the down-regulated group
        up_group (array): values in the up-regulated group
        colors (tuple): colors for (down, up) groups
        hist_range (tuple): range for histogram bins
        snr (float): signal-to-noise ratio value

    Returns:
        None: displays plot
    """
    down_color, up_color = colors
    n_bins = int(max(20, (len(down_group) + len(up_group)) / 10))
    n_bins = min(n_bins, 200)
    fig, ax = plt.subplots()
    tmp = ax.hist(
        down_group, bins=n_bins, alpha=0.5, color=down_color, range=hist_range
    )
    tmp = ax.hist(up_group, bins=n_bins, alpha=0.5, color=up_color, range=hist_range)
    # tmp = plt.title("{}:    SNR={:.2f},    neg={}, pos={}".format(feature_name,snr,len(down_group),len(up_group)))
    n_samples = min(len(down_group), len(up_group))
    # tmp = ax.set_title("SNR={:.2f},   n_samples={}".format(snr,n_samples))
    ax.text(
        0.05,
        0.95,
        feature_name,
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=24,
    )
    ax.text(
        0.95,
        0.95,
        "SNR=" + str(round(snr, 2)) + "\nn_samples=" + str(n_samples),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )
    # tmp = plt.savefig("figs_binarization/"+feature_name+".hist.svg", bbox_inches='tight', transparent=True)
    plt.show()


def plot_binarized_feature(row, direction, pos_mask, neg_mask, snr):
    gene = row.name
    row_values = row.values
    hist_range = row_values.min(), row_values.max()

    # set colors to two sample groups
    # red - overexpression
    # blue - under-expression
    # grey - background (group size > 1/2 of all samples)
    colors = ["grey", "grey"]

    if direction == "UP":
        colors[1] = "red"
    elif direction == "DOWN":
        colors[0] = "blue"
    # elif BOTH ...
    else:
        raise ValueError(f"direction = {direction}")

    # plotting
    plot_binarized_feature_impl(
        gene, row[pos_mask], row[neg_mask], colors, hist_range, snr
    )


def plot_binarization_results(stats, size_snr_trend, sizes, pval):
    """Plot binarization results showing SNR vs sample size with significance threshold.

    Args:
        stats (DataFrame): statistics DataFrame with columns 'SNR', 'SNR_threshold', 'size'
        size_snr_trend (function): function that computes SNR threshold for given sample sizes
        sizes (array): array of sample sizes for plotting the threshold line
        pval (float): p-value threshold used for significance testing

    Returns:
        None: displays plot
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # plot binarization results for real genes
    passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
    failed = stats.loc[stats["SNR"] <= stats["SNR_threshold"], :]
    ax.scatter(
        failed["size"], failed["SNR"], alpha=1, color="black", label="not passed"
    )
    for i, txt in enumerate(failed["size"].index.values):
        ax.annotate(txt, (failed["size"][i], failed["SNR"][i] + 0.1), fontsize=18)
    ax.scatter(passed["size"], passed["SNR"], alpha=1, color="red", label="passed")
    for i, txt in enumerate(passed["size"].index.values):
        ax.annotate(txt, (passed["size"][i], passed["SNR"][i] + 0.1), fontsize=18)

    # plot cutoff
    ax.plot(
        sizes,
        [size_snr_trend(x) for x in sizes],
        color="darkred",
        lw=2,
        ls="--",
        label="e.pval<" + str(pval),
    )
    ax.legend(("not passed", "passed", "e.pval<" + str(pval)), fontsize=18)
    ax.set_xlabel("n_samples", fontsize=18)
    ax.yaxis.tick_right()
    ax.set_ylabel("SNR", fontsize=18)
    ax.set_ylim(0, 4)
    # plt.savefig("figs_binarization/dist.svg", bbox_inches='tight', transparent=True)
    plt.show()

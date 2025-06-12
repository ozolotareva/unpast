import matplotlib.pyplot as plt


def plot_binarized_feature(feature_name, down_group, up_group, colors, hist_range, snr):
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


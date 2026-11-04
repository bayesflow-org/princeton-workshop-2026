"""Plotting helpers for the HNN tutorial."""

import matplotlib.pyplot as plt
import numpy as np

from helpers import PARAMETER_LABELS, PARAMETER_NAMES, TIME_GRID

METHOD_COLORS = {"NPE": "#2A9D8F", "NLE": "#E76F51"}
PRIOR_COLOR = "#8A8A8A"
OBSERVED_COLOR = "#161616"
PCA_COLOR = "#6F4E7C"


def plot_posterior_comparison(npe_draws, nle_draws, prior_draws, *, truth=None, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")

    for axis, name, label in zip(axes.flat, PARAMETER_NAMES, PARAMETER_LABELS):
        prior_values = prior_draws[name].ravel()
        displayed = np.concatenate([prior_values, npe_draws[name], nle_draws[name]])
        if truth is not None:
            displayed = np.append(displayed, truth[name])

        lower, upper = np.quantile(displayed, [0.0025, 0.9975])
        if truth is not None:
            lower = min(lower, truth[name])
            upper = max(upper, truth[name])
        padding = 0.04 * (upper - lower)
        bin_edges = np.linspace(max(0, lower - padding), upper + padding, 31)

        axis.hist(prior_values, bins=bin_edges, density=True, color=PRIOR_COLOR, alpha=0.18, label="Prior")

        prior_lower, prior_upper = np.quantile(prior_values, [0.05, 0.95])
        prior_width = prior_upper - prior_lower
        for method, draws in (("NPE", npe_draws[name]), ("NLE", nle_draws[name])):
            color = METHOD_COLORS[method]
            if np.std(draws) < 0.01 * prior_width:
                axis.axvline(np.median(draws), color=color, linewidth=2, label=method)
            else:
                axis.hist(draws, bins=bin_edges, density=True, histtype="step", color=color, linewidth=2, label=method)

        if truth is not None:
            axis.axvline(truth[name], color=OBSERVED_COLOR, linestyle="--", linewidth=1.5, label="Truth")

        axis.set(title=label, xlim=(bin_edges[0], bin_edges[-1]), ylabel=None)
        axis.set_yticks([])

    for axis in axes.flat[len(PARAMETER_NAMES) :]:
        axis.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False)
    fig.suptitle(title)
    return fig


def plot_predictive_comparison(predictives, observed, *, observed_label, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True, layout="constrained")

    for axis, method in zip(axes, ("NPE", "NLE")):
        lower, median, upper = np.quantile(predictives[method], [0.05, 0.5, 0.95], axis=0)
        color = METHOD_COLORS[method]

        axis.fill_between(TIME_GRID, lower, upper, color=color, alpha=0.18, label="90% predictive interval")
        axis.plot(TIME_GRID, median, color=color, linewidth=2, label="Predictive median")
        axis.plot(TIME_GRID, observed, color=OBSERVED_COLOR, linewidth=1.2, label=observed_label)

        axis.set(title=method, xlabel="Time (ms)", ylabel="Dipole (nAm)")
        axis.legend(frameon=False)

    fig.suptitle(title)
    return fig

# -*- coding: utf-8 -*-
"""
Plotting functions for PG NanoSIMS Simulations

This module contains all plotting functionality extracted from the main script,
maintaining progressive PDF saving capabilities.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np


def initialize_result_figures(iterations):
    """
    Initialize the result figures for plotting.

    Parameters
    ----------
    iterations : int
        Number of outer iterations

    Returns
    -------
    fres : matplotlib.figure.Figure
        Result figure with 3D subplots
    axres : numpy.ndarray
        Array of 3D axes for results
    f_adnesp : matplotlib.figure.Figure
        Figure for gradient descent parameters
    ax_adnesp : numpy.ndarray
        Array of axes for gradient descent plots
    """
    fres, axres = plt.subplots(2, iterations, subplot_kw={"projection": "3d"})
    axres = axres.ravel()
    f_adnesp, ax_adnesp = plt.subplots(4, iterations)
    # Ensure ax_adnesp is always 2D, even when iterations=1
    # When iterations=1, plt.subplots returns a 1D array, but we need 2D indexing
    if ax_adnesp.ndim == 1:
        ax_adnesp = ax_adnesp.reshape(4, 1)
    return fres, axres, f_adnesp, ax_adnesp


def plot_measured_grain(axres, k, iterations, grain_size, grain_delta):
    """
    Plot the measured presolar grain on result axes.

    Parameters
    ----------
    axres : numpy.ndarray
        Array of 3D axes
    k : int
        Current outer iteration index
    iterations : int
        Total number of outer iterations
    grain_size : float
        Measured grain size
    grain_delta : pandas.DataFrame
        DataFrame containing delta values (should have at least 2 columns)
    """
    axres[k].plot(
        grain_size,
        grain_delta.iloc[:, 0].item(),
        grain_delta.iloc[:, 1].item(),
        "sk",
        markersize=12,
        label="Measured presolar grain",
        zorder=10,
    )
    axres[k + iterations].plot(
        grain_size,
        grain_delta.iloc[:, 0].item(),
        grain_delta.iloc[:, 1].item(),
        "sk",
        markersize=12,
        label="Measured presolar grain",
        zorder=10,
    )


def plot_simulated_grain(axres, k, iterations, Diam, delta_values, color, i):
    """
    Plot a simulated grain on result axes.

    Parameters
    ----------
    axres : numpy.ndarray
        Array of 3D axes
    k : int
        Current outer iteration index
    iterations : int
        Total number of outer iterations
    Diam : float
        Measured diameter of simulated grain
    delta_values : array-like
        Array of delta values [delta1, delta2]
    color : array-like
        Color for the plot point
    i : int
        Grain index
    """
    axres[k].plot(
        Diam,
        delta_values[0],
        delta_values[1],
        "o",
        color=color[i],
        markersize=10,
        alpha=0.5,
    )
    axres[k + iterations].plot(
        Diam,
        delta_values[0],
        delta_values[1],
        "o",
        color=color[i],
        markersize=10,
        alpha=0.5,
    )


def plot_gradient_descent_parameters(
    ax_adnesp, k, j, norm3D, decay_adam, momentum_nesperov_adam, c, Nb_PG=9
):
    """
    Plot gradient descent parameters evolution.

    Parameters
    ----------
    ax_adnesp : numpy.ndarray
        Array of axes for gradient descent plots
    k : int
        Current outer iteration index
    j : int
        Current inner iteration index
    norm3D : numpy.ndarray
        Cost function values (norms) for each grain
    decay_adam : list
        Decay parameters for Adam optimizer [size, delta1, delta2]
    momentum_nesperov_adam : list
        Momentum parameters for NAdam optimizer [size, delta1, delta2]
    c : numpy.ndarray
        Color array for different grains
    Nb_PG : int, optional
        Number of presolar grains (default: 9)
    """
    for m in range(Nb_PG):  # Loop on simulated grains
        ax_adnesp[0, k].plot(
            j, norm3D[m], "v", mec=c[m], mfc="none", linewidth=3
        )  # Cost function (i.e., norm)
        ax_adnesp[1, k].plot(
            j,
            decay_adam[0].reshape(1, 9)[0, m],
            "o",
            mec=c[m],
            mfc="none",
            linewidth=3,
        )  # vt parameter in adam protocol on size
        ax_adnesp[1, k].plot(
            j,
            momentum_nesperov_adam[0].reshape(1, 9)[0, m],
            "s",
            mec=c[m],
            mfc="none",
            linewidth=3,
        )  # mt parameter in nadam protocol on size
        ax_adnesp[2, k].plot(
            j,
            decay_adam[1].reshape(1, 9)[0, m],
            "o",
            mec=c[m],
            mfc="None",
            linewidth=3,
        )  # vt parameter in adam protocol on d17O
        ax_adnesp[2, k].plot(
            j,
            momentum_nesperov_adam[1].reshape(1, 9)[0, m],
            "s",
            mec=c[m],
            mfc="None",
            linewidth=3,
        )  # mt parameter in nadam protocol on d17O
        ax_adnesp[3, k].plot(
            j,
            decay_adam[2].reshape(1, 9)[0, m],
            "o",
            mec=c[m],
            mfc="None",
            linewidth=3,
        )  # vt parameter in adam protocol on d18O
        ax_adnesp[3, k].plot(
            j,
            momentum_nesperov_adam[2].reshape(1, 9)[0, m],
            "s",
            mec=c[m],
            mfc="None",
            linewidth=3,
        )  # mt parameter in nadam protocol on d18O


def plot_closest_matches_iteration(axres, k, iterations, closest_match, Ratio_names):
    """
    Plot closest matches for a specific outer iteration.

    Parameters
    ----------
    axres : numpy.ndarray
        Array of 3D axes
    k : int
        Current outer iteration index
    iterations : int
        Total number of outer iterations
    closest_match : pandas.DataFrame
        DataFrame containing closest match data
    Ratio_names : list
        List of ratio names (e.g., ['d-17O/16O', 'd-18O/16O'])
    """
    axres[k].plot(
        closest_match["Measured diameter (nm)"],
        closest_match["Measured " + Ratio_names[0]],
        closest_match["Measured " + Ratio_names[1]],
        "s",
        mec="k",
        mfc="None",
        markersize=10,
        zorder=5,
        linewidth=10,
    )
    axres[k + iterations].plot(
        closest_match["Measured diameter (nm)"],
        closest_match["Measured " + Ratio_names[0]],
        closest_match["Measured " + Ratio_names[1]],
        "s",
        mec="k",
        mfc="None",
        markersize=10,
        zorder=5,
        linewidth=10,
    )
    axres[k].set_title("Iteration #" + str(k), fontsize=12)


def plot_closest_matches_final(
    axres, iterations, closest_match_final, Ratio_names, grain_size, grain_delta
):
    """
    Plot final closest matches across all iterations.

    Parameters
    ----------
    axres : numpy.ndarray
        Array of 3D axes
    iterations : int
        Total number of outer iterations
    closest_match_final : pandas.DataFrame
        DataFrame containing final closest match data
    Ratio_names : list
        List of ratio names (e.g., ['d-17O/16O', 'd-18O/16O'])
    grain_size : float
        Measured grain size
    grain_delta : pandas.DataFrame
        DataFrame containing delta values
    """
    for l in range(0, iterations):
        axres[l].plot(
            closest_match_final["Measured diameter (nm)"],
            closest_match_final["Measured " + Ratio_names[0]],
            closest_match_final["Measured " + Ratio_names[1]],
            "o",
            mec="g",
            mfc="None",
            markersize=10,
            zorder=5,
            linewidth=5,
        )
        axres[l + iterations].plot(
            closest_match_final["Measured diameter (nm)"],
            closest_match_final["Measured " + Ratio_names[0]],
            closest_match_final["Measured " + Ratio_names[1]],
            "o",
            mec="g",
            mfc="None",
            markersize=10,
            zorder=5,
            linewidth=5,
        )
        axres[l + iterations].set_xlim3d(
            [int(grain_size * 0.7), int(grain_size * 1.3)]
        )
        axres[l + iterations].set_ylim3d(
            [
                int(grain_delta[Ratio_names[0]].item() * 0.5),
                int(grain_delta[Ratio_names[0]].item() * 1.5),
            ]
        )
        axres[l + iterations].set_zlim3d(
            [
                int(grain_delta[Ratio_names[1]].item() * 0.5),
                int(grain_delta[Ratio_names[1]].item() * 1.5),
            ]
        )

        axres[l].set_xlabel("Grain diameter (nm)", fontsize=14)
        axres[l].set_ylabel(Ratio_names[0], fontsize=14)
        axres[l].set_zlabel(Ratio_names[1], fontsize=14)
        axres[l + iterations].set_xlabel("Grain diameter (nm)", fontsize=14)
        axres[l + iterations].set_ylabel(Ratio_names[0], fontsize=14)
        axres[l + iterations].set_zlabel(Ratio_names[1], fontsize=14)


def finalize_result_figures(
    fres,
    f_adnesp,
    axres,
    ax_adnesp,
    imagename,
    grain_name,
    Ratio_names,
    iterations,
    lines,
    labels,
    pp,
):
    """
    Finalize result figures with labels, titles, and save to PDF.

    Parameters
    ----------
    fres : matplotlib.figure.Figure
        Result figure
    f_adnesp : matplotlib.figure.Figure
        Gradient descent parameters figure
    axres : numpy.ndarray
        Array of 3D axes for results
    ax_adnesp : numpy.ndarray
        Array of axes for gradient descent plots
    imagename : str
        Name of the image
    grain_name : str
        Name of the grain
    Ratio_names : list
        List of ratio names
    iterations : int
        Total number of outer iterations
    lines : list
        List of Line2D objects for legend
    labels : list
        List of labels for legend
    pp : matplotlib.backends.backend_pdf.PdfPages
        PdfPages object for saving figures
    """
    # Set labels for gradient descent plots
    ax_adnesp[0, 0].set_ylabel("Cost function (norm)", fontsize=15)
    ax_adnesp[1, 0].set_ylabel("Size", fontsize=15)
    ax_adnesp[2, 0].set_ylabel(Ratio_names[0], fontsize=15)
    ax_adnesp[3, 0].set_ylabel(Ratio_names[1], fontsize=15)
    f_adnesp.suptitle("Evolution of the gradient descent parameters", fontsize=22)

    # Set title and legend for result figure
    fres.suptitle(imagename + "\n grain : " + str(grain_name), fontsize=15)
    axres[0].legend(lines, labels, loc="best", ncol=2)

    # Set figure sizes
    fres.set_size_inches(16, 10)
    f_adnesp.set_size_inches(16, 10)

    # Save figures to PDF
    pp.savefig(fres, transparent=True, dpi=100)
    pp.savefig(f_adnesp, transparent=True, dpi=100)


def save_figure_to_pdf(fig, pp, title=None, dpi=100):
    """
    Save a figure to PDF progressively.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    pp : matplotlib.backends.backend_pdf.PdfPages
        PdfPages object for saving figures
    title : str, optional
        Title to add to figure (default: None)
    dpi : int, optional
        Resolution for saving (default: 100)
    """
    if title is not None:
        fig.suptitle(title)
    pp.savefig(fig, transparent=True, dpi=dpi)
    plt.close(fig)


def set_gradient_descent_titles(ax_adnesp, k):
    """
    Set titles for gradient descent parameter plots.

    Parameters
    ----------
    ax_adnesp : numpy.ndarray
        Array of axes for gradient descent plots
    k : int
        Current outer iteration index
    """
    ax_adnesp[0, k].set_title("Outer iteration : " + str(k), fontsize=18)
    ax_adnesp[3, k].set_xlabel("Inner iteration", fontsize=15)


def create_legend_elements():
    """
    Create legend elements for result plots.

    Returns
    -------
    point_inner : matplotlib.lines.Line2D
        Line2D object for inner iteration matches
    point_matchfinal : matplotlib.lines.Line2D
        Line2D object for final matches
    label_points : list
        List of labels for legend
    """
    point_inner = Line2D(
        [0], [0], marker="s", mfc="None", mec="k", linestyle="", markersize=10
    )
    point_matchfinal = Line2D(
        [0], [0], marker="o", mfc="None", mec="g", linestyle="", markersize=10
    )
    label_points = ["Best matches of this iteration", "Best matches all iterations"]
    return point_inner, point_matchfinal, label_points


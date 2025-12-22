# -*- coding: utf-8 -*-
"""
Grain processing functions for PG NanoSIMS Simulations.

@author: Maximilien Verdier-Paoletti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from Modules.pg_grain_simulation import PG_simulationv6
from Modules.gradient_descent import (
    GD_AdamNesperov,
    initialize_gradient_descent_parameters,
    update_gradient_descent_parameters,
    apply_simulation_constraints,
    save_norm_to_summary,
    compute_cost,
)
from Modules.feature_extraction import (
    select_sigma_delta_maps,
    extract_grain_features,
)
from Modules.plots import (
    initialize_result_figures,
    plot_measured_grain,
    plot_simulated_grain,
    plot_gradient_descent_parameters,
    plot_closest_matches_iteration,
    plot_closest_matches_final,
    finalize_result_figures,
    save_figure_to_pdf,
    set_gradient_descent_titles,
    create_legend_elements,
)


def extract_grain_characteristics(grain, elem="O", delta_database=None):
    """
    Extract characteristics from a grain row.

    Parameters:
    -----------
    grain : pd.Series
        Single grain row from data
    elem : str
        Element name
    delta_database : list
        List of delta values for database

    Returns:
    --------
    grain_delta : pd.DataFrame
        Delta composition columns
    ER_grain_delta : pd.DataFrame
        Error columns for delta
    grain_size : float
        Grain size in nm
    sig_r : float
        Sigma ratio
    delta_range : list
        List of delta ranges for each ratio
    """
    grain_delta = grain.filter(regex="^d-.*" + elem, axis=1)
    ER_grain_delta = grain.filter(regex="^ER-d-.*" + elem, axis=1)
    grain_size = grain["ROIDIAM"].item() * 1000

    if grain.columns.str.contains("sig", case=False).any():
        sig_r = grain.iloc[:, np.where(grain.columns.str.contains("sig") == True)]
    else:
        sig_r = 0.5
        print(f"No information available on sigma ratio, fixed to {sig_r}")

    delta_range = []
    if delta_database is not None:
        for i in grain_delta.to_numpy()[0]:
            delta_range.append(
                [h for h in delta_database if np.sign(i) * h > np.abs(i)]
            )

    return grain_delta, ER_grain_delta, grain_size, sig_r, delta_range


def initialize_simulation_parameters(Nb_PG, delta_range, size_range, Ratio_names):
    """
    Initialize simulation parameters for gradient descent.

    Parameters:
    -----------
    Nb_PG : int
        Number of presolar grains
    delta_range : list
        List of delta ranges
    size_range : range
        Range of sizes
    Ratio_names : list
        List of ratio names

    Returns:
    --------
    PG_delta : list
        Initial delta values
    PG_size : np.ndarray
        Initial size values
    gradient_params : tuple
        Gradient descent parameters
    """
    PG_delta = [[[np.random.choice(i) for i in delta_range] for o in range(Nb_PG)]]
    PG_size = np.random.choice(size_range, Nb_PG).reshape(1, Nb_PG)

    gradient_params = initialize_gradient_descent_parameters(
        PG_size, PG_delta, Nb_PG, n_ratios=len(Ratio_names)
    )

    return PG_delta, PG_size, gradient_params


def process_single_simulation_iteration(
    file,
    grain,
    PG_size,
    PG_delta,
    elem,
    beam,
    boxcar,
    sig_r,
    Ratio_names,
    col,
    summary,
    all_simulations,
    imagename,
    k,
    j,
    SAVE_ALL_PLOTS,
    pp,
    axres,
    iterations,
    c,
):
    """
    Process a single simulation iteration.

    Returns:
    --------
    summary : pd.DataFrame
        Updated summary with new simulation results
    all_simulations : dict
        Updated simulations dictionary
    """
    verif = 1 if (k == 0) & (j == 0) else 0

    f, ax, plots, plots_title, PG_coor, raster, px, f_OG = PG_simulationv6(
        file=file,
        elem=elem,
        PG_delta=PG_delta,
        PG_size=PG_size[0, :],
        OG_grain=grain,
        beam_size=beam,
        boxcar_px=boxcar,
        smart=1,
        verif=verif,
        standard="average",
        display="OFF",
    )

    if f_OG is not None:
        save_figure_to_pdf(f_OG, pp, title=imagename)

    ax = ax.ravel()

    grain_delta = grain.filter(regex="^d-.*" + elem, axis=1)
    (
        sigma_map,
        delta_map,
        sigma_anomalous_map_index,
        delta_anomalous_map_index,
        anomalous_ratio_name,
    ) = select_sigma_delta_maps(plots, plots_title, grain_delta)

    for i in range(0, PG_size.shape[1]):
        Diam, delta_values, mask, mask_th, (xsel, ysel), (x, y) = (
            extract_grain_features(
                PG_size[0, :],
                PG_coor,
                raster,
                px,
                sig_r,
                sigma_map,
                delta_map,
                sigma_anomalous_map_index,
                delta_anomalous_map_index,
                i,
            )
        )

        plot_simulated_grain(
            axres, k, iterations, Diam, delta_values, c, i
        )

        for h in range(0, len(ax)):
            ax[h].plot(xsel, ysel, "--", color="w", linewidth=2)
            ax[h].plot(x, y, "-", color="r", linewidth=2)

        S = pd.Series(
            [
                imagename,
                grain.NAME.item(),
                k,
                j,
                i,
                PG_size[0, i],
                sig_r,
                Diam,
                np.round(delta_values[0], 2),
                np.round(delta_values[1], 2),
            ]
        )
        S = S.to_frame().T
        for o in range(len(PG_delta[0][i])):
            S.insert(5 + o, "", PG_delta[0][i][o], allow_duplicates=True)
        S = S.set_axis(col, axis=1)
        summary = pd.concat([summary, S], axis=0, ignore_index=True)

    initial = (
        "Outer Iteration : "
        + str(k)
        + "\n"
        + "Inner Iteration : "
        + str(j)
        + "\n"
        + "Presolar Grain Simulation #"
        + str(j)
        + "\n"
        + "Image is: "
        + str(file.rsplit("/", 1)[1])
        + "\n"
        + "Number of grains : "
        + str(PG_size.shape[1])
    )

    f.text(0.15, 0.92, initial, fontsize=11)
    f.set_size_inches(16, 10)

    if SAVE_ALL_PLOTS:
        save_figure_to_pdf(f, pp)
    else:
        plt.close(f)

    for im_it in range(0, len(plots_title)):
        all_simulations[imagename][k][j][plots_title[im_it]] = plots[im_it]

    return summary, all_simulations


def perform_gradient_descent_step(
    summary,
    grain,
    grain_size,
    grain_delta,
    Ratio_names,
    k,
    j,
    initial_simulations,
    learning_rate,
    norm_summary,
):
    """
    Perform one gradient descent step.

    Returns:
    --------
    new_simu : np.ndarray
        Updated simulation parameters
    norm3D : float
        Normalized 3D distance
    grad : np.ndarray
        Gradient values
    norm_summary : pd.DataFrame
        Updated norm summary
    """
    sim_selgrain = summary.loc[(summary["Grain"] == grain.NAME.item())]
    sim_outerin = sim_selgrain.loc[
        (summary["Outer Iteration"] == k) & (summary["Inner Iteration"] == j)
    ]

    target = np.array(
        [
            grain_size,
            grain_delta[Ratio_names[0]].item(),
            grain_delta[Ratio_names[1]].item(),
        ]
    )
    measured_simulations = np.array(
        [
            sim_outerin["Measured diameter (nm)"],
            sim_outerin["Measured d-17O/16O"],
            sim_outerin["Measured d-18O/16O"],
        ]
    )

    new_simu, norm3D, grad = GD_AdamNesperov(
        target, measured_simulations, initial_simulations, learning_rate
    )

    norm_summary = save_norm_to_summary(norm3D, norm_summary)

    return new_simu, norm3D, grad, norm_summary


def process_single_grain(
    file,
    grain,
    data,
    config,
    Ratio_names,
    col,
    summary,
    norm_summary,
    all_simulations,
    imagename,
    c,
    pp,
    pbar_overall,
    grain_counter,
    total_grains,
):
    """
    Process a single grain through all iterations.

    Returns:
    --------
    summary : pd.DataFrame
        Updated summary
    norm_summary : pd.DataFrame
        Updated norm summary
    all_simulations : dict
        Updated simulations
    match_summary : pd.DataFrame
        Match summary for this grain
    data_res : pd.DataFrame
        Results data for this grain
    grain_counter : int
        Updated grain counter
    """
    grain_delta, ER_grain_delta, grain_size, sig_r, delta_range = (
        extract_grain_characteristics(grain, config["elem"], config["delta_database"])
    )

    fres, axres, f_adnesp, ax_adnesp = initialize_result_figures(config["iterations"])
    point_inner, point_matchfinal, label_points = create_legend_elements()
    lines = []
    labels = []

    for k in tqdm(
        range(0, config["iterations"]),
        desc=f"Outer Iterations (Grain {grain_counter + 1}/{total_grains})",
        unit="iter",
        position=3,
        leave=False,
    ):
        all_simulations[imagename][k] = {}
        plot_measured_grain(axres, k, config["iterations"], grain_size, grain_delta)

        PG_delta, PG_size, gradient_params = initialize_simulation_parameters(
            config["Nb_PG"], delta_range, config["size_range"], Ratio_names
        )
        (
            eta,
            learning_rate,
            eps,
            beta_decay,
            beta_momentum,
            decay_mat,
            decay_adam,
            momentum_mat,
            momentum_adam,
        ) = gradient_params

        j = 0
        cost = 5
        pbar_inner = tqdm(
            total=config["max_iteration"] + 1,
            desc=f"Inner Iterations (Outer {k + 1}/{config['iterations']})",
            unit="iter",
            position=4,
            leave=False,
        )

        while True:
            all_simulations[imagename][k][j] = {}
            pbar_inner.update(1)
            pbar_inner.set_postfix(
                {"cost": f"{cost:.3f}", "goal": f"{config['cost_goal']:.3f}"}
            )

            if cost < config["cost_goal"] or j > config["max_iteration"]:
                pbar_inner.close()
                break

            summary, all_simulations = process_single_simulation_iteration(
                file,
                grain,
                PG_size,
                PG_delta,
                config["elem"],
                config["beam"],
                config["boxcar"],
                sig_r,
                Ratio_names,
                col,
                summary,
                all_simulations,
                imagename,
                k,
                j,
                config["SAVE_ALL_PLOTS"],
                pp,
                axres,
                config["iterations"],
                c,
            )

            sim_selgrain = summary.loc[(summary["Grain"] == grain.NAME.item())]
            sim_outerin = sim_selgrain.loc[
                (summary["Outer Iteration"] == k) & (summary["Inner Iteration"] == j)
            ]

            target = np.array(
                [
                    grain_size,
                    grain_delta[Ratio_names[0]].item(),
                    grain_delta[Ratio_names[1]].item(),
                ]
            )
            measured_simulations = np.array(
                [
                    sim_outerin["Measured diameter (nm)"],
                    sim_outerin["Measured d-17O/16O"],
                    sim_outerin["Measured d-18O/16O"],
                ]
            )
            initial_simulations = np.array(
                [
                    sim_outerin["Initial grain radius (nm)"],
                    sim_outerin["Initial d-17O/16O"],
                    sim_outerin["Initial d-18O/16O"],
                ]
            )

            new_simu, norm3D, grad, norm_summary = perform_gradient_descent_step(
                summary,
                grain,
                grain_size,
                grain_delta,
                Ratio_names,
                k,
                j,
                initial_simulations,
                learning_rate,
                norm_summary,
            )

            (
                decay_mat,
                decay_adam,
                momentum_mat,
                momentum_adam,
                momentum_nesperov_adam,
                learning_rate,
            ) = update_gradient_descent_parameters(
                grad, decay_mat, momentum_mat, eta, beta_decay, beta_momentum, eps, j
            )

            new_simu = apply_simulation_constraints(new_simu)

            plot_gradient_descent_parameters(
                ax_adnesp, k, j, norm3D, decay_adam, momentum_nesperov_adam, c
            )

            del PG_size, PG_delta
            PG_size = new_simu[:, 0].reshape(1, new_simu.shape[0])
            PG_delta = np.c_[new_simu[:, 1], new_simu[:, 2]]
            PG_delta = [PG_delta.tolist()]

            cost = compute_cost(
                norm_summary, sim_selgrain, k, config["nb_closest_match"]
            )
            j += 1

        norm_selgrain = norm_summary.iloc[sim_selgrain.index]
        norm_outer = norm_selgrain.loc[
            sim_selgrain.loc[sim_selgrain["Outer Iteration"] == k].index
        ]
        closest_match_index = norm_outer.sort_values(by="Norm", ascending=True)[
            0 : config["nb_closest_match"]
        ].index
        closest_match = sim_selgrain.loc[closest_match_index, :]

        plot_closest_matches_iteration(
            axres, k, config["iterations"], closest_match, Ratio_names
        )
        set_gradient_descent_titles(ax_adnesp, k)

    norm_selgrain = norm_summary.iloc[sim_selgrain.index]
    closest_match_index = norm_selgrain.sort_values(by="Norm", ascending=True)[
        0 : config["nb_closest_match"]
    ].index
    closest_match_final = sim_selgrain.loc[closest_match_index, :]

    plot_closest_matches_final(
        axres,
        config["iterations"],
        closest_match_final,
        Ratio_names,
        grain_size,
        grain_delta,
    )

    lines.extend((point_inner, point_matchfinal))
    labels.extend(label_points)

    Diam_res = closest_match_final["Initial grain radius (nm)"].mean()
    ER_Diam_res = closest_match_final["Initial grain radius (nm)"].std()
    Delta_res = closest_match_final.filter(regex="Initial d-*").mean()
    ER_Delta_res = closest_match_final.filter(regex="Initial d-*").std()
    Dilu_size = np.round((1 - grain_size / Diam_res) * 100, 2)
    Dilu_delta = (1 - grain_delta.div(Delta_res.values)) * 100

    data_res = pd.DataFrame(
        {
            "Filename": imagename,
            "Grain": grain.NAME.item(),
            "Grain measured diamter (nm)": grain.ROIDIAM * 1000,
            "Measured " + Ratio_names[0]: grain_delta[Ratio_names[0]].item(),
            "Measured " + Ratio_names[1]: grain_delta[Ratio_names[1]].item(),
            "Estimated true diameter (nm)": [int(Diam_res)],
            "ER-diam": [int(ER_Diam_res)],
            "Estimated " + Ratio_names[0]: [int(Delta_res.iloc[0])],
            "ER-" + Ratio_names[0]: [int(ER_Delta_res.iloc[0])],
            "Estimated " + Ratio_names[1]: [int(Delta_res.iloc[1])],
            "ER-" + Ratio_names[1]: [int(ER_Delta_res.iloc[0])],
            "Dilution on size (%)": [Dilu_size],
            "Dilution on " + Ratio_names[0] + "(%)": [
                Dilu_delta[Ratio_names[0]].item()
            ],
            "Dilution on " + Ratio_names[1] + "(%)": [
                Dilu_delta[Ratio_names[1]].item()
            ],
        }
    )

    print(
        f"Estimated size {int(Diam_res)} nm ({Dilu_size} %) and compositions "
        f"{'/'.join(map(str, Delta_res.astype(int).values))} permil "
        f"({str().join(map(str, Dilu_delta.astype(int).values))} %)"
    )

    grain_counter += 1
    pbar_overall.update(1)
    pbar_overall.set_postfix({"completed": f"{grain_counter}/{total_grains}"})

    finalize_result_figures(
        fres,
        f_adnesp,
        axres,
        ax_adnesp,
        imagename,
        grain.NAME.item(),
        Ratio_names,
        config["iterations"],
        lines,
        labels,
        pp,
    )

    return (
        summary,
        norm_summary,
        all_simulations,
        closest_match_final,
        data_res,
        grain_counter,
    )


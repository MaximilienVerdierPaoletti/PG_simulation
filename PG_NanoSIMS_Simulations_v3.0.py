# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:27:25 2023

Version 1.0:
Uses image files and excel database of those files to automatically estimates dilution on size and  one isotope ratio.

Version 2.0:
Enables to explore multiple isotpic ratios per grains at once.

Version 3.0:
Implement gradient descent methodoly

@author: Maximilien Verdier-Paoletti
"""

# %% Modules

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reset", "-f")

except:
    pass

import sys
# # sys.path.insert(0, 'F:/Work/Programmation/Presolar grains/Python functions/')
# sys.path.insert(0,'F:/Work/Programmation/Presolar grains/Simulations PG/')

import os
import numpy as np
import pandas as pd  # enables the use of dataframe
import matplotlib
import matplotlib.pyplot as plt  # Enables plotting of data
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import time
import tkinter as tk
from tkinter import filedialog

import cProfile
import pstats
from datetime import datetime
from tqdm import tqdm

from Modules.PG_simulations_func import PG_simulationv6
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

# plt.ioff()
matplotlib.rcParams["interactive"] = False


if __name__ == "__main__":
    # ========================================================================
    # PROFILING CONFIGURATION
    # Set ENABLE_PROFILING to True to enable profiling
    # ========================================================================
    ENABLE_PROFILING = False  # Set to False to disable profiling
    PROFILING_OUTPUT_DIR = "profiling_results"

    # ========================================================================
    # PLOT SAVING CONFIGURATION
    # Set SAVE_ALL_PLOTS to False to save only: original data, 3D plots, and
    # gradient descent parameter evolution plots (excludes simulation plots)
    # ========================================================================
    SAVE_ALL_PLOTS = False  # Set to False to save only essential plots

    # Create profiling output directory if it doesn't exist
    if ENABLE_PROFILING:
        os.makedirs(PROFILING_OUTPUT_DIR, exist_ok=True)
        pr = cProfile.Profile()
        pr.enable()
        print(
            f"[PROFILING] Profiling enabled. Results will be saved to '{PROFILING_OUTPUT_DIR}/'"
        )

    plt.close("all")

    # %% Grains and acquisition characteristics
    # ---- Grain parameters for simulations
    Nb_PG = 9
    nb_closest_match = 3  # Can't be bigger than Nb_PG
    beam = 100
    boxcar = 3
    elem = "O"

    Name_results = "test_alldata"

    delta_database = [*range(-900, 0, 50)]
    delta_database.extend(range(0, 200, 10))
    delta_database.extend(range(200, 2000, 100))
    delta_database.extend(range(2000, 21000, 1000))

    # ---- Number of outer and inner iterations
    iterations = 1
    max_iteration = 10
    cost_goal = 0.4

    # ---- Legend of summary figure (fres) for each measured grain
    lines = []
    labels = []
    point_inner, point_matchfinal, label_points = create_legend_elements()
    c = cm.rainbow(np.linspace(0, 1, Nb_PG))

    # --- Initialization of PDF figure summary
    pp = PdfPages(Name_results + ".pdf")

    # -----------------------------------------------------------------#
    # Data call
    root = tk.Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    file_list = filedialog.askopenfilename(multiple=True)
    root.withdraw()

    data = filedialog.askopenfilename()
    root.withdraw()
    data = pd.read_excel(data, header=0)
    data = data[data.NAME.str.contains("Bulk") == False]  # Drop bulk ROIs

    # ---- Saving variable allocation
    all_simulations = {}
    # Ratio_names=data.columns[data.columns.str.contains('^d-.*'+elem)].str.replace('d-','').to_list()
    Ratio_names = data.columns[data.columns.str.contains("^d-.*" + elem)].to_list()
    col = [
        "Image",
        "Grain",
        "Outer Iteration",
        "Inner Iteration",
        "Simulated grain index",
        "Initial grain radius (nm)",
        "sigma R",
        "Measured diameter (nm)",
    ]

    col_res = [
        "Grain",
        "sigma_r",
        "Measured diameter",
        "Simu true delta",
        "Simu true radius (nm)",
        "Simu measured radius (nm)",
    ]
    for m in range(len(Ratio_names)):
        s = Ratio_names[m]
        col_res.extend(["Simu measured " + s, "std", "Dilution on " + s + " (%)"])
        col_res.insert(3 + m, s)
        col.insert(5 + m, "Initial " + s)
        col.extend(["Measured " + s])
    col_res.append("Dilution on size (%)")

    summary = pd.DataFrame(columns=col)
    norm_summary = None  # Initialize norm_summary for gradient descent tracking

    # -----------------------------------------------------------------#
    #### Grain characteristics extraction
    start = time.time()

    # Calculate total number of grains for progress tracking
    total_grains = 0
    for file in file_list:
        imagename = file.rsplit("/", 1)[1].replace(".im", "")
        if data.NAME.str.contains("_corr").any() == False:
            imagename = imagename.replace("_corr", "")
        grains = data.loc[data.NAME.str.contains(imagename)]
        if grains.empty is False:
            total_grains += grains.shape[0]

    # Initialize overall progress bar
    pbar_overall = tqdm(
        total=total_grains,
        desc="Overall Progress",
        unit="grain",
        position=0,
        leave=True,
    )
    grain_counter = 0

    for file in tqdm(
        file_list, desc="Processing Files", unit="file", position=1, leave=False
    ):
        imagename = file.rsplit("/", 1)[1].replace(".im", "")
        all_simulations[imagename] = {}  # Arborescence of dictionnary on file name
        if data.NAME.str.contains("_corr").any() == False:
            imagename = imagename.replace("_corr", "")
        grains = data.loc[data.NAME.str.contains(imagename)]

        if grains.empty is True:
            print(
                f"\nNo presolar grain were detected prior by the user in acquisition {imagename}"
            )
            continue
        else:
            print(f"\n{grains.shape[0]} presolar grain detected by user in {imagename}")

        for z in tqdm(
            range(grains.shape[0]),
            desc=f"Grain ({imagename})",
            unit="grain",
            position=2,
            leave=False,
        ):  # Loop on number of detected grains by user in this file
            grain = grains.iloc[[z]]
            grain_delta = grain.filter(
                regex="^d-.*" + elem, axis=1
            )  # Extract automatically the delta composition of the grain based on the specified element
            ER_grain_delta = grain.filter(regex="^ER-d-.*" + elem, axis=1)
            grain_size = grain["ROIDIAM"].item() * 1000

            if grain.columns.str.contains("sig", case=False).any():
                sig_r = grain.iloc[
                    :, np.where(grain.columns.str.contains("sig") == True)
                ]
            else:
                sig_r = 0.5
                print(f"No information available on sigma ratio, fixed to {sig_r}")

            delta_range = []
            for i in grain_delta.to_numpy()[0]:
                delta_range.append(
                    [h for h in delta_database if np.sign(i) * h > np.abs(i)]
                )

            size_range = range(50, 900, 50)

            # %% Loops on simulations

            it = 0
            fres, axres, f_adnesp, ax_adnesp = initialize_result_figures(iterations)

            for k in tqdm(
                range(0, iterations),
                desc=f"Outer Iterations (Grain {z + 1}/{grains.shape[0]})",
                unit="iter",
                position=3,
                leave=False,
            ):
                all_simulations[imagename][k] = {}
                plot_measured_grain(axres, k, iterations, grain_size, grain_delta)

                # -----------------------------------------------------------------#
                # Grain simulations conditions

                PG_delta = [
                    [[np.random.choice(i) for i in delta_range] for o in range(Nb_PG)]
                ]  # Randomly select grain composition from initial ranges with shape
                PG_size = np.random.choice(size_range, Nb_PG).reshape(1, Nb_PG)

                # ---- Gradient descent parameters
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
                ) = initialize_gradient_descent_parameters(
                    PG_size, PG_delta, Nb_PG, n_ratios=len(Ratio_names)
                )

                # for j in range(0, zoom_iteration):
                j = 0
                cost = 5
                pbar_inner = tqdm(
                    total=max_iteration + 1,
                    desc=f"Inner Iterations (Outer {k + 1}/{iterations})",
                    unit="iter",
                    position=4,
                    leave=False,
                )
                while True:
                    all_simulations[imagename][k][j] = {}
                    pbar_inner.update(1)
                    pbar_inner.set_postfix(
                        {"cost": f"{cost:.3f}", "goal": f"{cost_goal:.3f}"}
                    )
                    if (
                        cost < cost_goal or j > max_iteration
                    ):  # As a while loop continues if one condition is True, this if loop is necessary to break the cycle if one condition is True
                        pbar_inner.close()
                        break
                    for u in range(0, PG_size.shape[0]):
                        # ----------- Simulation of PG images
                        if (k == 0) & (j == 0):
                            verif = 1
                        else:
                            verif = 0
                        f, ax, plots, plots_title, PG_coor, raster, px, f_OG = (
                            PG_simulationv6(
                                file=file,
                                elem=elem,
                                PG_delta=PG_delta,
                                PG_size=PG_size[u, :],
                                OG_grain=grain,
                                beam_size=beam,
                                boxcar_px=boxcar,
                                smart=1,
                                verif=verif,
                                standard="average",
                                display="OFF",
                            )
                        )

                        if f_OG is not None:
                            save_figure_to_pdf(f_OG, pp, title=imagename)
                        ax = ax.ravel()

                        # Sigma and Delta map selection based on anomalous ratio
                        (
                            sigma_map,
                            delta_map,
                            sigma_anomalous_map_index,
                            delta_anomalous_map_index,
                            anomalous_ratio_name,
                        ) = select_sigma_delta_maps(plots, plots_title, grain_delta)

                        for i in range(0, PG_size.shape[1]):  # Simulation on grains
                            # Extract features for this grain
                            (
                                Diam,
                                delta_values,
                                mask,
                                mask_th,
                                (xsel, ysel),
                                (x, y),
                            ) = extract_grain_features(
                                PG_size[u, :],
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

                            plot_simulated_grain(
                                axres, k, iterations, Diam, delta_values, c, i
                            )
                            ax = ax.ravel()
                            for h in range(0, len(ax)):
                                ax[h].plot(xsel, ysel, "--", color="w", linewidth=2)
                                ax[h].plot(x, y, "-", color="r", linewidth=2)
                                # axinsert[k].plot(x,y,'-',color='r',linewidth=2)

                            # final=r'$\sigma_{R}$ ='+str(sig_r[j])+r', $\delta^{17}O$ ='+str(int(np.mean(delta_map[X,Y])))+ r', $\delta^{18}O$ ='+str(int(np.mean(plots[10][X,Y])))+', Diameter ='+str(int(R*2))

                            # plt.gcf().text(0.7,0.98-j*0.01,final,fontsize=9)

                            S = pd.Series(
                                [
                                    imagename,
                                    grain.NAME.item(),
                                    k,
                                    j,
                                    i,
                                    PG_size[u, i],
                                    sig_r,
                                    Diam,
                                    np.round(delta_values[0], 2),
                                    np.round(delta_values[1], 2),
                                ]
                            )
                            S = S.to_frame().T
                            for o in range(len(PG_delta[u][i])):
                                S.insert(
                                    5 + o, "", PG_delta[u][i][o], allow_duplicates=True
                                )
                            S = S.set_axis(col, axis=1)
                            summary = pd.concat([summary, S], axis=0, ignore_index=True)

                        initial = (
                            "Outer Iteration : "
                            + str(k)
                            + "\n"
                            + "Inner Iteration : "
                            + str(j)
                            + "\n"
                            "Presolar Grain Simulation #"
                            + str(it)
                            + "\n"
                            + "Image is: "
                            + str(file.rsplit("/", 1)[1])
                            + "\n"
                            + "Number of grains : "
                            + str(PG_size.shape[1])
                        )

                        f.text(0.15, 0.92, initial, fontsize=11)

                        f.set_size_inches(16, 10)
                        it = +1

                        # Save simulation plots only if SAVE_ALL_PLOTS is True
                        if SAVE_ALL_PLOTS:
                            save_figure_to_pdf(f, pp)
                        else:
                            plt.close(f)  # Close figure without saving to save memory

                        for im_it in range(0, len(plots_title)):
                            all_simulations[imagename][k][j][plots_title[im_it]] = (
                                plots[im_it]
                            )

                    sim_selgrain = summary.loc[
                        (summary["Grain"] == grain.NAME.item())
                    ]  # Working solely on current grain
                    sim_outerin = sim_selgrain.loc[
                        (summary["Outer Iteration"] == k)
                        & (summary["Inner Iteration"] == j)
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

                    [new_simu, norm3D, grad] = GD_AdamNesperov(
                        target, measured_simulations, initial_simulations, learning_rate
                    )

                    # Update gradient descent parameters
                    (
                        decay_mat,
                        decay_adam,
                        momentum_mat,
                        momentum_adam,
                        momentum_nesperov_adam,
                        learning_rate,
                    ) = update_gradient_descent_parameters(
                        grad,
                        decay_mat,
                        momentum_mat,
                        eta,
                        beta_decay,
                        beta_momentum,
                        eps,
                        j,
                    )

                    # Apply constraints to updated simulations
                    new_simu = apply_simulation_constraints(new_simu)

                    # Save cost function evolution (normalized norms)
                    norm_summary = save_norm_to_summary(norm3D, norm_summary)

                    # Study of the behavior of parameters in gradient descent
                    plot_gradient_descent_parameters(
                        ax_adnesp, k, j, norm3D, decay_adam, momentum_nesperov_adam, c
                    )

                    del PG_size, PG_delta

                    PG_size = new_simu[:, 0].reshape(1, new_simu.shape[0])
                    PG_delta = np.c_[new_simu[:, 1], new_simu[:, 2]]
                    PG_delta = [PG_delta.tolist()]

                    # Compute cost function
                    cost = compute_cost(norm_summary, sim_selgrain, k, nb_closest_match)
                    j += 1

                # Look for the closest match in all simulation of this outer iteration
                # sim_outer = summary.loc[summary['Outer Iteration'] == k]
                # norm_outer = (np.abs(sim_outer['Measured diameter'].divide(grain_size) - 1) + (np.abs(sim_outer[Ratio_names].div(grain_delta.values) - 1)).sum(axis=1)) ** 0.5
                norm_selgrain = norm_summary.iloc[sim_selgrain.index]
                norm_outer = norm_selgrain.loc[
                    sim_selgrain.loc[sim_selgrain["Outer Iteration"] == k].index
                ]
                closest_match_index = norm_outer.sort_values(by="Norm", ascending=True)[
                    0:nb_closest_match
                ].index
                closest_match = sim_selgrain.loc[closest_match_index, :]

                plot_closest_matches_iteration(
                    axres, k, iterations, closest_match, Ratio_names
                )
                set_gradient_descent_titles(ax_adnesp, k)

            # Look for the closest match throughout all the iterations
            # norm = (np.abs(summary['Measured diameter'].divide(grain_size) - 1) + (np.abs(summary[Ratio_names].div(grain_delta.values) - 1)).sum(axis=1)) ** 0.5
            closest_match_index = norm_selgrain.sort_values(by="Norm", ascending=True)[
                0:nb_closest_match
            ].index
            closest_match_final = sim_selgrain.loc[closest_match_index, :]

            plot_closest_matches_final(
                axres,
                iterations,
                closest_match_final,
                Ratio_names,
                grain_size,
                grain_delta,
            )

            lines.extend((point_inner, point_matchfinal))
            labels.extend(label_points)

            # f_norm.suptitle(imagename+'\n grain : '+str(grain.NAME.item()),fontsize=15)
            # f_norm.set_size_inches(16, 10)
            # pp.savefig(f_norm, transparent=True, dpi=100)
            # plt.close(fres)

            if "match_summary" not in globals():
                match_summary = closest_match_final
            else:
                match_summary = pd.concat([match_summary, closest_match_final])

            # Result variables
            Diam_res = closest_match_final["Initial grain radius (nm)"].mean()
            ER_Diam_res = closest_match_final["Initial grain radius (nm)"].std()
            Delta_res = closest_match_final.filter(regex="Initial d-*").mean()
            ER_Delta_res = closest_match_final.filter(regex="Initial d-*").std()
            Dilu_size = np.round((1 - grain_size / Diam_res) * 100, 2)
            Dilu_delta = (1 - grain_delta.div(Delta_res.values)) * 100
            ER_Dilu_delta = np.abs(
                (Dilu_delta - 100)
                * (
                    np.divide(ER_Delta_res.values, Delta_res.values) ** 2
                    + np.divide(ER_grain_delta.values, grain_delta.values) ** 2
                )
                ** 0.5
            )

            if "data_res" not in globals():
                data_res = pd.DataFrame(
                    {
                        "Filename": imagename,
                        "Grain": grain.NAME.item(),
                        "Grain measured diamter (nm)": grain.ROIDIAM * 1000,
                        "Measured " + Ratio_names[0]: grain_delta[
                            Ratio_names[0]
                        ].item(),
                        "Measured " + Ratio_names[1]: grain_delta[
                            Ratio_names[1]
                        ].item(),
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
            else:
                Data_serie = pd.Series(
                    [
                        imagename,
                        grain.NAME.item(),
                        grain.ROIDIAM.item() * 1000,
                        grain_delta[Ratio_names[0]].item(),
                        grain_delta[Ratio_names[1]].item(),
                        int(Diam_res),
                        int(ER_Diam_res),
                        int(Delta_res.iloc[0]),
                        int(ER_Delta_res.iloc[0]),
                        int(Delta_res.iloc[1]),
                        int(ER_Delta_res.iloc[0]),
                        Dilu_size,
                        Dilu_delta[Ratio_names[0]].item(),
                        Dilu_delta[Ratio_names[1]].item(),
                    ]
                )
                Data_serie = Data_serie.to_frame().T
                Data_serie = Data_serie.set_axis(data_res.columns, axis=1)
                data_res = pd.concat([data_res, Data_serie], axis=0, ignore_index=True)

            # FIXME: Save all_simulation dictionnary into a pickle and or a HDF5 file

            print(
                f"Estimated size {int(Diam_res)} nm ({Dilu_size} %) and compositions {'/'.join(map(str, Delta_res.astype(int).values))} permil "
                f"({str().join(map(str, Dilu_delta.astype(int).values))} %)"
            )

            del closest_match, closest_match_index

            # Update overall progress bar
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
            iterations,
            lines,
            labels,
            pp,
        )

    plt.ion()
    plt.show()
    pp.close()

    # Close overall progress bar
    pbar_overall.close()

    # Saving results in Excel file
    with pd.ExcelWriter(Name_results + ".xlsx") as writer:
        summary.to_excel(writer, sheet_name="All_simulations")
        match_summary.to_excel(writer, sheet_name="Match Summary")
        data_res.to_excel(writer, sheet_name="Dilution results")

    end = time.time()
    print("Elapsed time: " + str(end - start) + " s")

    # ========================================================================
    # PROFILING RESULTS OUTPUT
    # ========================================================================
    if ENABLE_PROFILING:
        pr.disable()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed profiling statistics
        stats_file = os.path.join(
            PROFILING_OUTPUT_DIR, f"profile_stats_{timestamp}.txt"
        )
        with open(stats_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()
            print(f"\n[PROFILING] Detailed statistics saved to: {stats_file}")

        # Save cumulative time statistics (sorted by cumulative time)
        cumtime_file = os.path.join(
            PROFILING_OUTPUT_DIR, f"profile_cumtime_{timestamp}.txt"
        )
        with open(cumtime_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats(30)  # Top 30 functions
            print(f"[PROFILING] Cumulative time statistics saved to: {cumtime_file}")

        # Save per-call time statistics (sorted by per-call time)
        percall_file = os.path.join(
            PROFILING_OUTPUT_DIR, f"profile_percall_{timestamp}.txt"
        )
        with open(percall_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(30)  # Top 30 functions
            print(f"[PROFILING] Per-call time statistics saved to: {percall_file}")

        # Save profiling data in binary format for later analysis
        profile_binary = os.path.join(PROFILING_OUTPUT_DIR, f"profile_{timestamp}.prof")
        pr.dump_stats(profile_binary)
        print(f"[PROFILING] Binary profile data saved to: {profile_binary}")
        print(f"[PROFILING] To analyze later, use: python -m pstats {profile_binary}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY - Top 20 functions by total time")
        print("=" * 80)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(20)
        print("=" * 80 + "\n")


# %%

# import matplotlib.animation as animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.animation import PillowWriter

# out_it=0
# label_im='Sigma 17O/16O'


# S = summary.loc[summary['Outer Iteration'] == out_it]
# f_im = plt.figure()
# axani_im = plt.subplot(121)
# axani_im.set_axis_off()
# axani_im.set_title(label_im,fontsize=20)
# axani_grad = plt.subplot(122,projection = '3d')
# axani_grad.set_title('Gradient Descent',fontsize=20)
# axani_grad.tick_params(labelsize=20, pad=10)
# axani_grad.set_xlabel('Measured diameter (nm)', fontsize=20,labelpad=30)
# axani_grad.set_ylabel(grain_delta.columns[0], fontsize=20,labelpad=30)
# axani_grad.set_zlabel(grain_delta.columns[1], fontsize=20,labelpad=30)
# axani_grad.xaxis._axinfo['label']['space_factor'] = 5.0
# # ax.yaxis._axinfo['label']['space_factor'] = 2.0
# # ax.zaxis._axinfo['label']['space_factor'] = 2.0


# c_it = cm.rainbow(np.linspace(0, 1, S['Inner Iteration'].max()))
# axani_grad.plot(grain_size,grain_delta['d-17O/16O'].item(),grain_delta['d-18O/16O'].item(),'sk',markersize=12)

# def animate_im(i):
#     simu = all_simulations[imagename][out_it][i][label_im]
#     im = axani_im.imshow(simu.data)
#     divider = make_axes_locatable(axani_im)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = plt.colorbar(im,cax=cax)
#     cbar.ax.tick_params(labelsize=20)
#     # axs[i,j].title.set_text(titles[h])

#     return [im]

# # for im_it in range(0,S['Inner Iteration'].max()):
# #     animate(im_it)

# ani_im = animation.FuncAnimation(f_im, animate_im, interval = 500, frames = range(S['Inner Iteration'].max()), blit=True, repeat_delay=500)


# def animate_grad(i):
#     simu = all_simulations[imagename][out_it][i]['Sigma 17O/16O']
#     S_sel=S[S['Inner Iteration'] == i]
#     im_grad = axani_grad.plot(S_sel['Measured diameter (nm)'],S_sel['Measured ' + Ratio_names[0]],S_sel['Measured ' + Ratio_names[1]],color=c_it[i], marker='o', markersize=20,linestyle='none')

#     return im_grad

# # for im_it in range(0,S['Inner Iteration'].max()):
# #     animate(im_it)

# ani_grad = animation.FuncAnimation(f_im, animate_grad, interval = 500, frames = range(S['Inner Iteration'].max()), blit=True, repeat_delay=500)


# mng = plt.get_current_fig_manager()
# ### works on Ubuntu??? >> did NOT working on windows
# # mng.resize(*mng.window.maxsize())
# mng.window.state('zoomed') #works fine on Windows!
# plt.show(block = True)

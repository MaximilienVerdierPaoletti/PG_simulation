# -*- coding: utf-8 -*-
"""
Main processing orchestration for PG NanoSIMS Simulations.

@author: Maximilien Verdier-Paoletti
"""

import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from tqdm import tqdm

from Modules.pg_simulation_config import setup_profiling, stop_profiling
from Modules.pg_simulation_data import (
    load_data_files,
    initialize_data_structures,
    calculate_total_grains,
)
from Modules.pg_simulation_grain import process_single_grain
from Modules.pg_simulation_results import save_results

# plt.ioff()
matplotlib.rcParams["interactive"] = False


def process_all_grains(file_list, data, config, use_gui=True):
    """
    Main function to process all grains from files.

    Parameters:
    -----------
    file_list : list
        List of image file paths
    data : pd.DataFrame
        Loaded data
    config : dict
        Configuration dictionary
    use_gui : bool
        Whether to use GUI for file selection

    Returns:
    --------
    summary : pd.DataFrame
        Summary of all simulations
    match_summary : pd.DataFrame
        Summary of closest matches
    data_res : pd.DataFrame
        Dilution results
    all_simulations : dict
        All simulation data
    """
    # Load data if not provided
    if file_list is None or data is None:
        file_list, data = load_data_files(use_gui=use_gui)

    # Initialize data structures
    Ratio_names, col, col_res, summary = initialize_data_structures(
        data, config["elem"]
    )
    norm_summary = None
    all_simulations = {}
    f_OG_result = None  # Store the first f_OG figure

    # Setup profiling
    pr = setup_profiling(config["ENABLE_PROFILING"], config["PROFILING_OUTPUT_DIR"])

    plt.close("all")

    # Initialize PDF output
    pp = PdfPages(config["Name_results"] + ".pdf")

    # Calculate total grains for progress tracking
    total_grains = calculate_total_grains(file_list, data)
    pbar_overall = tqdm(
        total=total_grains,
        desc="Overall Progress",
        unit="grain",
        position=0,
        leave=True,
    )
    grain_counter = 0

    # Initialize result containers
    match_summary = None
    data_res = None

    # Color map for plotting
    c = cm.rainbow(np.linspace(0, 1, config["Nb_PG"]))

    start = time.time()

    # Process each file
    for file in tqdm(
        file_list, desc="Processing Files", unit="file", position=1, leave=False
    ):
        imagename = file.rsplit("/", 1)[1].replace(".im", "")
        all_simulations[imagename] = {}

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

        # Process each grain in the file
        for z in tqdm(
            range(grains.shape[0]),
            desc=f"Grain ({imagename})",
            unit="grain",
            position=2,
            leave=False,
        ):
            grain = grains.iloc[[z]]

            (
                summary,
                norm_summary,
                all_simulations,
                closest_match_final,
                grain_data_res,
                grain_counter,
                f_OG_grain,
            ) = process_single_grain(
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
            )

            # Accumulate results
            if match_summary is None:
                match_summary = closest_match_final
            else:
                match_summary = pd.concat(
                    [match_summary, closest_match_final], ignore_index=True
                )

            if data_res is None:
                data_res = grain_data_res
            else:
                data_res = pd.concat([data_res, grain_data_res], ignore_index=True)
            
            # Store first f_OG figure (or use the latest one)
            if f_OG_grain is not None:
                f_OG_result = f_OG_grain

    plt.ion()
    plt.show()
    pp.close()
    pbar_overall.close()

    # Save results
    save_results(summary, match_summary, data_res, config["Name_results"])

    end = time.time()
    print("Elapsed time: " + str(end - start) + " s")

    # Stop profiling
    stop_profiling(pr, config["PROFILING_OUTPUT_DIR"])

    return summary, match_summary, data_res, all_simulations, f_OG_result


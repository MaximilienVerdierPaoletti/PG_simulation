# -*- coding: utf-8 -*-
"""
Main entry point for PG NanoSIMS Simulations v3.0 (Functions version).

This script has been refactored into multiple modules for better organization:
- Modules/pg_simulation_config.py: Configuration and profiling
- Modules/pg_simulation_data.py: Data loading and initialization
- Modules/pg_simulation_grain.py: Grain processing functions
- Modules/pg_simulation_results.py: Result saving functions
- Modules/pg_simulation_core.py: Main processing orchestration

Version 1.0:
Uses image files and excel database of those files to automatically estimates dilution on size and  one isotope ratio.

Version 2.0:
Enables to explore multiple isotpic ratios per grains at once.

Version 3.0:
Implement gradient descent methodoly

Version 3.0_functions:
Refactored into functions for GUI integration

@author: Maximilien Verdier-Paoletti
"""

# %% Modules

import sys

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reset", "-f")

except Exception:
    pass

# # sys.path.insert(0, 'F:/Work/Programmation/Presolar grains/Python functions/')
# sys.path.insert(0,'F:/Work/Programmation/Presolar grains/Simulations PG/')

from Modules.pg_simulation_config import create_default_config
from Modules.pg_simulation_core import process_all_grains


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if GUI mode is requested
    use_gui_app = "--gui" in sys.argv or "-g" in sys.argv

    if use_gui_app:
        # Launch GUI application
        from gui_app import main

        main()
    else:
        # Command-line mode with default configuration
        # Create default configuration
        config = create_default_config()

        # Optionally modify config here
        # config["Nb_PG"] = 9
        # config["iterations"] = 1
        # etc.

        # ========================================================================
        # PROFILING CONFIGURATION
        # Set ENABLE_PROFILING to True to enable profiling
        # ========================================================================
        config["ENABLE_PROFILING"] = False  # Set to True to enable profiling
        config["PROFILING_OUTPUT_DIR"] = "profiling_results"

        # ========================================================================
        # PLOT SAVING CONFIGURATION
        # Set SAVE_ALL_PLOTS to False to save only: original data, 3D plots, and
        # gradient descent parameter evolution plots (excludes simulation plots)
        # ========================================================================
        config["SAVE_ALL_PLOTS"] = False  # Set to True to save all simulation plots

        # Process all grains
        summary, match_summary, data_res, all_simulations, f_OG = process_all_grains(
            file_list=None, data=None, config=config, use_gui=True
        )

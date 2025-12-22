# -*- coding: utf-8 -*-
"""
Result saving functions for PG NanoSIMS Simulations.

@author: Maximilien Verdier-Paoletti
"""

import pandas as pd


def save_results(summary, match_summary, data_res, output_name):
    """
    Save all results to Excel file.

    Parameters:
    -----------
    summary : pd.DataFrame
        Summary of all simulations
    match_summary : pd.DataFrame
        Summary of closest matches
    data_res : pd.DataFrame
        Dilution results
    output_name : str
        Base name for output file (without extension)
    """
    with pd.ExcelWriter(output_name + ".xlsx") as writer:
        summary.to_excel(writer, sheet_name="All_simulations")
        match_summary.to_excel(writer, sheet_name="Match Summary")
        data_res.to_excel(writer, sheet_name="Dilution results")


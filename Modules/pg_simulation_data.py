# -*- coding: utf-8 -*-
"""
Data loading and initialization functions for PG NanoSIMS Simulations.

@author: Maximilien Verdier-Paoletti
"""

import tkinter as tk
from tkinter import filedialog
import pandas as pd


def load_data_files(file_list=None, data_file=None, use_gui=True):
    """
    Load image files and Excel data file.

    Parameters:
    -----------
    file_list : list, optional
        List of image file paths. If None and use_gui=True, will prompt user.
    data_file : str, optional
        Path to Excel data file. If None and use_gui=True, will prompt user.
    use_gui : bool
        Whether to use GUI file dialogs (default: True)

    Returns:
    --------
    file_list : list
        List of image file paths
    data : pd.DataFrame
        Loaded and filtered data
    """
    if use_gui:
        if file_list is None:
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.after_idle(root.attributes, "-topmost", False)
            file_list = filedialog.askopenfilename(multiple=True)
            root.withdraw()

        if data_file is None:
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.after_idle(root.attributes, "-topmost", False)
            data_file = filedialog.askopenfilename()
            root.withdraw()

    if data_file is None:
        raise ValueError("data_file must be provided if use_gui=False")

    data = pd.read_excel(data_file, header=0)
    data = data[data.NAME.str.contains("Bulk") == False]  # Drop bulk ROIs

    return file_list, data


def initialize_data_structures(data, elem="O"):
    """
    Initialize data structures for storing results.

    Parameters:
    -----------
    data : pd.DataFrame
        Loaded data
    elem : str
        Element name for filtering ratio columns

    Returns:
    --------
    Ratio_names : list
        List of ratio column names
    col : list
        Column names for summary DataFrame
    col_res : list
        Column names for results DataFrame
    summary : pd.DataFrame
        Empty summary DataFrame
    """
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

    return Ratio_names, col, col_res, summary


def calculate_total_grains(file_list, data):
    """Calculate total number of grains across all files."""
    total_grains = 0
    for file in file_list:
        imagename = file.rsplit("/", 1)[1].replace(".im", "")
        if data.NAME.str.contains("_corr").any() == False:
            imagename = imagename.replace("_corr", "")
        grains = data.loc[data.NAME.str.contains(imagename)]
        if grains.empty is False:
            total_grains += grains.shape[0]
    return total_grains


# -*- coding: utf-8 -*-
"""
GUI Application for PG NanoSIMS Simulations v3.0

A customtkinter-based GUI for configuring and running PG NanoSIMS simulations.

@author: Maximilien Verdier-Paoletti
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for tkinter integration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from Modules.pg_simulation_config import create_default_config
from Modules.pg_simulation_core import process_all_grains


class PGSimulationGUI(ctk.CTk):
    """Main GUI application for PG NanoSIMS Simulations."""

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("PG NanoSIMS Simulations v3.0")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize configuration with defaults
        self.config = create_default_config()
        self.file_list = None
        self.data_file = None

        # Store simulation results for plotting
        self.simulation_results = {
            "summary": None,
            "match_summary": None,
            "data_res": None,
            "all_simulations": None,
            "original_data": None,
            "f_OG": None,  # Original grain figure
        }

        # Create main container
        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container - sidebar + main area
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Configuration sidebar (left side, fixed width)
        self.config_panel = ctk.CTkFrame(self.main_container, width=350)
        self.config_panel.pack(side="left", fill="y", padx=(0, 5))
        self.config_panel.pack_propagate(False)  # Maintain fixed width

        # Graph panel (right side, main area)
        self.graph_panel = ctk.CTkFrame(self.main_container)
        self.graph_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Create configuration widgets
        self.create_config_widgets()

        # Create graph widgets
        self.create_graph_widgets()

    def create_config_widgets(self):
        """Create configuration widgets in config panel."""
        # Title for sidebar
        sidebar_title = ctk.CTkLabel(
            self.config_panel,
            text="Configuration",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        sidebar_title.pack(pady=10)

        # Main scrollable frame for configuration
        self.config_scroll = ctk.CTkScrollableFrame(self.config_panel)
        self.config_scroll.pack(fill="both", expand=True, padx=10, pady=5)

        # Title
        title_label = ctk.CTkLabel(
            self.config_scroll,
            text="Simulation Configuration",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(pady=(0, 20))

        # File Selection Section
        self.create_file_section()

        # Grain Parameters Section
        self.create_grain_parameters_section()

        # Delta Database Section
        self.create_delta_database_section()

        # Iteration Parameters Section
        self.create_iteration_section()

        # Size Range Section
        self.create_size_range_section()

        # Output Settings Section
        self.create_output_section()

        # Control Buttons
        self.create_control_buttons()

    def create_graph_widgets(self):
        """Create graph display widgets."""
        # Title
        graph_title = ctk.CTkLabel(
            self.graph_panel,
            text="Simulation Results & Graphs",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        graph_title.pack(pady=10)

        # Create tabview for three graph tabs
        self.graph_tabview = ctk.CTkTabview(self.graph_panel)
        self.graph_tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Original Data
        self.original_tab = self.graph_tabview.add("Original Data")
        self.original_canvas_frame = ctk.CTkFrame(self.original_tab)
        self.original_canvas_frame.pack(fill="both", expand=True)

        # Tab 2: Results Graph
        self.results_tab = self.graph_tabview.add("Results")
        self.results_canvas_frame = ctk.CTkFrame(self.results_tab)
        self.results_canvas_frame.pack(fill="both", expand=True)

        # Tab 3: All Simulations
        self.simulations_tab = self.graph_tabview.add("All Simulations")
        self.simulations_canvas_frame = ctk.CTkFrame(self.simulations_tab)
        self.simulations_canvas_frame.pack(fill="both", expand=True)

        # Initialize empty plots
        self.init_graphs()

    def init_graphs(self):
        """Initialize empty graph canvases."""
        # Original Data plot - will display f_OG figure
        self.original_fig = None
        self.original_canvas = None
        # Placeholder label until f_OG is available
        self.original_placeholder = ctk.CTkLabel(
            self.original_canvas_frame,
            text="No original data yet.\nRun simulation to see original grain figure (f_OG).",
            font=ctk.CTkFont(size=14),
        )
        self.original_placeholder.pack(expand=True)

        # Results plot
        self.results_fig = Figure(figsize=(10, 6), dpi=100)
        self.results_ax = self.results_fig.add_subplot(111, projection="3d")
        self.results_ax.text(
            0.5,
            0.5,
            0.5,
            "No results yet.\nRun simulation to see results.",
            ha="center",
            va="center",
            fontsize=14,
        )
        self.results_canvas = FigureCanvasTkAgg(
            self.results_fig, self.results_canvas_frame
        )
        self.results_canvas.draw()
        self.results_canvas.get_tk_widget().pack(fill="both", expand=True)

        # All Simulations plot
        self.simulations_fig = Figure(figsize=(10, 6), dpi=100)
        self.simulations_ax = self.simulations_fig.add_subplot(111)
        self.simulations_ax.text(
            0.5,
            0.5,
            "No simulation data yet.\nRun simulation to see all simulations.",
            ha="center",
            va="center",
            fontsize=14,
        )
        self.simulations_ax.set_xticks([])
        self.simulations_ax.set_yticks([])
        self.simulations_canvas = FigureCanvasTkAgg(
            self.simulations_fig, self.simulations_canvas_frame
        )
        self.simulations_canvas.draw()
        self.simulations_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_graphs(self):
        """Update all graphs with simulation results."""
        # Update Original Data tab
        self.update_original_data_graph()

        # Update Results tab
        self.update_results_graph()

        # Update All Simulations tab
        self.update_simulations_graph()

    def update_original_data_graph(self):
        """Update the original data graph with f_OG figure."""
        # Remove placeholder if exists
        if self.original_placeholder is not None:
            self.original_placeholder.pack_forget()
            self.original_placeholder = None

        # Remove old canvas if exists
        if self.original_canvas is not None:
            self.original_canvas.get_tk_widget().pack_forget()

        if self.simulation_results["f_OG"] is not None:
            # Use the f_OG figure directly
            f_OG = self.simulation_results["f_OG"]
            self.original_canvas = FigureCanvasTkAgg(f_OG, self.original_canvas_frame)
            self.original_canvas.draw()
            self.original_canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            # Show placeholder if no f_OG available
            self.original_placeholder = ctk.CTkLabel(
                self.original_canvas_frame,
                text="No original grain figure (f_OG) available yet.\nRun simulation to see original grain data.",
                font=ctk.CTkFont(size=14),
            )
            self.original_placeholder.pack(expand=True)

    def update_results_graph(self):
        """Update the results graph with iteration results."""
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111, projection="3d")

        if self.simulation_results["summary"] is not None:
            summary = self.simulation_results["summary"]

            # Extract columns for 3D plot
            size_col = "Measured diameter (nm)"
            delta_cols = []

            # Find delta columns (e.g., "Measured d-17O/16O", "Measured d-18O/16O")
            for col in summary.columns:
                if col.startswith("Measured") and "d-" in col:
                    delta_cols.append(col)

            if size_col in summary.columns and len(delta_cols) >= 2:
                # Import for color mapping
                import matplotlib.cm as cm
                import numpy as np

                # Plot target values as black squares if original_data is available
                if (
                    self.simulation_results["original_data"] is not None
                    and "Grain" in summary.columns
                ):
                    original_data = self.simulation_results["original_data"]
                    # Get element for delta column matching
                    elem = self.config.get("elem", "O")

                    # Find delta columns in original data
                    orig_delta_cols = [
                        col
                        for col in original_data.columns
                        if col.startswith("d-") and elem in col
                    ]

                    # Get unique grains from summary
                    unique_grains = summary["Grain"].unique()

                    for grain_name in unique_grains:
                        # Find matching grain in original data (exact match or contains)
                        grain_data = original_data[original_data["NAME"] == grain_name]
                        # If no exact match, try contains
                        if grain_data.empty:
                            grain_data = original_data[
                                original_data["NAME"].str.contains(grain_name, na=False)
                            ]
                        if not grain_data.empty:
                            grain_row = grain_data.iloc[0]
                            # Get grain size (ROIDIAM is in micrometers, convert to nm)
                            if "ROIDIAM" in grain_row.index:
                                target_size = (
                                    grain_row["ROIDIAM"] * 1000
                                )  # Convert to nm
                            else:
                                continue

                            # Get target delta values
                            if len(orig_delta_cols) >= 2:
                                target_delta1 = grain_row[orig_delta_cols[0]]
                                target_delta2 = grain_row[orig_delta_cols[1]]

                                # Plot target as black square
                                ax.scatter(
                                    target_size,
                                    target_delta1,
                                    target_delta2,
                                    c="black",
                                    marker="s",
                                    s=100,
                                    alpha=1.0,
                                    zorder=10,
                                )

                # Color by simulated grain index
                if "Simulated grain index" in summary.columns:
                    # Get all unique simulated grain indices for color mapping
                    unique_grain_indices = sorted(
                        summary["Simulated grain index"].unique()
                    )

                    # Create color map based on simulated grain index
                    if len(unique_grain_indices) > 1:
                        colors = cm.rainbow(
                            np.linspace(0, 1, len(unique_grain_indices))
                        )
                    else:
                        colors = ["blue"]

                    # Plot all data points colored by simulated grain index
                    for grain_idx, sim_grain_idx in enumerate(unique_grain_indices):
                        grain_data = summary[
                            summary["Simulated grain index"] == sim_grain_idx
                        ]

                        sizes = grain_data[size_col].values
                        delta1 = grain_data[delta_cols[0]].values
                        delta2 = grain_data[delta_cols[1]].values

                        # Color based on simulated grain index
                        color = (
                            colors[grain_idx]
                            if len(unique_grain_indices) > 1
                            else colors[0]
                        )

                        ax.scatter(
                            sizes,
                            delta1,
                            delta2,
                            c=[color],
                            marker="o",
                            s=50,
                            alpha=0.6,
                        )
                else:
                    # Fallback: plot all points without iteration distinction
                    sizes = summary[size_col].values
                    delta1 = summary[delta_cols[0]].values
                    delta2 = summary[delta_cols[1]].values
                    ax.scatter(
                        sizes, delta1, delta2, c="blue", marker="o", s=50, alpha=0.6
                    )

                ax.set_xlabel("Diameter (nm)", fontsize=10)
                ax.set_ylabel(delta_cols[0], fontsize=10)
                ax.set_zlabel(delta_cols[1], fontsize=10)
                ax.set_title(
                    "Simulation Results - All Iterations",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                # Use 2D subplot for text message
                ax.remove()
                ax = self.results_fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Summary data available but insufficient columns for 3D plot.",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        elif self.simulation_results["data_res"] is not None:
            # Fallback to data_res if summary is not available
            data_res = self.simulation_results["data_res"]

            # Extract columns for 3D plot
            size_col = None
            delta_cols = []

            for col in data_res.columns:
                if "diameter" in col.lower() and (
                    "true" in col.lower() or "Estimated" in col
                ):
                    size_col = col
                elif col.startswith("Estimated") and "d-" in col:
                    delta_cols.append(col)

            if size_col and len(delta_cols) >= 2:
                sizes = data_res[size_col].values
                delta1 = data_res[delta_cols[0]].values
                delta2 = data_res[delta_cols[1]].values

                ax.scatter(sizes, delta1, delta2, c="blue", marker="o", s=50, alpha=0.6)
                ax.set_xlabel("Diameter (nm)", fontsize=10)
                ax.set_ylabel(delta_cols[0], fontsize=10)
                ax.set_zlabel(delta_cols[1], fontsize=10)
                ax.set_title("Simulation Results", fontsize=12, fontweight="bold")
            else:
                # Use 2D subplot for text message
                ax.remove()
                ax = self.results_fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Results data available but insufficient columns for 3D plot.",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Use 2D subplot for text message
            ax.remove()
            ax = self.results_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No results available yet.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.results_fig.tight_layout()
        self.results_canvas.draw()

    def update_simulations_graph(self):
        """Update the all simulations graph."""
        self.simulations_fig.clear()
        ax = self.simulations_fig.add_subplot(111)

        if self.simulation_results["summary"] is not None:
            summary = self.simulation_results["summary"]

            # Plot all simulations - example: measured diameter vs iterations
            if (
                "Measured diameter (nm)" in summary.columns
                and "Inner Iteration" in summary.columns
            ):
                iterations = summary["Inner Iteration"].values
                diameters = summary["Measured diameter (nm)"].values

                ax.scatter(iterations, diameters, alpha=0.5, s=20, c="green")
                ax.set_xlabel("Inner Iteration", fontsize=12)
                ax.set_ylabel("Measured Diameter (nm)", fontsize=12)
                ax.set_title(
                    "All Simulations - Diameter vs Iterations",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Summary data available but insufficient columns for plotting.",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No simulation data available yet.",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.simulations_fig.tight_layout()
        self.simulations_canvas.draw()

    def create_file_section(self):
        """Create file selection section."""
        file_frame = ctk.CTkFrame(self.config_scroll)
        file_frame.pack(fill="x", pady=10)

        file_label = ctk.CTkLabel(
            file_frame, text="File Selection", font=ctk.CTkFont(size=18, weight="bold")
        )
        file_label.pack(pady=10)

        # Image files
        image_frame = ctk.CTkFrame(file_frame)
        image_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(image_frame, text="Image Files:").pack(side="left", padx=5)
        self.image_files_label = ctk.CTkLabel(
            image_frame, text="No files selected", text_color="gray"
        )
        self.image_files_label.pack(side="left", padx=5, expand=True)

        image_btn = ctk.CTkButton(
            image_frame, text="Browse", command=self.select_image_files, width=100
        )
        image_btn.pack(side="right", padx=5)

        # Excel data file
        data_frame = ctk.CTkFrame(file_frame)
        data_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(data_frame, text="Excel Data File:").pack(side="left", padx=5)
        self.data_file_label = ctk.CTkLabel(
            data_frame, text="No file selected", text_color="gray"
        )
        self.data_file_label.pack(side="left", padx=5, expand=True)

        data_btn = ctk.CTkButton(
            data_frame, text="Browse", command=self.select_data_file, width=100
        )
        data_btn.pack(side="right", padx=5)

    def create_grain_parameters_section(self):
        """Create grain parameters section."""
        grain_frame = ctk.CTkFrame(self.config_scroll)
        grain_frame.pack(fill="x", pady=10)

        grain_label = ctk.CTkLabel(
            grain_frame,
            text="Grain Parameters",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        grain_label.pack(pady=10)

        # Grid layout for grain parameters
        params = [
            ("Nb_PG", "Number of Presolar Grains:", 9, 1, 100),
            ("nb_closest_match", "Number of Closest Matches:", 3, 1, 50),
            ("beam", "Beam Size:", 100, 1, 1000),
            ("boxcar", "Boxcar Pixels:", 3, 1, 20),
            ("elem", "Element:", "O", None, None),
        ]

        self.grain_vars = {}
        for i, (key, label, default, min_val, max_val) in enumerate(params):
            row_frame = ctk.CTkFrame(grain_frame)
            row_frame.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(row_frame, text=label, width=200).pack(side="left", padx=5)

            if key == "elem":
                var = ctk.CTkEntry(row_frame, width=100)
                var.insert(0, str(default))
                var.pack(side="left", padx=5)
            else:
                var = ctk.CTkEntry(row_frame, width=100)
                var.insert(0, str(default))
                var.pack(side="left", padx=5)

            self.grain_vars[key] = var

    def create_delta_database_section(self):
        """Create delta database configuration section."""
        delta_frame = ctk.CTkFrame(self.config_scroll)
        delta_frame.pack(fill="x", pady=10)

        delta_label = ctk.CTkLabel(
            delta_frame,
            text="Delta Database Ranges",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        delta_label.pack(pady=10)

        # Create sub-frames for each range
        ranges_info = [
            ("Range 1", "start1", "stop1", "step1", -900, 0, 50),
            ("Range 2", "start2", "stop2", "step2", 0, 200, 10),
            ("Range 3", "start3", "stop3", "step3", 200, 2000, 100),
            ("Range 4", "start4", "stop4", "step4", 2000, 21000, 1000),
        ]

        self.delta_vars = {}
        for i, (
            range_name,
            start_key,
            stop_key,
            step_key,
            start_def,
            stop_def,
            step_def,
        ) in enumerate(ranges_info, 1):
            range_frame = ctk.CTkFrame(delta_frame)
            range_frame.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(range_frame, text=f"{range_name}:").pack(side="left", padx=5)

            ctk.CTkLabel(range_frame, text="Start:").pack(side="left", padx=5)
            start_var = ctk.CTkEntry(range_frame, width=80)
            start_var.insert(0, str(start_def))
            start_var.pack(side="left", padx=2)
            self.delta_vars[start_key] = start_var

            ctk.CTkLabel(range_frame, text="Stop:").pack(side="left", padx=5)
            stop_var = ctk.CTkEntry(range_frame, width=80)
            stop_var.insert(0, str(stop_def))
            stop_var.pack(side="left", padx=2)
            self.delta_vars[stop_key] = stop_var

            ctk.CTkLabel(range_frame, text="Step:").pack(side="left", padx=5)
            step_var = ctk.CTkEntry(range_frame, width=80)
            step_var.insert(0, str(step_def))
            step_var.pack(side="left", padx=2)
            self.delta_vars[step_key] = step_var

    def create_iteration_section(self):
        """Create iteration parameters section."""
        iter_frame = ctk.CTkFrame(self.config_scroll)
        iter_frame.pack(fill="x", pady=10)

        iter_label = ctk.CTkLabel(
            iter_frame,
            text="Iteration Parameters",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        iter_label.pack(pady=10)

        params = [
            ("iterations", "Outer Iterations:", 1, 1, 100),
            ("max_iteration", "Max Inner Iterations:", 10, 1, 1000),
            ("cost_goal", "Cost Goal:", 0.4, 0.01, 10.0),
        ]

        self.iter_vars = {}
        for key, label, default, min_val, max_val in params:
            row_frame = ctk.CTkFrame(iter_frame)
            row_frame.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(row_frame, text=label, width=200).pack(side="left", padx=5)

            var = ctk.CTkEntry(row_frame, width=100)
            var.insert(0, str(default))
            var.pack(side="left", padx=5)

            self.iter_vars[key] = var

    def create_size_range_section(self):
        """Create size range section."""
        size_frame = ctk.CTkFrame(self.config_scroll)
        size_frame.pack(fill="x", pady=10)

        size_label = ctk.CTkLabel(
            size_frame,
            text="Size Range Parameters",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        size_label.pack(pady=10)

        range_frame = ctk.CTkFrame(size_frame)
        range_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(range_frame, text="Start:").pack(side="left", padx=5)
        self.size_start_var = ctk.CTkEntry(range_frame, width=100)
        self.size_start_var.insert(0, "50")
        self.size_start_var.pack(side="left", padx=5)

        ctk.CTkLabel(range_frame, text="Stop:").pack(side="left", padx=5)
        self.size_stop_var = ctk.CTkEntry(range_frame, width=100)
        self.size_stop_var.insert(0, "900")
        self.size_stop_var.pack(side="left", padx=5)

        ctk.CTkLabel(range_frame, text="Step:").pack(side="left", padx=5)
        self.size_step_var = ctk.CTkEntry(range_frame, width=100)
        self.size_step_var.insert(0, "50")
        self.size_step_var.pack(side="left", padx=5)

    def create_output_section(self):
        """Create output settings section."""
        output_frame = ctk.CTkFrame(self.config_scroll)
        output_frame.pack(fill="x", pady=10)

        output_label = ctk.CTkLabel(
            output_frame,
            text="Output Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        output_label.pack(pady=10)

        # Output name
        name_frame = ctk.CTkFrame(output_frame)
        name_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(name_frame, text="Results Name:", width=200).pack(
            side="left", padx=5
        )
        self.output_name_var = ctk.CTkEntry(name_frame, width=200)
        self.output_name_var.insert(0, "test_alldata")
        self.output_name_var.pack(side="left", padx=5)

        # Checkboxes
        checkbox_frame = ctk.CTkFrame(output_frame)
        checkbox_frame.pack(fill="x", padx=10, pady=5)

        self.save_all_plots_var = ctk.CTkCheckBox(checkbox_frame, text="Save All Plots")
        self.save_all_plots_var.pack(side="left", padx=10)

        self.enable_profiling_var = ctk.CTkCheckBox(
            checkbox_frame, text="Enable Profiling"
        )
        self.enable_profiling_var.pack(side="left", padx=10)

        # Profiling output directory
        prof_frame = ctk.CTkFrame(output_frame)
        prof_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(prof_frame, text="Profiling Output Dir:", width=200).pack(
            side="left", padx=5
        )
        self.profiling_dir_var = ctk.CTkEntry(prof_frame, width=200)
        self.profiling_dir_var.insert(0, "profiling_results")
        self.profiling_dir_var.pack(side="left", padx=5)

    def create_control_buttons(self):
        """Create control buttons."""
        button_frame = ctk.CTkFrame(self.config_scroll)
        button_frame.pack(fill="x", pady=20)

        # Status label
        self.status_label = ctk.CTkLabel(button_frame, text="Ready", text_color="green")
        self.status_label.pack(pady=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(button_frame, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)  # Initialize to 0
        self.progress_bar.pack_forget()  # Hide initially

        # Buttons
        btn_frame = ctk.CTkFrame(button_frame)
        btn_frame.pack(pady=10)

        run_btn = ctk.CTkButton(
            btn_frame,
            text="Run Simulation",
            command=self.run_simulation,
            width=200,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        run_btn.pack(side="left", padx=10)

        reset_btn = ctk.CTkButton(
            btn_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults,
            width=150,
            height=40,
        )
        reset_btn.pack(side="left", padx=10)

    def select_image_files(self):
        """Open file dialog to select image files."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilename(multiple=True)
        root.destroy()

        if files:
            # Convert to list for easier handling
            self.file_list = list(files) if isinstance(files, tuple) else [files]
            num_files = len(self.file_list)
            self.image_files_label.configure(
                text=f"{num_files} file(s) selected", text_color="white"
            )

    def select_data_file(self):
        """Open file dialog to select Excel data file."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        root.destroy()

        if file:
            self.data_file = file
            filename = os.path.basename(file)
            self.data_file_label.configure(text=filename, text_color="white")

    def reset_to_defaults(self):
        """Reset all fields to default values."""
        self.config = create_default_config()

        # Reset grain parameters
        self.grain_vars["Nb_PG"].delete(0, "end")
        self.grain_vars["Nb_PG"].insert(0, "9")
        self.grain_vars["nb_closest_match"].delete(0, "end")
        self.grain_vars["nb_closest_match"].insert(0, "3")
        self.grain_vars["beam"].delete(0, "end")
        self.grain_vars["beam"].insert(0, "100")
        self.grain_vars["boxcar"].delete(0, "end")
        self.grain_vars["boxcar"].insert(0, "3")
        self.grain_vars["elem"].delete(0, "end")
        self.grain_vars["elem"].insert(0, "O")

        # Reset delta database
        defaults = [
            ("start1", "-900"),
            ("stop1", "0"),
            ("step1", "50"),
            ("start2", "0"),
            ("stop2", "200"),
            ("step2", "10"),
            ("start3", "200"),
            ("stop3", "2000"),
            ("step3", "100"),
            ("start4", "2000"),
            ("stop4", "21000"),
            ("step4", "1000"),
        ]
        for key, value in defaults:
            self.delta_vars[key].delete(0, "end")
            self.delta_vars[key].insert(0, value)

        # Reset iteration parameters
        self.iter_vars["iterations"].delete(0, "end")
        self.iter_vars["iterations"].insert(0, "1")
        self.iter_vars["max_iteration"].delete(0, "end")
        self.iter_vars["max_iteration"].insert(0, "10")
        self.iter_vars["cost_goal"].delete(0, "end")
        self.iter_vars["cost_goal"].insert(0, "0.4")

        # Reset size range
        self.size_start_var.delete(0, "end")
        self.size_start_var.insert(0, "50")
        self.size_stop_var.delete(0, "end")
        self.size_stop_var.insert(0, "900")
        self.size_step_var.delete(0, "end")
        self.size_step_var.insert(0, "50")

        # Reset output settings
        self.output_name_var.delete(0, "end")
        self.output_name_var.insert(0, "test_alldata")
        self.save_all_plots_var.deselect()
        self.enable_profiling_var.deselect()
        self.profiling_dir_var.delete(0, "end")
        self.profiling_dir_var.insert(0, "profiling_results")

        self.status_label.configure(text="Reset to defaults", text_color="green")

    def validate_inputs(self):
        """Validate all input values."""
        errors = []

        # Validate grain parameters
        try:
            nb_pg = int(self.grain_vars["Nb_PG"].get())
            if nb_pg < 1:
                errors.append("Nb_PG must be >= 1")
        except ValueError:
            errors.append("Nb_PG must be an integer")

        try:
            nb_match = int(self.grain_vars["nb_closest_match"].get())
            nb_pg = int(self.grain_vars["Nb_PG"].get())
            if nb_match > nb_pg:
                errors.append("nb_closest_match cannot be > Nb_PG")
        except ValueError:
            errors.append("nb_closest_match must be an integer")

        # Validate delta ranges
        for i in range(1, 5):
            try:
                start = int(self.delta_vars[f"start{i}"].get())
                stop = int(self.delta_vars[f"stop{i}"].get())
                step = int(self.delta_vars[f"step{i}"].get())
                if step <= 0:
                    errors.append(f"Range {i} step must be > 0")
                if start >= stop:
                    errors.append(f"Range {i} start must be < stop")
            except ValueError:
                errors.append(f"Range {i} values must be integers")

        # Validate size range
        try:
            size_start = int(self.size_start_var.get())
            size_stop = int(self.size_stop_var.get())
            size_step = int(self.size_step_var.get())
            if size_step <= 0:
                errors.append("Size range step must be > 0")
            if size_start >= size_stop:
                errors.append("Size range start must be < stop")
        except ValueError:
            errors.append("Size range values must be integers")

        # Check file selection
        if not self.file_list:
            errors.append("Please select image files")
        if not self.data_file:
            errors.append("Please select Excel data file")

        return errors

    def collect_config(self):
        """Collect configuration from GUI inputs."""
        config = {}

        # Grain parameters
        config["Nb_PG"] = int(self.grain_vars["Nb_PG"].get())
        config["nb_closest_match"] = int(self.grain_vars["nb_closest_match"].get())
        config["beam"] = int(self.grain_vars["beam"].get())
        config["boxcar"] = int(self.grain_vars["boxcar"].get())
        config["elem"] = self.grain_vars["elem"].get()

        # Delta database
        delta_db = []
        for i in range(1, 5):
            start = int(self.delta_vars[f"start{i}"].get())
            stop = int(self.delta_vars[f"stop{i}"].get())
            step = int(self.delta_vars[f"step{i}"].get())
            delta_db.extend(range(start, stop, step))
        config["delta_database"] = delta_db

        # Iteration parameters
        config["iterations"] = int(self.iter_vars["iterations"].get())
        config["max_iteration"] = int(self.iter_vars["max_iteration"].get())
        config["cost_goal"] = float(self.iter_vars["cost_goal"].get())

        # Size range
        size_start = int(self.size_start_var.get())
        size_stop = int(self.size_stop_var.get())
        size_step = int(self.size_step_var.get())
        config["size_range"] = range(size_start, size_stop, size_step)

        # Output settings
        config["Name_results"] = self.output_name_var.get()
        config["SAVE_ALL_PLOTS"] = self.save_all_plots_var.get()
        config["ENABLE_PROFILING"] = self.enable_profiling_var.get()
        config["PROFILING_OUTPUT_DIR"] = self.profiling_dir_var.get()

        return config

    def run_simulation(self):
        """Run the simulation with current configuration."""
        # Validate inputs
        errors = self.validate_inputs()
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(f"• {e}" for e in errors)
            self.status_label.configure(text="Validation failed", text_color="red")
            # Show error dialog
            error_window = ctk.CTkToplevel(self)
            error_window.title("Validation Error")
            error_window.geometry("400x300")
            error_text = ctk.CTkTextbox(error_window, width=380, height=250)
            error_text.pack(padx=10, pady=10)
            error_text.insert("1.0", error_msg)
            error_text.configure(state="disabled")
            close_btn = ctk.CTkButton(
                error_window, text="Close", command=error_window.destroy
            )
            close_btn.pack(pady=10)
            return

        # Collect configuration
        try:
            config = self.collect_config()
        except Exception as e:
            self.status_label.configure(
                text=f"Config error: {str(e)}", text_color="red"
            )
            return

        # Update status
        self.status_label.configure(text="Running simulation...", text_color="yellow")

        # Show and reset progress bar
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # Store reference to run button for easier state management
        self.run_button_state = "disabled"
        self.disable_buttons()

        # Run simulation in a separate thread to avoid freezing GUI
        import threading
        from Modules.pg_simulation_data import calculate_total_grains

        def run_in_thread():
            try:
                # Load data file
                data = pd.read_excel(self.data_file, header=0)
                data_filtered = data[~data.NAME.str.contains("Bulk")]

                # Calculate total grains for progress tracking
                total_grains = calculate_total_grains(self.file_list, data_filtered)
                current_progress = [
                    0
                ]  # Use list to allow modification in nested function

                # Progress callback function
                def update_progress():
                    current_progress[0] += 1
                    progress_value = (
                        current_progress[0] / total_grains if total_grains > 0 else 0
                    )
                    self.after(0, lambda: self.progress_bar.set(progress_value))
                    self.after(
                        0,
                        lambda: self.status_label.configure(
                            text=f"Processing... {current_progress[0]}/{total_grains} grains",
                            text_color="yellow",
                        ),
                    )

                # Process all grains with progress callback
                summary, match_summary, data_res, all_simulations, f_OG = (
                    process_all_grains(
                        file_list=self.file_list,
                        data=data_filtered,
                        config=config,
                        use_gui=False,
                        progress_callback=update_progress,
                    )
                )

                # Store results for plotting
                self.simulation_results["summary"] = summary
                self.simulation_results["match_summary"] = match_summary
                self.simulation_results["data_res"] = data_res
                self.simulation_results["all_simulations"] = all_simulations
                self.simulation_results["original_data"] = data_filtered
                self.simulation_results["f_OG"] = f_OG

                # Update status on main thread
                self.after(
                    0, lambda: self.progress_bar.set(1.0)
                )  # Complete progress bar
                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text="Simulation completed successfully!", text_color="green"
                    ),
                )

                # Update graphs
                self.after(0, self.update_graphs)

                # Re-enable buttons
                self.after(0, self.enable_buttons)

                # Hide progress bar after a short delay
                self.after(2000, lambda: self.progress_bar.pack_forget())

            except Exception as e:
                error_msg = f"Simulation error: {str(e)}"
                self.after(0, lambda: self.progress_bar.set(0))  # Reset progress bar
                self.after(
                    0, lambda: self.progress_bar.pack_forget()
                )  # Hide progress bar
                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text=error_msg, text_color="red"
                    ),
                )
                self.after(0, self.enable_buttons)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

    def disable_buttons(self):
        """Disable all buttons during simulation."""

        # Find and disable all buttons in the main frame
        def disable_widget(widget):
            if isinstance(widget, ctk.CTkButton):
                widget.configure(state="disabled")
            else:
                for child in widget.winfo_children():
                    disable_widget(child)

        disable_widget(self.config_scroll)

    def enable_buttons(self):
        """Re-enable all buttons after simulation."""

        # Find and enable all buttons in the main frame
        def enable_widget(widget):
            if isinstance(widget, ctk.CTkButton):
                widget.configure(state="normal")
            else:
                for child in widget.winfo_children():
                    enable_widget(child)

        enable_widget(self.config_scroll)


def main():
    """Main entry point for GUI application."""
    app = PGSimulationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

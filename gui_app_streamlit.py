# -*- coding: utf-8 -*-
"""
Streamlit GUI Application for PG NanoSIMS Simulations v3.0

A Streamlit-based web GUI for configuring and running PG NanoSIMS simulations.

To run this application:
    streamlit run gui_app_streamlit.py

@author: Maximilien Verdier-Paoletti
"""

import streamlit as st
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for Streamlit
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import shutil
import time
import threading

from Modules.pg_simulation_config import create_default_config
from Modules.pg_simulation_core import process_all_grains
from Modules.pg_simulation_data import calculate_total_grains

# Page configuration
st.set_page_config(
    page_title="PG NanoSIMS Simulations v3.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "config" not in st.session_state:
    st.session_state.config = create_default_config()
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {
        "summary": None,
        "match_summary": None,
        "data_res": None,
        "all_simulations": None,
        "original_data": None,
        "f_OG": None,
    }
if "file_list" not in st.session_state:
    st.session_state.file_list = None
if "data_file" not in st.session_state:
    st.session_state.data_file = None
if "uploaded_image_files" not in st.session_state:
    st.session_state.uploaded_image_files = []
if "uploaded_data_file" not in st.session_state:
    st.session_state.uploaded_data_file = None


def reset_to_defaults():
    """Reset all configuration to default values."""
    st.session_state.config = create_default_config()
    st.session_state.simulation_results = {
        "summary": None,
        "match_summary": None,
        "data_res": None,
        "all_simulations": None,
        "original_data": None,
        "f_OG": None,
    }
    st.session_state.file_list = None
    st.session_state.data_file = None
    st.session_state.uploaded_image_files = []
    st.session_state.uploaded_data_file = None
    st.rerun()


def collect_config_from_ui():
    """Collect configuration from Streamlit UI inputs."""
    config = {}

    # Grain parameters
    config["Nb_PG"] = st.session_state.get("nb_pg", 9)
    config["nb_closest_match"] = st.session_state.get("nb_closest_match", 3)
    config["beam"] = st.session_state.get("beam", 100)
    config["boxcar"] = st.session_state.get("boxcar", 3)
    config["elem"] = st.session_state.get("elem", "O")

    # Delta database
    delta_db = []
    for i in range(1, 5):
        start = st.session_state.get(f"delta_start{i}", [-900, 0, 200, 2000][i - 1])
        stop = st.session_state.get(f"delta_stop{i}", [0, 200, 2000, 21000][i - 1])
        step = st.session_state.get(f"delta_step{i}", [50, 10, 100, 1000][i - 1])
        delta_db.extend(range(start, stop, step))
    config["delta_database"] = delta_db

    # Iteration parameters
    config["iterations"] = st.session_state.get("iterations", 1)
    config["max_iteration"] = st.session_state.get("max_iteration", 10)
    config["cost_goal"] = st.session_state.get("cost_goal", 0.4)

    # Size range
    size_start = st.session_state.get("size_start", 50)
    size_stop = st.session_state.get("size_stop", 900)
    size_step = st.session_state.get("size_step", 50)
    config["size_range"] = range(size_start, size_stop, size_step)

    # Output settings
    config["Name_results"] = st.session_state.get("output_name", "test_alldata")
    config["SAVE_ALL_PLOTS"] = st.session_state.get("save_all_plots", False)
    config["ENABLE_PROFILING"] = st.session_state.get("enable_profiling", False)
    config["PROFILING_OUTPUT_DIR"] = st.session_state.get(
        "profiling_dir", "profiling_results"
    )

    return config


def validate_inputs():
    """Validate all input values."""
    errors = []

    # Validate grain parameters
    try:
        nb_pg = st.session_state.get("nb_pg", 9)
        if nb_pg < 1:
            errors.append("Nb_PG must be >= 1")
    except (ValueError, TypeError):
        errors.append("Nb_PG must be an integer")

    try:
        nb_match = st.session_state.get("nb_closest_match", 3)
        nb_pg = st.session_state.get("nb_pg", 9)
        if nb_match > nb_pg:
            errors.append("nb_closest_match cannot be > Nb_PG")
    except (ValueError, TypeError):
        errors.append("nb_closest_match must be an integer")

    # Validate delta ranges
    for i in range(1, 5):
        try:
            start = st.session_state.get(f"delta_start{i}", 0)
            stop = st.session_state.get(f"delta_stop{i}", 100)
            step = st.session_state.get(f"delta_step{i}", 10)
            if step <= 0:
                errors.append(f"Range {i} step must be > 0")
            if start >= stop:
                errors.append(f"Range {i} start must be < stop")
        except (ValueError, TypeError):
            errors.append(f"Range {i} values must be integers")

    # Validate size range
    try:
        size_start = st.session_state.get("size_start", 50)
        size_stop = st.session_state.get("size_stop", 900)
        size_step = st.session_state.get("size_step", 50)
        if size_step <= 0:
            errors.append("Size range step must be > 0")
        if size_start >= size_stop:
            errors.append("Size range start must be < stop")
    except (ValueError, TypeError):
        errors.append("Size range values must be integers")

    # Check file selection
    if not st.session_state.uploaded_image_files:
        errors.append("Please upload image files")
    if st.session_state.uploaded_data_file is None:
        errors.append("Please upload Excel data file")

    return errors


def create_configuration_sidebar():
    """Create configuration sidebar."""
    # File Selection Section
    with st.sidebar.expander("Files", expanded=True):
        uploaded_images = st.file_uploader(
            "Image Files",
            type=None,
            accept_multiple_files=True,
            key="image_files_uploader",
        )
        if uploaded_images:
            st.session_state.uploaded_image_files = uploaded_images
            st.caption(f"✓ {len(uploaded_images)} file(s)")
        else:
            st.session_state.uploaded_image_files = []

        uploaded_data = st.file_uploader(
            "Excel Data",
            type=["xlsx", "xls"],
            key="data_file_uploader",
        )
        if uploaded_data:
            st.session_state.uploaded_data_file = uploaded_data
            st.caption(f"✓ {uploaded_data.name}")
        else:
            st.session_state.uploaded_data_file = None

    # Grain Parameters Section
    with st.sidebar.expander("Grain Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.nb_pg = st.number_input(
                "Nb PG",
                min_value=1,
                max_value=100,
                value=9,
                step=1,
                key="nb_pg_input",
            )
        with col2:
            st.session_state.nb_closest_match = st.number_input(
                "Closest Match",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                key="nb_closest_match_input",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.beam = st.number_input(
                "Beam",
                min_value=1,
                max_value=1000,
                value=100,
                step=1,
                key="beam_input",
            )
        with col2:
            st.session_state.boxcar = st.number_input(
                "Boxcar",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="boxcar_input",
            )

        st.session_state.elem = st.text_input(
            "Element",
            value="O",
            key="elem_input",
        )

    # Delta Database Section
    with st.sidebar.expander("Delta Ranges", expanded=False):
        delta_defaults = [
            ("R1", -900, 0, 50),
            ("R2", 0, 200, 10),
            ("R3", 200, 2000, 100),
            ("R4", 2000, 21000, 1000),
        ]

        for i, (range_name, start_def, stop_def, step_def) in enumerate(
            delta_defaults, 1
        ):
            st.markdown(f"**{range_name}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state[f"delta_start{i}"] = st.number_input(
                    "Start",
                    value=start_def,
                    key=f"delta_start{i}_input",
                    label_visibility="collapsed",
                )
            with col2:
                st.session_state[f"delta_stop{i}"] = st.number_input(
                    "Stop",
                    value=stop_def,
                    key=f"delta_stop{i}_input",
                    label_visibility="collapsed",
                )
            with col3:
                st.session_state[f"delta_step{i}"] = st.number_input(
                    "Step",
                    value=step_def,
                    min_value=1,
                    key=f"delta_step{i}_input",
                    label_visibility="collapsed",
                )

    # Iteration Parameters Section
    with st.sidebar.expander("Iterations", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.iterations = st.number_input(
                "Outer",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="iterations_input",
            )
        with col2:
            st.session_state.max_iteration = st.number_input(
                "Max Inner",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
                key="max_iteration_input",
            )

        st.session_state.cost_goal = st.number_input(
            "Cost Goal",
            min_value=0.01,
            max_value=10.0,
            value=0.4,
            step=0.01,
            format="%.2f",
            key="cost_goal_input",
        )

    # Size Range Section
    with st.sidebar.expander("Size Range", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.size_start = st.number_input(
                "Start",
                value=50,
                key="size_start_input",
            )
        with col2:
            st.session_state.size_stop = st.number_input(
                "Stop",
                value=900,
                key="size_stop_input",
            )
        with col3:
            st.session_state.size_step = st.number_input(
                "Step",
                value=50,
                min_value=1,
                key="size_step_input",
            )

    # Output Settings Section - Use expander for less frequently used options
    with st.sidebar.expander("Output Settings", expanded=False):
        st.session_state.output_name = st.text_input(
            "Results Name",
            value="test_alldata",
            key="output_name_input",
        )

        st.session_state.save_all_plots = st.checkbox(
            "Save All Plots",
            value=False,
            key="save_all_plots_input",
        )

        st.session_state.enable_profiling = st.checkbox(
            "Enable Profiling",
            value=False,
            key="enable_profiling_input",
        )

        st.session_state.profiling_dir = st.text_input(
            "Profiling Dir",
            value="profiling_results",
            key="profiling_dir_input",
        )

    # Control Buttons
    if st.sidebar.button("Reset to Defaults", use_container_width=True):
        reset_to_defaults()
        st.sidebar.success("Reset!")

    return st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True,
        key="run_simulation_button",
    )


def update_original_data_graph():
    """Update the original data graph with f_OG figure."""
    if st.session_state.simulation_results["f_OG"] is not None:
        f_OG = st.session_state.simulation_results["f_OG"]
        st.pyplot(f_OG)
    else:
        st.info(
            "No original grain figure (f_OG) available yet. Run simulation to see original grain data."
        )


def update_results_graph():
    """Update the results graph with iteration results."""
    summary = st.session_state.simulation_results["summary"]
    data_res = st.session_state.simulation_results["data_res"]
    original_data = st.session_state.simulation_results["original_data"]

    if summary is not None:
        # Extract columns for 3D plot
        size_col = "Measured diameter (nm)"
        delta_cols = []

        # Find delta columns
        for col in summary.columns:
            if col.startswith("Measured") and "d-" in col:
                delta_cols.append(col)

        if size_col in summary.columns and len(delta_cols) >= 2:
            # Create 3D scatter plot with Plotly
            fig = go.Figure()

            # Plot target values as black squares if original_data is available
            if original_data is not None and "Grain" in summary.columns:
                elem = st.session_state.get("elem", "O")
                orig_delta_cols = [
                    col
                    for col in original_data.columns
                    if col.startswith("d-") and elem in col
                ]

                unique_grains = summary["Grain"].unique()

                for grain_name in unique_grains:
                    grain_data = original_data[original_data["NAME"] == grain_name]
                    if grain_data.empty:
                        grain_data = original_data[
                            original_data["NAME"].str.contains(grain_name, na=False)
                        ]
                    if not grain_data.empty:
                        grain_row = grain_data.iloc[0]
                        if "ROIDIAM" in grain_row.index:
                            target_size = grain_row["ROIDIAM"] * 1000  # Convert to nm
                            if len(orig_delta_cols) >= 2:
                                target_delta1 = grain_row[orig_delta_cols[0]]
                                target_delta2 = grain_row[orig_delta_cols[1]]

                                fig.add_trace(
                                    go.Scatter3d(
                                        x=[target_size],
                                        y=[target_delta1],
                                        z=[target_delta2],
                                        mode="markers",
                                        marker=dict(
                                            size=10,
                                            color="black",
                                            symbol="square",
                                        ),
                                        name="Target",
                                        showlegend=True,
                                    )
                                )

            # Color by simulated grain index
            if "Simulated grain index" in summary.columns:
                unique_grain_indices = sorted(summary["Simulated grain index"].unique())

                for grain_idx, sim_grain_idx in enumerate(unique_grain_indices):
                    grain_data = summary[
                        summary["Simulated grain index"] == sim_grain_idx
                    ]

                    sizes = grain_data[size_col].values
                    delta1 = grain_data[delta_cols[0]].values
                    delta2 = grain_data[delta_cols[1]].values

                    # Create color based on index
                    color = px.colors.qualitative.Set3[
                        grain_idx % len(px.colors.qualitative.Set3)
                    ]

                    fig.add_trace(
                        go.Scatter3d(
                            x=sizes,
                            y=delta1,
                            z=delta2,
                            mode="markers",
                            marker=dict(size=5, color=color, opacity=0.6),
                            name=f"Grain {sim_grain_idx}",
                            showlegend=True,
                        )
                    )
            else:
                # Fallback: plot all points
                sizes = summary[size_col].values
                delta1 = summary[delta_cols[0]].values
                delta2 = summary[delta_cols[1]].values

                fig.add_trace(
                    go.Scatter3d(
                        x=sizes,
                        y=delta1,
                        z=delta2,
                        mode="markers",
                        marker=dict(size=5, color="blue", opacity=0.6),
                        name="All Simulations",
                    )
                )

            fig.update_layout(
                scene=dict(
                    xaxis_title="Diameter (nm)",
                    yaxis_title=delta_cols[0],
                    zaxis_title=delta_cols[1],
                ),
                title="Simulation Results - All Iterations",
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Summary data available but insufficient columns for 3D plot.")
    elif data_res is not None:
        # Fallback to data_res
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

            fig = go.Figure(
                data=go.Scatter3d(
                    x=sizes,
                    y=delta1,
                    z=delta2,
                    mode="markers",
                    marker=dict(size=5, color="blue", opacity=0.6),
                )
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="Diameter (nm)",
                    yaxis_title=delta_cols[0],
                    zaxis_title=delta_cols[1],
                ),
                title="Simulation Results",
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Results data available but insufficient columns for 3D plot.")
    else:
        st.info("No results available yet. Run simulation to see results.")


def update_simulations_graph():
    """Update the all simulations graph."""
    summary = st.session_state.simulation_results["summary"]

    if summary is not None:
        if (
            "Measured diameter (nm)" in summary.columns
            and "Inner Iteration" in summary.columns
        ):
            iterations = summary["Inner Iteration"].values
            diameters = summary["Measured diameter (nm)"].values

            fig = go.Figure(
                data=go.Scatter(
                    x=iterations,
                    y=diameters,
                    mode="markers",
                    marker=dict(size=5, color="green", opacity=0.5),
                )
            )

            fig.update_layout(
                xaxis_title="Inner Iteration",
                yaxis_title="Measured Diameter (nm)",
                title="All Simulations - Diameter vs Iterations",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Summary data available but insufficient columns for plotting.")
    else:
        st.info(
            "No simulation data available yet. Run simulation to see all simulations."
        )


def run_simulation():
    """Run the simulation with current configuration."""
    # Validate inputs
    errors = validate_inputs()
    if errors:
        error_msg = "Validation errors:\n" + "\n".join(f"• {e}" for e in errors)
        st.error(error_msg)
        return

    # Collect configuration
    try:
        config = collect_config_from_ui()
    except Exception as e:
        st.error(f"Config error: {str(e)}")
        return

    # Prepare file list from uploaded files
    if st.session_state.uploaded_image_files:
        # Save uploaded files to temporary directory
        temp_dir = tempfile.mkdtemp()
        file_list = []
        try:
            for uploaded_file in st.session_state.uploaded_image_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_list.append(temp_path)  # Keep actual path for file operations
            st.session_state.file_list = file_list
            st.session_state.temp_dir = temp_dir  # Store for cleanup
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            st.error(f"Error saving uploaded files: {str(e)}")
            return
    else:
        st.error("Please upload image files")
        return

    # Load data file
    if st.session_state.uploaded_data_file:
        try:
            data = pd.read_excel(st.session_state.uploaded_data_file, header=0)
            data_filtered = data[~data.NAME.str.contains("Bulk")]
            st.session_state.data_file = data_filtered
        except Exception as e:
            st.error(f"Error loading data file: {str(e)}")
            return
    else:
        st.error("Please upload Excel data file")
        return

    # Calculate total grains for progress tracking
    try:
        # Normalize paths for calculate_total_grains (same normalization as for process_all_grains)
        normalized_file_list = [
            f.replace("\\", "/") for f in st.session_state.file_list
        ]
        total_grains = calculate_total_grains(
            normalized_file_list, st.session_state.data_file
        )
    except Exception as e:
        st.warning(f"Could not calculate total grains: {str(e)}")
        total_grains = 1

    # Calculate total work units (accounting for iterations)
    outer_iterations = config.get("iterations", 1)
    max_inner_iterations = config.get("max_iteration", 10)
    # Estimate average inner iterations (assume 70% of max, but can stop early)
    estimated_avg_inner = int(max_inner_iterations * 0.7)
    total_work_units = total_grains * outer_iterations * estimated_avg_inner

    # Progress tracking - create a prominent progress section
    progress_container = st.container()
    with progress_container:
        st.markdown("### Simulation Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_details = st.empty()
        percentage_text = st.empty()

    current_grain = [0]
    current_work_units = [0]
    start_time = time.time()
    grain_start_time = [time.time()]
    is_running = [True]
    work_units_per_grain = outer_iterations * estimated_avg_inner

    def update_progress_display():
        """Update progress display with current estimates."""
        # Calculate work units completed
        completed_work_units = current_work_units[0]

        # Estimate progress within current grain based on elapsed time
        if current_grain[0] < total_grains:
            grain_elapsed = time.time() - grain_start_time[0]
            # Estimate time per grain based on previous grains
            if current_grain[0] > 0:
                avg_time_per_grain = (time.time() - start_time) / current_grain[0]
                # Estimate progress within current grain (0 to 1)
                grain_progress = (
                    min(grain_elapsed / avg_time_per_grain, 0.95)
                    if avg_time_per_grain > 0
                    else 0
                )
            else:
                grain_progress = 0

            # Add estimated progress within current grain
            estimated_current_grain_work = work_units_per_grain * grain_progress
            total_estimated_work = completed_work_units + estimated_current_grain_work
        else:
            total_estimated_work = completed_work_units

        # Calculate progress
        progress_value = (
            total_estimated_work / total_work_units if total_work_units > 0 else 0
        )
        progress_value = min(progress_value, 1.0)  # Cap at 100%
        percentage = int(progress_value * 100)

        # Update progress bar
        progress_bar.progress(progress_value)

        # Update status text
        if current_grain[0] < total_grains:
            status_text.markdown(
                f"**Status:** Processing grain {current_grain[0] + 1} of {total_grains} "
                f"(~{outer_iterations} outer × ~{estimated_avg_inner} inner iterations per grain)"
            )
        else:
            status_text.markdown(
                f"**Status:** Completed {current_grain[0]} of {total_grains} grains"
            )

        # Update percentage
        percentage_text.markdown(f"**Progress:** {percentage}%")

        # Calculate and display time information
        elapsed_time = time.time() - start_time
        if current_grain[0] > 0:
            avg_time_per_grain = elapsed_time / current_grain[0]
            remaining_grains = total_grains - current_grain[0]
            estimated_remaining = avg_time_per_grain * remaining_grains

            elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
            remaining_str = (
                f"{int(estimated_remaining // 60)}m {int(estimated_remaining % 60)}s"
            )

            # Calculate iterations info
            estimated_completed_iterations = int(total_estimated_work)
            total_iterations = total_work_units

            progress_details.markdown(
                f"⏱️ **Elapsed:** {elapsed_str} | **Remaining:** {remaining_str}<br>"
                f"🔄 **Iterations:** ~{estimated_completed_iterations}/{total_iterations} "
                f"({outer_iterations} outer × ~{estimated_avg_inner} inner per grain)",
                unsafe_allow_html=True,
            )

    def update_progress():
        # This is called once per grain completion
        current_grain[0] += 1

        # Calculate work units completed
        current_work_units[0] += work_units_per_grain

        # Update display
        update_progress_display()

        # Reset grain start time for next grain
        if current_grain[0] < total_grains:
            grain_start_time[0] = time.time()

    # Start periodic progress updates
    def periodic_update():
        while is_running[0] and current_grain[0] < total_grains:
            time.sleep(1)  # Update every second
            if is_running[0]:
                update_progress_display()

    progress_thread = threading.Thread(target=periodic_update, daemon=True)
    progress_thread.start()

    # Run simulation
    try:
        with progress_container:
            status_text.markdown(
                f"**Status:** Initializing simulation... "
                f"({total_grains} grains, {outer_iterations} outer × ~{estimated_avg_inner} inner iterations)"
            )
            percentage_text.markdown("**Progress:** 0%")
            progress_details.markdown(
                f"⏱️ Starting simulation... Total work: ~{total_work_units} iterations"
            )

        # Normalize file paths to use forward slashes for cross-platform compatibility
        # Python file operations on Windows can handle forward slashes, so this is safe
        normalized_file_list = [
            f.replace("\\", "/") for f in st.session_state.file_list
        ]

        summary, match_summary, data_res, all_simulations, f_OG = process_all_grains(
            file_list=normalized_file_list,
            data=st.session_state.data_file,
            config=config,
            use_gui=False,
            progress_callback=update_progress,
        )

        # Store results
        st.session_state.simulation_results["summary"] = summary
        st.session_state.simulation_results["match_summary"] = match_summary
        st.session_state.simulation_results["data_res"] = data_res
        st.session_state.simulation_results["all_simulations"] = all_simulations
        st.session_state.simulation_results["original_data"] = (
            st.session_state.data_file
        )
        st.session_state.simulation_results["f_OG"] = f_OG

        # Stop periodic updates
        is_running[0] = False

        # Complete progress
        with progress_container:
            progress_bar.progress(1.0)
            elapsed_time = time.time() - start_time
            elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
            status_text.success("✅ **Simulation completed successfully!**")
            percentage_text.markdown("**Progress:** 100%")
            progress_details.markdown(
                f"⏱️ **Total time:** {elapsed_str}<br>"
                f"🔄 **Total iterations:** ~{current_work_units[0]}/{total_work_units}",
                unsafe_allow_html=True,
            )

        # Cleanup temporary files
        if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except Exception as cleanup_error:
                st.warning(f"Could not clean up temporary files: {str(cleanup_error)}")

        # Wait a moment to show completion, then rerun
        time.sleep(2)
        st.rerun()

    except Exception as e:
        # Stop periodic updates
        is_running[0] = False

        with progress_container:
            progress_bar.progress(0)
            status_text.error(f"❌ **Simulation error:** {str(e)}")
            percentage_text.markdown("**Progress:** Failed")
            progress_details.markdown("⚠️ Simulation stopped due to error")
        st.exception(e)

        # Cleanup temporary files on error
        if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except Exception:
                pass


# Main app
def main():
    """Main Streamlit application."""
    # Create sidebar and get run button state
    run_simulation_clicked = create_configuration_sidebar()

    # Main content area
    st.header("Simulation Results & Graphs")

    # Create tabs for graphs
    tab1, tab2, tab3 = st.tabs(["Original Data", "Results", "All Simulations"])

    with tab1:
        st.subheader("Original Data")
        update_original_data_graph()

    with tab2:
        st.subheader("Results")
        update_results_graph()

    with tab3:
        st.subheader("All Simulations")
        update_simulations_graph()

    # Handle run simulation button
    if run_simulation_clicked:
        run_simulation()


if __name__ == "__main__":
    main()

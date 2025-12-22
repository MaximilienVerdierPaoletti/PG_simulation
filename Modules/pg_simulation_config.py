# -*- coding: utf-8 -*-
"""
Configuration and profiling functions for PG NanoSIMS Simulations.

@author: Maximilien Verdier-Paoletti
"""

import os
import cProfile
import pstats
from datetime import datetime


def create_default_config():
    """Create default configuration dictionary for simulations."""
    config = {
        # Grain parameters for simulations
        "Nb_PG": 9,
        "nb_closest_match": 3,  # Can't be bigger than Nb_PG
        "beam": 100,
        "boxcar": 3,
        "elem": "O",
        # Delta database ranges
        "delta_database": (
            [*range(-900, 0, 50)]
            + [*range(0, 200, 10)]
            + [*range(200, 2000, 100)]
            + [*range(2000, 21000, 1000)]
        ),
        # Iteration parameters
        "iterations": 1,
        "max_iteration": 10,
        "cost_goal": 0.4,
        # Size range for simulations
        "size_range": range(50, 900, 50),
        # Output settings
        "Name_results": "test_alldata",
        "SAVE_ALL_PLOTS": False,
        "ENABLE_PROFILING": False,
        "PROFILING_OUTPUT_DIR": "profiling_results",
    }
    return config


def setup_profiling(enable_profiling=False, output_dir="profiling_results"):
    """Setup profiling if enabled."""
    pr = None
    if enable_profiling:
        os.makedirs(output_dir, exist_ok=True)
        pr = cProfile.Profile()
        pr.enable()
        print(
            f"[PROFILING] Profiling enabled. Results will be saved to '{output_dir}/'"
        )
    return pr


def stop_profiling(pr, output_dir="profiling_results"):
    """Stop profiling and save results."""
    if pr is not None:
        pr.disable()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed profiling statistics
        stats_file = os.path.join(output_dir, f"profile_stats_{timestamp}.txt")
        with open(stats_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()
            print(f"\n[PROFILING] Detailed statistics saved to: {stats_file}")

        # Save cumulative time statistics
        cumtime_file = os.path.join(output_dir, f"profile_cumtime_{timestamp}.txt")
        with open(cumtime_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats(30)
            print(f"[PROFILING] Cumulative time statistics saved to: {cumtime_file}")

        # Save per-call time statistics
        percall_file = os.path.join(output_dir, f"profile_percall_{timestamp}.txt")
        with open(percall_file, "w") as f:
            stats = pstats.Stats(pr, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(30)
            print(f"[PROFILING] Per-call time statistics saved to: {percall_file}")

        # Save profiling data in binary format
        profile_binary = os.path.join(output_dir, f"profile_{timestamp}.prof")
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


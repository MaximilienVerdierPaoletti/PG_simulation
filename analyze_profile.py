# -*- coding: utf-8 -*-
"""
Profile Analysis Helper Script

This script helps analyze profiling results from PG_NanoSIMS_Simulations_v3.0.py

Usage:
    python analyze_profile.py <profile_file.prof>
    
Or to analyze the most recent profile:
    python analyze_profile.py
    
@author: Maximilien Verdier-Paoletti
"""

import sys
import os
import pstats
from pathlib import Path


def analyze_profile(profile_file=None, top_n=30):
    """
    Analyze a profiling file and display useful statistics.
    
    Parameters
    ----------
    profile_file : str, optional
        Path to the .prof file. If None, uses the most recent file in profiling_results/
    top_n : int
        Number of top functions to display
    """
    if profile_file is None:
        # Find the most recent profile file
        profiling_dir = Path("profiling_results")
        if not profiling_dir.exists():
            print("Error: profiling_results directory not found.")
            print("Run the main script with ENABLE_PROFILING=True first.")
            return
        
        profile_files = list(profiling_dir.glob("profile_*.prof"))
        if not profile_files:
            print("Error: No profile files found in profiling_results/")
            return
        
        profile_file = max(profile_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent profile: {profile_file}")
    
    # Convert Path object to string if needed
    profile_file_str = str(profile_file) if isinstance(profile_file, Path) else profile_file
    
    if not os.path.exists(profile_file_str):
        print(f"Error: Profile file not found: {profile_file_str}")
        return
    
    stats = pstats.Stats(profile_file_str)
    
    print("\n" + "="*80)
    print(f"PROFILE ANALYSIS: {os.path.basename(profile_file_str)}")
    print("="*80)
    
    # Total statistics
    print(f"\nTotal function calls: {stats.total_calls:,}")
    print(f"Total time: {stats.total_tt:.4f} seconds")
    print(f"Primitive calls: {stats.prim_calls:,}")
    
    # Top functions by total time
    print("\n" + "-"*80)
    print(f"TOP {top_n} FUNCTIONS BY TOTAL TIME")
    print("-"*80)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(top_n)
    
    # Top functions by cumulative time
    print("\n" + "-"*80)
    print(f"TOP {top_n} FUNCTIONS BY CUMULATIVE TIME")
    print("-"*80)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(top_n)
    
    # Top functions by per-call time (most expensive per call)
    print("\n" + "-"*80)
    print(f"TOP {top_n} FUNCTIONS BY PER-CALL TIME (most expensive per call)")
    print("-"*80)
    stats.sort_stats(pstats.SortKey.TIME)
    # Filter to show only functions with significant per-call time
    stats.print_stats(top_n)
    
    # Functions called most frequently
    print("\n" + "-"*80)
    print(f"TOP {top_n} FUNCTIONS BY CALL COUNT")
    print("-"*80)
    stats.sort_stats(pstats.SortKey.CALLS)
    stats.print_stats(top_n)
    
    print("\n" + "="*80)
    print("To analyze specific modules or functions, use:")
    print(f"  python -m pstats {profile_file_str}")
    print("="*80 + "\n")


if __name__ == "__main__":
    profile_file = sys.argv[1] if len(sys.argv) > 1 else None
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    analyze_profile(profile_file, top_n)


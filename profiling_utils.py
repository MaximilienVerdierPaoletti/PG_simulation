# -*- coding: utf-8 -*-
"""
Profiling Utilities for PG Simulations

This module provides utilities for profiling specific functions and code sections.

Usage:
    from profiling_utils import profile_function
    
    @profile_function
    def my_function():
        # Your code here
        pass

@author: Maximilien Verdier-Paoletti
"""

import cProfile
import pstats
import functools
import os
from datetime import datetime


class FunctionProfiler:
    """Context manager for profiling specific code sections."""
    
    def __init__(self, output_dir="profiling_results", name="section"):
        self.output_dir = output_dir
        self.name = name
        self.profiler = None
        
    def __enter__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir, 
            f"profile_{self.name}_{timestamp}.prof"
        )
        self.profiler.dump_stats(output_file)
        
        # Print summary
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(pstats.SortKey.TIME)
        print(f"\n[PROFILING] Section '{self.name}' profiling complete.")
        print(f"[PROFILING] Results saved to: {output_file}")
        print(f"[PROFILING] Top 10 functions:")
        stats.print_stats(10)
        print()


def profile_function(func):
    """
    Decorator to profile a specific function.
    
    Usage:
        @profile_function
        def my_function():
            # Your code here
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "profiling_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"profile_{func.__name__}_{timestamp}.prof"
            )
            profiler.dump_stats(output_file)
            
            # Print summary
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.TIME)
            print(f"\n[PROFILING] Function '{func.__name__}' profiling complete.")
            print(f"[PROFILING] Results saved to: {output_file}")
            stats.print_stats(5)
            print()
        return result
    return wrapper


def profile_section(name="section"):
    """
    Context manager for profiling a specific code section.
    
    Usage:
        with profile_section("my_section"):
            # Code to profile
            pass
    """
    return FunctionProfiler(name=name)


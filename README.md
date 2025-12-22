# PG Simulation - Profiling Guide

## Profiling Integration

This project includes comprehensive profiling capabilities to help identify performance bottlenecks.

### Quick Start

1. **Enable profiling** in `PG_NanoSIMS_Simulations_v3.0.py`:
   ```python
   ENABLE_PROFILING = True  # Set to True to enable profiling
   ```

2. **Run your simulation** as usual. Profiling data will be automatically collected.

3. **View results** in the `profiling_results/` directory:
   - `profile_stats_*.txt` - Detailed statistics sorted by total time
   - `profile_cumtime_*.txt` - Top functions by cumulative time
   - `profile_percall_*.txt` - Top functions by per-call time
   - `profile_*.prof` - Binary profile data for detailed analysis

### Analyzing Profiling Results

#### Method 1: Use the Analysis Script
```bash
python analyze_profile.py
```
This will analyze the most recent profile file and display:
- Top functions by total time
- Top functions by cumulative time
- Top functions by per-call time
- Most frequently called functions

To analyze a specific profile file:
```bash
python analyze_profile.py profiling_results/profile_20240101_120000.prof
```

#### Method 2: Use Python's pstats Module
```bash
python -m pstats profiling_results/profile_20240101_120000.prof
```

In the pstats interactive shell:
- `sort time` - Sort by total time
- `sort cumulative` - Sort by cumulative time
- `stats 20` - Show top 20 functions
- `stats PG_simulations_func` - Show stats for specific module
- `stats PG_simulationv6` - Show stats for specific function

### Profiling Specific Functions

For more granular profiling, you can profile specific functions using decorators:

```python
from profiling_utils import profile_function

@profile_function
def my_expensive_function():
    # Your code here
    pass
```

Or profile specific code sections:

```python
from profiling_utils import profile_section

with profile_section("grain_processing"):
    # Code to profile
    process_grains()
```

### Understanding Profiling Output

- **ncalls**: Number of calls
- **tottime**: Total time spent in this function (excluding sub-functions)
- **cumtime**: Cumulative time (including sub-functions)
- **percall**: Time per call
- **filename:lineno(function)**: Location of the function

### Performance Tips

1. Look for functions with high `tottime` - these are your bottlenecks
2. Functions with high `cumtime` but low `tottime` spend time in sub-functions
3. Functions with high `ncalls` might benefit from optimization even if per-call time is low
4. Focus on functions in your own code first (not library functions)

### Disabling Profiling

Set `ENABLE_PROFILING = False` in `PG_NanoSIMS_Simulations_v3.0.py` to disable profiling and avoid any performance overhead.


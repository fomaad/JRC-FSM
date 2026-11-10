# Sensitivity Analysis: Usage

OAT sensitivity study of `CC_human_driver` in the cut-in scenario, run over all 28 speed pairs (15 low + 13 high).

## Files

| File | Role |
|---|---|
| `param_catalog.py` | Configuration: speed pairs, grids, variations, parameter baselines, experiment matrix |
| `run_sensitivity.py` | Runs the simulations, compares each run with the baseline, writes logs, then calls the plotting step |
| `pcb_metrics.py` | Extracts the PCB for each lateral-velocity row and counts the point shifts against the baseline |
| `plot_and_report.py` | Rankings, tornado charts and Δv-grouped PCB plots |

## Requirements

`numpy`, `pandas`, `numba`, `scipy`, `matplotlib`

## Running

Run from the **repository root**. The scripts import from both `utility/` and `sensitivity/`, and the baseline check reads `results/`.

Full run (simulation, then plots):

```bash
python -m sensitivity.run_sensitivity
```

Plots only (reuses the existing CSV logs):

```bash
python -m sensitivity.plot_and_report
```

Notes:
- Simulations run in parallel on up to 8 worker processes (`min(cpu_count, 8)`).
- The console prints a check of the baseline `Crash` results against `results/cut_in_<range>_speed/`.
- The `plot_and_report.py` deletes and recreates `results_sensitivity_full/plots/` on every run.

## Configuration (`param_catalog.py`)

| Variable | Default | Notes |
|---|---|---|
| `PCT_VARIATIONS` | `[-40, -20, 20, 40]` | Plot colours are only defined for these values. If you change them, update `SWEEP_COLORS` and `SWEEP_MARKERS` in `plot_and_report.py` too. |
| `CC_PARAMETERS` | `CC_rt`, `CC_critical_ttc`, `CC_max_deceleration`, `CC_min_jerk`, `CC_release_deceleration` | Each key must be an attribute of the ego vehicle object, because it is applied with `setattr`. Overriding `CC_rt` also sets `CC_rt_counter`. |
| `FULL_SPEED_PAIRS` | 15 low + 13 high | The Δv groups in `plot_and_report.py` expect all 28 pairs. |
| `GRID_CONFIG` | low: 1–59 m, step 1 m. high: 1–119 m, step 2 m | Lateral velocity is fixed at 0–1.7 m/s in 0.1 m/s steps (in `run_sensitivity.py`). |

Experiment matrix: 1 baseline + (number of parameters × number of variations) runs. With the defaults this is 21 runs.

## Output (`results_sensitivity_full/`)

1. **`experiment_log.csv`** — Master log with one row per experiment (aggregates low-speed, high-speed, and total crash point changes and shift directions).
2. **`pcb_all_speed_pairs_log.csv`** — One row per experiment × speed pair (21 runs × 28 pairs = 588 entries), logging near/far shifts and swapped point counts.
3. **`pcb_all_rows_log.csv`** — Detailed row-level boundary log with one row per experiment × speed pair × lateral-velocity row (tracking boundary positions and flip directions).
4. **`parameter_ranking_low_speed.csv`** — Ranked parameter sensitivity table across all 15 low-speed pairs (sorted by total swapped points).
5. **`parameter_ranking_high_speed.csv`** — Ranked parameter sensitivity table across all 13 high-speed pairs (sorted by total swapped points).
6. **`exp_<NNN>_<param>_<minus|plus>_<pct>pct/`** (and **`exp_000_baseline/`**) — Individual experiment folders containing:
   - `cut_in_low_speed/` & `cut_in_high_speed/`: Raw simulation CSV outputs (`CC_human_driver_<ego>_<cut_in>.csv`) with distance, lateral velocity, Crash boolean, and minTTC.
   - `pcb_shift_by_row.csv`: Near/far boundary points and meter shifts per lateral-velocity row.
   - `pcb_shift_summary.csv`: Summary metrics for each of the 28 speed pairs.
   - `manifest.json`: Run metadata (exact parameter override value, execution timestamp, and runtime in seconds).
7. **`plots/`** — Generated visualization assets:
   - `tornado_low_speed_all_15_pairs.png`: Tornado sensitivity chart for all 15 low-speed pairs.
   - `tornado_high_speed_all_13_pairs.png`: Tornado sensitivity chart for all 13 high-speed pairs.
   - `<param>/low_speed_dv_<group>_<param>.png`: PCB boundary overlay curves for low-speed pairs grouped by Δv.
   - `<param>/high_speed_dv_<group>_<param>.png`: PCB boundary overlay curves for high-speed pairs grouped by Δv.

## Metric conventions

- Shifts are measured in grid points: 1 pt = 1 m on the low-speed grid, 2 m on the high-speed grid.
- `delta_far_points > 0`: the upper PCB moved right (the crash zone expanded).
- `total_swapped_points` = safe→crash flips + crash→safe flips.
- `net_crash_point_change` = experiment crash points − baseline crash points.
- Parameters are ranked by `total_swapped_points_all_sweeps`, with `mean_abs_far_shift_points` as the tie-breaker.

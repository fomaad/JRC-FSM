"""
Full-repository sensitivity runner across all 28 speed pairs (15 low-speed + 13 high-speed).
Executes the unmodified cut-in simulation logic with One-At-A-Time (OAT) parameter overrides
on ego_veh, logs all PCB point shifts, and generates rankings and plots.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd

from utility import movement as mvt
from utility import models as md
from utility import global_parameters as gp

from sensitivity.param_catalog import (
    MODEL_NAME,
    FULL_SPEED_PAIRS,
    GRID_CONFIG,
    build_experiment_matrix,
)
from sensitivity.pcb_metrics import compare_pcb_grids
from sensitivity.plot_and_report import generate_plots_and_rankings


def apply_vehicle_overrides(veh, overrides):
    """Apply parameter overrides to a vehicle instance without altering utility/vehicle.py."""
    for attr, val in overrides.items():
        setattr(veh, attr, val)
        if attr == "CC_rt":
            veh.CC_rt_counter = val


def simulate_cut_in_pair(speed_range, ego_speed, cut_in_speed, overrides):
    """
    Execute the exact simulation loop from utility/comparison_cut_in.py
    for CC_human_driver and a single (ego_speed, cut_in_speed) pair.
    """
    length = gp.length
    width = gp.width
    iterations = gp.iterations
    freq = gp.freq

    check = md.CC_check_safety_cut_in
    react = md.CC_react

    cfg = GRID_CONFIG[speed_range]
    long_dists = np.arange(cfg["dist_start"], cfg["dist_stop"], cfg["dist_step"])
    lat_vels = np.arange(0.0, 1.8, 0.1)

    result_d = {
        "long_dist": [],
        "lat_vel": [],
        "Crash": [],
        "Crash type": [],
        "minTTC": [],
    }

    for long_dist in long_dists:
        for lat_vel in lat_vels:
            init_pos_c = np.array([long_dist + length, 1.6 + width])
            init_pos_ego = np.array([0, 0])
            init_long_speed_c = cut_in_speed / 3.6
            init_long_speed_ego = ego_speed / 3.6
            lateral_speed = -lat_vel
            lat_accel = -1.5

            cut_in_veh = mvt.create_profile_cutting_in(
                init_pos_c, init_long_speed_c, lateral_speed, iterations, freq
            )
            ego_veh = mvt.create_profile_cutting_in(
                init_pos_ego, init_long_speed_ego, 0, iterations, freq
            )

            # Apply OAT parameter override on ego vehicle
            apply_vehicle_overrides(ego_veh, overrides)

            ego_veh.large_TTC = False

            # Cut-in vehicle pre-cut-in lateral acceleration phase
            lat_speed_append = np.arange(0, lateral_speed, lat_accel / freq)
            lon_speed_append = np.array([init_long_speed_c] * len(lat_speed_append))
            lat_pos_append = np.append(
                np.cumsum(-lat_speed_append[::-1] / freq)[::-1] + init_pos_c[1],
                init_pos_c[1],
            )
            lon_pos_append = np.append(
                np.cumsum(-lon_speed_append[::-1] / freq)[::-1] + init_pos_c[0],
                init_pos_c[0],
            )

            lat_speed_append = np.append(lat_speed_append, [lateral_speed])
            lon_speed_append = np.append(lon_speed_append, [init_long_speed_c])

            cut_in_veh.speed_profile_long = np.append(
                lon_speed_append, cut_in_veh.speed_profile_long
            )
            cut_in_veh.speed_profile_lat = np.append(
                lat_speed_append, cut_in_veh.speed_profile_lat
            )
            cut_in_veh.pos_profile_long = np.append(
                lon_pos_append, cut_in_veh.pos_profile_long
            )
            cut_in_veh.pos_profile_lat = np.append(
                lat_pos_append, cut_in_veh.pos_profile_lat
            )

            # Ego vehicle pre-cut-in constant-velocity phase
            lat_speed_append = np.array([0] * (len(lat_speed_append) - 1))
            lon_speed_append = np.array([init_long_speed_ego] * len(lat_speed_append))
            lat_pos_append = np.append(
                np.cumsum(-lat_speed_append[::-1] / freq)[::-1] + init_pos_ego[1],
                init_pos_ego[1],
            )
            lon_pos_append = np.append(
                np.cumsum(-lon_speed_append[::-1] / freq)[::-1] + init_pos_ego[0],
                init_pos_ego[0],
            )

            lat_speed_append = np.append(lat_speed_append, [0])
            lon_speed_append = np.append(lon_speed_append, [init_long_speed_ego])

            ego_veh.speed_profile_long = np.append(
                lon_speed_append, ego_veh.speed_profile_long
            )
            ego_veh.speed_profile_lat = np.append(
                lat_speed_append, ego_veh.speed_profile_lat
            )
            ego_veh.pos_profile_long = np.append(
                lon_pos_append, ego_veh.pos_profile_long
            )
            ego_veh.pos_profile_lat = np.append(
                lat_pos_append, ego_veh.pos_profile_lat
            )

            cut_in_veh.width = width
            cut_in_veh.length = length
            ego_veh.width = width
            ego_veh.length = length

            for i in range(iterations + len(lat_speed_append) - 1):
                ego_veh.cfs, ego_veh.pfs = 0, 0
                mvt.control(ego_veh, cut_in_veh, freq, check, react, i)
                if ego_veh.large_TTC:
                    break

            result_d["long_dist"].append(long_dist)
            result_d["lat_vel"].append(lat_vel)
            result_d["Crash"].append(ego_veh.crash)
            result_d["Crash type"].append(ego_veh.crash_type)

            # 2D TTC post-processing
            headway_lat = (
                abs(
                    np.array(ego_veh.pos_profile_lat)
                    - np.array(cut_in_veh.pos_profile_lat)
                )
                - ego_veh.width / 2
                - cut_in_veh.width / 2
            )
            headway_lon = (
                (
                    np.array(cut_in_veh.pos_profile_long)
                    - np.array(ego_veh.pos_profile_long)
                )
                - ego_veh.length / 2
                - cut_in_veh.length / 2
            )
            rel_speed_lon = np.array(ego_veh.speed_profile_long) - np.array(
                cut_in_veh.speed_profile_long
            )
            rel_speed_lat = -np.array(cut_in_veh.speed_profile_lat)
            rel_speed_lat[rel_speed_lat == 0] = 0.001
            max_ttc = 4

            length_of_series = len(rel_speed_lat)
            ttc_lon = np.array([max_ttc] * length_of_series)
            ttc_lat = np.array([max_ttc] * length_of_series)

            ttc_lon = np.minimum(headway_lon / rel_speed_lon, ttc_lon)
            ttc_lon[rel_speed_lon < 0] = max_ttc
            ttc_lon[headway_lon < -ego_veh.length - cut_in_veh.length] = 10
            ttc_lon[
                (headway_lon > -ego_veh.length - cut_in_veh.length)
                & (headway_lon <= 0)
            ] = 0

            ttc_lat = np.minimum(headway_lat / rel_speed_lat, ttc_lat)
            ttc_lat[headway_lat < 0] = 0

            ttc = np.maximum(ttc_lon, ttc_lat)
            result_d["minTTC"].append(np.nanmin(ttc))

    return pd.DataFrame(result_d)


def _run_single_experiment_sim(args):
    """Worker function to run all 28 speed pairs for a single experiment."""
    exp, base_out_dir = args
    exp_id = exp["exp_id"]
    exp_dir = os.path.join(base_out_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    t0 = time.time()
    sim_results = {}

    for speed_range, pairs in FULL_SPEED_PAIRS.items():
        range_subdir = os.path.join(exp_dir, f"cut_in_{speed_range}_speed")
        os.makedirs(range_subdir, exist_ok=True)
        for ego_speed, cut_in_speed in pairs:
            df_sim = simulate_cut_in_pair(
                speed_range, ego_speed, cut_in_speed, exp["overrides"]
            )
            csv_name = f"{MODEL_NAME}_{ego_speed}_{cut_in_speed}.csv"
            csv_path = os.path.join(range_subdir, csv_name)
            df_sim.to_csv(csv_path, index=False)
            sim_results[(speed_range, ego_speed, cut_in_speed)] = df_sim

    elapsed = round(time.time() - t0, 2)
    return exp_id, elapsed, sim_results


def run_full_experiments(base_out_dir="results_sensitivity_full"):
    os.makedirs(base_out_dir, exist_ok=True)
    experiments = build_experiment_matrix()

    n_low = len(FULL_SPEED_PAIRS["low"])
    n_high = len(FULL_SPEED_PAIRS["high"])
    print(
        f"Starting full-repo sensitivity study: {len(experiments)} runs x {n_low + n_high} speed pairs "
        f"({n_low} low + {n_high} high) -> '{base_out_dir}/'"
    )

    t_start = time.time()
    max_workers = min(os.cpu_count() or 4, 8)
    worker_args = [(exp, base_out_dir) for exp in experiments]

    all_sim_outputs = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for exp_id, elapsed, sim_results in executor.map(
            _run_single_experiment_sim, worker_args
        ):
            all_sim_outputs[exp_id] = (elapsed, sim_results)
            print(f"  Finished simulation for {exp_id} ({elapsed}s)")

    # Verify baseline against original results/ CSVs
    baseline_dfs = all_sim_outputs["exp_000_baseline"][1]
    all_matched = True
    for (speed_range, ego_speed, cut_in_speed), df_base in baseline_dfs.items():
        csv_name = f"{MODEL_NAME}_{ego_speed}_{cut_in_speed}.csv"
        orig_csv = os.path.join("results", f"cut_in_{speed_range}_speed", csv_name)
        if os.path.isfile(orig_csv):
            df_orig = pd.read_csv(orig_csv)
            if not (df_orig["Crash"] == df_base["Crash"]).all():
                all_matched = False
                print(f"  [WARNING] Mismatch on {speed_range} {ego_speed}_{cut_in_speed}")
    print(
        f"  [Baseline Verification Across All 28 Speed Pairs] 100% match with original repo = {all_matched}"
    )

    master_log_rows = []
    pair_log_rows = []
    all_row_shifts = []

    for exp in experiments:
        exp_id = exp["exp_id"]
        exp_dir = os.path.join(base_out_dir, exp_id)
        elapsed, sim_results = all_sim_outputs[exp_id]

        exp_row_shifts_list = []
        exp_pair_summaries = []

        for speed_range, pairs in FULL_SPEED_PAIRS.items():
            dist_step = GRID_CONFIG[speed_range]["dist_step"]
            for ego_speed, cut_in_speed in pairs:
                pair_key = (speed_range, ego_speed, cut_in_speed)
                df_base = baseline_dfs[pair_key]
                df_sim = sim_results[pair_key]

                row_shifts, summary = compare_pcb_grids(df_base, df_sim)

                row_shifts.insert(0, "cut_in_speed", cut_in_speed)
                row_shifts.insert(0, "ego_speed", ego_speed)
                row_shifts.insert(0, "speed_range", speed_range)
                row_shifts.insert(0, "new_val", exp["new_val"])
                row_shifts.insert(0, "delta_pct", exp["delta_pct"])
                row_shifts.insert(0, "param_name", exp["param_name"])
                row_shifts.insert(0, "exp_id", exp_id)
                exp_row_shifts_list.append(row_shifts)
                all_row_shifts.append(row_shifts)

                pair_entry = {
                    "exp_id": exp_id,
                    "param_name": exp["param_name"],
                    "baseline_val": exp["baseline_val"],
                    "delta_pct": exp["delta_pct"],
                    "new_val": exp["new_val"],
                    "speed_range": speed_range,
                    "ego_speed": ego_speed,
                    "cut_in_speed": cut_in_speed,
                    "rel_speed_kmh": ego_speed - cut_in_speed,
                    "dist_step_m": dist_step,
                    **summary,
                }
                exp_pair_summaries.append(pair_entry)
                pair_log_rows.append(pair_entry)

        df_exp_rows = pd.concat(exp_row_shifts_list, ignore_index=True)
        df_exp_rows.to_csv(os.path.join(exp_dir, "pcb_shift_by_row.csv"), index=False)

        df_exp_pairs = pd.DataFrame(exp_pair_summaries)
        df_exp_pairs.to_csv(os.path.join(exp_dir, "pcb_shift_summary.csv"), index=False)

        manifest = {
            **exp,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": elapsed,
            "speed_pairs": FULL_SPEED_PAIRS,
        }
        with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        df_low = df_exp_pairs[df_exp_pairs["speed_range"] == "low"]
        df_high = df_exp_pairs[df_exp_pairs["speed_range"] == "high"]

        master_entry = {
            "exp_id": exp_id,
            "param_name": exp["param_name"],
            "baseline_val": exp["baseline_val"],
            "delta_pct": exp["delta_pct"],
            "new_val": exp["new_val"],
            "unit": exp["unit"],
            "timestamp": manifest["timestamp"],
            "elapsed_sec": elapsed,
            # Low speed set aggregates (15 pairs)
            "low_base_crash_pts": int(df_low["base_total_crash_points"].sum()),
            "low_exp_crash_pts": int(df_low["exp_total_crash_points"].sum()),
            "low_net_crash_change": int(df_low["net_crash_point_change"].sum()),
            "low_swapped_pts": int(df_low["total_swapped_points"].sum()),
            "low_rows_right": int(df_low["rows_far_shifted_right"].sum()),
            "low_rows_left": int(df_low["rows_far_shifted_left"].sum()),
            "low_rows_unchanged": int(df_low["rows_far_unchanged"].sum()),
            # High speed set aggregates (13 pairs)
            "high_base_crash_pts": int(df_high["base_total_crash_points"].sum()),
            "high_exp_crash_pts": int(df_high["exp_total_crash_points"].sum()),
            "high_net_crash_change": int(df_high["net_crash_point_change"].sum()),
            "high_swapped_pts": int(df_high["total_swapped_points"].sum()),
            "high_rows_right": int(df_high["rows_far_shifted_right"].sum()),
            "high_rows_left": int(df_high["rows_far_shifted_left"].sum()),
            "high_rows_unchanged": int(df_high["rows_far_unchanged"].sum()),
            # Combined totals
            "total_swapped_points": int(df_exp_pairs["total_swapped_points"].sum()),
            "net_crash_point_change": int(df_exp_pairs["net_crash_point_change"].sum()),
        }
        master_log_rows.append(master_entry)

    df_master = pd.DataFrame(master_log_rows)
    df_master.to_csv(os.path.join(base_out_dir, "experiment_log.csv"), index=False)

    df_pair_log = pd.DataFrame(pair_log_rows)
    df_pair_log.to_csv(
        os.path.join(base_out_dir, "pcb_all_speed_pairs_log.csv"), index=False
    )

    df_all_rows = pd.concat(all_row_shifts, ignore_index=True)
    df_all_rows.to_csv(os.path.join(base_out_dir, "pcb_all_rows_log.csv"), index=False)

    total_wall = round(time.time() - t_start, 2)
    print(f"\nCompleted full 28-pair sensitivity study in {total_wall}s -> '{base_out_dir}/'.")
    return df_master, df_pair_log, df_all_rows


if __name__ == "__main__":
    out_dir = "results_sensitivity_full"
    run_full_experiments(out_dir)
    generate_plots_and_rankings(out_dir)

"""PCB boundary extraction and discrete point-shift quantification."""

import numpy as np
import pandas as pd


def extract_pcb_by_row(df):
    """
    Extract PCB boundary metrics for each lateral velocity row on the discrete grid.

    For each `lat_vel` row:
      - point_idx is 0, 1, 2, ... corresponding to sorted `long_dist`
      - near_point_idx / near_dist_m: first grid point where Crash == True (left boundary)
      - far_point_idx / far_dist_m: last grid point where Crash == True (right boundary / Upper PCB)
      - crash_points: total count of grid points with Crash == True in this row
    """
    rows = []
    df_sorted = df.sort_values(["lat_vel", "long_dist"]).reset_index(drop=True)
    all_dists = np.sort(df_sorted["long_dist"].unique())
    dist_to_idx = {d: idx for idx, d in enumerate(all_dists)}

    for lat_vel, group in df_sorted.groupby("lat_vel", sort=True):
        group = group.sort_values("long_dist")
        crashes = group[group["Crash"] == True]
        n_crash = int(len(crashes))
        if n_crash > 0:
            near_dist = float(crashes["long_dist"].min())
            far_dist = float(crashes["long_dist"].max())
            near_idx = int(dist_to_idx[near_dist])
            far_idx = int(dist_to_idx[far_dist])
            has_crash = True
        else:
            near_dist = np.nan
            far_dist = np.nan
            near_idx = np.nan
            far_idx = np.nan
            has_crash = False

        rows.append(
            {
                "lat_vel": round(float(lat_vel), 2),
                "has_crash": has_crash,
                "crash_points": n_crash,
                "near_point_idx": near_idx,
                "far_point_idx": far_idx,
                "near_dist_m": near_dist,
                "far_dist_m": far_dist,
            }
        )
    return pd.DataFrame(rows)


def compare_pcb_grids(df_base, df_exp):
    """
    Compare experiment grid against baseline grid and quantify PCB shifts in discrete points.

    Sign convention for boundary shift:
      - delta_far_points > 0: Right boundary shifted RIGHT (collision zone expanded outward)
      - delta_far_points < 0: Right boundary shifted LEFT (collision zone shrank inward)
      - delta_near_points > 0: Left boundary shifted RIGHT
      - delta_near_points < 0: Left boundary shifted LEFT
    """
    pcb_base = extract_pcb_by_row(df_base)
    pcb_exp = extract_pcb_by_row(df_exp)

    merged_pcb = pd.merge(
        pcb_base, pcb_exp, on="lat_vel", suffixes=("_base", "_exp")
    )

    # Cell-by-cell comparison per row
    df_b = df_base.sort_values(["lat_vel", "long_dist"]).reset_index(drop=True)
    df_e = df_exp.sort_values(["lat_vel", "long_dist"]).reset_index(drop=True)
    cell_comp = pd.DataFrame(
        {
            "lat_vel": df_b["lat_vel"].round(2),
            "long_dist": df_b["long_dist"],
            "crash_base": df_b["Crash"].astype(bool),
            "crash_exp": df_e["Crash"].astype(bool),
        }
    )
    cell_comp["safe_to_crash"] = (~cell_comp["crash_base"]) & cell_comp["crash_exp"]
    cell_comp["crash_to_safe"] = cell_comp["crash_base"] & (~cell_comp["crash_exp"])

    flips_by_row = (
        cell_comp.groupby("lat_vel", as_index=False)
        .agg(
            safe_to_crash_points=("safe_to_crash", "sum"),
            crash_to_safe_points=("crash_to_safe", "sum"),
        )
    )
    flips_by_row["lat_vel"] = flips_by_row["lat_vel"].round(2)

    row_shifts = pd.merge(merged_pcb, flips_by_row, on="lat_vel", how="left")

    # Point and meter shifts on the far (right) and near (left) PCB boundaries
    row_shifts["delta_far_points"] = (
        row_shifts["far_point_idx_exp"] - row_shifts["far_point_idx_base"]
    )
    row_shifts["delta_near_points"] = (
        row_shifts["near_point_idx_exp"] - row_shifts["near_point_idx_base"]
    )
    row_shifts["delta_crash_points"] = (
        row_shifts["crash_points_exp"] - row_shifts["crash_points_base"]
    )
    row_shifts["delta_far_dist_m"] = (
        row_shifts["far_dist_m_exp"] - row_shifts["far_dist_m_base"]
    )
    row_shifts["delta_near_dist_m"] = (
        row_shifts["near_dist_m_exp"] - row_shifts["near_dist_m_base"]
    )

    def classify_shift(row):
        if not row["has_crash_base"] and not row["has_crash_exp"]:
            return "no_crash_both"
        if not row["has_crash_base"] and row["has_crash_exp"]:
            return "new_crash_row"
        if row["has_crash_base"] and not row["has_crash_exp"]:
            return "eliminated_crash_row"
        d = row["delta_far_points"]
        if d > 0:
            return "shifted_right"
        if d < 0:
            return "shifted_left"
        return "unchanged"

    row_shifts["far_shift_direction"] = row_shifts.apply(classify_shift, axis=1)

    valid_far = row_shifts["delta_far_points"].dropna()
    valid_near = row_shifts["delta_near_points"].dropna()

    summary = {
        "total_grid_points": int(len(cell_comp)),
        "base_total_crash_points": int(cell_comp["crash_base"].sum()),
        "exp_total_crash_points": int(cell_comp["crash_exp"].sum()),
        "net_crash_point_change": int(
            cell_comp["crash_exp"].sum() - cell_comp["crash_base"].sum()
        ),
        "safe_to_crash_swaps": int(cell_comp["safe_to_crash"].sum()),
        "crash_to_safe_swaps": int(cell_comp["crash_to_safe"].sum()),
        "total_swapped_points": int(
            cell_comp["safe_to_crash"].sum() + cell_comp["crash_to_safe"].sum()
        ),
        "active_rows_both": int(len(valid_far)),
        "rows_far_shifted_right": int((valid_far > 0).sum()),
        "rows_far_shifted_left": int((valid_far < 0).sum()),
        "rows_far_unchanged": int((valid_far == 0).sum()),
        "rows_new_crash": int((row_shifts["far_shift_direction"] == "new_crash_row").sum()),
        "rows_eliminated_crash": int(
            (row_shifts["far_shift_direction"] == "eliminated_crash_row").sum()
        ),
        "sum_far_shift_points": float(valid_far.sum()) if len(valid_far) > 0 else 0.0,
        "mean_far_shift_points": round(float(valid_far.mean()), 4)
        if len(valid_far) > 0
        else 0.0,
        "mean_abs_far_shift_points": round(float(valid_far.abs().mean()), 4)
        if len(valid_far) > 0
        else 0.0,
        "max_right_far_shift_points": int(valid_far.max())
        if len(valid_far) > 0 and valid_far.max() > 0
        else 0,
        "max_left_far_shift_points": int(valid_far.min())
        if len(valid_far) > 0 and valid_far.min() < 0
        else 0,
        "sum_near_shift_points": float(valid_near.sum()) if len(valid_near) > 0 else 0.0,
        "mean_near_shift_points": round(float(valid_near.mean()), 4)
        if len(valid_near) > 0
        else 0.0,
    }

    return row_shifts, summary

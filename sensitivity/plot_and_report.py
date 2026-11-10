"""
Generate PCB boundary plots and parameter rankings for the full 28-speed-pair study.
Groups PCB result plots by relative velocity (Δv = ego_speed - cut_in_speed) so each figure
has large, easy-to-read subplots and a clear, legible legend on every panel.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sensitivity.param_catalog import CC_PARAMETERS, PCT_VARIATIONS

plt.style.use("ggplot")

# Speed pairs grouped by Δv = ego_speed - cut_in_speed
LOW_SPEED_DV_GROUPS = [
    {
        "group_id": "dv_10kmh",
        "title_label": "Δv = 10 km/h",
        "pairs": [(20, 10), (30, 20), (40, 30), (50, 40), (60, 50)],
        "nrows": 2,
        "ncols": 3,
        "figsize": (19, 10.5),
    },
    {
        "group_id": "dv_20kmh",
        "title_label": "Δv = 20 km/h",
        "pairs": [(30, 10), (40, 20), (50, 30), (60, 40)],
        "nrows": 2,
        "ncols": 2,
        "figsize": (14, 10),
    },
    {
        "group_id": "dv_30kmh",
        "title_label": "Δv = 30 km/h",
        "pairs": [(40, 10), (50, 20), (60, 30)],
        "nrows": 1,
        "ncols": 3,
        "figsize": (19, 5.6),
    },
    {
        "group_id": "dv_40kmh",
        "title_label": "Δv = 40 km/h",
        "pairs": [(50, 10), (60, 20)],
        "nrows": 1,
        "ncols": 2,
        "figsize": (14, 5.6),
    },
    {
        "group_id": "dv_50kmh",
        "title_label": "Δv = 50 km/h",
        "pairs": [(60, 10)],
        "nrows": 1,
        "ncols": 1,
        "figsize": (9.5, 6.2),
    },
]

HIGH_SPEED_DV_GROUPS = [
    {
        "group_id": "dv_10_to_30kmh",
        "title_label": "Δv = 10 to 30 km/h",
        "pairs": [(110, 100), (90, 70), (70, 40), (130, 100)],
        "nrows": 2,
        "ncols": 2,
        "figsize": (14, 10),
    },
    {
        "group_id": "dv_40_to_60kmh",
        "title_label": "Δv = 40 to 60 km/h",
        "pairs": [(110, 70), (90, 40), (70, 10), (130, 70)],
        "nrows": 2,
        "ncols": 2,
        "figsize": (14, 10),
    },
    {
        "group_id": "dv_70_to_120kmh",
        "title_label": "Δv = 70 to 120 km/h",
        "pairs": [(110, 40), (90, 10), (130, 40), (110, 10), (130, 10)],
        "nrows": 2,
        "ncols": 3,
        "figsize": (19, 10.5),
    },
]

SWEEP_COLORS = {-40: "#1f77b4", -20: "#2ca02c", 20: "#ff7f0e", 40: "#d62728"}
SWEEP_MARKERS = {-40: "v", -20: "s", 20: "^", 40: "D"}


def format_val_unit(val, unit):
    unit_display = unit.replace("m/s^2", "m/s²").replace("m/s^3", "m/s³")
    if unit_display == "s":
        return f"{val:.2f}s"
    return f"{val:.2f} {unit_display}"


def _plot_dv_group_figure(
    base_out_dir,
    df_rows,
    param_name,
    base_val,
    unit,
    s_range,
    dv_group,
    out_path,
):
    base_str = format_val_unit(base_val, unit)
    step_m = 1 if s_range == "low" else 2
    nrows, ncols = dv_group["nrows"], dv_group["ncols"]
    pairs = dv_group["pairs"]

    fig, axes = plt.subplots(nrows, ncols, figsize=dv_group["figsize"])
    axes_flat = np.array(axes).reshape(-1)

    fig.suptitle(
        f"{s_range.capitalize()} Speed Set ({dv_group['title_label']}, 1 pt = {step_m} m) — "
        f"{param_name} (Baseline = {base_str})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, ax in enumerate(axes_flat):
        if idx >= len(pairs):
            ax.axis("off")
            continue

        ego_v, cut_v = pairs[idx]
        dv = ego_v - cut_v

        sim_base_csv = os.path.join(
            base_out_dir,
            "exp_000_baseline",
            f"cut_in_{s_range}_speed",
            f"CC_human_driver_{ego_v}_{cut_v}.csv",
        )
        df_CC_base = pd.read_csv(sim_base_csv)
        df_CC_crash = df_CC_base[df_CC_base["Crash"]]

        # Original comparison style 'rx' markers for baseline CC crash region
        ax.plot(
            df_CC_crash["long_dist"],
            df_CC_crash["lat_vel"],
            "rx",
            markersize=8.5,
            alpha=0.25,
            label=f"CC Baseline Crash ({base_str})",
            zorder=1,
        )

        sub_base = df_rows[
            (df_rows["exp_id"] == "exp_000_baseline")
            & (df_rows["speed_range"] == s_range)
            & (df_rows["ego_speed"] == ego_v)
            & (df_rows["cut_in_speed"] == cut_v)
            & (df_rows["has_crash_base"] == True)
        ]

        # Semi-transparent Baseline Upper & Lower PCB boundaries
        ax.plot(
            sub_base["far_dist_m_base"],
            sub_base["lat_vel"],
            color="black",
            linestyle="-",
            marker="o",
            linewidth=2.3,
            markersize=5,
            alpha=0.42,
            label=f"{base_str} (Baseline Upper PCB)",
            zorder=2,
        )
        ax.plot(
            sub_base["near_dist_m_base"],
            sub_base["lat_vel"],
            color="#444444",
            linestyle="--",
            marker="s",
            linewidth=1.9,
            markersize=4,
            alpha=0.42,
            label=f"{base_str} (Baseline Lower PCB)",
            zorder=2,
        )

        # Experimental PCB curves
        for pct in PCT_VARIATIONS:
            new_val = base_val * (1.0 + pct / 100.0)
            val_str = format_val_unit(new_val, unit)
            sub_exp = df_rows[
                (df_rows["param_name"] == param_name)
                & (df_rows["delta_pct"] == pct)
                & (df_rows["speed_range"] == s_range)
                & (df_rows["ego_speed"] == ego_v)
                & (df_rows["cut_in_speed"] == cut_v)
                & (df_rows["has_crash_exp"] == True)
            ]
            ax.plot(
                sub_exp["far_dist_m_exp"],
                sub_exp["lat_vel"],
                linestyle="-",
                marker=SWEEP_MARKERS[pct],
                color=SWEEP_COLORS[pct],
                linewidth=1.8,
                markersize=5,
                alpha=0.92,
                label=f"{val_str} ({pct:+d}%)",
                zorder=4,
            )

        ax.set_title(
            f"Δv = {dv} km/h | Ego {ego_v} km/h, Cut-in {cut_v} km/h",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("Initial distance (m)", fontsize=10)
        ax.set_ylabel("Lateral velocity (m/s)", fontsize=10)
        x_max = 120 if s_range == "high" else 60
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, 1.8])

        # Automatically place legend in the empty quadrant (upper right vs lower left)
        upper_rows = sub_base[sub_base["lat_vel"] >= 1.0]
        max_upper_far = (
            float(upper_rows["far_dist_m_base"].max()) if len(upper_rows) > 0 else 0.0
        )
        leg_loc = "lower left" if max_upper_far > 0.50 * x_max else "upper right"

        leg = ax.legend(
            loc=leg_loc,
            fontsize=8.8,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.92,
        )
        leg.set_zorder(10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots_and_rankings(base_out_dir="results_sensitivity_full"):
    plots_dir = os.path.join(base_out_dir, "plots")
    if os.path.isdir(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    df_master = pd.read_csv(os.path.join(base_out_dir, "experiment_log.csv"))
    df_pairs = pd.read_csv(os.path.join(base_out_dir, "pcb_all_speed_pairs_log.csv"))
    df_rows = pd.read_csv(os.path.join(base_out_dir, "pcb_all_rows_log.csv"))

    df_pairs_sweeps = df_pairs[df_pairs["param_name"] != "NONE"].copy()

    # 1. Separate Rankings for Low Speed (15 pairs) and High Speed (13 pairs)
    for s_range in ["low", "high"]:
        sub = df_pairs_sweeps[df_pairs_sweeps["speed_range"] == s_range]
        exp_totals = (
            sub.groupby(
                ["exp_id", "param_name", "baseline_val", "delta_pct"], as_index=False
            )
            .agg(
                swapped_pts=("total_swapped_points", "sum"),
                net_change=("net_crash_point_change", "sum"),
                rows_right=("rows_far_shifted_right", "sum"),
                rows_left=("rows_far_shifted_left", "sum"),
                mean_abs_far_shift=("mean_abs_far_shift_points", "mean"),
                max_right_shift=("max_right_far_shift_points", "max"),
                max_left_shift=("max_left_far_shift_points", "min"),
            )
        )
        rank_df = (
            exp_totals.groupby("param_name", as_index=False)
            .agg(
                baseline_val=("baseline_val", "first"),
                total_swapped_points_all_sweeps=("swapped_pts", "sum"),
                max_swapped_points_single_sweep=("swapped_pts", "max"),
                mean_abs_far_shift_points=("mean_abs_far_shift", "mean"),
                max_right_shift_points=("max_right_shift", "max"),
                max_left_shift_points=("max_left_shift", "min"),
                total_rows_shifted_right=("rows_right", "sum"),
                total_rows_shifted_left=("rows_left", "sum"),
            )
            .sort_values(
                by=["total_swapped_points_all_sweeps", "mean_abs_far_shift_points"],
                ascending=False,
            )
            .reset_index(drop=True)
        )
        rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
        rank_df.to_csv(
            os.path.join(base_out_dir, f"parameter_ranking_{s_range}_speed.csv"),
            index=False,
        )

    # 2. Separate Tornado Plots for All 15 Low-Speed Pairs and All 13 High-Speed Pairs
    low_base_total = int(df_master.loc[0, "low_base_crash_pts"])
    high_base_total = int(df_master.loc[0, "high_base_crash_pts"])

    tornado_configs = [
        {
            "speed_range": "low",
            "title": "Full Low-Speed Set Sensitivity (All 15 Speed Pairs: 10–60 km/h, 1 pt = 1 m)",
            "xlabel": f"Net Change in Collision Points Across All 15 Low-Speed Pairs (Baseline = {low_base_total} pts)",
            "filename": "tornado_low_speed_all_15_pairs.png",
        },
        {
            "speed_range": "high",
            "title": "Full High-Speed Set Sensitivity (All 13 Speed Pairs: 70–130 km/h, 1 pt = 2 m)",
            "xlabel": f"Net Change in Collision Points Across All 13 High-Speed Pairs (Baseline = {high_base_total} pts)",
            "filename": "tornado_high_speed_all_13_pairs.png",
        },
    ]

    for tcfg in tornado_configs:
        s_range = tcfg["speed_range"]
        sub_range = df_pairs_sweeps[df_pairs_sweeps["speed_range"] == s_range]
        rank_df = pd.read_csv(
            os.path.join(base_out_dir, f"parameter_ranking_{s_range}_speed.csv")
        )
        param_names = list(rank_df["param_name"])

        fig, ax = plt.subplots(figsize=(11, 6))
        y_pos = np.arange(len(param_names))
        bar_height = 0.18

        for i, pct in enumerate(PCT_VARIATIONS):
            vals = []
            for p in param_names:
                rows_p = sub_range[
                    (sub_range["param_name"] == p) & (sub_range["delta_pct"] == pct)
                ]
                vals.append(int(rows_p["net_crash_point_change"].sum()))
            offset = (i - 1.5) * bar_height
            bars = ax.barh(
                y_pos + offset,
                vals,
                height=bar_height,
                color=SWEEP_COLORS[pct],
                label=f"{pct:+d}% change",
            )
            for bar, v in zip(bars, vals):
                if v != 0:
                    ha = "left" if v > 0 else "right"
                    pad = max(abs(x) for x in vals) * 0.015 * (1 if v > 0 else -1)
                    ax.text(
                        v + pad,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:+d}",
                        va="center",
                        ha=ha,
                        fontsize=8.8,
                    )

        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, fontsize=11)
        ax.invert_yaxis()
        ax.margins(x=0.14)
        ax.set_xlabel(tcfg["xlabel"])
        ax.set_title(tcfg["title"], fontsize=13, fontweight="bold")
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.92,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, tcfg["filename"]), dpi=150)
        plt.close()

    # 3. PCB Boundary Result Plots Grouped by Δv into Per-Parameter Subfolders
    for param_name, meta in CC_PARAMETERS.items():
        base_val = meta["baseline"]
        unit = meta["unit"]
        param_plot_dir = os.path.join(plots_dir, param_name)
        os.makedirs(param_plot_dir, exist_ok=True)

        for dv_group in LOW_SPEED_DV_GROUPS:
            out_path = os.path.join(
                param_plot_dir,
                f"low_speed_{dv_group['group_id']}_{param_name}.png",
            )
            _plot_dv_group_figure(
                base_out_dir,
                df_rows,
                param_name,
                base_val,
                unit,
                "low",
                dv_group,
                out_path,
            )

        for dv_group in HIGH_SPEED_DV_GROUPS:
            out_path = os.path.join(
                param_plot_dir,
                f"high_speed_{dv_group['group_id']}_{param_name}.png",
            )
            _plot_dv_group_figure(
                base_out_dir,
                df_rows,
                param_name,
                base_val,
                unit,
                "high",
                dv_group,
                out_path,
            )

    print(f"Rankings and Δv-grouped plots saved to '{plots_dir}/'.")


if __name__ == "__main__":
    generate_plots_and_rankings()

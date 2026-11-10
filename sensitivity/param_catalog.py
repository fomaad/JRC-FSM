"""Parameter catalog and scope configuration for CC_human_driver cut-in sensitivity study."""

MODEL_NAME = "CC_human_driver"

# Full repository speed pairs: 15 low-speed pairs + 13 high-speed pairs = 28 pairs
FULL_SPEED_PAIRS = {
    "low": [
        (ego, cut_in)
        for ego in [10, 20, 30, 40, 50, 60]
        for cut_in in range(10, ego, 10)
    ],
    "high": [
        (ego, cut_in)
        for ego in [70, 90, 110, 130]
        for cut_in in range(10, ego, 30)
    ],
}

# Grid step sizes (in meters) for low-speed and high-speed cut-in simulations
GRID_CONFIG = {
    "low": {"dist_start": 1, "dist_stop": 60, "dist_step": 1},
    "high": {"dist_start": 1, "dist_stop": 120, "dist_step": 2},
}

# Equal percentage variations to test (One-At-A-Time)
PCT_VARIATIONS = [-40, -20, 20, 40]

# Baseline CC_human_driver parameters from utility/vehicle.py
CC_PARAMETERS = {
    "CC_rt": {
        "baseline": 0.75,
        "unit": "s",
        "description": "Driver reaction time before braking starts",
    },
    "CC_critical_ttc": {
        "baseline": 2.0,
        "unit": "s",
        "description": "Critical TTC threshold to trigger safety reaction",
    },
    "CC_max_deceleration": {
        "baseline": 0.774 * 9.81,
        "unit": "m/s^2",
        "description": "Maximum braking deceleration",
    },
    "CC_min_jerk": {
        "baseline": 12.65,
        "unit": "m/s^3",
        "description": "Jerk limit (rate of deceleration ramp-up)",
    },
    "CC_release_deceleration": {
        "baseline": 0.4,
        "unit": "m/s^2",
        "description": "Accelerator release deceleration during reaction time",
    },
}


def build_experiment_matrix():
    """Return ordered list of experiment dicts (1 baseline + 20 OAT parameter sweeps)."""
    experiments = [
        {
            "exp_id": "exp_000_baseline",
            "param_name": "NONE",
            "baseline_val": None,
            "delta_pct": 0,
            "new_val": None,
            "unit": "-",
            "overrides": {},
        }
    ]
    exp_idx = 1
    for param_name, meta in CC_PARAMETERS.items():
        base_val = meta["baseline"]
        for pct in PCT_VARIATIONS:
            new_val = round(base_val * (1.0 + pct / 100.0), 6)
            sign_str = f"minus_{abs(pct)}pct" if pct < 0 else f"plus_{pct}pct"
            exp_id = f"exp_{exp_idx:03d}_{param_name}_{sign_str}"
            experiments.append(
                {
                    "exp_id": exp_id,
                    "param_name": param_name,
                    "baseline_val": round(base_val, 6),
                    "delta_pct": pct,
                    "new_val": new_val,
                    "unit": meta["unit"],
                    "overrides": {param_name: new_val},
                }
            )
            exp_idx += 1
    return experiments

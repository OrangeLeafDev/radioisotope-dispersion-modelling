import numpy as np
import pandas as pd
from scipy import stats

def scs_runoff(rainfall_mm, CN):
    """
    Step-wise SCS Curve Number direct runoff depth (mm) per time step
    from cumulative storm precipitation (mm).
    """
    cumulative_P = np.cumsum(rainfall_mm)
    S = (25400.0 / CN) - 254.0
    Ia = 0.2 * S
    cumulative_Q = np.where(
        cumulative_P > Ia,
        (cumulative_P - Ia) ** 2 / (cumulative_P + 0.8 * S),
        0.0
    )
    return np.diff(cumulative_Q, prepend=0.0)

def fit_rating_curve(stages_m, discharges_m3s, h0=0.0, exclude_indices=None):
    """
    Logbook Entry 20 & 26:
    Fit power-law stage-discharge rating curve Q = a * (h - h0)^b
    with optional outlier exclusion (e.g., incomplete flood vertical).
    """
    stages = np.asarray(stages_m, dtype=float)
    discharges = np.asarray(discharges_m3s, dtype=float)
    if exclude_indices:
        mask = np.ones(len(stages), dtype=bool)
        mask[exclude_indices] = False
        stages, discharges = stages[mask], discharges[mask]

    log_h = np.log(stages - h0)
    log_q = np.log(discharges)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_q)
    a = np.exp(intercept)
    b = slope
    return {"a": a, "b": b, "h0": h0, "r_squared": r_value ** 2, "p_value": p_value}

def evaluate_rq2_runoff(scs_runoff_mm, gauged_runoff_mm):
    """
    RQ 2 & Hypothesis 2:
    Test significant difference between SCS-CN direct runoff estimates
    and gauged direct runoff during monsoon events.
    """
    scs_arr = np.asarray(scs_runoff_mm, dtype=float)
    gauged_arr = np.asarray(gauged_runoff_mm, dtype=float)
    diff = scs_arr - gauged_arr

    # Normality check on paired differences
    is_normal = len(diff) >= 3 and stats.shapiro(diff).pvalue > 0.05
    if is_normal:
        stat_val, p_val = stats.ttest_rel(scs_arr, gauged_arr)
        test_name = "Paired t-test"
    else:
        stat_val, p_val = stats.wilcoxon(scs_arr, gauged_arr)
        test_name = "Wilcoxon Signed-Rank Test"

    rmse = np.sqrt(np.mean(diff ** 2))
    return {
        "test_used": test_name,
        "statistic": float(stat_val),
        "p_value": float(p_val),
        "rmse_mm": float(rmse),
        "significant_difference_H1": bool(p_val < 0.05)
    }

def manning_velocity(n, R, S0):
    """Manning's equation for mean channel flow velocity v (m/s)."""
    return (1.0 / n) * (R ** (2.0 / 3.0)) * (S0 ** 0.5)

def fischer_dispersion(v, W, H, S0, g=9.81):
    """
    Fischer's empirical formula for longitudinal dispersion coefficient D (m^2/s):
    D = 0.011 * v^2 * W^2 / (H * u*), where shear velocity u* = sqrt(g * H * S0).
    """
    u_star = np.sqrt(g * H * S0)
    return 0.011 * (v ** 2) * (W ** 2) / (H * u_star)
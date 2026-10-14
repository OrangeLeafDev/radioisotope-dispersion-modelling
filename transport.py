import numpy as np
import pandas as pd
from scipy import stats

def verify_courant(v, dt, dx):
    """Logbook Entry 29: Verify Courant number Cr = v * dt / dx <= 1 for stability."""
    cr = v * dt / dx
    Peclet = dx / 2.0  # grid check helper
    return cr, cr <= 1.0

def rainfall_source_term(
    runoff_mm, nx, source_node, runoff_conc_mg_L,
    catchment_area_m2, cross_section_area_m2, dx
):
    """
    Compute source loading term S(x,t) / A in (mg / L / s) for the source node.
    - runoff_mm (mm per step) * catchment_area_m2 (m^2) = runoff_volume_L (Liters!)
      because 1 mm * 1 m^2 = 0.001 m^3 = 1 Liter.
    - cell_volume_L = cross_section_area_m2 * dx * 1000.0 (Liters)
    """
    nt = len(runoff_mm)
    source = np.zeros((nt, nx))
    runoff_volume_L = runoff_mm * catchment_area_m2
    metal_mass_mg = runoff_volume_L * runoff_conc_mg_L
    cell_volume_L = cross_section_area_m2 * dx * 1000.0
    # Mass added per step divided by cell volume; divided by dt inside solver or here
    source[:, source_node] = metal_mass_mg / cell_volume_L
    return source

def partition_fractions(Kd_L_kg, ssc_mg_L):
    """
    Logbook Entry 38 & 39:
    Two-phase sediment-water partitioning submodel.
    Returns dissolved fraction (f_d) and particulate fraction (f_p)
    guaranteeing strict mass conservation (f_d + f_p == 1.0).
    - Kd_L_kg: Partition coefficient (L/kg)
    - ssc_mg_L: Suspended sediment concentration (mg/L) -> converted to kg/L via 1e-6
    """
    ssc_kg_L = ssc_mg_L * 1e-6
    f_d = 1.0 / (1.0 + Kd_L_kg * ssc_kg_L)
    f_p = 1.0 - f_d
    assert np.all((f_d >= 0.0) & (f_d <= 1.0)), "Mass balance error: f_d out of [0, 1] bounds!"
    return f_d, f_p

def adr_partition_model(
    C0, v, D, k_base, Kd_L_kg, ssc_mg_L, v_settling, H_depth,
    S_step_conc, dx, dt, nx, nt
):
    """
    1-D Advection-Dispersion-Reaction (ADR) solver with Kd sediment-water partitioning.
    Tracks Total Concentration C_total (mg/L), Dissolved C_w (mg/L),
    and Sorbed Sediment Concentration C_s (mg/kg).
    Effective attenuation k_eff combines dissolved reaction/sorption loss (k_base)
    and particulate settling loss (f_p * v_settling / H_depth).
    """
    cr, is_stable = verify_courant(v, dt, dx)
    diff_num = D * dt / (dx ** 2)
    if not is_stable or (cr + 2 * diff_num > 1.0):
        raise ValueError(
            f"Numerical instability detected (Courant={cr:.3f}, Diff={diff_num:.3f}). "
            f"Reduce dt (currently {dt}s) or increase dx (currently {dx}m)."
        )

    f_d, f_p = partition_fractions(Kd_L_kg, ssc_mg_L)
    k_settling = f_p * (v_settling / H_depth)
    k_eff = k_base + k_settling

    C_total = np.zeros((nt, nx))
    C_total[0] = C0

    for t in range(1, nt):
        prev = C_total[t - 1]
        d2Cdx2 = (prev[2:] - 2.0 * prev[1:-1] + prev[:-2]) / (dx ** 2)
        dCdx = (prev[1:-1] - prev[:-2]) / dx  # Upwind scheme for advection stability

        C_total[t, 1:-1] = (
            prev[1:-1]
            + dt * (D * d2Cdx2 - v * dCdx - k_eff * prev[1:-1])
            + S_step_conc[t, 1:-1]
        )
        C_total[t, 0] = C_total[t, 1]
        C_total[t, -1] = C_total[t, -2]

    C_dissolved = C_total * f_d
    C_particulate_water = C_total * f_p
    C_sediment_mg_kg = C_dissolved * Kd_L_kg

    return {
        "C_total": C_total,
        "C_dissolved": C_dissolved,
        "C_particulate": C_particulate_water,
        "C_sediment_mg_kg": C_sediment_mg_kg,
        "f_d": f_d,
        "f_p": f_p,
        "k_eff": k_eff,
        "courant": cr
    }

def analyze_rq3_pulse_metrics(C_matrix, dx, dt, source_node, threshold_mg_L=1e-4):
    """
    RQ 3 & Hypothesis 3:
    Compute downstream pulse travel velocity (m/s), maximum displacement distance (m),
    and spatial attenuation rate (1/m and 1/s) across downstream reaches.
    """
    nt, nx = C_matrix.shape
    downstream_nodes = np.arange(source_node + 1, nx)
    distances_m = (downstream_nodes - source_node) * dx

    peak_conc = C_matrix[:, downstream_nodes].max(axis=0)
    peak_time_idx = C_matrix[:, downstream_nodes].argmax(axis=0)

    # Pulse arrival times where peak exceeds threshold
    valid = peak_conc > threshold_mg_L
    if np.any(valid):
        max_displacement_m = float(distances_m[valid].max())
        travel_times_s = peak_time_idx[valid] * dt
        valid_dist = distances_m[valid]
        nonzero_t = travel_times_s > 0
        pulse_speeds = valid_dist[nonzero_t] / travel_times_s[nonzero_t]
        mean_speed_m_s = float(np.mean(pulse_speeds)) if len(pulse_speeds) > 0 else 0.0

        # Spatial attenuation slope: ln(C_peak) vs distance
        slope, _, _, _, _ = stats.linregress(valid_dist, np.log(peak_conc[valid]))
        spatial_attenuation_per_m = float(-slope)
    else:
        max_displacement_m = 0.0
        mean_speed_m_s = 0.0
        spatial_attenuation_per_m = 0.0

    return {
        "mean_pulse_speed_m_s": mean_speed_m_s,
        "max_displacement_m": max_displacement_m,
        "spatial_attenuation_per_m": spatial_attenuation_per_m,
        "peak_profile_mg_L": peak_conc,
        "distances_m": distances_m
    }

def lhs_sensitivity_analysis(param_bounds, n_samples, model_runner_fn, seed=42):
    """
    Logbook Entry 40 & 47:
    Latin Hypercube Sampling (LHS) over uncertain parameters (Kd, CN, Manning's n, k)
    using scipy.stats.qmc.LatinHypercube.
    param_bounds: dict of {param_name: (min_val, max_val)}
    """
    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    lows = np.array([param_bounds[k][0] for k in names])
    highs = np.array([param_bounds[k][1] for k in names])

    sampler = stats.qmc.LatinHypercube(d=len(names), seed=rng)
    unit_samples = sampler.random(n=n_samples)
    scaled_samples = stats.qmc.scale(unit_samples, lows, highs)

    records = []
    ensemble_peaks = []
    for i, row in enumerate(scaled_samples):
        params = dict(zip(names, row))
        out = model_runner_fn(params)
        ensemble_peaks.append(out["C_total"].max(axis=0))
        records.append({
            **params,
            "outlet_peak_mg_L": float(out["C_total"][:, -1].max()),
            "domain_max_mg_L": float(out["C_total"].max()),
            "k_eff": float(out["k_eff"])
        })

    return pd.DataFrame(records), np.array(ensemble_peaks)
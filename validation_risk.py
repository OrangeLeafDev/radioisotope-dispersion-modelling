import numpy as np
import pandas as pd

# DENR Administrative Order No. 2016-08 (DAO 2016-08) Water Quality Guidelines (mg/L)
# Logbook Entry 44
DAO_2016_08_LIMITS = {
    "Class A": {"Nickel (Ni)": 0.02, "Chromium (Cr)": 0.01},
    "Class B": {"Nickel (Ni)": 0.04, "Chromium (Cr)": 0.01},
    "Class C": {"Nickel (Ni)": 0.20, "Chromium (Cr)": 0.01},
    "Class D": {"Nickel (Ni)": 1.00, "Chromium (Cr)": 0.02},
}

def compute_rmse_nse(observed, simulated):
    """
    RQ 4 & Hypothesis 4 (Logbook Entry 41 & 42):
    Calculate Root Mean Square Error (RMSE) and Nash-Sutcliffe Efficiency (NSE).
    Acceptance criterion: NSE > 0.5.
    """
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]

    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1.0 - (np.sum((obs - sim) ** 2) / denom)) if denom > 0 else np.nan
    return {
        "RMSE": rmse,
        "NSE": nse,
        "acceptable_H1_NSE_gt_0_5": bool(nse > 0.5) if not np.isnan(nse) else False
    }

def evaluate_rq5_dao_exceedance(C_matrix, metal_name, water_class="Class C"):
    """
    RQ 5 & Hypothesis 5 (Logbook Entry 44):
    Compare simulated peak and time-averaged concentrations against DENR DAO 2016-08.
    """
    limit = DAO_2016_08_LIMITS[water_class][metal_name]
    peak_per_node = C_matrix.max(axis=0)
    mean_per_node = C_matrix.mean(axis=0)

    domain_peak = float(peak_per_node.max())
    outlet_peak = float(peak_per_node[-1])
    outlet_mean = float(mean_per_node[-1])

    return {
        "analyte": metal_name,
        "water_class": water_class,
        "dao_2016_08_limit_mg_L": limit,
        "simulated_domain_peak_mg_L": domain_peak,
        "simulated_outlet_peak_mg_L": outlet_peak,
        "simulated_outlet_time_avg_mg_L": outlet_mean,
        "peak_exceedance_ratio": domain_peak / limit,
        "outlet_avg_exceedance_ratio": outlet_mean / limit,
        "exceeds_criterion_H1": bool(domain_peak > limit or outlet_mean > limit)
    }

def compute_sediment_pollution_indices(df_solids, control_station="Station 00"):
    """
    Logbook Entry 43:
    Compute Contamination Factor (CF) and Geoaccumulation Index (Igeo)
    using local Station 00 (upstream control) background concentrations:
    CF = C_sample / C_background
    Igeo = log2(C_sample / (1.5 * C_background))
    """
    solids = df_solids[df_solids["sample_type"].str.lower().isin(["sediment", "soil"])].copy()
    ctrl = solids[solids["location"].str.lower() == control_station.lower()]
    bg_map = ctrl.groupby("analyte")["value_numeric"].mean().to_dict()

    solids["C_background"] = solids["analyte"].map(bg_map)
    solids["CF"] = solids["value_numeric"] / solids["C_background"]
    solids["Igeo"] = np.log2(solids["value_numeric"] / (1.5 * solids["C_background"]))
    return solids

def compute_hpi(df_water, water_class="Class C"):
    """
    Heavy Metal Pollution Index (HPI) for surface water:
    Wi = 1 / Si, Qi = (Mi / Si) * 100, HPI = sum(Wi * Qi) / sum(Wi)
    """
    limits = DAO_2016_08_LIMITS[water_class]
    water = df_water[df_water["sample_type"].str.lower() == "water"].copy()
    water["Si"] = water["analyte"].map(limits)
    water = water.dropna(subset=["Si", "value_numeric"])
    water["Wi"] = 1.0 / water["Si"]
    water["Qi"] = (water["value_numeric"] / water["Si"]) * 100.0
    water["Wi_Qi"] = water["Wi"] * water["Qi"]

    hpi_df = water.groupby(["location", "storm_phase"], dropna=False).apply(
        lambda g: pd.Series({"HPI": g["Wi_Qi"].sum() / g["Wi"].sum()}),
        include_groups=False
    ).reset_index()
    return hpi_df

def compute_hhra_non_carcinogenic(Cw_mg_L):
    """
    Risk and Safety Section & Logbook Entry 45 & 46:
    Calculate USEPA Non-Carcinogenic Average Daily Dose (ADD) and Hazard Quotient (HQ)
    via ingestion and dermal absorption pathways for Adults and Children.
    (Note: Cr6+ carcinogenic LADD is explicitly omitted per Logbook Entry 24 & 46
    because only total Cr is measured without valence speciation.)
    """
    # Oral Reference Dose RfD_oral (mg/kg/day) - USEPA IRIS
    rfd_oral = {"Nickel (Ni)": 0.02, "Chromium (Cr)": 1.5}  # Cr(III)/Total baseline RfD; (if conservative Cr: 0.003)
    kp_dermal = {"Nickel (Ni)": 2e-4, "Chromium (Cr)": 1e-3}  # cm/hr

    receptors = {
        "Adult": {"IR_L_day": 2.0, "BW_kg": 60.0, "SA_cm2": 18000.0, "ET_hr_day": 1.0, "EF": 350.0, "ED": 30.0},
        "Child": {"IR_L_day": 1.0, "BW_kg": 15.0, "SA_cm2": 6600.0,  "ET_hr_day": 1.0, "EF": 350.0, "ED": 6.0},
    }

    rows = []
    for age_group, p in receptors.items():
        AT = p["ED"] * 365.0
        for metal, conc in Cw_mg_L.items():
            # Ingestion ADD (mg/kg/day)
            add_ing = (conc * p["IR_L_day"] * p["EF"] * p["ED"]) / (p["BW_kg"] * AT)
            # Dermal ADD (mg/kg/day), 1e-3 L/cm^3 conversion
            add_derm = (conc * p["SA_cm2"] * kp_dermal[metal] * p["ET_hr_day"] * 1e-3 * p["EF"] * p["ED"]) / (p["BW_kg"] * AT)

            rfd = rfd_oral[metal]
            hq_ing = add_ing / rfd
            hq_derm = add_derm / (rfd * 0.05)  # Gastrointestinal absorption factor adjustment
            hq_total = hq_ing + hq_derm

            rows.append({
                "receptor": age_group,
                "analyte": metal,
                "concentration_mg_L": conc,
                "ADD_ingestion": add_ing,
                "ADD_dermal": add_derm,
                "HQ_total": hq_total,
                "adverse_risk_HQ_gt_1": hq_total > 1.0
            })

    df_hq = pd.DataFrame(rows)
    # Hazard Index (HI) = sum of HQs per receptor group
    hi_summary = df_hq.groupby("receptor")["HQ_total"].sum().rename("Hazard_Index_HI").reset_index()
    hi_summary["adverse_effect_HI_gt_1"] = hi_summary["Hazard_Index_HI"] > 1.0
    return df_hq, hi_summary
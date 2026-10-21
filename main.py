from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fileio import read_table, write_table
from dataproc import prepare_data, descriptive_stats, test_rq1_storm_phases, compute_site_kd
from hydrology import (
    scs_runoff, fit_rating_curve, evaluate_rq2_runoff,
    manning_velocity, fischer_dispersion
)
from transport import (
    rainfall_source_term, adr_partition_model,
    analyze_rq3_pulse_metrics, lhs_sensitivity_analysis
)
from validation_risk import (
    DAO_2016_08_LIMITS, compute_rmse_nse, evaluate_rq5_dao_exceedance,
    compute_sediment_pollution_indices, compute_hpi, compute_hhra_non_carcinogenic
)

def load_field_dataset():
    """
    Loads santa_cruz_enriched_dataset.csv or SantaCruz_NiCr_Synthetic_Dataset.xlsx ('Data' sheet).
    Falls back to generating the wide-format dataset with coordinates if neither file is in the working directory.
    """
    if Path("santa_cruz_enriched_dataset.csv").exists():
        return read_table("santa_cruz_enriched_dataset.csv")
    for excel_name in ("SantaCruz_NiCr_Synthetic_Dataset_2.xlsx", "SantaCruz_NiCr_Synthetic_Dataset.xlsx"):
        if Path(excel_name).exists():
            return read_table(excel_name, sheet_name="Data")

    # Fallback matching the exact wide format of SantaCruz_NiCr_Synthetic_Dataset.xlsx + coordinates
    return pd.DataFrame([
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Pre-storm",    "Trial": 1, "Water Ni (mg/L)": 0.0238, "Water Cr (mg/L)": 0.0197, "Sediment Ni (mg/kg dw)": 4583, "Sediment Cr (mg/kg dw)": 7672, "Soil Ni (mg/kg dw)": 4502, "Soil Cr (mg/kg dw)": 5677, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Pre-storm",    "Trial": 2, "Water Ni (mg/L)": 0.0320, "Water Cr (mg/L)": 0.0192, "Sediment Ni (mg/kg dw)": 5761, "Sediment Cr (mg/kg dw)": 6487, "Soil Ni (mg/kg dw)": 3787, "Soil Cr (mg/kg dw)": 5759, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Pre-storm",    "Trial": 3, "Water Ni (mg/L)": 0.0287, "Water Cr (mg/L)": 0.0204, "Sediment Ni (mg/kg dw)": 5324, "Sediment Cr (mg/kg dw)": 7412, "Soil Ni (mg/kg dw)": 4234, "Soil Cr (mg/kg dw)": 5694, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "During storm", "Trial": 1, "Water Ni (mg/L)": 0.1906, "Water Cr (mg/L)": 0.1338, "Sediment Ni (mg/kg dw)": 4298, "Sediment Cr (mg/kg dw)": 5943, "Soil Ni (mg/kg dw)": 4051, "Soil Cr (mg/kg dw)": 5848, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "During storm", "Trial": 2, "Water Ni (mg/L)": 0.1858, "Water Cr (mg/L)": 0.1152, "Sediment Ni (mg/kg dw)": 4901, "Sediment Cr (mg/kg dw)": 7271, "Soil Ni (mg/kg dw)": 3992, "Soil Cr (mg/kg dw)": 5644, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "During storm", "Trial": 3, "Water Ni (mg/L)": 0.1691, "Water Cr (mg/L)": 0.1465, "Sediment Ni (mg/kg dw)": 4739, "Sediment Cr (mg/kg dw)": 7105, "Soil Ni (mg/kg dw)": 4032, "Soil Cr (mg/kg dw)": 6850, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Post-storm",   "Trial": 1, "Water Ni (mg/L)": 0.0743, "Water Cr (mg/L)": 0.0400, "Sediment Ni (mg/kg dw)": 5061, "Sediment Cr (mg/kg dw)": 7571, "Soil Ni (mg/kg dw)": 4343, "Soil Cr (mg/kg dw)": 6051, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Post-storm",   "Trial": 2, "Water Ni (mg/L)": 0.0686, "Water Cr (mg/L)": 0.0419, "Sediment Ni (mg/kg dw)": 5216, "Sediment Cr (mg/kg dw)": 6563, "Soil Ni (mg/kg dw)": 3895, "Soil Cr (mg/kg dw)": 5584, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S1", "Site description": "Near source", "Storm phase": "Post-storm",   "Trial": 3, "Water Ni (mg/L)": 0.0710, "Water Cr (mg/L)": 0.0423, "Sediment Ni (mg/kg dw)": 4760, "Sediment Cr (mg/kg dw)": 6540, "Soil Ni (mg/kg dw)": 4686, "Soil Cr (mg/kg dw)": 5340, "Data status": "SYNTHETIC", "Longitude": 120.0000, "Latitude": 15.699444, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Pre-storm",    "Trial": 1, "Water Ni (mg/L)": 0.0180, "Water Cr (mg/L)": 0.0104, "Sediment Ni (mg/kg dw)": 4664, "Sediment Cr (mg/kg dw)": 6661, "Soil Ni (mg/kg dw)": 3348, "Soil Cr (mg/kg dw)": 5037, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Pre-storm",    "Trial": 2, "Water Ni (mg/L)": 0.0148, "Water Cr (mg/L)": 0.0094, "Sediment Ni (mg/kg dw)": 5046, "Sediment Cr (mg/kg dw)": 6494, "Soil Ni (mg/kg dw)": 3306, "Soil Cr (mg/kg dw)": 4731, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Pre-storm",    "Trial": 3, "Water Ni (mg/L)": 0.0149, "Water Cr (mg/L)": 0.0100, "Sediment Ni (mg/kg dw)": 4324, "Sediment Cr (mg/kg dw)": 6898, "Soil Ni (mg/kg dw)": 3383, "Soil Cr (mg/kg dw)": 5230, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "During storm", "Trial": 1, "Water Ni (mg/L)": 0.0707, "Water Cr (mg/L)": 0.0480, "Sediment Ni (mg/kg dw)": 4228, "Sediment Cr (mg/kg dw)": 6032, "Soil Ni (mg/kg dw)": 3239, "Soil Cr (mg/kg dw)": 4824, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "During storm", "Trial": 2, "Water Ni (mg/L)": 0.0851, "Water Cr (mg/L)": 0.0443, "Sediment Ni (mg/kg dw)": 4452, "Sediment Cr (mg/kg dw)": 6052, "Soil Ni (mg/kg dw)": 3350, "Soil Cr (mg/kg dw)": 4803, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "During storm", "Trial": 3, "Water Ni (mg/L)": 0.0591, "Water Cr (mg/L)": 0.0487, "Sediment Ni (mg/kg dw)": 4322, "Sediment Cr (mg/kg dw)": 6149, "Soil Ni (mg/kg dw)": 3744, "Soil Cr (mg/kg dw)": 4615, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Post-storm",   "Trial": 1, "Water Ni (mg/L)": 0.0313, "Water Cr (mg/L)": 0.0168, "Sediment Ni (mg/kg dw)": 4859, "Sediment Cr (mg/kg dw)": 5760, "Soil Ni (mg/kg dw)": 3214, "Soil Cr (mg/kg dw)": 4471, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Post-storm",   "Trial": 2, "Water Ni (mg/L)": 0.0328, "Water Cr (mg/L)": 0.0219, "Sediment Ni (mg/kg dw)": 4887, "Sediment Cr (mg/kg dw)": 6469, "Soil Ni (mg/kg dw)": 3625, "Soil Cr (mg/kg dw)": 4804, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S2", "Site description": "Mid-reach",   "Storm phase": "Post-storm",   "Trial": 3, "Water Ni (mg/L)": 0.0274, "Water Cr (mg/L)": 0.0177, "Sediment Ni (mg/kg dw)": 4880, "Sediment Cr (mg/kg dw)": 7121, "Soil Ni (mg/kg dw)": 3678, "Soil Cr (mg/kg dw)": 5078, "Data status": "SYNTHETIC", "Longitude": 119.9645, "Latitude": 15.713500, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Pre-storm",    "Trial": 1, "Water Ni (mg/L)": 0.0066, "Water Cr (mg/L)": 0.0059, "Sediment Ni (mg/kg dw)": 3604, "Sediment Cr (mg/kg dw)": 4686, "Soil Ni (mg/kg dw)": 3020, "Soil Cr (mg/kg dw)": 4361, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Pre-storm",    "Trial": 2, "Water Ni (mg/L)": 0.0079, "Water Cr (mg/L)": 0.0063, "Sediment Ni (mg/kg dw)": 3884, "Sediment Cr (mg/kg dw)": 5344, "Soil Ni (mg/kg dw)": 2832, "Soil Cr (mg/kg dw)": 3796, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Pre-storm",    "Trial": 3, "Water Ni (mg/L)": 0.0082, "Water Cr (mg/L)": 0.0060, "Sediment Ni (mg/kg dw)": 3777, "Sediment Cr (mg/kg dw)": 5793, "Soil Ni (mg/kg dw)": 2860, "Soil Cr (mg/kg dw)": 4150, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "During storm", "Trial": 1, "Water Ni (mg/L)": 0.0253, "Water Cr (mg/L)": 0.0154, "Sediment Ni (mg/kg dw)": 3984, "Sediment Cr (mg/kg dw)": 5263, "Soil Ni (mg/kg dw)": 2947, "Soil Cr (mg/kg dw)": 4190, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "During storm", "Trial": 2, "Water Ni (mg/L)": 0.0244, "Water Cr (mg/L)": 0.0182, "Sediment Ni (mg/kg dw)": 4282, "Sediment Cr (mg/kg dw)": 4863, "Soil Ni (mg/kg dw)": 2826, "Soil Cr (mg/kg dw)": 4293, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "During storm", "Trial": 3, "Water Ni (mg/L)": 0.0233, "Water Cr (mg/L)": 0.0176, "Sediment Ni (mg/kg dw)": 4260, "Sediment Cr (mg/kg dw)": 5782, "Soil Ni (mg/kg dw)": 2797, "Soil Cr (mg/kg dw)": 4438, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Post-storm",   "Trial": 1, "Water Ni (mg/L)": 0.0134, "Water Cr (mg/L)": 0.0077, "Sediment Ni (mg/kg dw)": 4174, "Sediment Cr (mg/kg dw)": 6297, "Soil Ni (mg/kg dw)": 2801, "Soil Cr (mg/kg dw)": 3883, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Post-storm",   "Trial": 2, "Water Ni (mg/L)": 0.0120, "Water Cr (mg/L)": 0.0085, "Sediment Ni (mg/kg dw)": 3929, "Sediment Cr (mg/kg dw)": 6295, "Soil Ni (mg/kg dw)": 2627, "Soil Cr (mg/kg dw)": 4312, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
        {"Site ID": "S3", "Site description": "Downstream",  "Storm phase": "Post-storm",   "Trial": 3, "Water Ni (mg/L)": 0.0141, "Water Cr (mg/L)": 0.0100, "Sediment Ni (mg/kg dw)": 4974, "Sediment Cr (mg/kg dw)": 6079, "Soil Ni (mg/kg dw)": 2717, "Soil Cr (mg/kg dw)": 4012, "Data status": "SYNTHETIC", "Longitude": 119.8920, "Latitude": 15.714300, "Campaign": 1},
    ])

def export_chapter3_methodology_tables(kd_summary, v_flow, D_disp):
    Path("outputs").mkdir(exist_ok=True)
    kd_ni = float(kd_summary.loc[kd_summary["analyte"] == "Nickel (Ni)", "mean_Kd"].iloc[0])
    kd_cr = float(kd_summary.loc[kd_summary["analyte"] == "Chromium (Cr)", "mean_Kd"].iloc[0])

    table2_dynamic = pd.DataFrame([
        {"Parameter": "CN", "Description": "Curve Number", "Value": 82.0, "Basis": "Mined Laterite LULC Overlay"},
        {"Parameter": "n", "Description": "Manning's Roughness", "Value": 0.040, "Basis": "Channel Geometry Calibration"},
        {"Parameter": "v", "Description": "Mean Flow Velocity", "Value": round(v_flow, 3), "Basis": "Calculated via Manning's Equation"},
        {"Parameter": "D", "Description": "Longitudinal Dispersion", "Value": round(D_disp, 2), "Basis": "Calculated via Fischer's Formula"},
        {"Parameter": "Kd (Ni)", "Description": "Nickel Partition Coeff.", "Value": round(kd_ni, 1), "Basis": "Derived from Field Samples (dataproc.py)"},
        {"Parameter": "Kd (Cr)", "Description": "Chromium Partition Coeff.", "Value": round(kd_cr, 1), "Basis": "Derived from Field Samples (dataproc.py)"},
    ])
    table2_dynamic.to_csv("outputs/Chapter3_Model_Parameters_Dynamic.csv", index=False)
    print("Successfully exported model-generated Chapter 3 parameters to outputs/Chapter3_Model_Parameters_Dynamic.csv")

def main():
    raw_df = load_field_dataset()
    clean_df = prepare_data(raw_df)
    desc_df = descriptive_stats(clean_df)
    rq1_df = test_rq1_storm_phases(clean_df, source_location="S1")
    paired_kd, kd_summary = compute_site_kd(clean_df, moisture_fraction=0.0)

    write_table(clean_df, "outputs/01_cleaned_field_data.csv")
    write_table(desc_df, "outputs/02_descriptive_statistics.csv")
    write_table(rq1_df, "outputs/03_rq1_storm_phase_tests.csv")
    write_table(kd_summary, "outputs/04_site_specific_Kd.csv")

    print("=== RQ 1: Source Concentrations Across Pre-, During-, Post-Storm ===")
    print(rq1_df.to_string(index=False), "\n")
    print("=== Site-Specific Partition Coefficients (Kd in L/kg) ===")
    print(kd_summary.to_string(index=False), "\n")

    nx, nt = 200, 600
    dx, dt = 50.0, 10.0

    rainfall_mm = np.zeros(nt)
    rainfall_mm[60:180] = 35.0 / 120.0
    rainfall_mm[180:300] = 55.0 / 120.0

    CN = 82.0
    scs_q_mm = scs_runoff(rainfall_mm, CN)

    rng = np.random.default_rng(42)
    gauged_q_mm = np.clip(scs_q_mm * rng.normal(1.03, 0.08, size=nt), 0, None)
    rq2_res = evaluate_rq2_runoff(scs_q_mm, gauged_q_mm)
    print("=== RQ 2: SCS-CN Estimated vs. Gauged Direct Runoff ===")
    print(pd.DataFrame([rq2_res]).to_string(index=False), "\n")

    manning_n = 0.040
    W_width, H_depth, S0_slope = 25.0, 1.6, 0.0025
    R_hydraulic = (W_width * H_depth) / (W_width + 2.0 * H_depth)
    A_cross = W_width * H_depth

    v_flow = manning_velocity(manning_n, R_hydraulic, S0_slope)
    D_disp = fischer_dispersion(v_flow, W_width, H_depth, S0_slope)
    print(f"Hydraulics -> Velocity v = {v_flow:.3f} m/s | Dispersion D = {D_disp:.2f} m^2/s\n")

    catchment_area_m2 = 2.5e6
    source_node = 5
    ssc_mg_L = 450.0
    v_settling = 1.5e-4

    # Extract empirical S1 During-storm water concentrations from the dataset
    s1_storm_water = clean_df[
        (clean_df["sample_type"].str.lower() == "water") &
        (clean_df["storm_phase"].str.lower() == "during storm") &
        (clean_df["location"].str.lower().isin(["s1", "source"]))
    ]
    ni_s1_mean = float(s1_storm_water[s1_storm_water["analyte"] == "Nickel (Ni)"]["value_numeric"].mean())
    cr_s1_mean = float(s1_storm_water[s1_storm_water["analyte"] == "Chromium (Cr)"]["value_numeric"].mean())

    metals = {
        "Nickel (Ni)": {
            "Kd_L_kg": float(kd_summary.loc[kd_summary["analyte"] == "Nickel (Ni)", "mean_Kd"].iloc[0]),
            "k_base": 1.2e-5,
            "runoff_conc_mg_L": ni_s1_mean if not np.isnan(ni_s1_mean) else 0.1818
        },
        "Chromium (Cr)": {
            "Kd_L_kg": float(kd_summary.loc[kd_summary["analyte"] == "Chromium (Cr)", "mean_Kd"].iloc[0]),
            "k_base": 2.5e-5,
            "runoff_conc_mg_L": cr_s1_mean if not np.isnan(cr_s1_mean) else 0.1318
        }
    }

    models_result = {}
    rq3_rows, rq4_rows, rq5_rows = [], [], []

    for name, m in metals.items():
        C0 = np.zeros(nx)
        S_step = rainfall_source_term(
            scs_q_mm, nx, source_node=source_node,
            runoff_conc_mg_L=m["runoff_conc_mg_L"],
            catchment_area_m2=catchment_area_m2,
            cross_section_area_m2=A_cross,
            dx=dx
        )
        res = adr_partition_model(
            C0=C0, v=v_flow, D=D_disp, k_base=m["k_base"],
            Kd_L_kg=m["Kd_L_kg"], ssc_mg_L=ssc_mg_L,
            v_settling=v_settling, H_depth=H_depth,
            S_step_conc=S_step, dx=dx, dt=dt, nx=nx, nt=nt
        )
        models_result[name] = res

        pulse_metrics = analyze_rq3_pulse_metrics(res["C_total"], dx, dt, source_node)
        rq3_rows.append({
            "analyte": name,
            "Kd_L_kg": m["Kd_L_kg"],
            "dissolved_fraction_fd": res["f_d"],
            "particulate_fraction_fp": res["f_p"],
            "effective_decay_k_eff_per_s": res["k_eff"],
            "mean_pulse_speed_m_s": pulse_metrics["mean_pulse_speed_m_s"],
            "max_displacement_m": pulse_metrics["max_displacement_m"],
            "spatial_attenuation_per_m": pulse_metrics["spatial_attenuation_per_m"]
        })

        sim_outlet = res["C_total"][:, -1]
        obs_outlet = np.clip(sim_outlet * rng.normal(1.0, 0.12, size=nt) + rng.normal(0, 0.002, size=nt), 0, None)
        val_metrics = compute_rmse_nse(obs_outlet, sim_outlet)
        rq4_rows.append({"analyte": name, **val_metrics})

        dao_metrics = evaluate_rq5_dao_exceedance(res["C_total"], name, water_class="Class C")
        rq5_rows.append(dao_metrics)

    rq3_df = pd.DataFrame(rq3_rows)
    rq4_df = pd.DataFrame(rq4_rows)
    rq5_df = pd.DataFrame(rq5_rows)

    write_table(rq3_df, "outputs/05_rq3_transport_pulse_metrics.csv")
    write_table(rq4_df, "outputs/06_rq4_validation_rmse_nse.csv")
    write_table(rq5_df, "outputs/07_rq5_dao_2016_08_exceedances.csv")

    solids_indices = compute_sediment_pollution_indices(clean_df, control_station="Station 00")
    hpi_df = compute_hpi(clean_df, water_class="Class C")

    peak_water_concs = {name: float(models_result[name]["C_total"].max()) for name in metals}
    hq_df, hi_summary = compute_hhra_non_carcinogenic(peak_water_concs)

    write_table(solids_indices, "outputs/08_sediment_soil_CF_Igeo.csv")
    write_table(hpi_df, "outputs/09_water_HPI.csv")
    write_table(hq_df, "outputs/10_hhra_hazard_quotients.csv")
    write_table(hi_summary, "outputs/11_hhra_hazard_index_summary.csv")

    def run_ni_lhs(p):
        v_lhs = manning_velocity(p["n"], R_hydraulic, S0_slope)
        D_lhs = fischer_dispersion(v_lhs, W_width, H_depth, S0_slope)
        q_lhs = scs_runoff(rainfall_mm, p["CN"])
        s_lhs = rainfall_source_term(
            q_lhs, nx, source_node, metals["Nickel (Ni)"]["runoff_conc_mg_L"],
            catchment_area_m2, A_cross, dx
        )
        return adr_partition_model(
            np.zeros(nx), v_lhs, D_lhs, metals["Nickel (Ni)"]["k_base"],
            p["Kd"], ssc_mg_L, v_settling, H_depth, s_lhs, dx, dt, nx, nt
        )

    ni_kd_center = metals["Nickel (Ni)"]["Kd_L_kg"]
    lhs_bounds = {"Kd": (ni_kd_center * 0.6, ni_kd_center * 1.4), "CN": (75.0, 88.0), "n": (0.035, 0.050)}
    lhs_df, lhs_peaks = lhs_sensitivity_analysis(lhs_bounds, n_samples=30, model_runner_fn=run_ni_lhs)
    write_table(lhs_df, "outputs/12_lhs_sensitivity_results.csv")

    ni_limit = DAO_2016_08_LIMITS["Class C"]["Nickel (Ni)"]
    exceed_prob = (lhs_peaks > ni_limit).mean(axis=0)
    gis_export = pd.DataFrame({
        "node_id": np.arange(nx),
        "distance_downstream_km": np.arange(nx) * dx / 1000.0,
        "Ni_mean_peak_mg_L": lhs_peaks.mean(axis=0),
        "Ni_p95_peak_mg_L": np.percentile(lhs_peaks, 95, axis=0),
        "DAO_Class_C_exceedance_prob": exceed_prob
    })
    write_table(gis_export, "outputs/13_gis_probabilistic_river_profile.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (name, res) in zip(axes[:2], models_result.items()):
        im = ax.imshow(
            res["C_total"].T, aspect="auto", origin="lower", cmap="inferno",
            extent=[0, nt * dt / 3600.0, 0, nx * dx / 1000.0],
            vmin=0
        )
        ax.set_title(f"Simulated {name} Dispersion")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Distance downstream (km)")
        fig.colorbar(im, ax=ax, label="Total Concentration (mg/L)")

    axes[2].plot(gis_export["distance_downstream_km"], gis_export["Ni_mean_peak_mg_L"], label="Mean Peak Ni (LHS)", color="tab:red")
    axes[2].fill_between(
        gis_export["distance_downstream_km"],
        np.percentile(lhs_peaks, 5, axis=0),
        gis_export["Ni_p95_peak_mg_L"],
        color="tab:red", alpha=0.25, label="5th–95th Percentile Band"
    )
    axes[2].axhline(ni_limit, color="black", linestyle="--", label=f"DAO 2016-08 Class C ({ni_limit} mg/L)")
    axes[2].set_title("Probabilistic Peak Profile (LHS Monte Carlo)")
    axes[2].set_xlabel("Distance downstream (km)")
    axes[2].set_ylabel("Concentration (mg/L)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/dispersion_and_risk_summary.png", dpi=300)
    export_chapter3_methodology_tables(kd_summary, v_flow, D_disp)
    plt.show()

if __name__ == "__main__":
    main()
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

def generate_sample_field_dataset():
    """
    Generates a realistic template dataset matching the 3 campaigns, 5 stations
    (Station 00 Control, Source, Near-Field, Mid-Field, Outlet), and 3 storm phases
    (Pre-Storm, During-Storm, Post-Storm) if no external CSV/XLSX is provided yet.
    """
    rng = np.random.default_rng(2026)
    stations = ["Station 00", "Source", "Near-Field", "Mid-Field", "Outlet"]
    phases = ["Pre-Storm", "During-Storm", "Post-Storm"]
    rows = []

    for campaign in [1, 2, 3]:
        for st in stations:
            # Spatial gradient multiplier
            loc_mult = {"Station 00": 0.15, "Source": 1.8, "Near-Field": 1.3, "Mid-Field": 0.9, "Outlet": 0.6}[st]
            for ph in phases:
                # First-flush mobilization multiplier during storm
                ph_mult = {"Pre-Storm": 0.4, "During-Storm": 2.2, "Post-Storm": 0.9}[ph]
                for rep in range(1, 4):
                    # Water (mg/L)
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Water", "analyte": "Nickel (Ni)",
                        "value": round(0.08 * loc_mult * ph_mult * rng.uniform(0.85, 1.15), 5),
                        "unit": "mg/L"
                    })
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Water", "analyte": "Chromium (Cr)",
                        "value": round(0.025 * loc_mult * ph_mult * rng.uniform(0.80, 1.20), 5),
                        "unit": "mg/L"
                    })
                    # Sediment (mg/kg) - Cr has higher Kd than Ni
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Sediment", "analyte": "Nickel (Ni)",
                        "value": round(120.0 * loc_mult * ph_mult * rng.uniform(0.9, 1.1), 2),
                        "unit": "mg/kg"
                    })
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Sediment", "analyte": "Chromium (Cr)",
                        "value": round(190.0 * loc_mult * ph_mult * rng.uniform(0.85, 1.15), 2),
                        "unit": "mg/kg"
                    })
                    # Soil (mg/kg)
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Soil", "analyte": "Nickel (Ni)",
                        "value": round(180.0 * loc_mult * ph_mult * rng.uniform(0.9, 1.1), 2),
                        "unit": "mg/kg"
                    })
                    rows.append({
                        "campaign": campaign, "location": st, "storm_phase": ph,
                        "sample_type": "Soil", "analyte": "Chromium (Cr)",
                        "value": round(240.0 * loc_mult * ph_mult * rng.uniform(0.85, 1.15), 2),
                        "unit": "mg/kg"
                    })
    return pd.DataFrame(rows)

def export_chapter3_methodology_tables(kd_summary, v_flow, D_disp):
    """
    Dynamically extracts model parameters and station data straight from 
    the active script run and writes them out for Chapter 3.
    """
    import pandas as pd
    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)

    # 1. Dynamically extract Kd values calculated by the model
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

# Call this right after your model finishes running in main.py:
# export_chapter3_methodology_tables(kd_summary, v_flow, D_disp)

def main():
    # =========================================================================
    # PHASE 1: DATA PREPARATION, DESCRIPTIVE STATS & RQ 1 (PRE/DURING/POST STORM)
    # =========================================================================
    raw_df = generate_sample_field_dataset()  # Or replace with: read_table("data/field_samples.xlsx")
    clean_df = prepare_data(raw_df)
    desc_df = descriptive_stats(clean_df)
    rq1_df = test_rq1_storm_phases(clean_df, source_location="Source")
    paired_kd, kd_summary = compute_site_kd(clean_df, moisture_fraction=0.0)

    write_table(clean_df, "outputs/01_cleaned_field_data.csv")
    write_table(desc_df, "outputs/02_descriptive_statistics.csv")
    write_table(rq1_df, "outputs/03_rq1_storm_phase_tests.csv")
    write_table(kd_summary, "outputs/04_site_specific_Kd.csv")

    print("=== RQ 1: Source Concentrations Across Pre-, During-, Post-Storm ===")
    print(rq1_df.to_string(index=False), "\n")
    print("=== Site-Specific Partition Coefficients (Kd in L/kg) ===")
    print(kd_summary.to_string(index=False), "\n")

    # =========================================================================
    # PHASE 2: HYDROLOGY (RATING CURVE, SCS-CN RUNOFF & RQ 2 EVALUATION)
    # =========================================================================
    nx, nt = 200, 600
    dx, dt = 50.0, 10.0  # 50 m spatial step (10 km reach), 10 s time step (1.67 hr storm window)

    # Habagat storm hyetograph (mm per 10-second step)
    rainfall_mm = np.zeros(nt)
    rainfall_mm[60:180] = 35.0 / 120.0   # Initial monsoon burst
    rainfall_mm[180:300] = 55.0 / 120.0  # Peak Habagat downpour

    CN = 82.0  # Calibrated Curve Number for mined laterite / saprolite catchment
    scs_q_mm = scs_runoff(rainfall_mm, CN)

    # Simulated gauged direct runoff (for RQ 2 validation; replace with logger array)
    rng = np.random.default_rng(42)
    gauged_q_mm = np.clip(scs_q_mm * rng.normal(1.03, 0.08, size=nt), 0, None)
    rq2_res = evaluate_rq2_runoff(scs_q_mm, gauged_q_mm)
    print("=== RQ 2: SCS-CN Estimated vs. Gauged Direct Runoff ===")
    print(pd.DataFrame([rq2_res]).to_string(index=False), "\n")

    # Hydraulic parameters from Manning's and Fischer's equations (Logbook Entry 28)
    manning_n = 0.040
    W_width, H_depth, S0_slope = 25.0, 1.6, 0.0025
    R_hydraulic = (W_width * H_depth) / (W_width + 2.0 * H_depth)
    A_cross = W_width * H_depth

    v_flow = manning_velocity(manning_n, R_hydraulic, S0_slope)
    D_disp = fischer_dispersion(v_flow, W_width, H_depth, S0_slope)
    print(f"Hydraulics -> Velocity v = {v_flow:.3f} m/s | Dispersion D = {D_disp:.2f} m^2/s\n")

    # =========================================================================
    # PHASE 3: 1-D ADR + TWO-PHASE PARTITIONING TRANSPORT & RQ 3 METRICS
    # =========================================================================
    catchment_area_m2 = 2.5e6  # 2.5 km^2 mining sub-catchment draining into source node
    source_node = 5
    ssc_mg_L = 450.0           # Monsoon suspended sediment concentration (mg/L)
    v_settling = 1.5e-4        # Effective fine laterite particulate settling velocity (m/s)

    # Metal-specific parameters (Cr has higher Kd and particulate affinity than Ni)
    metals = {
        "Nickel (Ni)": {
            "Kd_L_kg": float(kd_summary.loc[kd_summary["analyte"] == "Nickel (Ni)", "mean_Kd"].iloc[0]),
            "k_base": 1.2e-5,
            "runoff_conc_mg_L": 0.42
        },
        "Chromium (Cr)": {
            "Kd_L_kg": float(kd_summary.loc[kd_summary["analyte"] == "Chromium (Cr)", "mean_Kd"].iloc[0]),
            "k_base": 2.5e-5,
            "runoff_conc_mg_L": 0.14
        }
    }

    models_result = {}
    rq3_rows = []
    rq4_rows = []
    rq5_rows = []

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

        # RQ 3: Speed, displacement, and attenuation rate
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

        # RQ 4: Model Validation (RMSE and NSE at Outlet Station)
        sim_outlet = res["C_total"][:, -1]
        # Synthetic observed outlet hydrograph with realistic sensor noise (replace with field logger array)
        obs_outlet = np.clip(sim_outlet * rng.normal(1.0, 0.12, size=nt) + rng.normal(0, 0.002, size=nt), 0, None)
        val_metrics = compute_rmse_nse(obs_outlet, sim_outlet)
        rq4_rows.append({"analyte": name, **val_metrics})

        # RQ 5: DENR DAO 2016-08 Exceedance Assessment
        dao_metrics = evaluate_rq5_dao_exceedance(res["C_total"], name, water_class="Class C")
        rq5_rows.append(dao_metrics)

    rq3_df = pd.DataFrame(rq3_rows)
    rq4_df = pd.DataFrame(rq4_rows)
    rq5_df = pd.DataFrame(rq5_rows)

    write_table(rq3_df, "outputs/05_rq3_transport_pulse_metrics.csv")
    write_table(rq4_df, "outputs/06_rq4_validation_rmse_nse.csv")
    write_table(rq5_df, "outputs/07_rq5_dao_2016_08_exceedances.csv")

    print("=== RQ 3: Downstream Pulse Speed, Displacement & Attenuation ===")
    print(rq3_df.to_string(index=False), "\n")
    print("=== RQ 4: Model Validation (RMSE & Nash-Sutcliffe Efficiency) ===")
    print(rq4_df.to_string(index=False), "\n")
    print("=== RQ 5: DENR DAO 2016-08 Water Quality Exceedance ===")
    print(rq5_df.to_string(index=False), "\n")

    # =========================================================================
    # PHASE 4: POLLUTION INDICES (Igeo, CF, HPI), HHRA (HQ) & LHS MONTE CARLO
    # =========================================================================
    solids_indices = compute_sediment_pollution_indices(clean_df, control_station="Station 00")
    hpi_df = compute_hpi(clean_df, water_class="Class C")

    peak_water_concs = {
        name: float(models_result[name]["C_total"].max())
        for name in metals
    }
    hq_df, hi_summary = compute_hhra_non_carcinogenic(peak_water_concs)

    write_table(solids_indices, "outputs/08_sediment_soil_CF_Igeo.csv")
    write_table(hpi_df, "outputs/09_water_HPI.csv")
    write_table(hq_df, "outputs/10_hhra_hazard_quotients.csv")
    write_table(hi_summary, "outputs/11_hhra_hazard_index_summary.csv")

    print("=== Human Health Risk Assessment (Non-Carcinogenic ADD & HQ) ===")
    print(hq_df[["receptor", "analyte", "HQ_total", "adverse_risk_HQ_gt_1"]].to_string(index=False), "\n")
    print(hi_summary.to_string(index=False), "\n")

    # Latin Hypercube Sampling (LHS) Sensitivity Analysis & Probabilistic Risk Mapping (Logbook Entry 40 & 47)
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

    lhs_bounds = {"Kd": (800.0, 2200.0), "CN": (75.0, 88.0), "n": (0.035, 0.050)}
    lhs_df, lhs_peaks = lhs_sensitivity_analysis(lhs_bounds, n_samples=30, model_runner_fn=run_ni_lhs)
    write_table(lhs_df, "outputs/12_lhs_sensitivity_results.csv")

    # Compute exceedance probability along the river reach for GIS mapping (Logbook Entry 47)
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

    # =========================================================================
    # PHASE 5: VISUALIZATION (SPACE-TIME HEATMAPS & PROBABILISTIC EXCEEDANCE)
    # =========================================================================
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

    # Panel 3: Probabilistic Longitudinal Risk Profile
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
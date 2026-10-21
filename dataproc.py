import re
import numpy as np
import pandas as pd
from scipy import stats

ALIASES = {
    "sampleid": "sample_id",
    "sampleno": "sample_id",
    "site_id": "location",
    "site": "location",
    "sampletype": "sample_type",
    "matrix": "sample_type",
    "element": "analyte",
    "parameter": "analyte",
    "concentration": "value",
    "activity": "value",
    "measurement": "value",
    "result": "value",
    "units": "unit",
    "sampling_date": "date",
    "sample_location": "location",
    "station": "location",
    "sampling_season": "season",
    "storm_phase": "storm_phase",
    "phase": "storm_phase",
    "campaign": "campaign",
    "detectionlimit": "detection_limit",
}

# Maps wide-format columns from SantaCruz_NiCr_Synthetic_Dataset into (sample_type, analyte, unit)
WIDE_MEASUREMENT_MAP = {
    "water_ni_mg_l": ("Water", "Nickel (Ni)", "mg/L"),
    "water_cr_mg_l": ("Water", "Chromium (Cr)", "mg/L"),
    "sediment_ni_mg_kg_dw": ("Sediment", "Nickel (Ni)", "mg/kg"),
    "sediment_cr_mg_kg_dw": ("Sediment", "Chromium (Cr)", "mg/kg"),
    "sediment_ni_mg_kg": ("Sediment", "Nickel (Ni)", "mg/kg"),
    "sediment_cr_mg_kg": ("Sediment", "Chromium (Cr)", "mg/kg"),
    "soil_ni_mg_kg_dw": ("Soil", "Nickel (Ni)", "mg/kg"),
    "soil_cr_mg_kg_dw": ("Soil", "Chromium (Cr)", "mg/kg"),
    "soil_ni_mg_kg": ("Soil", "Nickel (Ni)", "mg/kg"),
    "soil_cr_mg_kg": ("Soil", "Chromium (Cr)", "mg/kg"),
}

SITE_COORDS_DEFAULT = {
    "s0": (120.025000, 15.699722),
    "station 00": (120.025000, 15.699722),
    "s1": (120.000000, 15.699444),
    "source": (120.000000, 15.699444),
    "near-field": (119.989000, 15.704200),
    "s2": (119.964500, 15.713500),
    "mid-field": (119.964500, 15.713500),
    "lower reach": (119.937874, 15.719390),
    "s3": (119.892000, 15.714300),
    "outlet": (119.892000, 15.714300),
}

UNIT_FACTORS = {
    "mg/l": (1.0, "mg/L"),
    "ppm": (1.0, "mg/L"),
    "ug/l": (1e-3, "mg/L"),
    "µg/l": (1e-3, "mg/L"),
    "ppb": (1e-3, "mg/L"),
    "mg/kg": (1.0, "mg/kg"),
    "mg/kg dw": (1.0, "mg/kg"),
    "ug/kg": (1e-3, "mg/kg"),
    "µg/kg": (1e-3, "mg/kg"),
    "g/kg": (1e3, "mg/kg"),
}

def prepare_data(df):
    """Clean column names, reshape wide format if detected, populate coordinates, and standardize units."""
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    df = df.rename(columns={c: ALIASES[c] for c in df.columns if c in ALIASES})

    # Automatically reshape wide-format spreadsheets/CSVs into long format
    wide_cols = [c for c in df.columns if c in WIDE_MEASUREMENT_MAP]
    if wide_cols:
        id_vars = [c for c in df.columns if c not in wide_cols]
        melted_parts = []
        for col in wide_cols:
            s_type, analyte_name, unit_str = WIDE_MEASUREMENT_MAP[col]
            part = df[id_vars].copy()
            part["sample_type"] = s_type
            part["analyte"] = analyte_name
            part["value"] = df[col]
            part["unit"] = unit_str
            melted_parts.append(part)
        df = pd.concat(melted_parts, ignore_index=True)

    if df.columns.duplicated().any():
        raise ValueError("Multiple columns map to the same name; resolve them manually.")

    # Auto-fill missing coordinates and campaign if absent
    if "location" in df.columns:
        loc_key = df["location"].astype(str).str.strip().str.lower()
        if "longitude" not in df.columns:
            df["longitude"] = loc_key.map(lambda k: SITE_COORDS_DEFAULT.get(k, (120.0, 15.7))[0])
        if "latitude" not in df.columns:
            df["latitude"] = loc_key.map(lambda k: SITE_COORDS_DEFAULT.get(k, (120.0, 15.7))[1])
    if "campaign" not in df.columns:
        df["campaign"] = 1

    if "value" in df.columns:
        raw = df["value"].astype("string").str.strip()
        df["below_detection_limit"] = raw.str.match(r"^<\s*\d", na=False)
        numeric = raw.str.extract(
            r"^\s*<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            expand=False
        )
        df["value_numeric"] = pd.to_numeric(numeric, errors="coerce")
        if "detection_limit" in df.columns:
            dl_num = pd.to_numeric(df["detection_limit"], errors="coerce")
            df.loc[df["below_detection_limit"] & dl_num.notna(), "value_numeric"] = dl_num / 2.0
        df["missing_value"] = raw.isna() | raw.eq("")

    if "unit" in df.columns and "value_numeric" in df.columns:
        unit_clean = df["unit"].astype(str).str.strip().str.lower()
        factors = unit_clean.map(lambda u: UNIT_FACTORS.get(u, (1.0, u))[0])
        std_units = unit_clean.map(lambda u: UNIT_FACTORS.get(u, (1.0, u))[1])
        df["unit_converted_flag"] = factors != 1.0
        df["value_numeric"] = df["value_numeric"] * factors
        df["unit_standardized"] = std_units

    return df

def descriptive_stats(df):
    """Calculate mean, median, std, min, max, Q1, Q3, and IQR across groups."""
    if "value_numeric" not in df.columns:
        raise ValueError("Expected value_numeric; call prepare_data() first.")
    groups = [
        c for c in ("analyte", "sample_type", "location", "storm_phase", "season")
        if c in df.columns
    ]
    values = df.dropna(subset=["value_numeric"])
    aggregation = {
        "n": "count",
        "mean": "mean",
        "median": "median",
        "std": "std",
        "minimum": "min",
        "maximum": "max",
        "q1": lambda s: s.quantile(0.25),
        "q3": lambda s: s.quantile(0.75),
    }
    if groups:
        out = values.groupby(groups, dropna=False)["value_numeric"].agg(
            **aggregation
        ).reset_index()
    else:
        out = pd.DataFrame([{
            name: func(values["value_numeric"]) if callable(func) else getattr(values["value_numeric"], func)()
            for name, func in aggregation.items()
        }])
    out["IQR"] = out["q3"] - out["q1"]
    return out

def test_rq1_storm_phases(df, source_location="S1"):
    """
    RQ 1 & Hypothesis 1:
    Test significant differences in Ni and Cr concentrations in surface water,
    sediment, and soil at the source (S1 or Source) across pre-storm, during-storm, and post-storm.
    """
    sub = df.dropna(subset=["value_numeric"]).copy()
    if "location" in sub.columns and source_location is not None:
        loc_lower = sub["location"].astype(str).str.lower()
        if (loc_lower == source_location.lower()).any():
            sub = sub[loc_lower == source_location.lower()]
        elif (loc_lower == "s1").any():
            sub = sub[loc_lower == "s1"]
        elif (loc_lower == "source").any():
            sub = sub[loc_lower == "source"]

    results = []
    for (analyte, matrix), grp in sub.groupby(["analyte", "sample_type"]):
        phases = [g["value_numeric"].values for _, g in grp.groupby("storm_phase") if len(g) > 0]
        if len(phases) < 2:
            continue

        normal = all(len(p) >= 3 and stats.shapiro(p).pvalue > 0.05 for p in phases)
        homoscedastic = len(phases) >= 2 and stats.levene(*phases).pvalue > 0.05

        if normal and homoscedastic:
            stat_val, p_val = stats.f_oneway(*phases)
            test_used = "One-Way ANOVA"
        else:
            stat_val, p_val = stats.kruskal(*phases)
            test_used = "Kruskal-Wallis"

        results.append({
            "analyte": analyte,
            "sample_type": matrix,
            "test_used": test_used,
            "statistic": stat_val,
            "p_value": p_val,
            "significant_H1": p_val < 0.05
        })
    return pd.DataFrame(results)

def compute_site_kd(df, moisture_fraction=0.0):
    """
    Calculate site-specific partition coefficient Kd = Cs (mg/kg dry) / Cw (mg/L)
    from paired surface water and sediment concentrations (paired by trial, phase, and site).
    """
    water = df[df["sample_type"].str.lower() == "water"].copy()
    sed = df[df["sample_type"].str.lower() == "sediment"].copy()

    sed["Cs_dry"] = sed["value_numeric"] / (1.0 - moisture_fraction)

    merge_cols = [c for c in ("analyte", "location", "storm_phase", "campaign", "trial") if c in df.columns]
    paired = pd.merge(
        sed[merge_cols + ["Cs_dry"]],
        water[merge_cols + ["value_numeric"]].rename(columns={"value_numeric": "Cw"}),
        on=merge_cols
    )
    paired = paired[paired["Cw"] > 0]
    paired["Kd_L_per_kg"] = paired["Cs_dry"] / paired["Cw"]

    summary = paired.groupby("analyte")["Kd_L_per_kg"].agg(
        n="count", mean_Kd="mean", std_Kd="std", median_Kd="median"
    ).reset_index()
    return paired, summary
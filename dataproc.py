import re
import pandas as pd

ALIASES = {
    "sampleid": "sample_id",
    "sampleno": "sample_id",
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
    "sampling_season": "season",
    "detectionlimit": "detection_limit",
}

def prepare_data(df):
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    df = df.rename(columns={c: ALIASES[c] for c in df.columns if c in ALIASES})
    if df.columns.duplicated().any():
        raise ValueError("Multiple columns map to the same name; resolve them manually.")
    if "value" in df.columns:
        raw = df["value"].astype("string").str.strip()
        df["below_detection_limit"] = raw.str.match(r"^<\s*\d", na=False)
        numeric = raw.str.extract(
            r"^\s*<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            expand=False
        )
        df["value_numeric"] = pd.to_numeric(numeric, errors="coerce")
        df["missing_value"] = raw.isna() | raw.eq("")
    return df

def descriptive_stats(df):
    if "value_numeric" not in df.columns:
        raise ValueError("Expected value_numeric; call prepare_data() first.")
    groups = [
        c for c in ("analyte", "saple_type", "location", "season")
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
        stats = values.groupby(groups, dropna=False)["value_numeric"].agg(
            **aggregation
        ).reset_index()
    else:
        stats = pd.DataFrame([{
            name: func(values["value_numeric"])
            for name, func in aggregation.items()
        }])
    stats["IQR"] = stats["q3"] - stats["q1"]
    return stats
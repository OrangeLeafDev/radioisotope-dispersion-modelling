from pathlib import Path
import pandas as pd

def read_table(path, **kw):
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path, **kw)
        case ".xlsx" | ".xls":
            return pd.read_excel(path, **kw)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")

def write_table(df, path):
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            return df.to_csv(path, index=False)
        case ".xlsx" | ".xls":
            return df.to_excel(path, index=False)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
"""Load and save tabular data in whatever format the client sends."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # pandas' default NA strings, minus "None" — a real category in many datasets
    from pandas._libs.parsers import STR_NA_VALUES
    NA_VALUES = sorted(set(STR_NA_VALUES) - {"None"})
except ImportError:  # pragma: no cover
    NA_VALUES = ["", "#N/A", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]

READERS = {
    ".csv": lambda p, **kw: pd.read_csv(p, **kw),
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".txt": lambda p, **kw: pd.read_csv(p, sep=None, engine="python", **kw),
    ".xlsx": lambda p, **kw: pd.read_excel(p, **kw),
    ".xls": lambda p, **kw: pd.read_excel(p, **kw),
    ".json": lambda p, **kw: pd.read_json(p, **kw),
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".parquet": lambda p, **kw: pd.read_parquet(p, **kw),
}


def load(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV/TSV/Excel/JSON/JSONL/Parquet based on the file extension.

    CSVs are tried with UTF-8 first, then latin-1 (common in client exports).
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(READERS))}")
    if ext in (".csv", ".tsv", ".txt"):
        kwargs.setdefault("keep_default_na", False)
        kwargs.setdefault("na_values", NA_VALUES)
        try:
            return READERS[ext](p, **kwargs)
        except UnicodeDecodeError:
            return READERS[ext](p, encoding="latin-1", **kwargs)
    return READERS[ext](p, **kwargs)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".csv":
        df.to_csv(p, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    elif ext == ".parquet":
        df.to_parquet(p, index=False)
    elif ext == ".json":
        df.to_json(p, orient="records", indent=2, date_format="iso")
    else:
        raise ValueError(f"Unsupported output type '{ext}'")
    return p

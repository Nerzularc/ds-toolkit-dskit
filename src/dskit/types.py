"""Semantic column-type inference shared by profile, clean and baseline."""
from __future__ import annotations

import re

import pandas as pd

ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|^key$|_key$|^no$|number$|^nik$)", re.I)


def looks_like_dates(s: pd.Series, sample: int = 200) -> bool:
    vals = s.dropna().astype(str).head(sample)
    if len(vals) < 5 or vals.str.len().median() < 6:
        return False
    if not vals.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}", regex=True).mean() > 0.8:
        return False
    parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def looks_numeric_text(s: pd.Series, sample: int = 300) -> bool:
    vals = s.dropna().astype(str).head(sample).str.replace(r"[,\s$€£%]|Rp\.?", "", regex=True)
    if len(vals) < 5:
        return False
    return pd.to_numeric(vals, errors="coerce").notna().mean() > 0.95


def infer_type(s: pd.Series) -> str:
    """Return one of: id, boolean, numeric, numeric_text, datetime, categorical, text, constant, empty."""
    n = len(s)
    nn = s.dropna()
    if nn.empty:
        return "empty"
    nunique = nn.nunique()
    if nunique == 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s) or (nunique == 2 and {str(v).strip().lower() for v in nn.unique()}.issubset(
            {"0", "1", "true", "false", "yes", "no", "y", "n", "1.0", "0.0"})):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        if ID_NAME_RE.search(str(s.name)) and nunique > 0.95 * len(nn):
            return "id"
        return "numeric"
    # object / string
    if looks_like_dates(s):
        return "datetime"
    if looks_numeric_text(s):
        return "numeric_text"
    if ID_NAME_RE.search(str(s.name)) and nunique > 0.9 * len(nn):
        return "id"
    avg_len = nn.astype(str).str.len().mean()
    if nunique > 0.9 * len(nn) and n > 50:
        return "text" if avg_len > 25 else "id"
    if avg_len > 60:
        return "text"
    return "categorical"


def infer_types(df: pd.DataFrame) -> dict[str, str]:
    return {c: infer_type(df[c]) for c in df.columns}

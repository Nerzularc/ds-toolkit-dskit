"""Standard, auditable cleaning for messy client files.

    dskit clean raw.csv -o clean.csv --report reports/cleaning.html

Every step is logged (what changed, how many cells/rows) so the client can see
exactly what was done to their data. Imputation is intentionally NOT done here:
it belongs inside the modelling pipeline, fitted on training data only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .report import Report
from .types import infer_types


@dataclass
class Step:
    step: str
    column: str
    detail: str
    affected: int


@dataclass
class CleanResult:
    df: pd.DataFrame
    steps: list[Step] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0


def snake_case(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "column"


def parse_number_text(s: pd.Series) -> pd.Series:
    """'Rp 1,234.50' -> 1234.5 ; '1.234,50' (Indonesian/European) -> 1234.5 ; '12%' -> 12."""
    t = s.astype("string").str.strip()
    t = t.str.replace(r"(?i)rp\.?|idr|usd|\$|€|£|%|\s", "", regex=True)
    euro = t.str.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?|-?\d+,\d{1,2}").fillna(False)
    t = t.where(~euro, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    t = t.where(euro, t.str.replace(",", "", regex=False))
    t = t.replace("", pd.NA)
    return pd.to_numeric(t, errors="coerce").astype("float64")


def clean(df: pd.DataFrame, drop_constant: bool = True, dedupe: bool = True,
          unify_labels: bool = True) -> CleanResult:
    res = CleanResult(df=df.copy(), rows_before=len(df))
    d = res.df
    log = res.steps.append

    # 1. column names
    new_cols, seen = [], {}
    for c in d.columns:
        sc = snake_case(c)
        if sc in seen:
            seen[sc] += 1
            sc = f"{sc}_{seen[sc]}"
        else:
            seen[sc] = 0
        new_cols.append(sc)
    renamed = sum(a != b for a, b in zip(d.columns, new_cols))
    if renamed:
        log(Step("rename columns", "(all)", "snake_case, unique", renamed))
    d.columns = new_cols

    # 2. whitespace + empty strings
    for c in d.columns:
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            before = d[c]
            stripped = before.where(before.isna(), before.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
            changed = int((stripped.fillna("<NA>") != before.fillna("<NA>")).sum())
            empties = stripped.isin(["", "nan", "NaN", "null", "NULL", "-", "?"])
            if empties.any():
                stripped = stripped.mask(empties)
                log(Step("blank → missing", c, "empty / placeholder strings set to missing", int(empties.sum())))
            if changed:
                log(Step("trim whitespace", c, "leading/trailing/double spaces removed", changed))
            d[c] = stripped

    types = infer_types(d)

    # 3. numbers stored as text
    for c, t in types.items():
        if t == "numeric_text":
            parsed = parse_number_text(d[c])
            lost = int((d[c].notna() & parsed.isna()).sum())
            d[c] = parsed
            log(Step("text → number", c, f"currency/thousand separators removed; {lost} unparsable → missing", int(parsed.notna().sum())))

    # 4. dates stored as text
    for c, t in types.items():
        if t == "datetime" and not pd.api.types.is_datetime64_any_dtype(d[c]):
            parsed = pd.to_datetime(d[c], errors="coerce", format="mixed")
            lost = int((d[c].notna() & parsed.isna()).sum())
            d[c] = parsed
            log(Step("text → date", c, f"parsed to datetime; {lost} unparsable → missing", int(parsed.notna().sum())))

    # 5. booleans
    for c, t in types.items():
        if t == "boolean" and not pd.api.types.is_bool_dtype(d[c]) and not pd.api.types.is_numeric_dtype(d[c]):
            m = {"yes": True, "y": True, "true": True, "1": True, "no": False, "n": False, "false": False, "0": False}
            d[c] = d[c].map(lambda v: m.get(str(v).strip().lower(), pd.NA) if pd.notna(v) else pd.NA).astype("boolean")
            log(Step("text → boolean", c, "yes/no, true/false, y/n → True/False", int(d[c].notna().sum())))

    # 6. inconsistent category labels (case variants) → most frequent spelling
    if unify_labels:
        for c, t in types.items():
            if t == "categorical":
                s = d[c].dropna().astype(str)
                key = s.str.lower()
                canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
                mapped = d[c].map(lambda v: canon.get(str(v).lower(), v) if pd.notna(v) else v)
                changed = int((mapped.fillna("<NA>") != d[c].fillna("<NA>")).sum())
                if changed:
                    d[c] = mapped
                    log(Step("unify labels", c, "case variants mapped to the most common spelling", changed))

    # 7. empty & constant columns
    for c, t in list(types.items()):
        if t == "empty" or (drop_constant and t == "constant"):
            d.drop(columns=c, inplace=True)
            log(Step("drop column", c, "completely empty" if t == "empty" else "single constant value", len(d)))

    # 8. duplicates
    if dedupe:
        dups = int(d.duplicated().sum())
        if dups:
            d.drop_duplicates(inplace=True)
            d.reset_index(drop=True, inplace=True)
            log(Step("drop duplicate rows", "(rows)", "exact duplicates, first kept", dups))

    # 9. flag (don't fix) suspicious negatives — business decision for the client
    for c in d.select_dtypes(include=[np.number]).columns:
        if any(k in c for k in ("price", "amount", "charge", "revenue", "cost", "qty", "quantity", "age", "fee")):
            neg = int((d[c] < 0).sum())
            if neg:
                log(Step("flag only", c, "negative values left unchanged — confirm with client (refunds? data entry?)", neg))

    res.df = d
    res.rows_after = len(d)
    return res


def cleaning_report(res: CleanResult, source: str = "") -> Report:
    rep = Report("Data cleaning log", f"{source} · {res.rows_before:,} → {res.rows_after:,} rows")
    rep.section("Summary")
    rep.kpis([("Rows before", f"{res.rows_before:,}"), ("Rows after", f"{res.rows_after:,}"),
              ("Steps applied", str(len(res.steps))), ("Columns", str(res.df.shape[1]))])
    rep.text("Imputation of missing values was intentionally not performed: it should be fitted on training data "
             "inside the modelling pipeline to avoid leakage.")
    rep.section("Steps")
    rep.table(pd.DataFrame([s.__dict__ for s in res.steps]), max_rows=500)
    rep.section("Resulting schema")
    schema = pd.DataFrame({"column": res.df.columns, "dtype": [str(t) for t in res.df.dtypes],
                           "missing %": (res.df.isna().mean() * 100).round(1).values})
    rep.table(schema, max_rows=300)
    return rep

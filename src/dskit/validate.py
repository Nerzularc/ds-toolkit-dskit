"""Rule-based data validation ("data contracts") for incoming client files.

    dskit validate data.csv --rules rules.yaml        # exit code 1 if an error-level rule fails
    dskit validate data.csv --suggest > rules.yaml    # draft rules from a known-good file

Rules file:

    min_rows: 1000
    columns:
      customer_id: {required: true, not_null: true, unique: true, regex: '^CUST-\\d{5}$'}
      age:         {min: 17, max: 100, max_missing: 0.10, severity: warn}
      contract:    {allowed: [Month-to-month, One year, Two year]}
      signup_date: {type: date, not_after: today}
    row_checks:
      - name: total charges at least monthly charges
        expr: total_charges >= monthly_charges
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yaml

from .report import Report, badge
from .types import infer_types


@dataclass
class CheckResult:
    column: str
    check: str
    passed: bool
    failing: int
    severity: str  # error | warn
    detail: str = ""


def _to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(r"[,\s$€£]|Rp\.?", "", regex=True), errors="coerce")


def _sample(values: pd.Series, k: int = 5) -> str:
    vals = values.dropna().astype(str).unique()[:k]
    return ", ".join(v[:30] for v in vals)


def run_rules(df: pd.DataFrame, rules: dict) -> list[CheckResult]:
    out: list[CheckResult] = []
    n = len(df)
    if "min_rows" in rules:
        ok = n >= rules["min_rows"]
        out.append(CheckResult("(table)", f"min_rows ≥ {rules['min_rows']}", ok, 0 if ok else 1, "error", f"{n:,} rows"))
    if "max_duplicate_rows" in rules:
        d = int(df.duplicated().sum())
        out.append(CheckResult("(table)", f"duplicate rows ≤ {rules['max_duplicate_rows']}", d <= rules["max_duplicate_rows"], d, "error"))

    for col, spec in (rules.get("columns") or {}).items():
        sev = spec.get("severity", "error")
        if col not in df.columns:
            if spec.get("required", True):
                out.append(CheckResult(col, "column exists", False, n, "error", "column missing from file"))
            continue
        s = df[col]
        if spec.get("not_null"):
            bad = s.isna() | (s.astype(str).str.strip() == "")
            out.append(CheckResult(col, "not null", not bad.any(), int(bad.sum()), sev))
        if "max_missing" in spec:
            m = float(s.isna().mean())
            out.append(CheckResult(col, f"missing ≤ {spec['max_missing']:.0%}", m <= spec["max_missing"], int(s.isna().sum()), sev, f"{m:.1%} missing"))
        if spec.get("unique"):
            dup = s.dropna().duplicated(keep=False)
            out.append(CheckResult(col, "unique", not dup.any(), int(dup.sum()), sev, _sample(s.dropna()[dup])))
        if "allowed" in spec:
            allowed = {str(a) for a in spec["allowed"]}
            bad = s.notna() & ~s.astype(str).isin(allowed)
            out.append(CheckResult(col, "allowed values", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
        if "regex" in spec:
            pat = re.compile(spec["regex"])
            bad = s.notna() & ~s.astype(str).map(lambda v: bool(pat.fullmatch(v)))
            out.append(CheckResult(col, f"matches /{spec['regex']}/", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
        if "min" in spec or "max" in spec:
            x = _to_num(s)
            unparsable = s.notna() & x.isna() & (s.astype(str).str.strip() != "")
            if unparsable.any():
                out.append(CheckResult(col, "numeric", False, int(unparsable.sum()), sev, _sample(s[unparsable])))
            if "min" in spec:
                bad = x < spec["min"]
                out.append(CheckResult(col, f"≥ {spec['min']}", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
            if "max" in spec:
                bad = x > spec["max"]
                out.append(CheckResult(col, f"≤ {spec['max']}", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
        if spec.get("type") == "date":
            d = pd.to_datetime(s, errors="coerce", format="mixed")
            bad = s.notna() & d.isna()
            out.append(CheckResult(col, "valid date", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
            for key, op in (("not_before", lambda a, b: a < b), ("not_after", lambda a, b: a > b)):
                if key in spec:
                    ref = pd.Timestamp(date.today()) if str(spec[key]) == "today" else pd.Timestamp(spec[key])
                    bad = op(d, ref)
                    out.append(CheckResult(col, f"{key.replace('_', ' ')} {ref.date()}", not bad.any(), int(bad.sum()), sev, _sample(s[bad])))
        if "max_age_days" in spec:  # freshness: newest record must be recent
            d = pd.to_datetime(s, errors="coerce", format="mixed")
            age = (pd.Timestamp(date.today()) - d.max()).days
            out.append(CheckResult(col, f"freshness ≤ {spec['max_age_days']} days", age <= spec["max_age_days"], 0 if age <= spec["max_age_days"] else 1, sev, f"newest record {age} days old"))

    for rc in rules.get("row_checks") or []:
        sev = rc.get("severity", "error")
        try:
            tmp = df.copy()
            for c in tmp.columns:
                if not pd.api.types.is_numeric_dtype(tmp[c]):
                    conv = _to_num(tmp[c])
                    if conv.notna().mean() > 0.9:
                        tmp[c] = conv
            ok_mask = tmp.eval(rc["expr"])
            bad = ~ok_mask.fillna(False).astype(bool)
            out.append(CheckResult("(row)", rc.get("name", rc["expr"]), not bad.any(), int(bad.sum()), sev))
        except Exception as exc:  # bad expression should not crash the run
            out.append(CheckResult("(row)", rc.get("name", rc["expr"]), False, n, "error", f"expression error: {exc}"))
    return out


def suggest_rules(df: pd.DataFrame) -> dict:
    """Draft a rules file from a known-good dataset. Review before using!"""
    types = infer_types(df)
    cols: dict = {}
    for c, t in types.items():
        s = df[c]
        spec: dict = {}
        miss = float(s.isna().mean())
        if miss == 0:
            spec["not_null"] = True
        else:
            spec["max_missing"] = round(min(1.0, miss * 1.5 + 0.01), 2)
        if t == "id":
            spec["unique"] = bool(not s.dropna().duplicated().any())
            sample = s.dropna().astype(str).head(200)
            pattern = re.sub(r"\d", r"\\d", sample.iloc[0]) if len(sample) else None
            if pattern and sample.map(lambda v: bool(re.fullmatch(pattern, v))).all():
                spec["regex"] = "^" + pattern + "$"
        elif t in ("numeric", "numeric_text"):
            x = _to_num(s).dropna()
            if len(x):
                spec["min"] = float(0 if x.min() >= 0 else x.quantile(0.001))
                spec["max"] = float(round(x.quantile(0.999) * 1.5, 2))
        elif t in ("categorical", "boolean") and s.nunique() <= 30:
            spec["allowed"] = sorted(map(str, s.dropna().unique()))
        elif t == "datetime":
            spec["type"] = "date"
            spec["not_after"] = "today"
        cols[c] = spec
    return {"min_rows": max(1, int(len(df) * 0.5)), "max_duplicate_rows": 0, "columns": cols}


def validate(df: pd.DataFrame, rules: dict, source: str = "") -> tuple[list[CheckResult], Report]:
    results = run_rules(df, rules)
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warns = [r for r in results if not r.passed and r.severity != "error"]
    rep = Report("Data validation report", f"{source} · {len(df):,} rows · {len(results)} checks")
    rep.section("Result")
    rep.kpis([("Checks", str(len(results))), ("Passed", str(sum(r.passed for r in results))),
              ("Errors", str(len(errors))), ("Warnings", str(len(warns)))])
    if errors:
        rep.callout(f"<b>FAILED</b> — {len(errors)} error-level check(s) failed. Do not load this file into production.", "bad")
    elif warns:
        rep.callout("<b>PASSED with warnings</b> — review the items below.", "warn")
    else:
        rep.callout("<b>PASSED</b> — every check passed.", "good")
    rep.section("All checks")
    table = pd.DataFrame([{
        "status": badge("good", "pass") if r.passed else badge("bad" if r.severity == "error" else "warn", "fail"),
        "column": r.column, "check": r.check, "failing rows": r.failing, "examples / detail": r.detail}
        for r in sorted(results, key=lambda r: (r.passed, r.severity != "error"))])
    rep.table(table, max_rows=500, html_cols=["status"])
    return results, rep


def load_rules(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_rules(rules: dict) -> str:
    return yaml.safe_dump(rules, sort_keys=False, allow_unicode=True, width=120)

"""Exploratory data profiling: one command, one client-ready HTML report.

    dskit profile data.csv --target churn -o reports/profile.html
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .report import INK2, PALETTE, Report, badge, style_axes
from .types import infer_types


@dataclass
class Alert:
    level: str  # good | warn | bad
    column: str
    message: str


@dataclass
class Profile:
    n_rows: int
    n_cols: int
    types: dict[str, str]
    missing_pct: dict[str, float]
    duplicate_rows: int
    alerts: list[Alert] = field(default_factory=list)
    columns: pd.DataFrame | None = None

    @property
    def quality_score(self) -> int:
        """0-100 heuristic: starts at 100, minus penalties per alert."""
        penalty = sum(8 if a.level == "bad" else 3 for a in self.alerts)
        return max(0, 100 - penalty)


def _column_summary(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        t = types[c]
        row = {"column": c, "type": t, "missing %": round(s.isna().mean() * 100, 1),
               "unique": int(s.nunique(dropna=True))}
        if t == "numeric":
            row.update({"mean / top": round(float(s.mean()), 3), "min": float(s.min()), "max": float(s.max())})
        elif t == "datetime":
            d = pd.to_datetime(s, errors="coerce", format="mixed")
            row.update({"mean / top": "", "min": str(d.min())[:10], "max": str(d.max())[:10]})
        else:
            top = s.mode(dropna=True)
            row.update({"mean / top": str(top.iloc[0])[:30] if len(top) else "", "min": "", "max": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def detect_alerts(df: pd.DataFrame, types: dict[str, str], target: str | None = None) -> list[Alert]:
    alerts: list[Alert] = []
    n = len(df)
    dups = int(df.duplicated().sum())
    if dups:
        alerts.append(Alert("bad" if dups / n > 0.01 else "warn", "(rows)", f"{dups:,} duplicate rows ({dups / n:.1%})."))
    for c in df.columns:
        s, t = df[c], types[c]
        miss = s.isna().mean()
        if t == "empty":
            alerts.append(Alert("bad", c, "Column is completely empty — drop it."))
            continue
        if miss > 0.5:
            alerts.append(Alert("bad", c, f"{miss:.0%} missing — consider dropping or a missing indicator."))
        elif miss > 0.05:
            alerts.append(Alert("warn", c, f"{miss:.0%} missing — needs an imputation strategy."))
        if t == "constant":
            alerts.append(Alert("warn", c, "Only one distinct value — carries no information."))
        if t == "categorical" and s.nunique() > 50:
            alerts.append(Alert("warn", c, f"High cardinality ({s.nunique():,} categories) — group rare levels or target-encode."))
        if t == "numeric_text":
            alerts.append(Alert("warn", c, "Numbers stored as text (currency symbols / separators?) — convert to numeric."))
        if t == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            alerts.append(Alert("warn", c, "Dates stored as text — parse to datetime."))
        if t == "categorical":
            raw = s.dropna().astype(str)
            if raw.str.strip().nunique() < raw.nunique() or raw.str.lower().str.strip().nunique() < raw.nunique():
                alerts.append(Alert("warn", c, "Inconsistent labels (case / whitespace variants of the same value)."))
        if t == "numeric":
            x = s.dropna().astype(float)
            if len(x) > 20:
                skew = float(x.skew())
                if abs(skew) > 2:
                    alerts.append(Alert("warn", c, f"Highly skewed (skew {skew:.1f}) — consider a log transform."))
                q1, q3 = np.percentile(x, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    out = ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean()
                    if out > 0.01:
                        alerts.append(Alert("warn", c, f"{out:.1%} extreme outliers (beyond 3×IQR)."))
                if (x < 0).any() and any(k in c.lower() for k in ("price", "amount", "qty", "quantity", "age", "revenue", "sales", "charge", "cost", "fee", "income", "salary", "tenure")):
                    alerts.append(Alert("bad", c, f"{int((x < 0).sum())} negative values in a column that should be ≥ 0."))
    if target and target in df.columns:
        vc = df[target].value_counts(normalize=True)
        if types.get(target) in ("categorical", "boolean") and len(vc) and vc.iloc[-1] < 0.1:
            alerts.append(Alert("warn", target, f"Imbalanced target: minority class is {vc.iloc[-1]:.1%} — use stratified CV and PR-AUC."))
        # leakage: a feature that almost perfectly predicts the target
        y = df[target]
        if pd.api.types.is_numeric_dtype(y) or types.get(target) == "boolean":
            if pd.api.types.is_numeric_dtype(y) and types.get(target) != "boolean":
                yn = y.astype(float)
            else:
                yn = pd.to_numeric(y.astype(str).str.strip().str.lower().map(
                    {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0, "1.0": 1, "0.0": 0}), errors="coerce")
            for c in df.columns:
                if c == target or types[c] != "numeric":
                    continue
                r = df[c].corr(yn)
                if pd.notna(r) and abs(r) > 0.95:
                    alerts.append(Alert("bad", c, f"Correlation {r:.2f} with target — possible data leakage."))
    return alerts


# ---------------------------------------------------------------- charts
def _numeric_grid(df: pd.DataFrame, cols: list[str]):
    cols = cols[:12]
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        x = df[c].dropna().astype(float)
        ax.hist(x, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
        ax.set_title(c, fontsize=9)
        ax.axvline(x.median(), color=INK2, linewidth=0.8)
        style_axes(ax)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _categorical_grid(df: pd.DataFrame, cols: list[str]):
    cols = cols[:8]
    ncol = 2
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        vc = df[c].astype("string").fillna("(missing)").value_counts().head(8)[::-1]
        ax.barh([str(i)[:22] for i in vc.index], vc.values, color=PALETTE[0], height=0.6)
        ax.set_title(c, fontsize=9)
        ax.tick_params(axis="y", labelsize=7.5, colors=INK2, length=0)
        style_axes(ax, "x")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _corr_heatmap(corr: pd.DataFrame):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(9.5, 1.2 + 0.55 * n), min(8, 1 + 0.5 * n)))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(n), corr.index, fontsize=7.5)
    if n <= 14:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for s in ax.spines.values():
        s.set_visible(False)
    return fig


def _missing_chart(miss: pd.Series):
    miss = miss[miss > 0].sort_values()[-20:]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(miss) + 1.0))
    ax.barh(miss.index.astype(str), miss.values * 100, color=PALETTE[1], height=0.5)
    for i, v in enumerate(miss.values):
        ax.text(v * 100 + 0.5, i, f"{v:.1%}", va="center", fontsize=7.5)
    ax.set_xlabel("% missing")
    ax.set_xlim(0, min(100, miss.max() * 100 * 1.25 + 2))
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    style_axes(ax, "x")
    return fig


# ---------------------------------------------------------------- main
def profile(df: pd.DataFrame, target: str | None = None, title: str = "Data profile",
            source: str = "") -> tuple[Profile, Report]:
    types = infer_types(df)
    miss = df.isna().mean()
    alerts = detect_alerts(df, types, target)
    prof = Profile(n_rows=len(df), n_cols=df.shape[1], types=types, missing_pct=miss.round(4).to_dict(),
                   duplicate_rows=int(df.duplicated().sum()), alerts=alerts, columns=_column_summary(df, types))

    rep = Report(title, f"{source} · {len(df):,} rows × {df.shape[1]} columns")
    rep.section("Overview")
    type_counts = pd.Series(types).value_counts()
    rep.kpis([
        ("Rows", f"{len(df):,}"), ("Columns", f"{df.shape[1]}"),
        ("Missing cells", f"{df.isna().mean().mean():.1%}"), ("Duplicate rows", f"{prof.duplicate_rows:,}"),
        ("Quality score", f"{prof.quality_score}/100", "heuristic: 100 minus alert penalties"),
        ("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB"),
    ])
    rep.text("Column types: " + ", ".join(f"{k} {v}" for k, v in type_counts.items()) + ".")

    rep.section(f"Data-quality alerts ({len(alerts)})")
    if alerts:
        order = {"bad": 0, "warn": 1, "good": 2}
        adf = pd.DataFrame([{"severity": badge(a.level, "fix" if a.level == "bad" else "review"),
                             "column": a.column, "finding": a.message} for a in sorted(alerts, key=lambda a: order[a.level])])
        rep.table(adf, max_rows=100, html_cols=["severity"])
    else:
        rep.callout("No data-quality issues detected.", "good")

    rep.section("Columns")
    rep.table(prof.columns, max_rows=200)

    num_cols = [c for c, t in types.items() if t == "numeric"]
    cat_cols = [c for c, t in types.items() if t in ("categorical", "boolean")]
    if num_cols:
        rep.section("Numeric distributions")
        rep.figure(_numeric_grid(df, num_cols), "Histograms (30 bins); vertical line = median. First 12 numeric columns.")
        rep.table(df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(3).reset_index(names="column"))
    if cat_cols:
        rep.section("Categorical distributions")
        rep.figure(_categorical_grid(df, cat_cols), "Top 8 values per column (first 8 categorical columns).")
    if miss.sum() > 0:
        rep.section("Missing values")
        rep.figure(_missing_chart(miss), "Share of missing values per column.")
    if len(num_cols) >= 2:
        rep.section("Correlations")
        corr = df[num_cols].corr(numeric_only=True)
        rep.figure(_corr_heatmap(corr), "Pearson correlation (blue = positive, red = negative).")
        cols_ = list(corr.columns)
        pairs = pd.DataFrame([{"feature a": cols_[i], "feature b": cols_[j], "r": corr.iloc[i, j]}
                              for i in range(len(cols_)) for j in range(i + 1, len(cols_))]).dropna()
        pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(10)
        rep.subheading("Strongest pairs")
        rep.table(pairs.round(3))

    if target and target in df.columns:
        rep.section(f"Target: {target}")
        y = df[target]
        if types[target] in ("categorical", "boolean") or y.nunique() <= 10:
            vc = y.value_counts(dropna=False)
            fig, ax = plt.subplots(figsize=(6, 0.4 * len(vc) + 0.8))
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=PALETTE[0], height=0.6)
            for i, v in enumerate(vc.values[::-1]):
                ax.text(v, i, f"  {v:,} ({v / len(y):.1%})", va="center", fontsize=8)
            style_axes(ax, "x")
            rep.figure(fig, "Class balance.")
        else:
            fig, ax = plt.subplots(figsize=(6, 2.4))
            ax.hist(y.dropna(), bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
            style_axes(ax)
            rep.figure(fig, "Target distribution.")
        if types[target] == "boolean":
            yn = y.astype(str).str.strip().str.lower().map({"yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
                                                             "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0})
        elif types[target] == "categorical":
            yn = y.astype("category").cat.codes
        else:
            yn = pd.to_numeric(y, errors="coerce")
        assoc = {c: abs(df[c].corr(yn)) for c in num_cols if c != target}
        if assoc:
            top = pd.Series(assoc).dropna().sort_values(ascending=False).head(10)
            rep.subheading("Numeric features most associated with the target (|Pearson r|)")
            rep.table(top.round(3).rename_axis("feature").rename("abs r").reset_index())

    rep.section("Sample rows")
    rep.table(df.head(10))
    return prof, rep

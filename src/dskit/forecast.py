"""Time-series forecasting with rolling-origin backtests — no black boxes.

    dskit forecast sales.csv --date date --value orders --horizon 30 -o reports/forecast.html

Models compared on the same backtest windows:
  naive · seasonal naive · moving average · trend + seasonality regression
The winner is chosen by MAE on the backtests; prediction intervals come from the
empirical distribution of backtest errors (so they reflect real performance).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .report import INK2, PALETTE, Report, style_axes

SEASON = {"D": 7, "W": 52, "MS": 12, "M": 12}


@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    best: str
    freq: str
    filled: int


def infer_freq(idx: pd.DatetimeIndex) -> str:
    step = pd.Series(idx).diff().dt.days.median()
    if step <= 1.5:
        return "D"
    if step <= 8:
        return "W"
    return "MS"


def prepare_series(df: pd.DataFrame, date: str, value: str, freq: str | None = None, agg: str = "sum") -> tuple[pd.Series, str, int]:
    d = df[[date, value]].copy()
    d[date] = pd.to_datetime(d[date], errors="coerce", format="mixed")
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna(subset=[date])
    s = d.groupby(date)[value].agg(agg).sort_index()
    freq = freq or infer_freq(s.index)
    if freq == "W":
        s = s.resample("W-MON").agg(agg)
    elif freq in ("M", "MS"):
        freq = "MS"
        s = s.resample("MS").agg(agg)
    full = pd.date_range(s.index.min(), s.index.max(), freq=s.index.freq or ("W-MON" if freq == "W" else freq))
    missing = len(full.difference(s.index))
    s = s.reindex(full).interpolate(limit_direction="both")
    s.name = value
    return s, freq, missing


# ------------------------------------------------------------------ models
def _features(idx: pd.DatetimeIndex, t0: pd.Timestamp, freq: str) -> pd.DataFrame:
    t = ((idx - t0).days.values / 365.25).astype(float)
    X = pd.DataFrame({"t": t}, index=idx)
    doy = idx.dayofyear.values
    for k in (1, 2):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    if freq == "D":
        for dow in range(6):
            X[f"dow{dow}"] = (idx.dayofweek == dow).astype(float)
        X["month_end"] = (idx.day >= 25).astype(float)  # payday effect (common in Indonesia)
    if freq == "MS":
        for m in range(1, 12):
            X[f"m{m}"] = (idx.month == m).astype(float)
    return X


def predict_model(name: str, train: pd.Series, horizon_idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    m = SEASON.get(freq, 7)
    h = len(horizon_idx)
    if name == "Naive (last value)":
        return np.repeat(train.iloc[-1], h)
    if name == "Seasonal naive":
        last = train.iloc[-m:].to_numpy()
        return np.array([last[i % m] for i in range(h)])
    if name == "Moving average":
        w = {"D": 28, "W": 8, "MS": 6}.get(freq, 8)
        return np.repeat(train.iloc[-w:].mean(), h)
    if name == "Trend + seasonality":
        t0 = train.index[0]
        X = _features(train.index, t0, freq)
        y = np.log1p(train.clip(lower=0)) if (train > 0).all() else train
        model = Ridge(alpha=0.3).fit(X, y)
        p = model.predict(_features(horizon_idx, t0, freq))
        return np.expm1(p) if (train > 0).all() else p
    raise ValueError(name)


MODELS = ["Naive (last value)", "Seasonal naive", "Moving average", "Trend + seasonality"]


def backtest(s: pd.Series, horizon: int, freq: str, folds: int = 4) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, errors = [], {m: [] for m in MODELS}
    n = len(s)
    min_train = max(2 * SEASON.get(freq, 7), n // 3)
    for k in range(folds, 0, -1):
        cut = n - k * horizon
        if cut < min_train:
            continue
        train, test = s.iloc[:cut], s.iloc[cut:cut + horizon]
        for m in MODELS:
            p = predict_model(m, train, test.index, freq)
            e = test.to_numpy() - p
            errors[m].extend((e / np.maximum(np.abs(p), 1e-9)).tolist())  # relative errors for intervals
            denom = np.abs(test.to_numpy())
            rows.append({"model": m, "fold_start": test.index[0].date(), "mae": np.mean(np.abs(e)),
                         "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "mape": float(np.mean(np.abs(e) / np.where(denom == 0, np.nan, denom)))})
    if not rows:
        raise ValueError("Series too short for backtesting — need at least ~3× the horizon.")
    bt = pd.DataFrame(rows)
    return bt, {m: np.array(v) for m, v in errors.items()}


def run_forecast(df: pd.DataFrame, date: str, value: str, horizon: int = 30, freq: str | None = None,
                 agg: str = "sum", level: float = 0.8) -> ForecastResult:
    s, freq, filled = prepare_series(df, date, value, freq, agg)
    bt, rel_err = backtest(s, horizon, freq)
    summary = bt.groupby("model")[["mae", "rmse", "mape"]].mean()
    snaive = summary.loc["Seasonal naive", "mae"]
    summary["mase_vs_seasonal_naive"] = summary["mae"] / snaive
    summary = summary.sort_values("mae").reset_index()
    best = summary.iloc[0]["model"]
    step = s.index.freq or pd.tseries.frequencies.to_offset("W-MON" if freq == "W" else freq)
    fidx = pd.date_range(s.index[-1] + step, periods=horizon, freq=step)
    point = predict_model(best, s, fidx, freq)
    lo_q, hi_q = np.nanquantile(rel_err[best], [(1 - level) / 2, 1 - (1 - level) / 2])
    fc = pd.DataFrame({"date": fidx, "forecast": point, "lower": point * (1 + lo_q), "upper": point * (1 + hi_q)})
    return ForecastResult(history=s, forecast=fc, backtest=summary, best=best, freq=freq, filled=filled)


# ------------------------------------------------------------------ report
def forecast_report(r: ForecastResult, source: str = "", level: float = 0.8) -> Report:
    unit = {"D": "days", "W": "weeks", "MS": "months"}[r.freq]
    h = len(r.forecast)
    rep = Report(f"Forecast: {r.history.name}", f"{source} · {len(r.history):,} {unit} of history · horizon {h} {unit}")
    best = r.backtest.iloc[0]
    rep.section("Summary")
    ly_dates = r.forecast["date"] - pd.DateOffset(years=1)
    ly = r.history.reindex(ly_dates)
    ly_total = f"{ly.sum():,.0f}" if ly.notna().all() else "—"
    rep.kpis([("Best model", r.best), ("Backtest MAPE", f"{best['mape']:.1%}"),
              ("vs seasonal naive", f"{(1 - best['mase_vs_seasonal_naive']):+.0%}", "error reduction"),
              (f"Next {h} {unit} total", f"{r.forecast['forecast'].sum():,.0f}"),
              ("Same period last year", ly_total)])
    if r.filled:
        rep.callout(f"{r.filled} missing {unit} were filled by linear interpolation before modelling.", "warn")
    rep.section("Forecast")
    hist = r.history.iloc[-min(len(r.history), max(4 * h, 120)):]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(hist.index, hist.values, color=INK2, linewidth=1.2, label="Actual")
    ax.plot(r.forecast["date"], r.forecast["forecast"], color=PALETTE[0], linewidth=2, label=f"Forecast ({r.best})")
    ax.fill_between(r.forecast["date"], r.forecast["lower"], r.forecast["upper"], color=PALETTE[0], alpha=0.15,
                    linewidth=0, label=f"{level:.0%} interval")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    style_axes(ax, "y")
    rep.figure(fig, f"Interval derived from {r.best} errors in rolling backtests.")
    rep.table(r.forecast.assign(date=r.forecast["date"].dt.date).round(1), max_rows=60)
    rep.section("Model comparison (rolling-origin backtest)")
    rep.text(f"Each model was trained on data up to a cut-off and asked to forecast the next {h} {unit}, "
             f"repeated over several cut-offs. MASE < 1 means better than simply repeating last season.")
    rep.table(r.backtest.round(3))
    return rep

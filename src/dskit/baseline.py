"""Honest ML baselines in one command: classification or regression, detected
automatically, with cross-validation, a hold-out test, explainability and a
model card the client can read.

    dskit baseline data.csv --target churn -o reports/churn_baseline.html --model-out models/churn.joblib
    dskit predict models/churn.joblib new_customers.csv -o predictions.csv
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clean import clean
from .report import INK2, PALETTE, Report, style_axes
from .types import infer_types

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class BaselineResult:
    task: str
    target: str
    features: dict[str, list[str]]
    leaderboard: pd.DataFrame
    best_name: str
    holdout: dict[str, float]
    model: Pipeline
    importance: pd.DataFrame
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ prep
def prepare(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]], str, dict]:
    """Clean, drop unusable columns, engineer date features, detect the task."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {', '.join(map(str, df.columns))}")
    target_raw = target
    res = clean(df, drop_constant=True)
    d = res.df
    from .clean import snake_case

    target = snake_case(target)
    d = d[d[target].notna()].reset_index(drop=True)
    y = d.pop(target)
    types = infer_types(d)
    dropped = {c: t for c, t in types.items() if t in ("id", "text", "constant", "empty")}
    d = d.drop(columns=list(dropped))
    ref_dates = {}
    for c in list(types):
        if c in d.columns and pd.api.types.is_datetime64_any_dtype(d[c]):
            ref = d[c].max()
            ref_dates[c] = ref
            d[f"{c}_days_before_latest"] = (ref - d[c]).dt.days
            d[f"{c}_month"] = d[c].dt.month
            d = d.drop(columns=c)
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
    cat = [c for c in d.columns if c not in num]
    for c in cat:
        d[c] = d[c].astype("string").astype(object).where(d[c].notna(), np.nan)
    ynum = pd.to_numeric(y, errors="coerce") if not pd.api.types.is_bool_dtype(y) else y.astype(int)
    if pd.api.types.is_bool_dtype(y) or y.dtype == object or pd.api.types.is_string_dtype(y) or (ynum.nunique() <= 10 and (ynum.dropna() % 1 == 0).all()):
        task = "classification"
        y = y.astype(str)
    else:
        task = "regression"
        y = ynum.astype(float)
    meta = {"target_raw": target_raw, "dropped": dropped, "date_refs": {k: str(v) for k, v in ref_dates.items()},
            "cleaning_steps": len(res.steps)}
    return d, y, {"numeric": num, "categorical": cat}, task, meta


def make_preprocessor(features: dict[str, list[str]], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), features["numeric"]),
                              ("cat", Pipeline(cat_steps), features["categorical"])], remainder="drop")


def candidates(task: str, seed: int) -> dict[str, tuple[object, bool]]:
    if task == "classification":
        return {
            "Baseline (majority class)": (DummyClassifier(strategy="prior"), False),
            "Logistic regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), True),
            "Random forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample", random_state=seed), False),
            "Gradient boosting": (HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300, early_stopping=True, random_state=seed), False),
        }
    return {
        "Baseline (mean)": (DummyRegressor(), False),
        "Ridge regression": (Ridge(alpha=1.0), True),
        "Random forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed), False),
        "Gradient boosting": (HistGradientBoostingRegressor(learning_rate=0.06, max_iter=400, early_stopping=True, random_state=seed), False),
    }


# ------------------------------------------------------------------ train
def run_baseline(df: pd.DataFrame, target: str, seed: int = 42, folds: int = 5, test_size: float = 0.2) -> BaselineResult:
    X, y, feats, task, meta = prepare(df, target)
    binary = task == "classification" and y.nunique() == 2
    if task == "classification":
        classes = sorted(y.unique())
        pos = next((c for c in classes if c.lower() in ("true", "1", "yes")), classes[-1]) if binary else None
        meta["positive_class"] = pos
    strat = y if task == "classification" and y.value_counts().min() >= folds else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed) if strat is not None else KFold(folds, shuffle=True, random_state=seed)

    if task == "classification":
        scoring = {"roc_auc": "roc_auc" if binary else "roc_auc_ovr_weighted", "f1": "f1_weighted" if not binary else "f1",
                   "accuracy": "accuracy"}
        if binary:
            scoring["pr_auc"] = "average_precision"
        primary = "roc_auc"
        ytr_fit = (ytr == pos).astype(int) if binary else ytr
    else:
        scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
        primary = "rmse"
        ytr_fit = ytr

    rows, fitted = [], {}
    for name, (est, scale) in candidates(task, seed).items():
        pipe = Pipeline([("prep", make_preprocessor(feats, scale)), ("model", est)])
        t0 = time.perf_counter()
        cvres = cross_validate(pipe, Xtr, ytr_fit, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for k in scoring:
            vals = cvres[f"test_{k}"]
            if k in ("rmse", "mae"):
                vals = -vals
            row[f"cv_{k}"] = float(np.mean(vals))
            row[f"cv_{k}_sd"] = float(np.std(vals))
        row["fit_seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
        fitted[name] = pipe
    lb = pd.DataFrame(rows)
    asc = primary in ("rmse", "mae")
    lb = lb.sort_values(f"cv_{primary}", ascending=asc).reset_index(drop=True)
    best_name = lb.iloc[0]["model"]
    best = fitted[best_name].fit(Xtr, ytr_fit)

    extra: dict = {"meta": meta, "n_train": len(Xtr), "n_test": len(Xte), "primary": primary}
    if task == "classification":
        yte_fit = (yte == pos).astype(int) if binary else yte
        if binary:
            proba = best.predict_proba(Xte)[:, 1]
            prec, rec, thr = precision_recall_curve(yte_fit, proba)
            f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            best_t = float(thr[np.argmax(f1s[:-1])]) if len(thr) else 0.5
            pred = (proba >= best_t).astype(int)
            holdout = {"roc_auc": roc_auc_score(yte_fit, proba), "pr_auc": average_precision_score(yte_fit, proba),
                       "f1_at_best_threshold": f1_score(yte_fit, pred), "accuracy": accuracy_score(yte_fit, pred),
                       "best_threshold": best_t, "positive_rate": float(yte_fit.mean())}
            extra.update(proba=proba, y_true=yte_fit.to_numpy(), pred=pred)
        else:
            pred = best.predict(Xte)
            proba = best.predict_proba(Xte)
            holdout = {"roc_auc": roc_auc_score(yte, proba, multi_class="ovr", average="weighted"),
                       "f1_weighted": f1_score(yte, pred, average="weighted"), "accuracy": accuracy_score(yte, pred)}
            extra.update(y_true=yte.to_numpy(), pred=pred, labels=list(best.classes_))
        score_name = "roc_auc"
    else:
        pred = best.predict(Xte)
        holdout = {"rmse": root_mean_squared_error(yte, pred), "mae": mean_absolute_error(yte, pred),
                   "r2": r2_score(yte, pred)}
        if (yte > 0).all():
            holdout["mape"] = mean_absolute_percentage_error(yte, pred)
        extra.update(y_true=yte.to_numpy(), pred=pred)
        score_name = "neg_root_mean_squared_error"

    yte_imp = (yte == extra["meta"].get("positive_class")).astype(int) if binary else yte
    pi = permutation_importance(best, Xte, yte_imp, n_repeats=5, random_state=seed, scoring=score_name, n_jobs=1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "sd": pi.importances_std}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    return BaselineResult(task, target, feats, lb, best_name, {k: float(v) for k, v in holdout.items()}, best, imp, extra)


# ------------------------------------------------------------------ report
def _leaderboard_chart(lb: pd.DataFrame, primary: str):
    d = lb.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 0.8))
    colors = [PALETTE[0] if i == len(d) - 1 else "#9ec5f4" for i in range(len(d))]
    ax.barh(d["model"], d[f"cv_{primary}"], xerr=d[f"cv_{primary}_sd"], color=colors, height=0.55,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    for i, v in enumerate(d[f"cv_{primary}"]):
        ax.text(v, i + 0.32, f"{v:,.3f}", fontsize=8, ha="center", color=INK2)
    ax.set_xlabel(f"Cross-validated {primary.upper()} (mean ± sd)")
    ax.tick_params(axis="y", colors=INK2, length=0)
    style_axes(ax, "x")
    return fig


def _clf_curves(y, p):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, color=PALETTE[0], linewidth=2)
    axes[1].axhline(y.mean(), color="#c3c2b7", linewidth=1)
    axes[1].text(0.02, y.mean() + 0.02, "random", fontsize=7.5, color=INK2)
    axes[1].set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    for ax in axes:
        style_axes(ax, "both")
    fig.tight_layout()
    return fig


def _threshold_table(y, p) -> pd.DataFrame:
    rows = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append({"threshold": t, "flagged %": round(pred.mean() * 100, 1),
                     "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                     "true positives": tp, "false positives": fp, "missed": fn})
    return pd.DataFrame(rows)


def _reg_charts(y, pred):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].scatter(y, pred, s=8, alpha=0.4, color=PALETTE[0], edgecolor="none")
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    axes[0].plot([lo, hi], [lo, hi], color=INK2, linewidth=1)
    axes[0].set(title="Predicted vs actual (hold-out)", xlabel="Actual", ylabel="Predicted")
    axes[1].hist(y - pred, bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color=INK2, linewidth=1)
    axes[1].set(title="Residuals (actual − predicted)", xlabel="Residual")
    for ax in axes:
        style_axes(ax, "both")
    fig.tight_layout()
    return fig


def _importance_chart(imp: pd.DataFrame):
    d = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 0.8))
    ax.barh(d["feature"], d["importance"], xerr=d["sd"], color=PALETTE[0], height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 0.8})
    ax.set_xlabel("Drop in hold-out score when the feature is shuffled")
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=8)
    style_axes(ax, "x")
    return fig


def baseline_report(r: BaselineResult, source: str = "") -> Report:
    m = r.extra["meta"]
    rep = Report(f"ML baseline: predicting '{m['target_raw']}'", f"{source} · {r.task} · train {r.extra['n_train']:,} / test {r.extra['n_test']:,}")
    rep.section("Summary")
    h = r.holdout
    if r.task == "classification" and "pr_auc" in h:
        dummy = r.leaderboard[r.leaderboard["model"].str.startswith("Baseline")]["cv_roc_auc"].iloc[0]
        rep.kpis([("Best model", r.best_name), ("ROC-AUC (hold-out)", f"{h['roc_auc']:.3f}", f"majority-class baseline {dummy:.2f}"),
                  ("PR-AUC", f"{h['pr_auc']:.3f}", f"positive rate {h['positive_rate']:.1%}"),
                  ("F1 @ best threshold", f"{h['f1_at_best_threshold']:.3f}", f"threshold {h['best_threshold']:.2f}")])
        verdict = "good" if h["roc_auc"] >= 0.75 else "warn" if h["roc_auc"] >= 0.65 else "bad"
        rep.callout(f"The {r.best_name.lower()} ranks a random positive case above a random negative one "
                    f"<b>{h['roc_auc']:.0%}</b> of the time (ROC-AUC). "
                    + ("Useful signal — worth piloting." if verdict == "good" else "Moderate signal — improve features before relying on it." if verdict == "warn" else "Weak signal — collect better features first."), verdict)
    elif r.task == "classification":
        rep.kpis([("Best model", r.best_name), ("ROC-AUC (weighted OvR)", f"{h['roc_auc']:.3f}"),
                  ("Accuracy", f"{h['accuracy']:.3f}"), ("F1 (weighted)", f"{h['f1_weighted']:.3f}")])
    else:
        dummy_rmse = r.leaderboard[r.leaderboard["model"].str.startswith("Baseline")]["cv_rmse"].iloc[0]
        kp = [("Best model", r.best_name), ("RMSE (hold-out)", f"{h['rmse']:,.2f}", f"mean baseline {dummy_rmse:,.2f}"),
              ("MAE", f"{h['mae']:,.2f}"), ("R²", f"{h['r2']:.3f}")]
        if "mape" in h:
            kp.append(("MAPE", f"{h['mape']:.1%}"))
        rep.kpis(kp)
        lvl = "good" if h["r2"] >= 0.7 else "warn" if h["r2"] >= 0.4 else "bad"
        rep.callout(f"The model explains <b>{h['r2']:.0%}</b> of the variance in hold-out data and cuts error by "
                    f"<b>{1 - h['rmse'] / dummy_rmse:.0%}</b> versus always predicting the mean.", lvl)

    rep.section("Model comparison (5-fold cross-validation on training data)")
    rep.figure(_leaderboard_chart(r.leaderboard, r.extra["primary"]), "Best model highlighted. Error bars = standard deviation across folds.")
    rep.table(r.leaderboard.round(4))

    rep.section("Hold-out evaluation")
    if r.task == "classification" and "proba" in r.extra:
        rep.figure(_clf_curves(r.extra["y_true"], r.extra["proba"]), "Evaluated once on the 20% test split never seen during model selection.")
        rep.subheading("Choosing a decision threshold")
        rep.text("Lower thresholds catch more positives (recall) at the cost of more false alarms (precision). "
                 "Pick the row that matches the cost of a missed case vs. the cost of acting on a false alarm.")
        rep.table(_threshold_table(r.extra["y_true"], r.extra["proba"]))
        cm = confusion_matrix(r.extra["y_true"], r.extra["pred"])
        rep.subheading(f"Confusion matrix at threshold {h['best_threshold']:.2f}")
        rep.table(pd.DataFrame(cm, index=["actual negative", "actual positive"], columns=["predicted negative", "predicted positive"]), index=True)
    elif r.task == "classification":
        labels = r.extra["labels"]
        cm = confusion_matrix(r.extra["y_true"], r.extra["pred"], labels=labels)
        rep.table(pd.DataFrame(cm, index=[f"actual {l}" for l in labels], columns=[f"pred {l}" for l in labels]), index=True)
    else:
        rep.figure(_reg_charts(r.extra["y_true"], r.extra["pred"]), "Points on the diagonal are perfect predictions.")

    rep.section("What drives the predictions")
    rep.figure(_importance_chart(r.importance), "Permutation importance on hold-out data (top 15). Measures predictive value, not causation.")

    rep.section("Model card")
    rep.table(pd.DataFrame([
        {"item": "Task", "value": f"{r.task} on '{m['target_raw']}'" + (f" (positive class: {m.get('positive_class')})" if m.get("positive_class") else "")},
        {"item": "Features used", "value": f"{len(r.features['numeric'])} numeric, {len(r.features['categorical'])} categorical"},
        {"item": "Columns excluded", "value": ", ".join(f"{k} ({v})" for k, v in m["dropped"].items()) or "none"},
        {"item": "Preprocessing", "value": "auto-clean; median / mode imputation with missing indicators; one-hot (rare levels grouped); scaling for linear models"},
        {"item": "Validation", "value": f"{len(r.leaderboard)} models × 5-fold CV on train; single hold-out test (20%)"},
        {"item": "Intended use", "value": "Baseline to judge whether the problem is predictable and which features matter — not a production model."},
        {"item": "Known limitations", "value": "Random (not time-based) split; no hyper-parameter tuning; check for leakage in top features; "
                                               "evaluate fairness across customer groups before deployment."},
    ]))
    return rep


def save_model(r: BaselineResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": r.model, "task": r.task, "target": r.target, "features": r.features,
                 "meta": r.extra["meta"], "holdout": r.holdout, "model_name": r.best_name}, path)
    path.with_suffix(".json").write_text(json.dumps({"model": r.best_name, "task": r.task, "holdout": r.holdout,
                                                     "leaderboard": r.leaderboard.to_dict("records")}, indent=2, default=str))
    return path


def predict(model_path: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    from .clean import snake_case

    res = clean(df, drop_constant=False, dedupe=False)
    d = res.df
    for c, ref in bundle["meta"].get("date_refs", {}).items():
        if c in d.columns:
            dt = pd.to_datetime(d[c], errors="coerce", format="mixed")
            d[f"{c}_days_before_latest"] = (pd.Timestamp(ref) - dt).dt.days
            d[f"{c}_month"] = dt.dt.month
    cols = bundle["features"]["numeric"] + bundle["features"]["categorical"]
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    X = d[cols].copy()
    for c in bundle["features"]["categorical"]:
        X[c] = X[c].astype("string").astype(object).where(X[c].notna(), np.nan)
    out = df.copy()
    pipe = bundle["pipeline"]
    if bundle["task"] == "classification" and bundle["meta"].get("positive_class") is not None:
        out[f"prob_{snake_case(bundle['meta']['target_raw'])}"] = pipe.predict_proba(X)[:, 1]
    else:
        out["prediction"] = pipe.predict(X)
    return out

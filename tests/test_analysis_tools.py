import numpy as np
import pandas as pd
import pytest

from dskit.abtest import analyze, compare, holm, sample_size, srm_check
from dskit.baseline import baseline_report, predict, run_baseline, save_model
from dskit.forecast import prepare_series, run_forecast
from dskit.scaffold import new_project
from dskit.segment import rfm, rfm_report


# ---------------------------------------------------------------- baseline
def test_classification_baseline_beats_dummy(churn_df, tmp_path):
    r = run_baseline(churn_df, "churn")
    assert r.task == "classification"
    assert r.holdout["roc_auc"] > 0.7
    dummy = r.leaderboard[r.leaderboard["model"].str.startswith("Baseline")]["cv_roc_auc"].iloc[0]
    assert r.leaderboard["cv_roc_auc"].iloc[0] > dummy + 0.15
    assert "customer_id" in r.extra["meta"]["dropped"]          # IDs never used as features
    assert r.importance["feature"].iloc[0] in {"contract", "tenure_months", "signup_date_days_before_latest"}
    assert "Model card" in baseline_report(r).render()
    path = save_model(r, tmp_path / "m.joblib")
    preds = predict(path, churn_df.head(50))
    col = [c for c in preds.columns if c.startswith("prob_")][0]
    assert preds[col].between(0, 1).all() and len(preds) == 50


def test_regression_baseline(house_df):
    r = run_baseline(house_df, "price_idr")
    assert r.task == "regression"
    assert r.holdout["r2"] > 0.8
    assert "listing_id" in r.extra["meta"]["dropped"]


def test_unknown_target_raises(churn_df):
    with pytest.raises(ValueError, match="not found"):
        run_baseline(churn_df, "nope")


# ---------------------------------------------------------------- forecast
def test_prepare_series_fills_gaps(sales_df):
    s, freq, filled = prepare_series(sales_df, "date", "orders")
    assert freq == "D" and filled == 9 and s.isna().sum() == 0


def test_forecast_beats_seasonal_naive(sales_df):
    r = run_forecast(sales_df, "date", "orders", horizon=28)
    assert len(r.forecast) == 28
    assert (r.forecast["lower"] <= r.forecast["forecast"]).all() and (r.forecast["forecast"] <= r.forecast["upper"]).all()
    best = r.backtest.iloc[0]
    assert best["mase_vs_seasonal_naive"] < 1


def test_weekly_aggregation(sales_df):
    r = run_forecast(sales_df, "date", "orders", horizon=8, freq="W")
    assert r.freq == "W" and len(r.forecast) == 8


# ---------------------------------------------------------------- A/B testing
def test_sample_size_matches_reference():
    # Standard calculators give ~39.4k per arm for 4% baseline, +10% relative, alpha .05, power .8
    assert 39_000 < sample_size(0.04, 0.10) < 40_000


def test_compare_detects_real_lift_and_null():
    rng = np.random.default_rng(0)
    a = pd.Series((rng.random(40000) < 0.05).astype(int))
    b = pd.Series((rng.random(40000) < 0.06).astype(int))
    r = compare(a, b)
    assert r["significant"] and r["prob_treatment_better"] > 0.99
    lo, hi = r["rel_ci"]
    assert lo < 0.2 < hi
    same = compare(a, pd.Series((rng.random(40000) < 0.05).astype(int)))
    assert same["rel_ci"][0] < 0 < same["rel_ci"][1]


def test_srm_and_holm():
    assert srm_check(pd.Series([5000, 5050])) > 0.05
    assert srm_check(pd.Series([5000, 5600])) < 0.001
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_analyze_end_to_end(ab_df):
    res, rep = analyze(ab_df, "group", "converted", segment="device", guardrails=["revenue"])
    assert res["control_name"] == "control"
    assert res["recommendation"] in {"good", "warn", "bad"}
    assert set(res["segments"]["segment"]) == {"mobile", "desktop"}
    assert "Sample ratio mismatch" in rep.render()


def test_analyze_flags_broken_split(ab_df):
    broken = pd.concat([ab_df, ab_df[ab_df["group"] == "treatment"].head(3000)])
    res, _ = analyze(broken, "group", "converted")
    assert res["recommendation"] == "bad" and res["srm_p"] < 0.001


# ---------------------------------------------------------------- RFM
def test_rfm_segments(tx_df):
    g = rfm(tx_df, "customer_id", "order_date", "amount")
    assert len(g) == tx_df["customer_id"].nunique()
    assert g[["R", "F", "M"]].isin([1, 2, 3, 4, 5]).all().all()
    assert (g["segment"] != "Other").mean() > 0.95
    champs = g[g["segment"] == "Champions"]
    lost = g[g["segment"] == "Lost"]
    assert champs["monetary"].mean() > lost["monetary"].mean()
    assert champs["recency_days"].mean() < lost["recency_days"].mean()
    assert "Recency × frequency" in rfm_report(g).render()


# ---------------------------------------------------------------- scaffold
def test_new_project(tmp_path):
    p = new_project("Acme Churn", client="Acme", goal="Reduce churn", root=tmp_path)
    for rel in ["README.md", ".gitignore", "docs/project_brief.md", "docs/weekly_update.md", "docs/handover_checklist.md",
                "rules/data_contract.yaml", "notebooks/01_exploration.ipynb", "src/acme_churn/__init__.py", "data/raw/.gitkeep"]:
        assert (p / rel).exists(), rel
    assert "data/raw/" in (p / ".gitignore").read_text()
    assert "Acme" in (p / "README.md").read_text()
    with pytest.raises(FileExistsError):
        new_project("Acme Churn", root=tmp_path)

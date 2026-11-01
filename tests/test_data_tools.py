import pandas as pd

from dskit.clean import clean, parse_number_text, snake_case
from dskit.io import load
from dskit.profile import profile
from dskit.report import Report, fmt_num
from dskit.types import infer_type
from dskit.validate import run_rules, suggest_rules, validate


# ---------------------------------------------------------------- io / types
def test_load_keeps_none_as_a_category(churn_csv):
    df = load(churn_csv)
    assert "None" in set(df["internet_service"].dropna())


def test_load_rejects_unknown_extension(tmp_path):
    p = tmp_path / "x.foo"
    p.write_text("a")
    try:
        load(p)
    except ValueError as e:
        assert "Unsupported" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_type_inference():
    assert infer_type(pd.Series(["Rp 1,200.00", "Rp 3,400.50"] * 10, name="total")) == "numeric_text"
    assert infer_type(pd.Series(["2026-01-01", "2026-02-03"] * 10, name="d")) == "datetime"
    assert infer_type(pd.Series(["Yes", "No"] * 10, name="churn")) == "boolean"
    assert infer_type(pd.Series([f"ID-{i}" for i in range(100)], name="customer_id")) == "id"
    assert infer_type(pd.Series(["a"] * 10, name="c")) == "constant"
    assert infer_type(pd.Series([None] * 5, name="e", dtype=object)) == "empty"


# ---------------------------------------------------------------- profile
def test_profile_detects_planted_problems(churn_csv):
    prof, rep = profile(load(churn_csv), target="churn")
    msgs = {(a.column, a.message.split(" ")[0]) for a in prof.alerts}
    cols = {a.column for a in prof.alerts}
    assert {"(rows)", "city", "total_charges", "country", "monthly_charges", "age", "signup_date"} <= cols
    assert any(c == "monthly_charges" and m.isdigit() for c, m in msgs)  # "N negative values"
    assert 0 < prof.quality_score < 100
    html = rep.render()
    assert "Data-quality alerts" in html and "data:image/png;base64" in html


def test_profile_flags_leakage():
    df = pd.DataFrame({"x": range(200), "y": [i * 2 + 1 for i in range(200)], "noise": [i % 7 for i in range(200)]})
    prof, _ = profile(df, target="y")
    assert any("leakage" in a.message for a in prof.alerts if a.column == "x")


# ---------------------------------------------------------------- validate
def test_validation_catches_contract_breaks(churn_df):
    rules = {"min_rows": 100, "max_duplicate_rows": 0,
             "columns": {"customer_id": {"unique": True, "regex": r"^CUST-\d{5}$"},
                         "monthly_charges": {"min": 0},
                         "contract": {"allowed": ["Month-to-month", "One year", "Two year"]},
                         "missing_col": {"required": True}},
             "row_checks": [{"name": "bad expr", "expr": "nonexistent > 1"}]}
    res = {(r.column, r.check): r for r in run_rules(churn_df, rules)}
    assert not res[("(table)", "duplicate rows ≤ 0")].passed
    assert not res[("customer_id", "unique")].passed
    assert not res[("monthly_charges", "≥ 0")].passed
    assert res[("contract", "allowed values")].passed
    assert not res[("missing_col", "column exists")].passed
    assert "expression error" in res[("(row)", "bad expr")].detail


def test_suggested_rules_pass_on_their_own_data(house_df):
    rules = suggest_rules(house_df)
    results, _ = validate(house_df, rules)
    assert all(r.passed for r in results), [r for r in results if not r.passed]


# ---------------------------------------------------------------- clean
def test_parse_number_text_handles_local_formats():
    out = parse_number_text(pd.Series(["Rp 1,234.50", "1.234,50", "12%", " ", "abc", "-5"]))
    assert out.tolist()[:3] == [1234.5, 1234.5, 12.0]
    assert out.isna().tolist()[3:5] == [True, True]
    assert out.iloc[5] == -5


def test_snake_case():
    assert snake_case("Total Charges (Rp)") == "total_charges_rp"
    assert snake_case("customerID") == "customer_id"


def test_clean_fixes_and_logs(churn_df):
    res = clean(churn_df)
    d = res.df
    assert res.rows_after == res.rows_before - churn_df.duplicated().sum()
    assert pd.api.types.is_float_dtype(d["total_charges"])
    assert pd.api.types.is_datetime64_any_dtype(d["signup_date"])
    assert set(d["city"].dropna().unique()) <= {"Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", "Semarang"}
    assert "country" not in d.columns
    assert (d["monthly_charges"] < 0).any(), "negatives must be flagged, not silently changed"
    steps = {s.step for s in res.steps}
    assert {"text → number", "text → date", "unify labels", "drop duplicate rows", "flag only"} <= steps


def test_clean_does_not_mutate_input(churn_df):
    before = churn_df.copy()
    clean(churn_df)
    pd.testing.assert_frame_equal(before, churn_df)


# ---------------------------------------------------------------- report
def test_report_escapes_untrusted_text(tmp_path):
    rep = Report("<script>x</script>")
    rep.section("S").text("<b>hi</b>").table(pd.DataFrame({"<c>": ["<img src=x>"]}))
    html = rep.render()
    assert "<script>x</script>" not in html and "&lt;img src=x&gt;" in html
    assert rep.save(tmp_path / "r.html").exists()


def test_fmt_num():
    assert fmt_num(6.0) == "6"
    assert fmt_num(2.6) == "2.6"
    assert fmt_num(3_915_484_700.0) == "3.92B"
    assert fmt_num(None) == "—"

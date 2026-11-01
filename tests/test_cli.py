from dskit.cli import main


def test_cli_profile_clean_validate(churn_csv, tmp_path, capsys):
    out = tmp_path / "p.html"
    assert main(["profile", str(churn_csv), "--target", "churn", "-o", str(out)]) == 0
    assert out.exists()
    clean_csv = tmp_path / "clean.csv"
    assert main(["clean", str(churn_csv), "-o", str(clean_csv), "--report", str(tmp_path / "c.html")]) == 0
    assert clean_csv.exists()
    rules = tmp_path / "rules.yaml"
    rules.write_text("columns:\n  monthly_charges: {min: 0}\n")
    assert main(["validate", str(churn_csv), "--rules", str(rules), "-o", str(tmp_path / "v.html")]) == 1  # negatives → fail
    assert main(["validate", str(churn_csv), "--suggest"]) == 0
    assert "columns:" in capsys.readouterr().out


def test_cli_samplesize(capsys):
    assert main(["samplesize", "--baseline", "0.05", "--mde", "0.2", "--daily-traffic", "1000"]) == 0
    assert "users per group" in capsys.readouterr().out


def test_cli_reports_missing_file(capsys):
    assert main(["profile", "does_not_exist.csv"]) == 2
    assert "✕" in capsys.readouterr().err

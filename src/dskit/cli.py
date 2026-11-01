"""Command-line interface: `dskit <command> --help` for details."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _out(path: str | None, default: str) -> Path:
    return Path(path or default)


def cmd_profile(a):
    from .io import load
    from .profile import profile

    df = load(a.file)
    prof, rep = profile(df, target=a.target, source=Path(a.file).name)
    path = rep.save(_out(a.output, f"reports/{Path(a.file).stem}_profile.html"))
    print(f"✓ {len(df):,} rows · {len(prof.alerts)} alerts · quality score {prof.quality_score}/100 → {path}")


def cmd_validate(a):
    from .io import load
    from .validate import dump_rules, load_rules, suggest_rules, validate

    df = load(a.file)
    if a.suggest:
        sys.stdout.write(dump_rules(suggest_rules(df)))
        return 0
    if not a.rules:
        sys.exit("Provide --rules rules.yaml or use --suggest to draft one.")
    results, rep = validate(df, load_rules(a.rules), source=Path(a.file).name)
    path = rep.save(_out(a.output, f"reports/{Path(a.file).stem}_validation.html"))
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warns = [r for r in results if not r.passed and r.severity != "error"]
    for r in errors + warns:
        print(f"  {'✕' if r.severity == 'error' else '!'} {r.column}: {r.check} — {r.failing:,} rows {r.detail[:60]}")
    print(f"{'✕ FAILED' if errors else '✓ PASSED'}: {len(results) - len(errors) - len(warns)}/{len(results)} checks passed → {path}")
    return 1 if errors else 0


def cmd_clean(a):
    from .clean import clean, cleaning_report
    from .io import load, save

    df = load(a.file)
    res = clean(df, drop_constant=not a.keep_constant, dedupe=not a.keep_duplicates)
    out = save(res.df, _out(a.output, f"data/processed/{Path(a.file).stem}_clean.csv"))
    rep = cleaning_report(res, source=Path(a.file).name).save(_out(a.report, f"reports/{Path(a.file).stem}_cleaning.html"))
    for s in res.steps:
        print(f"  • {s.step:<20} {s.column:<22} {s.affected:>8,}  {s.detail}")
    print(f"✓ {res.rows_before:,} → {res.rows_after:,} rows → {out} (log: {rep})")


def cmd_baseline(a):
    from .baseline import baseline_report, run_baseline, save_model
    from .io import load

    df = load(a.file)
    r = run_baseline(df, a.target, seed=a.seed)
    rep = baseline_report(r, source=Path(a.file).name).save(_out(a.output, f"reports/{Path(a.file).stem}_baseline.html"))
    print(r.leaderboard.drop(columns=[c for c in r.leaderboard.columns if c.endswith("_sd")]).round(4).to_string(index=False))
    print(f"✓ {r.task}: best = {r.best_name} · hold-out " + ", ".join(f"{k}={v:.4g}" for k, v in r.holdout.items()) + f" → {rep}")
    if a.model_out:
        print(f"  model saved → {save_model(r, a.model_out)}")


def cmd_predict(a):
    from .baseline import predict
    from .io import load, save

    out = predict(a.model, load(a.file))
    print(f"✓ {len(out):,} predictions → {save(out, _out(a.output, 'predictions.csv'))}")


def cmd_forecast(a):
    from .forecast import forecast_report, run_forecast
    from .io import load, save

    r = run_forecast(load(a.file), a.date, a.value, horizon=a.horizon, freq=a.freq, agg=a.agg, level=a.level)
    rep = forecast_report(r, source=Path(a.file).name, level=a.level).save(_out(a.output, f"reports/{Path(a.file).stem}_forecast.html"))
    if a.csv:
        save(r.forecast, a.csv)
    print(r.backtest.round(3).to_string(index=False))
    print(f"✓ best = {r.best} · next {a.horizon} total {r.forecast['forecast'].sum():,.0f} → {rep}")


def cmd_abtest(a):
    from .abtest import analyze
    from .io import load

    res, rep = analyze(load(a.file), a.group, a.metric, control=a.control, segment=a.segment,
                       guardrails=a.guardrail, alpha=a.alpha, source=Path(a.file).name)
    path = rep.save(_out(a.output, f"reports/{Path(a.file).stem}_abtest.html"))
    lo, hi = res["rel_ci"]
    print(f"{res['control_name']}: {res['control']:.4g} · {res['treatment_name']}: {res['treatment']:.4g} · "
          f"lift {res['rel_lift']:+.1%} [{lo:+.1%}, {hi:+.1%}] · p={res['p_value']:.4f} · "
          f"P(better)={res['prob_treatment_better']:.1%} · SRM p={res['srm_p']:.3f}")
    print(f"✓ recommendation: {res['recommendation'].upper()} → {path}")


def cmd_samplesize(a):
    from .abtest import sample_size

    n = sample_size(a.baseline, a.mde, a.alpha, a.power)
    print(f"{n:,} users per group ({2 * n:,} total) to detect a {a.mde:.0%} relative lift on a {a.baseline:.2%} baseline "
          f"(alpha={a.alpha}, power={a.power}).")
    if a.daily_traffic:
        print(f"≈ {2 * n / a.daily_traffic:.1f} days at {a.daily_traffic:,} users/day.")


def cmd_rfm(a):
    from .io import load, save
    from .segment import rfm, rfm_report

    g = rfm(load(a.file), a.customer, a.date, a.amount, as_of=a.as_of)
    rep = rfm_report(g, source=Path(a.file).name, currency=a.currency).save(_out(a.output, f"reports/{Path(a.file).stem}_rfm.html"))
    csv = save(g, _out(a.csv, f"data/processed/{Path(a.file).stem}_rfm.csv"))
    print(g["segment"].value_counts().to_string())
    print(f"✓ {len(g):,} customers → {rep} · customer list → {csv}")


def cmd_new(a):
    from .scaffold import new_project

    p = new_project(a.name, client=a.client, goal=a.goal, owner=a.owner, root=a.root)
    print(f"✓ project created at {p}\n  next: cd {p} && git init && cat docs/project_brief.md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dskit", description="Data science toolkit for client work: profile, validate, clean, model, forecast, test, segment.")
    p.add_argument("--version", action="version", version=f"dskit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("profile", help="EDA + data-quality report")
    s.add_argument("file"); s.add_argument("--target"); s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("validate", help="check a file against a YAML data contract (exit 1 on error)")
    s.add_argument("file"); s.add_argument("--rules"); s.add_argument("--suggest", action="store_true", help="print draft rules to stdout")
    s.add_argument("-o", "--output"); s.set_defaults(func=cmd_validate)

    s = sub.add_parser("clean", help="standard auditable cleaning")
    s.add_argument("file"); s.add_argument("-o", "--output"); s.add_argument("--report")
    s.add_argument("--keep-constant", action="store_true"); s.add_argument("--keep-duplicates", action="store_true")
    s.set_defaults(func=cmd_clean)

    s = sub.add_parser("baseline", help="auto ML baseline (classification/regression) + model card")
    s.add_argument("file"); s.add_argument("--target", required=True); s.add_argument("-o", "--output")
    s.add_argument("--model-out"); s.add_argument("--seed", type=int, default=42); s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("predict", help="score a new file with a saved baseline model")
    s.add_argument("model"); s.add_argument("file"); s.add_argument("-o", "--output"); s.set_defaults(func=cmd_predict)

    s = sub.add_parser("forecast", help="time-series forecast with backtests")
    s.add_argument("file"); s.add_argument("--date", required=True); s.add_argument("--value", required=True)
    s.add_argument("--horizon", type=int, default=30); s.add_argument("--freq", choices=["D", "W", "MS"])
    s.add_argument("--agg", default="sum", choices=["sum", "mean"]); s.add_argument("--level", type=float, default=0.8)
    s.add_argument("-o", "--output"); s.add_argument("--csv", help="also write the forecast table"); s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("abtest", help="analyse an A/B test")
    s.add_argument("file"); s.add_argument("--group", required=True); s.add_argument("--metric", required=True)
    s.add_argument("--control"); s.add_argument("--segment"); s.add_argument("--guardrail", action="append")
    s.add_argument("--alpha", type=float, default=0.05); s.add_argument("-o", "--output"); s.set_defaults(func=cmd_abtest)

    s = sub.add_parser("samplesize", help="users needed for an A/B test")
    s.add_argument("--baseline", type=float, required=True, help="baseline conversion rate, e.g. 0.04")
    s.add_argument("--mde", type=float, required=True, help="relative lift to detect, e.g. 0.1")
    s.add_argument("--alpha", type=float, default=0.05); s.add_argument("--power", type=float, default=0.8)
    s.add_argument("--daily-traffic", type=int); s.set_defaults(func=cmd_samplesize)

    s = sub.add_parser("rfm", help="RFM customer segmentation from transactions")
    s.add_argument("file"); s.add_argument("--customer", required=True); s.add_argument("--date", required=True)
    s.add_argument("--amount", required=True); s.add_argument("--as-of"); s.add_argument("--currency", default="")
    s.add_argument("-o", "--output"); s.add_argument("--csv"); s.set_defaults(func=cmd_rfm)

    s = sub.add_parser("new", help="scaffold a client project (folders + remote-work docs)")
    s.add_argument("name"); s.add_argument("--client", default="Client"); s.add_argument("--goal", default="Describe the business problem.")
    s.add_argument("--owner", default="Your Name"); s.add_argument("--root", default="."); s.set_defaults(func=cmd_new)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc = args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"✕ {exc}", file=sys.stderr)
        return 2
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())

"""A/B test analysis that a product manager can act on.

    dskit abtest experiment.csv --group group --metric converted --control control --segment device
    dskit samplesize --baseline 0.04 --mde 0.10

Includes: sample-ratio-mismatch check, frequentist test with confidence
interval, Bayesian probability-to-beat-control, achieved power / MDE,
segment breakdown with Holm correction, and a plain-language recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .report import INK2, PALETTE, Report, style_axes


@dataclass
class ArmStats:
    name: str
    n: int
    mean: float
    sd: float


def sample_size(baseline: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group needed to detect a relative lift `mde_rel` on a conversion rate."""
    p1, p2 = baseline, baseline * (1 + mde_rel)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    return int(np.ceil(n))


def srm_check(counts: pd.Series, expected_split: list[float] | None = None) -> float:
    exp = np.array(expected_split or [1 / len(counts)] * len(counts)) * counts.sum()
    return float(stats.chisquare(counts.values, exp).pvalue)


def _is_binary(x: pd.Series) -> bool:
    return set(pd.unique(x.dropna())).issubset({0, 1})


def compare(a: pd.Series, b: pd.Series, alpha: float = 0.05, seed: int = 0) -> dict:
    """b = treatment vs a = control."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    binary = _is_binary(a) and _is_binary(b)
    ma, mb = a.mean(), b.mean()
    diff = mb - ma
    if binary:
        se = np.sqrt(ma * (1 - ma) / len(a) + mb * (1 - mb) / len(b))
        pooled = (a.sum() + b.sum()) / (len(a) + len(b))
        se0 = np.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
        z = diff / se0 if se0 > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        test = "two-proportion z-test"
        rng = np.random.default_rng(seed)
        pa = rng.beta(1 + a.sum(), 1 + len(a) - a.sum(), 40000)
        pb = rng.beta(1 + b.sum(), 1 + len(b) - b.sum(), 40000)
        rel_post = pb / pa - 1
    else:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = stats.ttest_ind(b, a, equal_var=False)
        p = float(t.pvalue)
        test = "Welch t-test"
        rng = np.random.default_rng(seed)
        ia = rng.integers(0, len(a), (2000, len(a)))
        ib = rng.integers(0, len(b), (2000, len(b)))
        rel_post = b.to_numpy()[ib].mean(1) / a.to_numpy()[ia].mean(1) - 1  # bootstrap
    zc = stats.norm.ppf(1 - alpha / 2)
    lo, hi = diff - zc * se, diff + zc * se
    # power achieved for the observed sample at the observed baseline: smallest detectable relative lift
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(0.8)
    mde_abs = (za + zb) * se
    return {
        "test": test, "binary": binary, "n_control": len(a), "n_treatment": len(b), "control": ma, "treatment": mb,
        "abs_lift": diff, "abs_ci": (lo, hi), "rel_lift": diff / ma if ma else np.nan,
        "rel_ci": (lo / ma, hi / ma) if ma else (np.nan, np.nan), "p_value": float(p),
        "prob_treatment_better": float((rel_post > 0).mean()), "rel_lift_samples": rel_post,
        "mde_rel_80": mde_abs / ma if ma else np.nan, "significant": p < alpha,
    }


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def recommendation(r: dict, srm_p: float, guardrail_ok: bool = True) -> tuple[str, str]:
    if srm_p < 0.001:
        return "bad", ("<b>Do not trust these results.</b> The traffic split is significantly different from the plan "
                       f"(SRM p = {srm_p:.1e}) — assignment or logging is broken. Fix and rerun.")
    lo, hi = r["rel_ci"]
    if r["significant"] and r["abs_lift"] > 0:
        caveat = ""
        if abs(r["rel_lift"]) < r["mde_rel_80"]:
            caveat = (f" Note: the test was under-powered for effects this size (80%-power MDE ±{r['mde_rel_80']:.1%}), "
                      "so the true lift is probably smaller than measured — plan on the lower end of the interval.")
        return "good", (f"<b>Ship the treatment.</b> Lift of {r['rel_lift']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}), "
                        f"p = {r['p_value']:.3f}; {r['prob_treatment_better']:.0%} probability it beats control.{caveat}")
    if r["significant"] and r["abs_lift"] < 0:
        return "bad", (f"<b>Do not ship.</b> The treatment is worse: {r['rel_lift']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}).")
    return "warn", (f"<b>No conclusive difference yet.</b> Observed {r['rel_lift']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}). "
                    f"With this sample the test could only reliably detect lifts of about ±{r['mde_rel_80']:.1%}. "
                    "Keep it running or accept that any effect is smaller than that.")


def _ci_chart(r: dict, labels=("control", "treatment")):
    fig, ax = plt.subplots(figsize=(7, 1.9))
    for i, (_lab, m, n) in enumerate([(labels[0], r["control"], r["n_control"]), (labels[1], r["treatment"], r["n_treatment"])]):
        se = np.sqrt(m * (1 - m) / n) if r["binary"] else None
        ax.errorbar(m, i, xerr=1.96 * se if se else None, fmt="o", color=PALETTE[i], ms=7, capsize=3, elinewidth=1.4)
        ax.text(m, i + 0.25, f"{m:.2%}" if r["binary"] else f"{m:,.2f}", ha="center", fontsize=8.5)
    ax.set_yticks([0, 1], labels, color=INK2)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("Conversion rate (95% CI)" if r["binary"] else "Mean (95% CI)")
    ax.tick_params(axis="y", length=0)
    style_axes(ax, "x")
    return fig


def _posterior_chart(samples: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 2.3))
    lo, hi = np.quantile(samples, [0.001, 0.999])
    s = samples[(samples > lo) & (samples < hi)] * 100
    bins = np.linspace(s.min(), s.max(), 60)
    ax.hist(s[s < 0], bins=bins, color=PALETTE[7], alpha=0.85, label="treatment worse")
    ax.hist(s[s >= 0], bins=bins, color=PALETTE[0], alpha=0.85, label="treatment better")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Relative lift (%) — plausible values given the data")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")
    style_axes(ax, "x")
    return fig


def _segment_chart(seg: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(seg) + 0.8))
    for i, row in enumerate(seg.itertuples()):
        ax.errorbar(row.rel_lift * 100, i, xerr=[[ (row.rel_lift - row.rel_ci_low) * 100], [(row.rel_ci_high - row.rel_lift) * 100]],
                    fmt="o", color=PALETTE[0], capsize=3, ms=6)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(seg)), seg["segment"].astype(str), color=INK2)
    ax.set_xlabel("Relative lift % (95% CI, unadjusted)")
    ax.tick_params(axis="y", length=0)
    style_axes(ax, "x")
    return fig


def analyze(df: pd.DataFrame, group: str, metric: str, control: str | None = None, segment: str | None = None,
            alpha: float = 0.05, guardrails: list[str] | None = None, source: str = "") -> tuple[dict, Report]:
    groups = df[group].astype(str)
    names = sorted(groups.unique())
    if len(names) != 2:
        raise ValueError(f"Expected exactly 2 groups in '{group}', found {names}")
    control = control or next((g for g in names if g.lower() in ("control", "a", "ctrl", "baseline")), names[0])
    treatment = next(g for g in names if g != control)
    a, b = df.loc[groups == control, metric], df.loc[groups == treatment, metric]
    r = compare(a, b, alpha)
    srm_p = srm_check(groups.value_counts().reindex([control, treatment]))
    level, rec = recommendation(r, srm_p)

    rep = Report(f"A/B test: {metric}", f"{source} · {control} vs {treatment} · {len(df):,} units")
    rep.section("Recommendation")
    rep.callout(rec, level)
    lo, hi = r["rel_ci"]
    rep.kpis([(f"{control}", f"{r['control']:.2%}" if r["binary"] else f"{r['control']:,.2f}", f"n = {r['n_control']:,}"),
              (f"{treatment}", f"{r['treatment']:.2%}" if r["binary"] else f"{r['treatment']:,.2f}", f"n = {r['n_treatment']:,}"),
              ("Relative lift", f"{r['rel_lift']:+.1%}", f"95% CI {lo:+.1%} … {hi:+.1%}"),
              ("p-value", f"{r['p_value']:.4f}", r["test"]),
              ("P(treatment better)", f"{r['prob_treatment_better']:.1%}", "Bayesian, flat prior" if r["binary"] else "bootstrap"),
              ("Detectable lift (80% power)", f"±{r['mde_rel_80']:.1%}")])
    rep.section("Result")
    rep.figure(_ci_chart(r, (control, treatment)))
    rep.figure(_posterior_chart(r["rel_lift_samples"]), "Share of the distribution right of zero = probability the treatment is better.")
    rep.section("Health checks")
    counts = groups.value_counts()
    rep.table(pd.DataFrame([
        {"check": "Sample ratio mismatch (expected 50/50)", "result": f"p = {srm_p:.3f}",
         "status": "OK" if srm_p >= 0.001 else "FAIL — investigate assignment"},
        {"check": "Group sizes", "result": " / ".join(f"{k}: {v:,}" for k, v in counts.items()), "status": ""},
        {"check": "Minimum detectable effect at this sample", "result": f"±{r['mde_rel_80']:.1%} relative", "status": "smaller real effects will look like noise"},
    ]))

    for g in guardrails or []:
        gr = compare(df.loc[groups == control, g], df.loc[groups == treatment, g], alpha)
        rep.subheading(f"Guardrail: {g}")
        rep.text(f"{g}: {gr['rel_lift']:+.1%} (95% CI {gr['rel_ci'][0]:+.1%} to {gr['rel_ci'][1]:+.1%}), p = {gr['p_value']:.3f}.")

    seg_df = None
    if segment:
        rows = []
        for sv, part in df.groupby(segment):
            gp = part[group].astype(str)
            sr = compare(part.loc[gp == control, metric], part.loc[gp == treatment, metric], alpha)
            rows.append({"segment": sv, "n": len(part), "control": sr["control"], "treatment": sr["treatment"],
                         "rel_lift": sr["rel_lift"], "rel_ci_low": sr["rel_ci"][0], "rel_ci_high": sr["rel_ci"][1], "p_value": sr["p_value"]})
        seg_df = pd.DataFrame(rows)
        seg_df["p_holm"] = holm(seg_df["p_value"].tolist())
        rep.section(f"By {segment}")
        rep.figure(_segment_chart(seg_df))
        rep.table(seg_df.round(4))
        rep.text("Segment results are exploratory: p-values are Holm-adjusted for multiple comparisons. "
                 "Confirm any segment-specific decision with a follow-up test.")
    out = {**{k: v for k, v in r.items() if k != "rel_lift_samples"}, "srm_p": srm_p, "control_name": control,
           "treatment_name": treatment, "recommendation": level, "segments": seg_df}
    return out, rep

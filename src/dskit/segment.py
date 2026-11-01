"""RFM customer segmentation from raw transactions.

    dskit rfm transactions.csv --customer customer_id --date order_date --amount amount -o reports/rfm.html

Recency, Frequency and Monetary value are scored 1–5 by quintile and mapped to
named segments with a suggested action for each, so marketing can use the
output directly (the per-customer CSV can be uploaded to an ad/email tool).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .report import INK2, PALETTE, SEQ_BLUE, Report, fmt_num, style_axes

# (segment, rule on R and FM scores, action)
SEGMENTS = [
    ("Champions", lambda r, fm: (r >= 4) & (fm >= 4.5), "Reward, ask for reviews/referrals, early access to launches"),
    ("Loyal", lambda r, fm: (r >= 3) & (fm >= 3.5), "Upsell higher-value products, loyalty programme"),
    ("Potential loyalists", lambda r, fm: (r >= 4) & (fm >= 2), "Membership offer, personalised recommendations"),
    ("New customers", lambda r, fm: (r >= 4) & (fm < 2), "Onboarding sequence, second-purchase incentive"),
    ("Need attention", lambda r, fm: (r == 3) & (fm >= 2), "Limited-time offer based on past purchases"),
    ("At risk", lambda r, fm: (r <= 2) & (fm >= 3.5), "Win-back campaign with strong offer, survey why they left"),
    ("About to sleep", lambda r, fm: (r == 3) & (fm < 2), "Reactivation with popular products / social proof"),
    ("Hibernating", lambda r, fm: (r <= 2) & (fm >= 2), "Low-cost reminders; exclude from expensive channels"),
    ("Lost", lambda r, fm: (r <= 2) & (fm < 2), "Suppress from paid campaigns; occasional newsletter only"),
]


def _score(x: pd.Series, reverse: bool = False) -> pd.Series:
    ranks = x.rank(method="first")
    s = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return 6 - s if reverse else s


def rfm(df: pd.DataFrame, customer: str, date: str, amount: str, as_of: str | None = None) -> pd.DataFrame:
    d = df[[customer, date, amount]].copy()
    d[date] = pd.to_datetime(d[date], errors="coerce", format="mixed")
    d[amount] = pd.to_numeric(d[amount], errors="coerce")
    d = d.dropna()
    ref = pd.Timestamp(as_of) if as_of else d[date].max() + pd.Timedelta(days=1)
    g = d.groupby(customer).agg(last=(date, "max"), first=(date, "min"), frequency=(date, "count"), monetary=(amount, "sum"))
    g["recency_days"] = (ref - g["last"]).dt.days
    g["avg_order_value"] = g["monetary"] / g["frequency"]
    g["tenure_days"] = (ref - g["first"]).dt.days
    g["R"] = _score(g["recency_days"], reverse=True)
    g["F"] = _score(g["frequency"])
    g["M"] = _score(g["monetary"])
    fm = (g["F"] + g["M"]) / 2
    g["segment"] = "Other"
    g["action"] = ""
    assigned = pd.Series(False, index=g.index)
    for name, rule, action in SEGMENTS:
        mask = rule(g["R"], fm) & ~assigned
        g.loc[mask, "segment"] = name
        g.loc[mask, "action"] = action
        assigned |= mask
    g["rfm_code"] = g["R"].astype(str) + g["F"].astype(str) + g["M"].astype(str)
    return g.reset_index().drop(columns=["first"]).rename(columns={"last": "last_order"})


def rfm_report(g: pd.DataFrame, source: str = "", currency: str = "") -> Report:
    order = [s[0] for s in SEGMENTS] + ["Other"]
    summ = (g.groupby("segment").agg(customers=("segment", "size"), revenue=("monetary", "sum"),
                                     avg_recency_days=("recency_days", "mean"), avg_orders=("frequency", "mean"),
                                     avg_order_value=("avg_order_value", "mean"))
            .reindex([o for o in order if o in g["segment"].unique()]))
    summ["% customers"] = summ["customers"] / summ["customers"].sum() * 100
    summ["% revenue"] = summ["revenue"] / summ["revenue"].sum() * 100
    actions = {s[0]: s[2] for s in SEGMENTS}
    summ["suggested action"] = [actions.get(i, "") for i in summ.index]

    rep = Report("Customer segmentation (RFM)", f"{source} · {len(g):,} customers")
    rep.section("Summary")
    top = summ.sort_values("% revenue", ascending=False)
    champ = summ.loc["Champions"] if "Champions" in summ.index else None
    risk = summ.loc["At risk"] if "At risk" in summ.index else None
    rep.kpis([("Customers", f"{len(g):,}"), ("Revenue", f"{currency}{fmt_num(float(g['monetary'].sum()))}"),
              ("Repeat customers", f"{(g['frequency'] > 1).mean():.1%}"),
              ("Champions' revenue share", f"{champ['% revenue']:.0f}%" if champ is not None else "—",
               f"from {champ['% customers']:.0f}% of customers" if champ is not None else ""),
              ("Revenue at risk", f"{currency}{fmt_num(float(risk['revenue']))}" if risk is not None else "—", "past value of 'At risk' customers")])
    rep.callout(f"<b>{top.index[0]}</b> generate the largest share of revenue ({top.iloc[0]['% revenue']:.0f}%). "
                + (f"<b>{int(risk['customers']):,} 'At risk'</b> customers used to buy often and spend a lot but haven't ordered in "
                   f"~{risk['avg_recency_days']:.0f} days — the highest-value win-back target." if risk is not None else ""), "info")

    rep.section("Segments")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.42 * len(summ) + 1), sharey=True)
    labels = summ.index[::-1]
    axes[0].barh(labels, summ["% customers"][::-1], color="#9ec5f4", height=0.6)
    axes[0].set_title("% of customers")
    axes[1].barh(labels, summ["% revenue"][::-1], color=PALETTE[0], height=0.6)
    axes[1].set_title("% of revenue")
    for ax, col in zip(axes, ["% customers", "% revenue"]):
        for i, v in enumerate(summ[col][::-1]):
            ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
        ax.tick_params(axis="y", colors=INK2, length=0)
        style_axes(ax, "x")
    fig.tight_layout()
    rep.figure(fig, "Segments compared by headcount and revenue contribution.")
    rep.table(summ.round(1).reset_index())

    rep.section("Recency × frequency map")
    pivot = g.pivot_table(index="F", columns="R", values="monetary", aggfunc="size").reindex(index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    from matplotlib.colors import ListedColormap

    ax.imshow(pivot.values, cmap=ListedColormap(SEQ_BLUE))
    for i in range(5):
        for j in range(5):
            v = int(pivot.values[i, j])
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > pivot.values.max() * 0.55 else "#0b0b0b")
    ax.set_xticks(range(5), ["1\nlong ago", "2", "3", "4", "5\nrecent"])
    ax.set_yticks(range(5), ["5 frequent", "4", "3", "2", "1 once"])
    ax.set_xlabel("Recency score")
    ax.set_ylabel("Frequency score")
    for s in ax.spines.values():
        s.set_visible(False)
    rep.figure(fig, "Number of customers per recency/frequency score.")

    rep.section("Sample of customer-level output")
    rep.table(g.sort_values("monetary", ascending=False).head(15)[["customer_id" if "customer_id" in g.columns else g.columns[0],
                                                                    "recency_days", "frequency", "monetary", "rfm_code", "segment"]].round(0))
    return rep

"""Self-contained HTML report builder shared by every dskit command.

Reports are single files (CSS inline, charts embedded as PNG) so they can be
emailed, attached to a ticket or opened offline by a client — no server needed.
"""
from __future__ import annotations

import base64
import html
import io
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (colour-blind safe on adjacent pairs) + chrome.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, WARN, BAD = "#006300", "#b5651d", "#d03b3b"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False,
})


def style_axes(ax, grid: str = "y") -> None:
    """Hairline grid behind the data, recessive axes."""
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt_num(v, digits: int | None = None) -> str:
    """Integers without decimals, small ratios with 3 decimals, large values with separators."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            v = v.item()
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        a = abs(v)
        if v.is_integer() and a < 1e9:
            return f"{int(v):,}"
        if a >= 1e9:
            return f"{v / 1e9:,.2f}B"
        if a >= 1e6:
            return f"{v / 1e6:,.2f}M"
        if digits is not None:
            return f"{v:,.{digits}f}"
        if a >= 100:
            return f"{v:,.1f}"
        out = f"{v:,.2f}" if a >= 10 else f"{v:,.3f}"
        return out.rstrip("0").rstrip(".") if "." in out else out
    return html.escape(str(v))


CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--surface:#fcfcfb;--page:#f6f6f3;--blue:#2a78d6;
--good:#006300;--warn:#b5651d;--bad:#d03b3b}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 28px 64px}
header{margin-bottom:22px}h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}.sub{color:var(--ink2);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 4px}nav a{font-size:12px;color:var(--ink2);text-decoration:none;border:1px solid var(--grid);background:var(--surface);border-radius:99px;padding:3px 10px}
section{background:var(--surface);border:1px solid rgba(11,11,11,.08);border-radius:12px;padding:20px 22px;margin:16px 0}
h2{font-size:16px;margin:0 0 10px}h3{font-size:13.5px;margin:16px 0 6px;color:var(--ink2)}
p{margin:6px 0 10px;max-width:80ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{border:1px solid var(--grid);border-radius:10px;padding:12px 14px;background:#fff}
.kpi .l{font-size:12px;color:var(--ink2)}.kpi .v{font-size:22px;font-weight:650;margin-top:2px}.kpi .n{font-size:11.5px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px}
th{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid #c3c2b7;padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:6px 8px;font-variant-numeric:tabular-nums;vertical-align:top}
td.num,th.num{text-align:right}.tablewrap{overflow-x:auto}
img{max-width:100%;display:block;margin:6px 0}
.callout{border-left:3px solid var(--blue);background:#f2f6fc;padding:10px 14px;border-radius:6px;margin:8px 0}
.callout.good{border-color:var(--good);background:#eef6ee}.callout.warn{border-color:var(--warn);background:#fbf3ea}.callout.bad{border-color:var(--bad);background:#fbeeee}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:1px 7px;border-radius:99px;border:1px solid}
.badge.good{color:var(--good);border-color:var(--good)}.badge.warn{color:var(--warn);border-color:var(--warn)}.badge.bad{color:var(--bad);border-color:var(--bad)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.grid2{grid-template-columns:1fr}}
code{background:#f0efec;padding:1px 5px;border-radius:4px;font-size:12px}
footer{color:var(--muted);font-size:11.5px;margin-top:24px}
"""

ICON = {"good": "✓", "warn": "!", "bad": "✕", "info": "i"}


class Report:
    """Collects blocks and renders one self-contained HTML file."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.sections: list[tuple[str, list[str]]] = []

    # -- structure -----------------------------------------------------------
    def section(self, heading: str) -> Report:
        self.sections.append((heading, []))
        return self

    def _add(self, block: str) -> Report:
        if not self.sections:
            self.section("Summary")
        self.sections[-1][1].append(block)
        return self

    # -- blocks --------------------------------------------------------------
    def text(self, s: str, raw: bool = False) -> Report:
        return self._add(f"<p>{s if raw else html.escape(s)}</p>")

    def subheading(self, s: str) -> Report:
        return self._add(f"<h3>{html.escape(s)}</h3>")

    def callout(self, s: str, level: str = "info") -> Report:
        icon = ICON.get(level, "i")
        return self._add(f'<div class="callout {level}"><b>{icon}</b> {s}</div>')

    def kpis(self, items: Iterable[tuple[str, str] | tuple[str, str, str]]) -> Report:
        cells = []
        for it in items:
            label, value, note = (list(it) + [""])[:3]
            cells.append(f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(str(value))}</div>'
                         f'<div class="n">{html.escape(note)}</div></div>')
        return self._add(f'<div class="kpis">{"".join(cells)}</div>')

    def table(self, df: pd.DataFrame, max_rows: int = 50, index: bool = False, html_cols: Iterable[str] = ()) -> Report:
        html_cols = set(html_cols)
        d = df.head(max_rows)
        cols = list(d.columns)
        numeric = {c for c in cols if pd.api.types.is_numeric_dtype(d[c]) and c not in html_cols}
        head = ("<th></th>" if index else "") + "".join(
            f'<th class="{"num" if c in numeric else ""}">{html.escape(str(c))}</th>' for c in cols)
        rows = []
        for idx, r in d.iterrows():
            tds = (f"<td>{html.escape(str(idx))}</td>" if index else "")
            for c in cols:
                v = r[c]
                if c in html_cols:
                    tds += f"<td>{v}</td>"
                elif c in numeric:
                    tds += f'<td class="num">{fmt_num(v)}</td>'
                else:
                    tds += f"<td>{html.escape('' if pd.isna(v) else str(v))}</td>"
            rows.append(f"<tr>{tds}</tr>")
        more = f'<p class="sub">Showing {max_rows} of {len(df):,} rows.</p>' if len(df) > max_rows else ""
        return self._add(f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{more}')

    def figure(self, fig, caption: str = "") -> Report:
        cap = f'<p class="sub">{html.escape(caption)}</p>' if caption else ""
        return self._add(f'<img alt="{html.escape(caption or "chart")}" src="{fig_to_data_uri(fig)}">{cap}')

    def html(self, raw: str) -> Report:
        return self._add(raw)

    # -- output --------------------------------------------------------------
    def render(self) -> str:
        nav = "".join(f'<a href="#s{i}">{html.escape(h)}</a>' for i, (h, _) in enumerate(self.sections))
        body = "".join(f'<section id="s{i}"><h2>{html.escape(h)}</h2>{"".join(blocks)}</section>'
                       for i, (h, blocks) in enumerate(self.sections))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f"<title>{html.escape(self.title)}</title><style>{CSS}</style></head><body><main>"
                f'<header><h1>{html.escape(self.title)}</h1><div class="sub">{html.escape(self.subtitle)}</div><nav>{nav}</nav></header>'
                f"{body}<footer>Generated by dskit on {ts}.</footer></main></body></html>")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path


def badge(level: str, text: str) -> str:
    return f'<span class="badge {level}">{ICON.get(level, "")} {html.escape(text)}</span>'

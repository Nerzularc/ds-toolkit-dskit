"""Scaffold a client project with the structure and documents remote work needs.

    dskit new acme-churn --client "Acme Telco" --goal "Reduce monthly churn by 10%"

Async remote work lives or dies on written communication, so besides code
folders the template ships a project brief, data dictionary, decision log,
weekly update template and a handover checklist.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

GITIGNORE = """# data never goes to git — share via the client's drive/bucket
data/raw/
data/interim/
data/processed/
models/
*.joblib
.env
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
"""

README = """# {title}

**Client:** {client} · **Started:** {today} · **Owner:** {owner}

## Objective
{goal}

## Deliverables
- [ ] Data-quality report (`make profile validate`)
- [ ] Baseline model / analysis with written findings
- [ ] Final report + handover (see `docs/handover_checklist.md`)

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # credentials, never committed
# put the client's files in data/raw/
make profile                      # EDA report → reports/
make validate                     # data contract check → reports/
make baseline TARGET=your_target  # baseline model → reports/ + models/
```

## Project docs
| File | Purpose |
|---|---|
| `docs/project_brief.md` | Scope, success metric, assumptions, open questions — agree this with the client first |
| `docs/data_dictionary.md` | What every column means, source and owner |
| `docs/decision_log.md` | Every non-obvious decision with date and reason |
| `docs/weekly_update.md` | Template for the async Friday update |
| `docs/handover_checklist.md` | What the client receives at the end |
"""

BRIEF = """# Project brief — {title}

_Draft {today}. Send to the client and get explicit confirmation before starting work._

## 1. Business problem
{goal}

## 2. Decision this work will inform
> Who will do what differently based on the result?

## 3. Success metric
| Metric | Current | Target | How measured |
|---|---|---|---|
| | | | |

## 4. Scope
**In scope:**
-
**Out of scope:**
-

## 5. Data
| Source | Owner (client side) | Access method | Refresh | Status |
|---|---|---|---|---|
| | | | | ☐ received |

## 6. Assumptions & risks
-

## 7. Timeline & milestones
| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Data received & profiled | Data-quality report |
| 2 | Baseline | Baseline report + findings call |
| 3–4 | Iteration | Improved model / analysis |
| 5 | Handover | Final report, code, documentation |

## 8. Communication
- Async update every Friday (`docs/weekly_update.md`), 30-min call every second week.
- Response time: within 1 business day. Time zone: Asia/Jakarta (UTC+7).
- Questions that block work are flagged in the update under **Blockers**.

## 9. Open questions for the client
1.
"""

DICTIONARY = """# Data dictionary

| Table / file | Column | Type | Meaning | Allowed values / unit | Source | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

_Tip: `dskit profile data/raw/<file>` gives you types, missing % and examples to start this table._
"""

DECISIONS = """# Decision log

Record every choice a reviewer might question later. Newest first.

| Date | Decision | Why | Alternatives considered | Agreed with |
|---|---|---|---|---|
| {today} | Project scaffolded with dskit | Standard structure for reproducibility | — | — |
"""

WEEKLY = """# Weekly update — week of YYYY-MM-DD

**Status:** 🟢 on track / 🟡 at risk / 🔴 blocked

## Done this week
-

## Key findings (with numbers)
-

## Next week
-

## Blockers / decisions needed from you
- [ ]

## Hours used: __ / __ budgeted
"""

HANDOVER = """# Handover checklist

- [ ] Final report (PDF/HTML) with executive summary, method, results, limitations
- [ ] Code in the client's repository; `README.md` explains how to rerun everything
- [ ] Environment file (`requirements.txt`) pinned; secrets documented in `.env.example`
- [ ] Data contract (`rules/data_contract.yaml`) so future files can be validated
- [ ] Model file + model card (if a model was delivered); retraining instructions
- [ ] Data dictionary complete
- [ ] Decision log up to date
- [ ] 30-minute walkthrough recorded (Loom) and shared
- [ ] Access to client systems removed / credentials rotated
- [ ] Invoice sent; testimonial requested
"""

MAKEFILE = """DATA ?= data/raw/data.csv
TARGET ?= target

profile:
\tdskit profile $(DATA) -o reports/profile.html

validate:
\tdskit validate $(DATA) --rules rules/data_contract.yaml -o reports/validation.html

suggest-rules:
\tdskit validate $(DATA) --suggest > rules/data_contract.yaml

clean:
\tdskit clean $(DATA) -o data/processed/clean.csv --report reports/cleaning.html

baseline:
\tdskit baseline $(DATA) --target $(TARGET) -o reports/baseline.html --model-out models/baseline.joblib
"""

RULES = """# Data contract — generate a draft with `make suggest-rules`, then edit by hand.
min_rows: 1
columns: {}
row_checks: []
"""

REQUIREMENTS = """dskit @ git+https://github.com/YOUR_USERNAME/ds-toolkit.git
jupyter
"""

ENV = """# Credentials and connection strings. Copy to .env — never commit .env.
DATABASE_URL=
CLIENT_BUCKET=
"""


def _notebook(title: str) -> str:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title} — exploration\n", "\n", "Goal of this notebook: …"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ["import pandas as pd\n", "from dskit.io import load\n", "from dskit.profile import profile\n", "\n",
                    "df = load('../data/raw/data.csv')\n", "prof, report = profile(df)\n", "report.save('../reports/profile.html')\n",
                    "prof.columns"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Findings\n", "- "]},
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1)


def new_project(name: str, client: str = "Client", goal: str = "Describe the business problem.",
                owner: str = "Your Name", root: str | Path = ".") -> Path:
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    base = Path(root) / slug
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"{base} already exists and is not empty")
    title = name.replace("-", " ").title()
    ctx = {"title": title, "client": client, "goal": goal, "owner": owner, "today": date.today().isoformat()}
    files = {
        "README.md": README.format(**ctx), ".gitignore": GITIGNORE, "Makefile": MAKEFILE, "requirements.txt": REQUIREMENTS,
        ".env.example": ENV, "rules/data_contract.yaml": RULES, "docs/project_brief.md": BRIEF.format(**ctx),
        "docs/data_dictionary.md": DICTIONARY, "docs/decision_log.md": DECISIONS.format(**ctx),
        "docs/weekly_update.md": WEEKLY, "docs/handover_checklist.md": HANDOVER,
        "notebooks/01_exploration.ipynb": _notebook(title),
        f"src/{slug.replace('-', '_')}/__init__.py": '"""Project code: reusable functions live here, not in notebooks."""\n',
    }
    for d in ["data/raw", "data/interim", "data/processed", "data/external", "models", "reports"]:
        (base / d).mkdir(parents=True, exist_ok=True)
        (base / d / ".gitkeep").write_text("")
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return base

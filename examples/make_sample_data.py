"""Create realistic, deliberately messy sample datasets for every dskit command.

    python examples/make_sample_data.py

Outputs (examples/data/):
  customers_churn.csv   telco-style churn data with typical client-data problems
  daily_sales.csv       2.5 years of daily sales with trend, weekly & yearly seasonality
  ab_test.csv           checkout experiment, one row per user
  transactions.csv      e-commerce orders for RFM segmentation
  house_prices.csv      regression example
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"


def churn(rng: np.random.Generator, n: int = 5000) -> pd.DataFrame:
    tenure = rng.integers(1, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    monthly = np.round(rng.normal(70, 25, n).clip(18, 140), 2)
    support_calls = rng.poisson(1.2, n)
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.4, 0.15])
    payment = rng.choice(["E-wallet", "Bank transfer", "Credit card", "Virtual account"], n)
    age = rng.normal(38, 12, n).clip(17, 85).round()
    logit = (-1.2 + 1.4 * (contract == "Month-to-month") - 0.035 * tenure + 0.012 * (monthly - 70)
             + 0.35 * support_calls + 0.5 * (internet == "Fiber") - 0.6 * (contract == "Two year"))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "signup_date": (pd.Timestamp("2026-10-01") - pd.to_timedelta(tenure * 30, unit="D")).strftime("%Y-%m-%d"),
        "age": age,
        "city": rng.choice(["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", "Semarang"], n, p=[.35, .2, .15, .12, .1, .08]),
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2),
        "support_calls": support_calls,
        "country": "Indonesia",
        "churn": np.where(churn, "Yes", "No"),
    })
    # --- typical client-data problems ---
    df.loc[rng.choice(n, 400, replace=False), "age"] = np.nan                       # missing values
    df.loc[rng.choice(n, 60, replace=False), "monthly_charges"] *= -1               # negative amounts
    idx = rng.choice(n, 250, replace=False)
    df.loc[idx, "city"] = df.loc[idx, "city"].str.upper() + " "                     # inconsistent labels
    df["total_charges"] = df["total_charges"].map(lambda v: f"Rp {v:,.2f}")         # numbers as text
    df.loc[rng.choice(n, 30, replace=False), "total_charges"] = " "
    df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)          # duplicates
    return df


def sales(rng: np.random.Generator) -> pd.DataFrame:
    d = pd.date_range("2024-05-01", "2026-10-31", freq="D")
    t = np.arange(len(d))
    weekly = np.array([0.92, 0.95, 0.97, 1.0, 1.08, 1.18, 1.1])[d.dayofweek]
    yearly = 1 + 0.18 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365.25)
    payday = np.where(d.day >= 25, 1.12, 1.0)
    y = (1200 + 1.1 * t) * weekly * yearly * payday * rng.lognormal(0, 0.06, len(d))
    df = pd.DataFrame({"date": d, "store": "Online", "orders": np.round(y).astype(int)})
    return df.drop(rng.choice(len(df), 9, replace=False))  # a few missing days


def ab_test(rng: np.random.Generator, n: int = 24000) -> pd.DataFrame:
    group = rng.choice(["control", "treatment"], n, p=[0.5, 0.5])
    device = rng.choice(["mobile", "desktop"], n, p=[0.7, 0.3])
    base = np.where(device == "mobile", 0.038, 0.052)
    lift = np.where(group == "treatment", np.where(device == "mobile", 1.16, 1.02), 1.0)
    converted = rng.random(n) < base * lift
    revenue = np.where(converted, np.round(rng.lognormal(12.4, 0.5, n), -2), 0)  # IDR
    return pd.DataFrame({"user_id": np.arange(n), "group": group, "device": device,
                         "converted": converted.astype(int), "revenue": revenue})


def transactions(rng: np.random.Generator, customers: int = 2500) -> pd.DataFrame:
    rows = []
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-31")
    for c in range(customers):
        kind = rng.choice(["loyal", "regular", "one-off", "lapsed"], p=[0.12, 0.33, 0.35, 0.2])
        n = {"loyal": rng.integers(8, 25), "regular": rng.integers(3, 8), "one-off": 1, "lapsed": rng.integers(2, 6)}[kind]
        last = end if kind != "lapsed" else end - pd.Timedelta(days=int(rng.integers(200, 400)))
        first = start + pd.Timedelta(days=int(rng.integers(0, 200)))
        if last <= first:
            last = first + pd.Timedelta(days=30)
        dates = pd.to_datetime(rng.uniform(first.value, last.value, n)).normalize()
        spend = {"loyal": 420_000, "regular": 260_000, "one-off": 180_000, "lapsed": 220_000}[kind]
        for d in dates:
            rows.append((f"C{c:05d}", d, round(float(rng.lognormal(np.log(spend), 0.45)), -2)))
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    df.insert(0, "order_id", [f"ORD{i:07d}" for i in range(len(df))])
    return df.sort_values("order_date").reset_index(drop=True)


def houses(rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    area = rng.normal(120, 45, n).clip(30, 400)
    bedrooms = np.clip(np.round(area / 40 + rng.normal(0, 0.7, n)), 1, 7)
    district = rng.choice(["Central", "North", "South", "East", "West"], n)
    dist_mult = pd.Series(district).map({"Central": 1.6, "South": 1.35, "North": 0.95, "East": 0.85, "West": 1.0}).values
    age = rng.integers(0, 40, n)
    price = (area * 14_000_000 * dist_mult * (1 - 0.006 * age) + bedrooms * 40_000_000) * rng.lognormal(0, 0.12, n)
    df = pd.DataFrame({"listing_id": np.arange(n), "district": district, "land_area_m2": area.round(1),
                       "bedrooms": bedrooms, "building_age": age, "has_garage": rng.choice([0, 1], n),
                       "price_idr": price.round(-6)})
    df.loc[rng.choice(n, 120, replace=False), "building_age"] = np.nan
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for name, df in [("customers_churn", churn(rng)), ("daily_sales", sales(rng)), ("ab_test", ab_test(rng)),
                     ("transactions", transactions(rng)), ("house_prices", houses(rng))]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(df):,} rows")

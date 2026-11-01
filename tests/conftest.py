import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import make_sample_data as msd  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def churn_df():
    return msd.churn(np.random.default_rng(1), n=1500)


@pytest.fixture(scope="session")
def sales_df():
    return msd.sales(np.random.default_rng(2))


@pytest.fixture(scope="session")
def ab_df():
    return msd.ab_test(np.random.default_rng(3), n=20000)


@pytest.fixture(scope="session")
def tx_df():
    return msd.transactions(np.random.default_rng(4), customers=600)


@pytest.fixture(scope="session")
def house_df():
    return msd.houses(np.random.default_rng(5), n=1200)


@pytest.fixture()
def churn_csv(tmp_path, churn_df):
    p = tmp_path / "churn.csv"
    churn_df.to_csv(p, index=False)
    return p


@pytest.fixture()
def tiny_df():
    return pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "x", "x"]})

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subrec.config import RecurrenceConfig  # noqa: E402
from subrec.data import exact_distinct, prepare_history  # noqa: E402
from subrec.pipeline import run_analysis  # noqa: E402
from subrec.synthetic import MONTHS, make_synthetic_history  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return make_synthetic_history(seed=0)


@pytest.fixture(scope="session")
def hist(raw):
    return prepare_history(raw)


@pytest.fixture(scope="session")
def analysis(hist):
    return run_analysis({"original": hist, "exact_distinct": exact_distinct(hist)})


@pytest.fixture(scope="session")
def M():
    return MONTHS


def status(analysis, acct, merchant, month, method, dataset="original", clf="v2"):
    pm = analysis.runs[(dataset, clf)].pm
    row = pm[(pm.account_key == acct) & (pm.merchant == merchant) & (pm.month == month)]
    assert len(row) == 1, f"expected one pair-month for {acct}/{merchant}/{month}, got {len(row)}"
    return row.iloc[0][f"status_{method}"]


def pair(analysis, acct, merchant, dataset="original", clf="v2"):
    pm = analysis.runs[(dataset, clf)].pm
    return pm[(pm.account_key == acct) & (pm.merchant == merchant)].set_index("month")


@pytest.fixture
def st(analysis):
    return lambda *a, **k: status(analysis, *a, **k)


@pytest.fixture
def pr(analysis):
    return lambda *a, **k: pair(analysis, *a, **k)

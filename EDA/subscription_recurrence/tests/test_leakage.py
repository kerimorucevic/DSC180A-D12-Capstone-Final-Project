"""Future-data leakage: M1-M3 decisions for month t must not change when later data is removed."""
import pandas as pd
import pytest

from subrec.config import PROSPECTIVE_METHODS, RecurrenceConfig
from subrec.pipeline import run_one

KEY = ["account_key", "merchant", "month"]


def _compare(hist, cutoff, clf, cfg):
    full = run_one(hist, "full", clf, cfg)
    trunc_hist = hist[hist["month"] <= cutoff]
    trunc = run_one(trunc_hist, "trunc", clf, cfg)
    # Accounts with data in the cutoff month: their coverage statuses before the cutoff are unaffected
    # by truncation (an account whose data ends earlier would get a different partial_edge month).
    keep = set(trunc_hist.loc[trunc_hist["month"] == cutoff, "account_key"])
    cols = KEY + [f"status_{m}" for m in ["M0", *PROSPECTIVE_METHODS]]
    a = full.pm[(full.pm.month < cutoff) & full.pm.account_key.isin(keep)][cols]
    b = trunc.pm[(trunc.pm.month < cutoff) & trunc.pm.account_key.isin(keep)][cols]
    j = a.merge(b, on=KEY, suffixes=("_full", "_trunc"), validate="one_to_one")
    assert len(j) == len(a) == len(b) and len(j) > 0
    feat = [f"rec_{m}" for m in PROSPECTIVE_METHODS] + [f"pot_{m}" for m in PROSPECTIVE_METHODS]
    fa = full.am[(full.am.month < cutoff) & full.am.account_key.isin(keep)].set_index(["account_key", "month"])
    fb = trunc.am[(trunc.am.month < cutoff) & trunc.am.account_key.isin(keep)].set_index(["account_key", "month"])
    return j, fa[feat + ["eligible_common"]], fb[feat + ["eligible_common"]]


@pytest.mark.parametrize("clf", ["baseline", "v2"])
@pytest.mark.parametrize("gap_policy", ["break", "bridge"])
@pytest.mark.parametrize("cut", [10, 20, 30])
def test_prospective_methods_ignore_future(hist, M, clf, gap_policy, cut):
    j, fa, fb = _compare(hist, M[cut], clf, RecurrenceConfig(gap_policy=gap_policy))
    for m in PROSPECTIVE_METHODS:
        assert (j[f"status_{m}_full"] == j[f"status_{m}_trunc"]).all(), m
    pd.testing.assert_frame_equal(fa.sort_index(), fb.sort_index())


def test_M0_does_use_the_future(hist, M):
    j, _, _ = _compare(hist, M[10], "v2", RecurrenceConfig())
    assert (j["status_M0_full"] != j["status_M0_trunc"]).any()


def test_adding_future_charges_does_not_change_the_past(hist, M):
    """Inject new subscription charges after the cutoff for every account; past decisions unchanged."""
    cut = M[20]
    extra = hist[(hist.month > cut) & hist.is_debit].groupby("account_key").head(1).copy()
    extra["memo"] = "NETFLIX.COM 866-579-7172 CA"
    extra["text"] = "netflix.com 866-579-7172 ca"
    cfg = RecurrenceConfig()
    a = run_one(hist, "a", "v2", cfg).pm
    b = run_one(pd.concat([hist, extra], ignore_index=True), "b", "v2", cfg).pm
    cols = KEY + [f"status_{m}" for m in PROSPECTIVE_METHODS]
    pd.testing.assert_frame_equal(a[a.month <= cut][cols].reset_index(drop=True),
                                  b[b.month <= cut][cols].reset_index(drop=True))

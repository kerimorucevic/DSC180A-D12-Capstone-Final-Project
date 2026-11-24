"""Scripted synthetic scenarios: expected status of every method, month by month.

Months: M[0] = 2023-01 (partial first month) ... M[35] = 2025-12 (partial last month).
"""
import numpy as np
import pytest

from subrec.config import RecurrenceConfig
from subrec.data import exact_distinct
from subrec.pipeline import run_one

C, P, N = "confirmed", "potential", "not_confirmed"
M3S = ["M3_10", "M3_15", "M3_20"]


def test_monthly_subscription(st, pr, M):
    p = pr(1001, "netflix")
    assert (p["status_M0"] == C).all()
    assert [st(1001, "netflix", M[i], "M1") for i in range(4)] == [P, P, P, C]   # partial M[0] is not history
    # M[0] is partial, so the first window with three covered months ends at M[3]
    assert [st(1001, "netflix", M[i], "M2") for i in range(5)] == [P, P, P, C, C]
    for meth in ["M2", *M3S]:
        assert (p.loc[M[3]:M[34], f"status_{meth}"] == C).all()


def test_irregular_platform_purchases_never_confirmed_by_M2_M3(pr, M):
    for merch in ["apple_platform", "google_platform"]:
        p = pr(1002, merch)
        assert (p["status_M0"] == C).all()                 # M0: >= 3 months anywhere -> "recurring"
        assert p["tier"].eq("ambiguous_platform").all()
        for meth in ["M2", *M3S]:
            assert not (p[f"status_{meth}"] == C).any()
    p = pr(1002, "apple_platform")
    assert p.loc[M[6], "status_M1"] == C and p.loc[M[5], "status_M1"] == P


def test_annual_subscription(pr, M):
    p = pr(1003, "amazon_prime")
    assert list(p.index) == [M[1], M[13], M[25]]
    assert (p["status_M0"] == C).all()                     # retrospective M0 calls an annual plan monthly-recurring
    assert list(p["status_M1"]) == [P, P, C]
    for meth in ["M2", *M3S]:
        assert not (p[f"status_{meth}"] == C).any()
    assert list(p["annual_candidate"]) == [False, True, True]   # diagnostic flag, past-only


def test_price_change_threshold_sensitivity(st, M):
    # 11.99 -> 13.99 at M[18] (+16.7%)
    got = {k: [st(1004, "spotify", M[i], k) for i in (17, 18, 19, 20)] for k in ["M2", *M3S]}
    assert got["M2"] == [C, C, C, C]
    assert got["M3_10"] == [C, N, N, C]
    assert got["M3_15"] == [C, N, C, C]
    assert got["M3_20"] == [C, C, C, C]


def test_paused_subscription(analysis, pr, M):
    p = pr(1005, "planet_fitness")
    assert not any(M[i] in p.index for i in range(11, 16))
    assert p.loc[M[16], "status_M1"] == C                  # M1 counts months anywhere in the past
    assert [p.loc[M[i], "status_M2"] for i in (16, 17, 18)] == [N, N, C]   # history exists: not "potential"
    am = analysis.runs[("original", "v2")].am
    pause = am[(am.account_key == 1005) & am.month.isin(M[11:16])]
    assert (pause["status"] == "covered").all() and (pause["rec_M2"] == 0).all()   # real zeros, not missing


def test_missing_month_is_missing_not_zero(analysis, pr, M):
    am = analysis.runs[("original", "v2")].am
    gap = am[(am.account_key == 1006) & (am.month == M[12])].iloc[0]
    assert gap["status"] == "no_data_gap" and not gap["eligible_covered"] and not gap["eligible_common"]
    p = pr(1006, "hulu")
    assert [p.loc[M[i], "status_M2"] for i in (13, 14, 15)] == [P, P, C]   # break policy: window unobservable


def test_missing_month_bridge_policy(hist, M):
    r = run_one(hist[hist.account_key == 1006], "original", "v2", RecurrenceConfig(gap_policy="bridge"))
    p = r.pm[r.pm.merchant == "hulu"].set_index("month")
    assert [p.loc[M[i], "status_M2"] for i in (13, 14, 15)] == [C, C, C]
    assert not r.grid.set_index("month").loc[M[12], "eligible_common"]


def test_repeated_records(analysis, pr, M):
    orig, dist = pr(1007, "disney_plus"), pr(1007, "disney_plus", dataset="exact_distinct")
    assert orig.loc[M[8], "amount"] == pytest.approx(3 * 13.99)
    assert dist.loc[M[8], "amount"] == pytest.approx(13.99)
    assert (orig.loc[M[3]:M[34], "status_M2"] == C).all() and (dist.loc[M[3]:M[34], "status_M2"] == C).all()
    for i in (8, 9, 10):                                   # windows containing the tripled month
        assert orig.loc[M[i], "status_M3_20"] == N
        assert dist.loc[M[i], "status_M3_20"] == C


def test_whole_account_copies_double_spend_but_stay_stable(analysis):
    o = analysis.runs[("original", "v2")].am
    d = analysis.runs[("exact_distinct", "v2")].am
    so = o[o.account_key == 1010]["rec_M3_10"].sum()
    sd = d[d.account_key == 1010]["rec_M3_10"].sum()
    assert sd > 0 and so == pytest.approx(2 * sd)


def test_new_subscription_is_potential_not_absent(st, M):
    assert [st(1011, "openai", M[i], "M2") for i in (33, 34)] == [P, P]
    assert [st(1011, "openai", M[i], "M1") for i in (33, 34)] == [P, P]


def test_short_account(analysis, st, M):
    assert [st(1015, "netflix", M[i], "M2") for i in (31, 32, 33)] == [P, P, C]
    am = analysis.runs[("original", "v2")].am
    a = am[am.account_key == 1015]
    assert list(a.loc[a.eligible_common, "month"]) == [M[33], M[34]]


def test_gambling_events_and_false_positives_excluded(analysis):
    for clf in ["baseline", "v2"]:
        assert 1008 not in set(analysis.runs[("original", clf)].pm.account_key)
    v2 = analysis.runs[("original", "v2")].pm
    base = analysis.runs[("original", "baseline")].pm
    assert 1012 not in set(v2.account_key)
    assert {"google", "disney"} <= set(base[base.account_key == 1012].merchant)


def test_missing_keywords_and_fee_memos(analysis):
    v2 = analysis.runs[("original", "v2")].pm
    base = analysis.runs[("original", "baseline")].pm
    assert set(v2[v2.account_key == 1013].merchant) == {"adobe", "crunch", "youtube_premium_tv"}
    assert 1013 not in set(base.account_key)
    assert set(v2[v2.account_key == 1014].merchant) == {"paramount_plus"}


def test_confirmed_and_potential_dollars_are_disjoint(analysis):
    am = analysis.runs[("original", "v2")].am
    pm = analysis.runs[("original", "v2")].pm
    for k in ["M1", "M2", "M3_15"]:
        assert am[f"rec_{k}"].sum() + am[f"pot_{k}"].sum() <= pm["amount"].sum() + 1e-6
        assert am[f"amb_{k}"].le(am[f"rec_{k}"] + 1e-9).all()


def test_stricter_methods_are_nested(analysis):
    """By construction (not a claim of accuracy): M3_t confirmed => M2 confirmed; M3_10 subset of M3_20."""
    pm = analysis.runs[("original", "v2")].pm
    c = {k: pm[f"status_{k}"].eq(C) for k in ["M1", "M2", *M3S]}
    assert not (c["M3_20"] & ~c["M2"]).any()
    assert not (c["M3_10"] & ~c["M3_15"]).any() and not (c["M3_15"] & ~c["M3_20"]).any()


def test_M1_excludes_partial_edge_months_from_history(st, hist, M):
    # 1011: charges in M[33], M[34] (covered) and M[35] (partial last month)
    assert st(1011, "openai", M[35], "M1") == P
    assert st(1011, "openai", M[35], "M0") == C                      # M0 preserved as the retrospective benchmark
    r = run_one(hist, "original", "v2", RecurrenceConfig(history_requires_covered=False))
    assert r.pm[r.pm.account_key == 1011].set_index("month").loc[M[35], "status_M1"] == C


def test_M1_first_confirmation_is_third_covered_charge(pr, M):
    p = pr(1001, "netflix")
    assert p.index[p["status_M1"] == C][0] == M[3]


def test_primary_specification_defaults():
    cfg = RecurrenceConfig()
    assert cfg.gap_policy == "break" and cfg.history_requires_covered and cfg.window == 3
    assert cfg.m3_thresholds == {"M3_10": .10, "M3_15": .15, "M3_20": .20}


def test_platforms_excluded_view_is_exact(analysis):
    r = analysis.runs[("original", "v2")]
    non = r.pm[~r.pm.is_ambiguous]
    for k in ["M0", "M1", "M2", "M3_15"]:
        for col, status_ in [("rec", "confirmed"), ("pot", "potential")]:
            assert r.am_high[f"{col}_{k}"].sum() == pytest.approx(non.loc[non[f"status_{k}"] == status_, "amount"].sum())
        assert (r.am_high[f"amb_{k}"] == 0).all()
        assert r.am_high[f"rec_{k}"].sum() == pytest.approx(r.am[f"rec_{k}"].sum() - r.am[f"amb_{k}"].sum())

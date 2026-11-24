"""Data preparation, repeated-record diagnostics, metrics, review sampling and the summary generator."""
import numpy as np
import pandas as pd
import pytest

from subrec.config import FAKE_ACCOUNT
from subrec.coverage import build_grid
from subrec.data import repeated_record_diagnostics, exact_distinct, prepare_history
from subrec.metrics import summarize
from subrec.report import SYNTHETIC_BANNER, build_summary
from subrec.review import build_review_sample, cohen_kappa, evaluate_review


def test_fake_account_excluded(raw, hist, analysis):
    assert FAKE_ACCOUNT in set(raw.account_key)
    assert FAKE_ACCOUNT not in set(hist.account_key)
    for r in analysis.runs.values():
        assert FAKE_ACCOUNT not in set(r.am.account_key)


def test_primary_dataset_not_deduplicated(raw, hist):
    assert len(hist) == (raw.account_key != FAKE_ACCOUNT).sum()
    assert hist.duplicated(subset=["account_key", "posted_date", "amount", "credit_or_debit", "memo"]).any()


def test_repeated_record_diagnostics_counts():
    df = pd.DataFrame({"account_key": [1, 1, 1, 2], "posted_date": ["2024-01-02"] * 4,
                       "amount": [5.0, 5.0, 5.0, 5.0], "credit_or_debit": [2] * 4, "memo": ["A", "A", "A", "A"]})
    overall, copies, per_acct = repeated_record_diagnostics(prepare_history(df))
    assert overall["rows"] == 4 and overall["distinct_key_combinations"] == 2
    assert overall["rows_repeating_an_existing_combination"] == 2 and overall["repeat_share"] == 0.5
    assert copies.loc[3, "combinations"] == 1 and copies.loc[1, "combinations"] == 1


def test_exact_distinct_keeps_coverage_statuses(hist):
    g1, g2 = build_grid(hist), build_grid(exact_distinct(hist))
    assert (g1[["account_key", "month", "status"]].values == g2[["account_key", "month", "status"]].values).all()


def test_injected_copies_removed_by_exact_distinct(raw, hist):
    d = exact_distinct(hist)
    assert len(hist) - len(d) == hist["injected_copy"].sum()


def test_common_population_identical_across_methods_and_datasets(analysis):
    m = analysis.metrics
    c = m[m.population == "common"]
    assert c["eligible_account_months"].nunique() == 1


def test_summarize_simple():
    am = pd.DataFrame({"account_key": [1, 1, 2, 2], "eligible_common": [True] * 4, "eligible_covered": [True] * 4,
                       "rec_M2": [0.0, 10.0, 0.0, 0.0], "amb_M2": [0.0, 4.0, 0.0, 0.0], "pot_M2": [1.0, 0, 0, 0]})
    s = summarize(am, "M2", "common")
    assert s["accounts_detected"] == 1 and s["pct_accounts_detected"] == 0.5
    assert s["zero_share"] == 0.75 and s["median_positive"] == 10 and s["total_recurring"] == 10
    assert s["ambiguous_share"] == 0.4 and s["potential_total"] == 1


def test_review_sample_reproducible_and_weighted(analysis):
    r = analysis.runs[("original", "v2")]
    a = build_review_sample(r.pm, r.charges, n_per_stratum=5, seed=1)
    b = build_review_sample(r.pm, r.charges, n_per_stratum=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a.groupby("stratum")["sample_weight"].sum().round(6)
            == a.groupby("stratum")["stratum_size"].first()).all()
    assert (a["human_label"] == "").all() and a["example_memos"].str.len().gt(0).all()
    assert a["stratum"].nunique() > 3


def test_evaluate_review_oracle():
    rev = pd.DataFrame({"status_M2": ["confirmed", "not_confirmed", "confirmed", "potential"],
                        "status_M0": ["confirmed"] * 4, "sample_weight": [1.0, 2.0, 1.0, 1.0],
                        "human_label": ["recurring", "not_recurring", "recurring", "unsure"]})
    e = evaluate_review(rev, methods=["M0", "M2"]).set_index("method")
    assert e.loc["M2", "weighted_precision"] == 1 and e.loc["M2", "weighted_recall"] == 1
    assert e.loc["M0", "weighted_precision"] == pytest.approx(2 / 4)
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1


def test_summary_marks_synthetic_and_writes_no_findings(analysis):
    text = build_summary(analysis, "synthetic", {"rows": 1}, {"rows": 1}, pd.DataFrame({"account_months": [1]}))
    assert SYNTHETIC_BANNER in text and "to be written by the team" in text
    real = build_summary(analysis, "real", {"rows": 1}, {"rows": 1}, pd.DataFrame({"account_months": [1]}))
    assert SYNTHETIC_BANNER not in real

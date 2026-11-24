"""Write analysis_summary.md from computed results only.

The generator fills tables with numbers computed in this run. It writes no interpretation:
the 'Findings' section is left for the team to write after reading the numbers and the review.
"""
import datetime as dt
import math

import pandas as pd

SYNTHETIC_BANNER = ("> **SYNTHETIC DATA. These are not research results.** The numbers below come from the "
                    "generated test histories in `subrec/synthetic.py` and only show that the pipeline runs.")


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "money":
        return f"${v:,.2f}"
    if kind == "int" or (isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer()):
        return f"{int(v):,}"
    return f"{v:,.3f}" if isinstance(v, float) else str(v)


def md_table(df, kinds=None, index=True):
    kinds = kinds or {}
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(_fmt(v, kinds.get(c, "")) for c, v in zip(d.columns, row)) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


KINDS = {"eligible_account_months": "int", "accounts_in_population": "int", "accounts_detected": "int",
         "pct_accounts_detected": "pct", "zero_share": "pct", "median_all": "money", "median_positive": "money",
         "p90_all": "money", "p95_all": "money", "p99_all": "money", "p90_positive": "money",
         "p95_positive": "money", "p99_positive": "money", "total_recurring": "money", "ambiguous_total": "money",
         "ambiguous_share": "pct", "potential_total": "money"}
MAIN_COLS = ["eligible_account_months", "accounts_detected", "pct_accounts_detected", "zero_share", "median_all",
             "median_positive", "p90_all", "p95_all", "p99_all", "total_recurring", "ambiguous_share",
             "potential_total"]


def _sel(m, **kw):
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.set_index("method")


def build_summary(analysis, mode, meta, dup_overall, coverage_counts, gap_table=None, status_table=None,
                  figures=()):
    m, cfg = analysis.metrics, analysis.cfg
    methods = cfg.methods
    L = [f"# Subscription recurrence analysis: summary ({mode} run)", ""]
    if mode != "real":
        L += [SYNTHETIC_BANNER, ""]
    L += [f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `subscription_recurrence_analysis.ipynb`.", "",
          "## Run", "",
          f"- Data: {meta.get('source', 'n/a')}",
          f"- Rows after excluding the synthetic account: {meta.get('rows', 0):,}; accounts: {meta.get('accounts', 0):,}; "
          f"dates {meta.get('date_min', '?')} to {meta.get('date_max', '?')}",
          f"- Recurrence config: min_months={cfg.min_months}, window={cfg.window}, gap_policy={cfg.gap_policy}, "
          f"history_requires_covered={cfg.history_requires_covered}, M3 thresholds={cfg.m3_thresholds}",
          "- Primary specification (research-lead decisions): v2 classifier (provisional, pending human review), original "
          "records, **ambiguous platforms excluded**, common population (covered months whose preceding window is "
          "observable, identical for every method). M3 at 15% is the primary M3 variant (10% / 20% are sensitivity "
          "checks); missing-data months break windows (bridge is a sensitivity check); partial edge months are not "
          "recurrence history for M1-M3; M0 is the retrospective benchmark and uses future months.", "",
          "## Data quality: repeated records", "",
          md_table(pd.Series(dup_overall).to_frame("value"),
                   {"value": ""}).replace("| index | value |", "| measure | value |"), "",
          "Repeated rows are described, not judged: without a transaction id, an exact repeat may be an export "
          "artefact or a genuine second identical purchase; repeats are not treated as confirmed duplicates.", "",
          "## Coverage (original records)", "", md_table(coverage_counts, {"account_months": "int"}), "",
          "## 1. Recurrence definitions (primary specification: ambiguous platforms excluded)", "",
          md_table(_sel(m, dataset="original", classifier="v2", platforms="excluded", population="common")
                   .loc[methods, MAIN_COLS], KINDS), "",
          "### Same, all covered months (pa7 denominator)", "",
          md_table(_sel(m, dataset="original", classifier="v2", platforms="excluded", population="covered")
                   .loc[methods, MAIN_COLS], KINDS), ""]
    amb = pd.DataFrame({
        "v2 ambiguous share": _sel(m, dataset="original", classifier="v2", platforms="included",
                                   population="common").loc[methods, "ambiguous_share"],
        "baseline ambiguous share": _sel(m, dataset="original", classifier="baseline", platforms="included",
                                         population="common").loc[methods, "ambiguous_share"],
        "v2 total incl. platforms": _sel(m, dataset="original", classifier="v2", platforms="included",
                                         population="common").loc[methods, "total_recurring"],
        "v2 total excl. platforms": _sel(m, dataset="original", classifier="v2", platforms="excluded",
                                         population="common").loc[methods, "total_recurring"],
        "v2 % accounts excl. platforms": _sel(m, dataset="original", classifier="v2", platforms="excluded",
                                              population="common").loc[methods, "pct_accounts_detected"],
        "v2 % accounts incl. platforms": _sel(m, dataset="original", classifier="v2", platforms="included",
                                              population="common").loc[methods, "pct_accounts_detected"],
        "v2 zero share excl.": _sel(m, dataset="original", classifier="v2", platforms="excluded",
                                    population="common").loc[methods, "zero_share"],
        "v2 zero share incl.": _sel(m, dataset="original", classifier="v2", platforms="included",
                                    population="common").loc[methods, "zero_share"]})
    L += ["## 2. Ambiguous platform merchants (sensitivity: included vs excluded)", "",
          md_table(amb, {"v2 ambiguous share": "pct", "baseline ambiguous share": "pct",
                         "v2 total incl. platforms": "money", "v2 total excl. platforms": "money",
                         "v2 % accounts excl. platforms": "pct", "v2 % accounts incl. platforms": "pct",
                         "v2 zero share excl.": "pct", "v2 zero share incl.": "pct"}), ""]
    dd = []
    for M in methods:
        a = _sel(m, dataset="original", classifier="v2", platforms="excluded", population="common").loc[M]
        b = _sel(m, dataset="exact_distinct", classifier="v2", platforms="excluded", population="common").loc[M]
        dd.append({"method": M, "total original": a.total_recurring, "total exact-distinct": b.total_recurring,
                   "total change": (b.total_recurring / a.total_recurring - 1) if a.total_recurring else float("nan"),
                   "% accts original": a.pct_accounts_detected, "% accts exact-distinct": b.pct_accounts_detected,
                   "median+ original": a.median_positive, "median+ exact-distinct": b.median_positive})
    L += ["## 3. Original vs exact-distinct records (v2, ambiguous platforms excluded, common population)", "",
          md_table(pd.DataFrame(dd), {"total original": "money", "total exact-distinct": "money",
                                      "total change": "pct", "% accts original": "pct",
                                      "% accts exact-distinct": "pct", "median+ original": "money",
                                      "median+ exact-distinct": "money"}, index=False), ""]
    cc = pd.DataFrame({f"{c} {k}": _sel(m, dataset="original", classifier=c, platforms="included",
                                       population="common").loc[methods, k]
                       for c in ["baseline", "v2"] for k in ["pct_accounts_detected", "total_recurring"]})
    L += ["## 4. Baseline (pa7) vs v2 classifier (original records, ambiguous platforms included as in pa7, common population)", "",
          md_table(cc, {c: ("pct" if "pct" in c else "money") for c in cc.columns}), ""]
    if gap_table is not None:
        L += ["## 5. Gap policy sensitivity (v2, original records, ambiguous platforms excluded)", "", md_table(gap_table, KINDS), ""]
    if status_table is not None:
        L += ["## 6. Confirmed vs potential vs not confirmed dollars (v2, covered months, ambiguous platforms excluded)", "",
              md_table(status_table, {c: "money" for c in status_table.columns}), ""]
    if figures:
        L += ["## Figures", "", *[f"- `{f}`" for f in figures], ""]
    L += ["## Findings (to be written by the team)", "",
          "_Not generated automatically. Stricter definitions change what is counted; they are not shown "
          "to be more accurate until the human review sample has been labelled and scored._", "",
          "## Standing caveats", "",
          "- Classifier and recurrence rules are keyword / pattern rules without human validation yet. Stricter recurrence "
          "definitions count fewer charges; that does not make them more accurate.",
          "- M0 (the original PA7 recurrence rule, distinct from the PA7 baseline classifier) uses future months and is for "
          "retrospective comparison only.",
          "- Repeated records are described, not treated as confirmed duplicates: without a transaction id they may be "
          "export artefacts or genuine identical purchases. The original / exact-distinct comparison is a sensitivity range.",
          "- Coverage is retrospective: first / last (partial_edge) months and the covered / gap status come from the "
          "account's full observed span, which would not be known in real time.",
          "- No refund netting: only debits are summed; credits and refunds from subscription merchants are not subtracted "
          "(as in PA7).",
          "- Annual and quarterly plans are not detected by M1-M3 within a monthly window "
          "(diagnostic `annual_candidate` flag only).",
          "- Ambiguous platforms mix subscriptions and one-off purchases under one memo.",
          "- Merchant grouping is platform-level: several products billed through one platform share one series, so a "
          "monthly total is a sum and M3 can reject genuine subscriptions or accept coincidental sums.",
          "- The v2 classifier is provisional, pending human review.",
          "- Accounts were sampled by activity; percentiles pool account-months, so long, active accounts weigh more.",
          ""]
    return "\n".join(L)


def write_summary(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path

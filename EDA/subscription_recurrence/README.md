# Subscription recurrence sensitivity analysis

> **Status: exploratory feature validation, tested on synthetic data only.**
> This folder contains code, synthetic tests and documentation. It contains **no real transaction data, no real-data
> results, no account identifiers, no transaction memos and no private review samples.**
> **No financial-distress prediction has been conducted here.** The analysis only examines how different definitions of
> "recurring subscription spending" change an account-month feature that *may* later be considered for distress modelling.

## Research question

How do alternative definitions of *recurrence* change estimated monthly subscription spending per account, and do
stricter definitions better separate ongoing commitments from repeated discretionary purchases?

The second half of that question is **open**. A stricter definition counts fewer charges; it is not automatically more
accurate. Nothing in this folder shows that any definition matches reality: that needs the human-review procedure
described below, which has not been carried out.

## Two things are called "baseline": keep them apart

| Name | What it is | Where |
|---|---|---|
| **PA7 baseline classifier** | The original PA7 keyword rules that decide which debits are subscription-type. Preserved verbatim from `pa7_subscriptions.ipynb`. | `subrec/rules_baseline.py` |
| **M0** | The original PA7 *recurrence rule*: a merchant is recurring if the account is charged in ≥ 3 distinct months **anywhere in its history**. A recurrence definition, not a classifier. It uses future months, so it is a retrospective benchmark only. | `subrec/recurrence.py` |
| **v2 classifier** | Alternative keyword classifier: fixes unreachable subtype keywords and obvious false positives, splits *high-confidence* merchants from *ambiguous platforms* (Apple, Google, Microsoft ...), excludes gambling and one-time events. **Provisional, pending human review.** | `subrec/rules_v2.py` |
| **M1, M2, M3** | New recurrence definitions using only the present and past (details in [docs/METHODS.md](docs/METHODS.md)). | `subrec/recurrence.py` |

PA7's original feature = **PA7 baseline classifier + M0**. A test reproduces it exactly. Every other combination
(v2 classifier, M1–M3) is new.

## Primary specification and sensitivity checks

| | Primary | Sensitivity |
|---|---|---|
| Classifier | v2 (provisional) | PA7 baseline classifier |
| Records | original (nothing removed) | exact-distinct records (see limitations) |
| Population | common eligible population | all covered months |
| Ambiguous platforms | **excluded** | included |
| M3 amount stability | 15 % | 10 %, 20 % |
| Missing-data months | break consecutive-month windows | `bridge` policy |
| Partial edge months | not measured, not history for M1–M3 | — |

## Folder contents

```
EDA/subscription_recurrence/
├── README.md                              this file
├── AI_USAGE.md                            how AI assistance was used
├── requirements.txt
├── .gitignore                             blocks data, caches, outputs, private files
├── subscription_recurrence_analysis.ipynb the analysis (committed with no outputs)
├── subrec/                                reusable modules (classifiers, coverage, methods, metrics, plots, review, report)
├── tests/                                 synthetic tests (pytest)
├── scripts/check_publication_ready.py     pre-commit hygiene check
└── docs/METHODS.md, docs/TESTING.md       definitions, decisions, test coverage
```

## Quick start on a laptop (synthetic data)

```bash
git clone https://github.com/kerimorucevic/DSC180A-D12-Capstone-Final-Project.git
cd DSC180A-D12-Capstone-Final-Project/EDA/subscription_recurrence

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. tests (synthetic only, a few seconds)
python -m pytest -q -rs tests

# 2. run the notebook end to end on generated data (writes outputs_subscription_recurrence/synthetic/, git-ignored)
jupyter nbconvert --to notebook --execute subscription_recurrence_analysis.ipynb --output executed_synthetic.ipynb
```

Synthetic mode is the default; no data files are needed. For interactive use, `pip install jupyterlab`, start it **from this
folder**, open the notebook and Run All. Executed copies (`executed_*.ipynb`) are git-ignored; do not commit an executed
notebook.

**Baseline fidelity tests** compare the preserved PA7 rules and feature against the reference notebook
`pa7_subscriptions.ipynb`. The tests look for it in the parent folders of this directory (for example the repository
root) and then anywhere in the repository. If it lives elsewhere, set
`export SUBREC_PA7_NOTEBOOK=/path/to/pa7_subscriptions.ipynb`. If it cannot be found those 3 tests are skipped, and
`-rs` prints why: expect `88 passed` when it is found and `85 passed, 3 skipped` when it is not.

Synthetic results show that the pipeline runs and that each definition behaves as specified on constructed cases. They are
**not research results** (the generated `analysis_summary.md` says so in a banner).

## Real-data run (team members with access, on UCSD DSMLP)

The real run reads the **existing** cache `full_histories_5000accts.parquet` written by `income_feature.ipynb`. It never
scans the restricted data share. Run it only on DSMLP; the cache contains real memo text and must not leave the hub.

```bash
git clone https://github.com/kerimorucevic/DSC180A-D12-Capstone-Final-Project.git                   # on DSMLP (or `git pull` if already cloned)
cd DSC180A-D12-Capstone-Final-Project/EDA/subscription_recurrence
find ~ -maxdepth 6 -name "full_histories_5000accts.parquet" 2>/dev/null     # locate the existing cache
export SUBREC_MODE=real
export SUBREC_HIST_CACHE=/full/path/to/full_histories_5000accts.parquet
python -m pytest -q -rs tests                    # sanity check of the environment (synthetic)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
    subscription_recurrence_analysis.ipynb --output executed_real.ipynb
```

Interactive alternative: in the notebook's first code cell set `MODE = "real"` and `HIST_CACHE_PATH = "/full/path/..."`.

Checks: the first cell must print `MODE=real`; section 1 reports `rows_before_fake_exclusion`, which should equal the row
count you obtained from the earlier `income_feature.ipynb` run on the same cache. The notebook stops with a clear
`FileNotFoundError` if the cache path is wrong. Resource needs at full size have not been measured; request a generous
amount of memory (8 GB or more).

Outputs go to `outputs_subscription_recurrence/real/` (git-ignored): aggregate tables, six figures, `analysis_summary.md`
(computed tables only; its *Findings* section is for the team to write) and, in `private/`, memo-level files
(`classifier_disagreements.csv`, `review_sample_v2.csv`). **Keep `private/`, the cache and executed real notebooks on DSMLP.**
Review any aggregate output before it leaves DSMLP, and never commit real-data results to this public repository.

## Before every commit

```bash
python scripts/check_publication_ready.py        # notebooks have no outputs; no data, outputs, home paths, e-mails, account key
```

To clear an executed notebook: `jupyter nbconvert --clear-output --inplace subscription_recurrence_analysis.ipynb`.

## Limitations (read before interpreting any output)

* **Stricter ≠ more accurate.** M2 and M3 count fewer charges than M0 and M1. Whether they better identify genuine
  ongoing commitments is untested. Validation requires human labels (see below).
* **Repeated records.** The real transaction table has no transaction id, and rows that repeat an existing
  (account, date, amount, direction, memo) combination are kept in the primary analysis. They are **not** confirmed
  duplicates: a repeat may be an export artefact or a genuine second identical purchase. The separate exact-distinct
  analysis is a sensitivity range, not a correction. Repeats raise monthly totals and can make a stable subscription look
  like it changed price, which affects M3.
* **Platform-level merchant grouping.** A merchant is identified at platform level, so several products billed through one
  platform (two Apple subscriptions, or a subscription plus an app purchase) form one series whose monthly total is a sum.
  M3 may then reject genuine subscriptions or accept coincidental sums. Ambiguous platforms are excluded from the primary
  view for this reason, and the notebook reports a diagnostic for them.
* **Retrospective coverage.** Coverage statuses (`covered`, `no_data_gap`, `partial_edge`) use each account's full observed
  span, so "last month of the account" is known only in hindsight, and M0 uses future months. M1–M3 use no future
  *transactions*, but the eligible-month definition itself is retrospective, so these features are not a real-time
  simulation.
* **No refund netting.** Only debits are summed. Credits and refunds from subscription merchants are not subtracted
  (same as PA7), so spend may be overstated for accounts with refunds.
* **Unvalidated rules.** Both classifiers are keyword patterns; the v2 classifier is provisional.
* **Annual or quarterly plans** are invisible to M1–M3 (a diagnostic flag only) and M0 labels annual plans as recurring.
* **Sampling.** Accounts were sampled by activity; pooled account-month percentiles over-weight long, active accounts.

## Human validation procedure (not yet carried out)

`subrec/review.py` and notebook section 7 draw a stratified sample of account-merchant-months (stratified by merchant tier and
by the pattern of decisions across M0/M1/M2/M3, so disagreements are over-sampled, with inverse-probability weights). Two
reviewers independently label each row (`recurring`, `not_recurring`, `unsure`) using only the memo examples and the amount
history, without seeing the method columns. Then report reviewer agreement (`cohen_kappa`) and weighted precision/recall per
method (`evaluate_review`). Only after that can any definition be called more or less accurate. The sample contains real memos
and stays on DSMLP.

## Further reading

* [docs/METHODS.md](docs/METHODS.md): exact definitions, populations, configuration and the decisions applied.
* [docs/TESTING.md](docs/TESTING.md): which test covers which requirement and the latest run.
* [AI_USAGE.md](AI_USAGE.md): how AI assistance was used and what still needs human review.

# Methods, decisions and configuration

This document describes what the code does. It contains no real-data results. Stricter recurrence definitions are not
claimed to be more accurate; see the README.

## Pipeline

1. **Prepare** (`subrec/data.py`): exclude the known synthetic account (`FAKE_ACCOUNT` in `subrec/config.py`); keep every other row
   (nothing removed or collapsed); keep debits (`credit_or_debit == 2`).
2. **Classify** each distinct debit memo (`subrec/rules_baseline.py` or `subrec/rules_v2.py`).
3. **Pair-months** (`subrec/recurrence.py`): total of an account's charges from one merchant in one calendar month.
4. **Recurrence status** per pair-month and method: `confirmed` / `potential` / `not_confirmed`.
5. **Account-month features** (`subrec/features.py`): `rec_<M>` (confirmed dollars), `pot_<M>` (potential dollars, never in
   `rec_`), `amb_<M>` (ambiguous-platform part of `rec_`), `nsvc_<M>`.
6. **Coverage and populations** (`subrec/coverage.py`), **metrics** (`subrec/metrics.py`), **figures** (`subrec/plots.py`),
   **summary** (`subrec/report.py`).

## Classifiers

* **PA7 baseline classifier**: rules copied verbatim from `pa7_subscriptions.ipynb` (cell 3). Not edited. A test checks the copy.
* **v2 classifier** (provisional): matches memos directly against a merchant dictionary (first match wins):
  bank-fee memos, gambling, advertising → excluded; named services → `high_confidence`
  (streaming, software / apps, gym); one-time events, P2P / wallet transfers, retail stores → excluded; billing platforms
  → `ambiguous_platform`. A match only says "subscription-type merchant"; recurrence decides the rest.

## Recurrence definitions

Notation: *t* is the observation month. "Covered" has the coverage meaning below.

| Method | Confirmed at *t* if | Uses future data? |
|---|---|---|
| **M0** | ≥ 3 distinct charged months anywhere in the account history. The original PA7 recurrence rule. | **Yes**: retrospective benchmark only |
| **M1** | ≥ 3 distinct charged **covered** months up to and including *t* | No |
| **M2** | a charge in each of *t−2*, *t−1*, *t*, with *t−2* and *t−1* covered | No |
| **M3_x** | M2 and every monthly total in the window within *x* of the window median: max\|total − median\| / median ≤ *x* | No |

* **Primary M3**: 15 %. **Sensitivity**: 10 % and 20 %.
* `potential`: a subscription-type charge whose observed history up to *t* is too short to decide (new merchant inside the
  window, or window months unobserved / partial). Kept separate from `confirmed`.
* `not_confirmed`: enough history was observable and the rule failed (irregular, paused, unstable amount).
* **Gap policy**: `break` (primary) uses consecutive *calendar* months, so a month with no data makes the window
  unobservable; `bridge` (sensitivity) skips such months.
* `annual_candidate`: diagnostic flag (previous charge exactly 12 months earlier, similar amount). Not a method.

## Coverage and populations

Each account-month between an account's first and last transaction has a status (unchanged from PA7 / `income_feature`):

| Status | Meaning | Measured? |
|---|---|---|
| `covered` | full month with ≥ 1 transaction of any kind | yes: $0 is a real zero |
| `no_data_gap` | no transactions at all inside the account's span | no: missing coverage, not zero |
| `partial_edge` | first or last month of the account | no; not history for M1–M3 |

* **common** population (primary): covered months whose two preceding months are also covered. Same rows for every method,
  classifier and dataset.
* **covered** population: all covered months (the PA7 denominator).

## Sensitivity analyses

* **Repeated records.** `exact_distinct` keeps one row per exact (account, date, amount, direction, memo) combination.
  The primary analysis never uses it. Repeats are *not* assumed to be duplicates or errors.
* **Ambiguous platforms** included vs excluded; **classifier** PA7 baseline vs v2; **gap policy**; **M3 threshold**.

## Decisions applied (research lead, 2026-11-24)

1. Partial edge months are not recurrence history for M1, M2 and M3 (`history_requires_covered=True`); M0 unchanged.
2. Missing-data months break consecutive-month windows; `bridge` retained as a sensitivity check.
3. M3 primary = 15 % median-based stability; 10 % / 20 % sensitivity.
4. Platform-level merchant grouping retained, limitations flagged.
5. Primary analysis = v2 classifier, original records, common population, ambiguous platforms excluded.
6. v2 classifier retained provisionally, pending human review.

## Configuration (`RecurrenceConfig` in `subrec/config.py`)

`min_months=3`, `window=3`, `m3_thresholds={"M3_10": .10, "M3_15": .15, "M3_20": .20}`, `gap_policy="break"`,
`history_requires_covered=True`, `annual_tolerance=0.20`.

## Known limitations

See the README. In short: no validation; repeated records are ambiguous; platform grouping; retrospective coverage;
no refund netting; annual plans invisible to M1–M3; unvalidated rules; activity-based sampling.

# Test coverage and results (synthetic data only)

Latest run: 2026-11-24, macOS, Python 3.11.5, pandas 2.0.3, numpy 1.24.3 (pytest 7.4.0). No real data was used.

```
python -m pytest -q -rs tests
88 passed
```

Executing `subscription_recurrence_analysis.ipynb` in synthetic mode runs the same suite (88 passed) and finished with no errors.
The 3 baseline-fidelity tests need the reference notebook `pa7_subscriptions.ipynb`; see the README for how it is located.
Without it they are skipped (`85 passed, 3 skipped`).

## Coverage by requirement

| Requirement | Test(s) | What is asserted |
|---|---|---|
| Monthly subscription | `test_monthly_subscription` | M0 confirms every month; M1 from the 3rd *covered* charged month (the partial first month is not history, so M[3] not M[2]); M2/M3 from the first window of 3 covered months |
| Irregular purchases | `test_irregular_platform_purchases_never_confirmed_by_M2_M3` | Apple/Google one-off purchases in 6 / 4 scattered months: M0 and M1 call them recurring, M2/M3 never do |
| Annual subscription | `test_annual_subscription` | 3 yearly charges: M0 confirms all (retrospective flaw), M1 only the 3rd, M2/M3 never; `annual_candidate` flags years 2 and 3 |
| Price change | `test_price_change_threshold_sensitivity` | +16.7 % price change: M2 continuous; M3_10 drops 2 months, M3_15 drops 1, M3_20 drops 0 |
| Paused subscription | `test_paused_subscription` | No charges during the pause; pause months are covered real zeros; after resuming, M2 is `not_confirmed` (not `potential`) for 2 months |
| Missing months | `test_missing_month_is_missing_not_zero`, `test_missing_month_bridge_policy` | Gap month is `no_data_gap`, outside both populations; M2 `potential` for 2 months under `break`, confirmed straight away under `bridge` |
| Repeated records | `test_repeated_records`, `test_whole_account_copies_double_spend_but_stay_stable`, `test_injected_copies_removed_by_exact_distinct`, `test_exact_distinct_keeps_coverage_statuses`, `test_primary_dataset_not_deduplicated` | A tripled month triples spend and fails M3 in the 3 windows containing it (original) but passes on exact-distinct; whole-account copies double spend but stay stable; exact-distinct removes exactly the injected copies; coverage statuses unchanged; primary data not deduplicated |
| Future-data leakage | `test_prospective_methods_ignore_future` (12 cases), `test_M0_does_use_the_future`, `test_adding_future_charges_does_not_change_the_past` | Truncating at 3 cutoffs × 2 classifiers × 2 gap policies leaves every M1–M3 status and feature value before the cutoff unchanged; M0 does change (shows the leak); injecting new future charges changes nothing in the past |
| Potential vs confirmed | `test_new_subscription_is_potential_not_absent`, `test_short_account`, `test_confirmed_and_potential_dollars_are_disjoint` | New subscriptions and short accounts are `potential`; confirmed and potential dollars never overlap |
| Classifier v2 | `test_classifier.py` (42) | Missing keywords now matched; false positives (Google Pay transfer, Disneyland, Google Ads, bank FX fee, Apple Store) excluded; gambling / events excluded; Apple / Google / Microsoft / Xbox / Amazon digital / generic YouTube tiered `ambiguous_platform` |
| Baseline preserved | `test_rules_copy_is_verbatim`, `test_classify_matches_pa7_on_all_synthetic_memos`, `test_baseline_M0_reproduces_pa7_feature` | `rules_baseline.py` contains pa7 cell 3 verbatim; classifications identical; executing pa7 cells 3/7/11/13 gives the same coverage status and `recurring_sub_spend` as baseline + M0 for every account-month |
| Research-lead decisions (2026-11-24) | `test_M1_excludes_partial_edge_months_from_history`, `test_M1_first_confirmation_is_third_covered_charge`, `test_primary_specification_defaults`, `test_platforms_excluded_view_is_exact` | M1 ignores partial edge months (and reverts with `history_requires_covered=False`) while M0 is unchanged; defaults = break policy, covered-only history, 3-month window, M3 at 10/15/20 %; the platforms-excluded feature table equals the non-platform part of the full table exactly (rec, pot, amb) |
| Synthetic account | `test_fake_account_excluded` | The known synthetic account (`FAKE_ACCOUNT`) is present in the raw synthetic data and absent after preparation and from every output |
| Metrics, review, summary | `test_summarize_simple`, `test_common_population_identical_across_methods_and_datasets`, `test_review_sample_reproducible_and_weighted`, `test_evaluate_review_oracle`, `test_summary_marks_synthetic_and_writes_no_findings` | Metric arithmetic; one common population for all variants; seeded, weighted review sample; weighted precision/recall; synthetic banner and no auto-written findings |

## What the tests do not show

They check that each definition does what it says on constructed cases. They say nothing about which
definition better matches real subscriptions; that needs the human-review sample (notebook section 7).


## Full listing

```
tests/test_baseline_fidelity.py::test_rules_copy_is_verbatim PASSED
tests/test_baseline_fidelity.py::test_classify_matches_pa7_on_all_synthetic_memos PASSED
tests/test_baseline_fidelity.py::test_baseline_M0_reproduces_pa7_feature PASSED
tests/test_classifier.py::test_high_confidence_merchants[NETFLIX.COM 866-579-7172 CA-streaming-netflix] PASSED
tests/test_classifier.py::test_high_confidence_merchants[YOUTUBE PREMIUM 650-253-0000-streaming-youtube_premium_tv] PASSED
tests/test_classifier.py::test_high_confidence_merchants[HBO MAX 855-442-6629-streaming-max_hbo] PASSED
tests/test_classifier.py::test_high_confidence_merchants[SIRIUSXM RADIO 888-635-5144-streaming-siriusxm] PASSED
tests/test_classifier.py::test_high_confidence_merchants[DISNEY PLUS 888-905-7888 CA-streaming-disney_plus] PASSED
tests/test_classifier.py::test_high_confidence_merchants[AMZN PRIME*2K4LM8-streaming-amazon_prime] PASSED
tests/test_classifier.py::test_high_confidence_merchants[AMAZON PRIME*2K4LM8-streaming-amazon_prime] PASSED
tests/test_classifier.py::test_high_confidence_merchants[PARAMOUNT+ MONTHLY FEE-streaming-paramount_plus] PASSED
tests/test_classifier.py::test_high_confidence_merchants[AMC+ SUBSCRIPTION-streaming-amc_plus] PASSED
tests/test_classifier.py::test_high_confidence_merchants[ADOBE *CREATIVE CLD 408-536-6000-software / apps-adobe] PASSED
tests/test_classifier.py::test_high_confidence_merchants[OPENAI *CHATGPT SUBSCR-software / apps-openai] PASSED
tests/test_classifier.py::test_high_confidence_merchants[DROPBOX*7HJ2K-software / apps-dropbox] PASSED
tests/test_classifier.py::test_high_confidence_merchants[XBOX GAME PASS ULTIMATE-software / apps-xbox_game_pass] PASSED
tests/test_classifier.py::test_high_confidence_merchants[CRUNCH FITNESS #221-gym-crunch] PASSED
tests/test_classifier.py::test_high_confidence_merchants[ANYTIME FITNESS 4411-gym-anytime_fitness] PASSED
tests/test_classifier.py::test_high_confidence_merchants[YMCA OF SAN DIEGO-gym-ymca] PASSED
tests/test_classifier.py::test_high_confidence_merchants[PLANET FITNESS CLUB FEES-gym-planet_fitness] PASSED
tests/test_classifier.py::test_ambiguous_platforms[APPLE.COM/BILL 866-712-7753 CA-apple_platform] PASSED
tests/test_classifier.py::test_ambiguous_platforms[GOOGLE *GOOGLE PLAY APP-google_platform] PASSED
tests/test_classifier.py::test_ambiguous_platforms[MICROSOFT*STORE PURCHASE-microsoft_platform] PASSED
tests/test_classifier.py::test_ambiguous_platforms[XBOX DIGITAL PURCHASE-xbox_platform] PASSED
tests/test_classifier.py::test_ambiguous_platforms[AMZN DIGITAL*2K81 WA-amazon_digital] PASSED
tests/test_classifier.py::test_ambiguous_platforms[YOUTUBE 650-253-0000-youtube_generic] PASSED
tests/test_classifier.py::test_exclusions[DRAFTKINGS SPORTSBOOK-gambling] PASSED
tests/test_classifier.py::test_exclusions[FANDUEL SPORTSBOOK-gambling] PASSED
tests/test_classifier.py::test_exclusions[TICKETMASTER 800-653-8000-events / recreation] PASSED
tests/test_classifier.py::test_exclusions[AMC THEATRES 2231-events / recreation] PASSED
tests/test_classifier.py::test_exclusions[DISNEYLAND RESORT TICKETS-events / recreation] PASSED
tests/test_classifier.py::test_exclusions[ROUND1 BOWLING SAN DIEGO-events / recreation] PASSED
tests/test_classifier.py::test_exclusions[GOOGLE PAY TRANSFER 4821-transfer_or_p2p] PASSED
tests/test_classifier.py::test_exclusions[APPLE CASH SENT MONEY-transfer_or_p2p] PASSED
tests/test_classifier.py::test_exclusions[GOOGLE *ADS4821-advertising] PASSED
tests/test_classifier.py::test_exclusions[FOREIGN TRANSACTION FEE NETFLIX-bank_fee] PASSED
tests/test_classifier.py::test_exclusions[APPLE STORE R102 UTC-retail_store] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[TRADER JOE'S #552] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[OFFICE MAX 1123] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[MONTHLY MAINTENANCE FEE] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[PANDORA JEWELRY 22] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[SHELL OIL 57442] PASSED
tests/test_classifier.py::test_non_subscriptions_unmatched[TIDAL WAVE CAR WASH] PASSED
tests/test_classifier.py::test_baseline_gaps_that_v2_fixes PASSED
tests/test_classifier.py::test_baseline_false_positives_that_v2_excludes PASSED
tests/test_data_review_report.py::test_fake_account_excluded PASSED
tests/test_data_review_report.py::test_primary_dataset_not_deduplicated PASSED
tests/test_data_review_report.py::test_repeated_record_diagnostics_counts PASSED
tests/test_data_review_report.py::test_exact_distinct_keeps_coverage_statuses PASSED
tests/test_data_review_report.py::test_injected_copies_removed_by_exact_distinct PASSED
tests/test_data_review_report.py::test_common_population_identical_across_methods_and_datasets PASSED
tests/test_data_review_report.py::test_summarize_simple PASSED
tests/test_data_review_report.py::test_review_sample_reproducible_and_weighted PASSED
tests/test_data_review_report.py::test_evaluate_review_oracle PASSED
tests/test_data_review_report.py::test_summary_marks_synthetic_and_writes_no_findings PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[10-break-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[10-break-v2] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[10-bridge-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[10-bridge-v2] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[20-break-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[20-break-v2] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[20-bridge-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[20-bridge-v2] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[30-break-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[30-break-v2] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[30-bridge-baseline] PASSED
tests/test_leakage.py::test_prospective_methods_ignore_future[30-bridge-v2] PASSED
tests/test_leakage.py::test_M0_does_use_the_future PASSED
tests/test_leakage.py::test_adding_future_charges_does_not_change_the_past PASSED
tests/test_recurrence_scenarios.py::test_monthly_subscription PASSED
tests/test_recurrence_scenarios.py::test_irregular_platform_purchases_never_confirmed_by_M2_M3 PASSED
tests/test_recurrence_scenarios.py::test_annual_subscription PASSED
tests/test_recurrence_scenarios.py::test_price_change_threshold_sensitivity PASSED
tests/test_recurrence_scenarios.py::test_paused_subscription PASSED
tests/test_recurrence_scenarios.py::test_missing_month_is_missing_not_zero PASSED
tests/test_recurrence_scenarios.py::test_missing_month_bridge_policy PASSED
tests/test_recurrence_scenarios.py::test_repeated_records PASSED
tests/test_recurrence_scenarios.py::test_whole_account_copies_double_spend_but_stay_stable PASSED
tests/test_recurrence_scenarios.py::test_new_subscription_is_potential_not_absent PASSED
tests/test_recurrence_scenarios.py::test_short_account PASSED
tests/test_recurrence_scenarios.py::test_gambling_events_and_false_positives_excluded PASSED
tests/test_recurrence_scenarios.py::test_missing_keywords_and_fee_memos PASSED
tests/test_recurrence_scenarios.py::test_confirmed_and_potential_dollars_are_disjoint PASSED
tests/test_recurrence_scenarios.py::test_stricter_methods_are_nested PASSED
tests/test_recurrence_scenarios.py::test_M1_excludes_partial_edge_months_from_history PASSED
tests/test_recurrence_scenarios.py::test_M1_first_confirmation_is_third_covered_charge PASSED
tests/test_recurrence_scenarios.py::test_primary_specification_defaults PASSED
tests/test_recurrence_scenarios.py::test_platforms_excluded_view_is_exact PASSED
```

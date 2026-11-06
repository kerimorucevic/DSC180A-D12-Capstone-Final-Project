# Measuring Subscriptions & Entertainment Spending with NLP Transaction Categorization

DSC 180A Capstone, Section D12, Group 3 (Prism Data): Aiden Lai, Vanessa Feng, Katie Hannigan, Kerim Orucevic

Bank transaction memos are short, inconsistent and full of reference numbers, so the same purchase can look
different from one bank to the next. This project builds a reviewed 20-category label set for anonymized Prism
Data transactions, trains a fastText classifier that generalizes to future months and unseen accounts, and uses it
to answer one question: **how does Subscriptions & Entertainment spending (streaming, events and games, gambling)
move over time, and does it track national recreation spending and prices from FRED?**

The repository follows the path of the quarter: each lecture's question is one stage of the pipeline, one
command, and one notebook.

## Quarter roadmap

| Lecture | Question | Command (`--steps`) | Notebook | Key outputs |
|---|---|---|---|---|
| L2 Data exploration + memo cleaning | What is in the data, and which tokens carry merchant / type cues? | `l2` (explore, merchants) | `01_data_exploration` | `data_summary.json`, monthly volume + cumulative plots, rule coverage, top patterns per category |
| L3 Cleaning thresholds + labels | Does TF-IDF help cleaning? Can four labelers apply one guide? | `l3` (tfidf, agreement) | `02_cleaning_thresholds_and_labels` | `tfidf_cleaning_study.csv`, `agreement.json` |
| L4 Reviewed labels + first fastText | With a frozen account + time split, how do simple classifiers do? | `l4` (prepare, baseline) | `03_first_classifier` | `class_counts.csv`, `baseline.csv` |
| L5 Scale labels + tune | How good is tuned fastText, where does it fail, how much can it label without review? | `sample`, `python -m src.review`, `audit`, `build-labels`, then `l5` | `04_scale_labels_and_tune` | `experiments.csv`, `scores.csv`, `confusion.png`, `coverage_val.csv`, runtime |
| Project A | Does Subscriptions & Entertainment spending track FRED? | `project-a` (monthly) | `05_subscriptions_entertainment` | `theme_monthly.csv`, `theme_quarterly.csv`, `theme_monthly.png` |

Earlier assignment notebooks are kept in `notebooks/archive/` (Vanessa's PA4, Kerim's and Aiden's PA5); their
code now lives in `src/`.

## Repository layout

```
config.yaml              paths, split cutoffs, sampling, fastText grid, review cutoff, Project A theme + FRED
requirements.txt
src/
  main.py                entrypoint: python -m src.main --config config.yaml --steps <steps or stage>
  explore.py             L2  DuckDB profile, account/balance/link checks, merchant frequencies
  clean.py               L2  text_lower (case + spacing), text_norm (masked IDs), memo pattern
  tfidf_study.py         L3  TF-IDF token-removal thresholds vs the baseline
  taxonomy.py            L3  the 20-category guide
  rules.py               L3  ordered pre-tag rules (AI suggestions with quoted cue + confidence)
  agreement.py           L3  agreement on the shared 100 memos
  split.py               L4  frozen account + time split
  model.py               L4  fastText fit / predict / scores
  tune.py                L4/L5  baselines (majority, TF-IDF + logistic regression, earlier fastText) + staged tuning
  sample.py              L5  next labeling batch (sparse categories, focus category, Pareto head, random)
  review.py              L5  review file, timed review loop, audit decision, labels_v2.csv export
  evaluate.py            L5  per-class report, confusion, errors, review cutoff, Test A / Test B
  runtime.py             L5  fit / inference timing, human review effort
  monthly.py             Project A  monthly series, Subscriptions & Entertainment groups, FRED download
  data.py                parquet readers (DataHub only)
notebooks/
  01-05 stage notebooks (outputs not committed)
  archive/               original PA4 / PA5 notebooks
data/                    labels and review files (git-ignored: memos stay on DataHub)
outputs/                 model, metrics, figures (git-ignored)
```

## Setup and running

```bash
pip install -r requirements.txt      # on numpy >= 2 use fasttext-numpy2-wheel instead of fasttext-wheel
python -m src.main --config config.yaml --steps l2,l3,l4,l5,project-a     # the whole quarter
python -m src.main --config config.yaml --steps l5                        # or one stage
python -m src.main --config config.yaml --steps evaluate                  # or single steps
```

Edit `config.yaml` first: `paths.data_dir` (parquet files), `paths.labels` (reviewed labels), and
`agreement.file` (the shared 100 memos). Label files need `item_id, split, account_key, date, amount, direction,
memo_raw, category`. The notebooks call the same steps; set `CAPSTONE_CONFIG` to use another config file.

**Growing the label set (Lecture 5, DataHub):**

```bash
python -m src.main --config config.yaml --steps sample        # scan accounts, freeze split, write review_batch.csv
python -m src.review --config config.yaml                     # timed review: Enter accept, 7 / name correct, u unresolved, b undo, q quit
python -m src.main --config config.yaml --steps audit,build-labels
```

Each batch over-samples Subscriptions & Entertainment (`sampling.focus_category`, `focus_target`) across its
groups, with gambling capped at a third, so streaming and events get enough training rows.

## Category guide

| # | Category | Examples |
|---|---|---|
| 1 | Transfers | Zelle to a friend, Venmo, Cash App, moving money between own checking and savings |
| 2 | Food & Drink | Restaurants, Starbucks, Chipotle, McDonald's, DoorDash, bars, boba shops |
| 3 | Shopping | Amazon, Target, Zara, Sephora, Nike, Best Buy |
| 4 | Groceries | Trader Joe's, Whole Foods, Ralphs, Costco and Walmart groceries |
| 5 | **Subscriptions & Entertainment** | **Netflix, Spotify, Hulu, movies, concerts, video games, amusement parks, gambling** |
| 6 | Miscellaneous | Recognizable purchases that fit no other category |
| 7 | Gas | Shell, Chevron, Exxon, Mobil, Arco |
| 8 | Income / Deposits | Payroll, direct deposit, tax refund, interest, cash deposit |
| 9 | Cash Out (ATM) | ATM withdrawal, cash withdrawal at a bank, cash back |
| 10 | Fees & Adjustments | Overdraft, ATM fee, late fee, maintenance fee, bank correction |
| 11 | Professional Services | Lawyer, accountant, tax preparation, consulting, notary |
| 12 | Debt / Credit Payments | Credit card payments, student and personal loans, buy-now-pay-later |
| 13 | Finance & Investments | Brokerage contributions, stocks, ETFs, mutual funds, crypto |
| 14 | Automotive | Car repair, oil change, tires, car wash, vehicle registration |
| 15 | Travel | Airlines, hotels, Airbnb, Uber, Lyft, rental cars |
| 16 | Bills & Utilities | Electricity, water, internet, phone, rent, insurance |
| 17 | Health & Wellness | Doctor, pharmacy, dentist, therapy, gym membership |
| 18 | Gifts & Donations | Charities, GoFundMe, gifts, nonprofits |
| 19 | Education | Tuition, textbooks, school supplies |
| 20 | Uncertain | The memo does not support any category (truncated merchant name, bare reference number) |

**Rulebook.** Transaction type over merchant (an "ATM fee" is a Fee). Direction matters: cash-advance apps are
Income / Deposits on credits and Debt / Credit Payments on debits. Buy-now-pay-later is Debt / Credit Payments.
Gambling and sportsbooks are Subscriptions & Entertainment, tagged so they can be reported separately.
Uncertain only when the memo cannot support any category. `src/rules.py` encodes this order (first match wins).

## Method by stage

**L2 Exploration and cleaning.** DuckDB profile of all 15 parquet files (2,017,117,849 transactions, Jan 2022 to
Sep 2026) plus the balance and consumer-link files (duplicates, null / zero / negative balances, currencies,
accounts linked to several consumers). Cleaning keeps `memo_raw`; `text_lower` lowercases and collapses spaces;
`text_norm` also masks dates, card masks and long IDs. Pre-tag rule coverage and the most common memo patterns per
category are measured on a random sample.

**L3 Cleaning thresholds and labels.** Three TF-IDF token-removal settings (`max_df = 0.10`, `min_df = 2`,
`TF-IDF >= 0.3`) are compared with the case/spacing baseline on useful cues lost, noise removed and empty outputs.
Removal lost useful cues, so TF-IDF is kept for model features only. The four of us labeled the same 100 memos
independently; `agreement` reports unanimous and majority agreement, pairwise agreement and Fleiss' kappa.

**L4 Split and first classifiers.** The split is frozen before sampling or tuning and saved to
`split_plan.json`: train, then validation, then test by date, with about 20% of accounts held out entirely.
Test A = later transactions from seen accounts (time drift); Test B = later transactions from held-out accounts
(new customers). Baselines on the same validation rows: majority class, TF-IDF + logistic regression, the
Lecture 4 starter fastText setting and our PA4 setting.

**L5 Labels, tuning and coverage.** Rule suggestions with quoted cues, timed human review, and a 25% audit of
high-confidence suggestions (auto-accept only if 5% or fewer need correcting; ours was 14%, so every row was
reviewed). fastText is tuned on validation macro-F1 one group at a time (text column, word and character n-grams,
minCount; epochs x learning rate; vector size; character n-gram recheck), the top three settings are rerun with
three seeds, and the test sets are scored once. A prediction is auto-accepted when its score clears a cutoff
chosen on validation (highest coverage with at most 5% error among accepted rows); row and dollar coverage are
both reported, along with accuracy on seen vs unseen memo patterns and the share of unseen words.

**Project A.** Accepted debits predicted as Subscriptions & Entertainment are aggregated by month into counts,
dollars, dollars per active account and share of active accounts, split into streaming & subscriptions, events &
games, gambling and other, with and without gambling. Complete quarters are compared with FRED:

| FRED series | What it is | Compare with |
|---|---|---|
| `DRCARC1Q027SBEA` | PCE: Recreation services, billions of dollars SAAR, quarterly (BEA; includes gambling) | `theme_quarterly.csv` dollars |
| `CUUR0000SERA02` | CPI: Cable, satellite, and live streaming TV service, monthly | streaming & subscriptions group |
| `CPIRECSL` | CPI: Recreation, monthly, seasonally adjusted | dollars per active account |

Compare growth rates and turning points, not levels: the Prism sample is not the US population and the FRED
series are seasonally adjusted.

## Split dates: configured vs current labels

`config.yaml` holds the team cutoffs from our data profile (train before 2026-04-01, validation April to May
2026, test June to September 2026). The current `labels_v2.csv` was drawn from the accounts in our first 100
labels, whose history ends in January 2026, so the frozen plan fell back to the same 70% / 85% rule on those
accounts' own dates: **train before 2025-03-01, validation March to July 2025, test August 2025 to January 2026.**
Notebook 03 prints the date windows the labels actually use. Report these dates, or re-split new labels with the
team cutoffs, before the final report.

## Notes

- fastText 0.9.x occasionally raises "Encountered NaN" depending on memory state. Failed fits are retried and
  logged; if the final fit fails, the next seed or candidate is used and `final_model.json` says so.
- `fasttext.bucket` is 200,000 instead of 2,000,000, cutting the model from about 1.6 GB to about 160 MB.
- Use local AI only for transaction-level work, and never commit memo text: `data/` and `outputs/` are ignored,
  and notebooks print aggregate tables only.
- Code sources: L2 profile and agreement file from Vanessa's PA4; cleaning, split, sampling, review loop, audit,
  coverage and Project A scaffold from Kerim's PA5; gambling rules, character n-gram recheck, seen / unseen
  patterns and the smaller bucket from Aiden's PA5.

## References

- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. EACL.
- Jørgensen, R. K., & Igel, C. (2021). Machine learning for financial transaction classification across companies
  using character-level word embeddings of text fields. *Intelligent Systems in Accounting, Finance and Management*.
- Toran, L., Van Der Walt, C., Sammarone, A., & Keller, A. (2023). Scalable and weakly supervised bank transaction
  classification. arXiv:2305.18430.
- U.S. Bureau of Economic Analysis and U.S. Bureau of Labor Statistics series via FRED, Federal Reserve Bank of St. Louis.

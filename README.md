# NLP-Based Classification of Financial Transactions

DSC 180A Capstone, Section D12, Group 3 (Prism Data): Aiden Lai, Vanessa Feng, Katie Hannigan, Kerim Orucevic

Bank transaction memos are short, inconsistent and full of abbreviations, reference numbers and boilerplate. This
project classifies anonymized Prism Data transactions into a 20-category taxonomy. We build a reviewed label set
with rule-based AI suggestions and human review, train a fastText classifier, and evaluate it on future
transactions from seen accounts (Test A) and from held-out accounts (Test B).

## Repository layout

```
config.yaml          paths, split cutoffs, sampling targets, fastText grid, review cutoff rule
requirements.txt
src/
  main.py            entrypoint: python -m src.main --config config.yaml
  taxonomy.py        the 20 categories, name parsing, fastText label tokens
  rules.py           ordered pre-tag rules (AI label suggestions with quoted cue + confidence)
  clean.py           memo cleaning: text_lower, text_norm (masked IDs), memo pattern
  data.py            parquet readers (DataHub only)
  split.py           frozen time + account split
  sample.py          choose the next batch to label (sparse categories, Pareto head, random)
  review.py          review file, interactive review loop, audit decision, export of labels_v2.csv
  model.py           fastText fit / predict / scores
  tune.py            staged validation tuning + seed stability
  evaluate.py        per-class report, confusion matrix, errors, review cutoff, Test A / Test B
  runtime.py         fit, load and inference timing; human review effort
  monthly.py         Project A scaffold: monthly count + dollar series per category
  explore.py         DuckDB profile of the raw parquet files (totals, by month, 70%/85% cutoff months)
  agreement.py       labeler agreement on the shared multiply-labeled memos
notebooks/           earlier PA notebooks (outputs stripped)
aiden/               Aiden's PA5 notebook and outputs
Vanessa_PA4.ipynb    Vanessa's PA4 notebook
data/                labels and review files (git-ignored: transaction memos stay on DataHub)
outputs/             model, metrics and figures (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt      # on numpy >= 2 use fasttext-numpy2-wheel instead of fasttext-wheel
```

Edit `config.yaml` so `paths.data_dir` points at the Prism parquet files and `paths.labels` at the reviewed
label file. Label files need these columns: `item_id, split, account_key, date, amount, direction, memo_raw,
category` (optional: `label_source, review_status, note`). `split` is one of `train, val, test_A, test_B,
excluded`.

## Running

**Model (rerun any time from `labels_v2.csv`):**

```bash
python -m src.main --config config.yaml                       # prepare, train, evaluate, runtime
python -m src.main --config config.yaml --steps evaluate      # any subset, in order
python -m src.main --config config.yaml --steps monthly       # Project A series (needs parquet access)
```

| Step | What it does |
|---|---|
| `prepare` | Drops unlabeled rows, adds cleaned text, checks for leakage (no Test B account in train/val, no transaction in two splits), writes class counts and a Pareto check |
| `train` | Majority-class and previous-settings baselines, staged tuning on validation, final fit and saved model |
| `evaluate` | Per-class P/R/F1, confusion matrix, error table, review cutoff, Test A and Test B scored once |
| `runtime` | Fit time, model load time, inference rows/second, estimated time for 1M memos and the full dataset, human review time |
| `monthly` | Project A: monthly series per predicted category, Subscriptions & Entertainment broken into groups, FRED download |

**Data and label quality:**

```bash
python -m src.main --config config.yaml --steps explore       # dataset profile (needs parquet access + duckdb)
python -m src.main --config config.yaml --steps agreement     # agreement on the shared 100 memos
```

`explore` reproduces the report's data description (2,017,117,849 transactions, 1,052,849 accounts,
Jan 2022 to Sep 2026) and the months nearest 70% and 85% of transactions, which set the split cutoffs.
`agreement` reads the 100-memo file with one column per group member and reports how often all four agreed,
majority agreement, pairwise agreement, Fleiss' kappa and each person vs the final agreed label.

**Growing the label set (DataHub):**

```bash
python -m src.main --config config.yaml --steps sample        # scan accounts, freeze split, write review_batch.csv
python -m src.review --config config.yaml                     # interactive, timed review (rerun to continue)
python -m src.main --config config.yaml --steps audit         # accept deferred AI labels only if the audit passes
python -m src.main --config config.yaml --steps build-labels  # seed labels + reviewed batch -> labels_v2.csv
```

Review loop keys: `Enter` accept the suggestion, `7` or `groceries` correct (optionally followed by a note),
`u <what is missing>` unresolved, `b` undo, `s` skip, `?` category numbers, `q` save and quit.

## Category guide

| # | Category | Examples |
|---|---|---|
| 1 | Transfers | Zelle to a friend, Venmo, Cash App, moving money between own checking and savings |
| 2 | Food & Drink | Restaurants, Starbucks, Chipotle, McDonald's, DoorDash, bars, boba shops |
| 3 | Shopping | Amazon, Target, Zara, Sephora, Nike, Best Buy |
| 4 | Groceries | Trader Joe's, Whole Foods, Ralphs, Costco and Walmart groceries |
| 5 | Subscriptions & Entertainment | Netflix, Spotify, Hulu, movies, concerts, video games, amusement parks, gambling |
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

### Labeling rulebook

- **Transaction type over merchant:** an "ATM fee" is a Fee, not Cash Out (ATM).
- **Direction matters:** cash-advance apps (Dave, Earnin, Brigit, MoneyLion, ...) are Income / Deposits when
  money comes in and Debt / Credit Payments when a repayment goes out.
- **Edge cases:** buy-now-pay-later is Debt / Credit Payments; gambling and sportsbooks are Subscriptions &
  Entertainment (tagged `subcategory = gambling` so they can be reported separately).
- **Uncertain** only when the memo lacks the information to support any category.

The pre-tag rules in `src/rules.py` encode this order: first match wins, so fees and income are checked before
merchants, P2P apps before the person on the other end, and broad merchant lists and generic words last.

## Method

**Labels.** Each memo gets a proposed label from the pre-tag rules (with the matched text and a high or low
confidence); where no rule matches, an earlier fastText model can give a second suggestion. Reviewers accept,
correct with a note, or mark unresolved, and every decision is timed. A random 25% of high-confidence training
suggestions is audited; the rest would be auto-accepted only if 5% or fewer needed correcting. In our PA5 batch
14% needed correcting, so every row was reviewed by hand. Training memos are chosen on purpose (sparse
categories, common memo patterns, random memos); validation and test memos are random samples.

**Cleaning.** `memo_raw` is never changed. `text_lower` lowercases and collapses spaces; `text_norm` also masks
dates, card masks and long IDs (`<date>`, `<mask>`, `<num>`). Both are compared during tuning.

**Split.** Fixed before any sampling or tuning and saved to `split_plan.json`:
train before 2026-04-01, validation April-May 2026, test June-September 2026. About 20% of accounts are held out
entirely. Test A = later transactions from accounts seen in training; Test B = later transactions from held-out
accounts.

**Model.** fastText supervised classifier, tuned on validation macro-F1 one group at a time: features (text
column, word n-grams, character n-grams, minCount), then epochs x learning rate, then vector size, then a recheck
of character n-grams at the tuned learning settings. The top three settings are rerun with three seeds and the
best mean validation macro-F1 wins. Test sets are scored once with the final model.

**Evaluation.** Accuracy, macro-F1, majority-class baseline, per-class P/R/F1, confusion matrix, accuracy on
seen vs unseen memo patterns, share of memo words never seen in training, Test A vs Test B with class mix and the gambling slice. Review coverage: a
prediction is accepted when its score clears a cutoff chosen on validation (highest row coverage with at most 5%
error among accepted rows); the rest goes to human review. Row and dollar coverage are both reported.

## Project A theme: Subscriptions & Entertainment

The `monthly` step runs the final model over the random sample, keeps accepted predictions (score above the
review cutoff), and builds monthly transaction counts, dollars and active-account measures for Subscriptions &
Entertainment. Within the category, memos are grouped by the pre-tag rule that matches them:

| Group | Examples |
|---|---|
| streaming & subscriptions | Netflix, Spotify, Hulu, Apple.com/bill, YouTube, memberships |
| events & games | Ticketmaster, AMC, Xbox, Steam, theme parks, bowling |
| gambling | DraftKings, FanDuel, PrizePicks, lottery, casinos |
| other | memos the model put in the category that no rule recognizes |

Gambling is reported separately and every series is also given excluding gambling, because it was a large,
volatile share of this category (about 30% of Test B memos).

External comparison (`project_a.fred_series` in `config.yaml`):

| FRED series | What it is | Compare with |
|---|---|---|
| `DRCARC1Q027SBEA` | PCE: Recreation services, billions of dollars SAAR, quarterly (BEA; includes gambling) | `theme_quarterly.csv` dollars, series "all" |
| `CUUR0000SERA02` | CPI: Cable, satellite, and live streaming TV service, monthly, not seasonally adjusted | streaming & subscriptions group |
| `CPIRECSL` | CPI: Recreation, monthly, seasonally adjusted | dollars per active account |

Compare growth rates and turning points, not levels: our sample covers Prism accounts, not the US population,
and the national series are seasonally adjusted while ours are not.

Outputs: `theme_monthly.csv`, `theme_monthly_by_group.csv`, `theme_quarterly.csv`, `theme_monthly.png`,
`fred_series.csv` (when FRED is reachable).

## Outputs (`outputs/`)

`metrics.json` (everything below in one file), `class_counts.csv`, `baseline.csv`, `experiments.csv`,
`stability.csv`, `final_model.json`, `data_summary.json`, `transactions_by_month.csv`, `agreement.json`, `model.bin`, `scores.csv`, `per_class_*.csv`, `confusion.png`,
`errors_val.csv`, `coverage_val.csv`, `class_mix.csv`, `predictions.csv`, `monthly_category_series.csv`,
`manifest.json`.

## Notes

- fastText 0.9.x occasionally raises "Encountered NaN" depending on memory state rather than the setting alone.
  Failed fits are retried and logged in `experiments.csv`; if the final fit fails, the next seed or candidate is
  used and `final_model.json` says so.
- `fasttext.bucket` is 200,000 instead of the default 2,000,000, which cuts the model from about 1.6 GB to about
  160 MB with character n-grams on, with little effect at this vocabulary size.
- Code sources: cleaning, split, sampling, review loop and audit come from Kerim's PA5 notebook; gambling rules,
  the char n-gram recheck, seen/unseen pattern analysis and the smaller bucket come from Aiden's PA5 notebook;
  the DuckDB data profile, 70%/85% cutoff derivation, unseen-word share and the 4-labeler 100-memo file
  (agreement step) come from Vanessa's PA4 notebook.

## References

- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. EACL.
- Jørgensen, R. K., & Igel, C. (2021). Machine learning for financial transaction classification across companies
  using character-level word embeddings of text fields. *Intelligent Systems in Accounting, Finance and Management*.
- Toran, L., Van Der Walt, C., Sammarone, A., & Keller, A. (2023). Scalable and weakly supervised bank transaction
  classification. arXiv:2305.18430.

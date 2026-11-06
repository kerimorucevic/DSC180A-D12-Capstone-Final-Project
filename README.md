# NLP-Based Classification of Financial Transactions

DSC 180A Capstone, Section D12, Group 3 (Prism Data): Aiden Lai, Vanessa Feng, Katie Hannigan, Kerim Orucevic

Bank transaction memos are short, inconsistent and full of abbreviations, reference numbers and bank
boilerplate. This project builds a reviewed label set for anonymized Prism Data transactions and trains a fastText
classifier that assigns each memo to one of 20 spending categories, evaluated on future transactions from seen
and unseen accounts.

## Repository

```
config.yaml        paths, split cutoffs, sampling targets, review audit rule, fastText settings
src/
  rules.py         the 20 categories and the rulebook as ordered pre-tag rules
  labeling.py      cleaning, frozen split, choosing memos, AI-suggested labels, review loop, audit
  model.py         fastText training format, tuning on validation, evaluation on Test A / Test B
  main.py          entrypoint
notebooks/         our assignment notebooks (Vanessa PA4, Kerim PA5, Aiden PA5)
data/, outputs/    labels and results (git-ignored: memos stay on DataHub)
```

## Running

```bash
pip install -r requirements.txt
python -m src.main --config config.yaml                      # train + evaluate from data/labels_v2.csv
```

Growing the label set (DataHub):

```bash
python -m src.main --config config.yaml --steps sample        # choose memos, write review_batch.csv
python -m src.labeling --config config.yaml                   # interactive review (rerun to continue)
python -m src.main --config config.yaml --steps audit,build-labels
```

Review loop keys: `Enter` accept the suggestion, a number or category name (plus an optional note) to correct,
`u <note>` unresolved, `b` undo, `q` save and quit.

## Method

### Categories and rulebook

We defined 20 transaction categories modeled on the spending categories used by Chase, Bank of America and
American Express:

Transfers · Food & Drink · Shopping · Groceries · Subscriptions & Entertainment · Miscellaneous · Gas ·
Income / Deposits · Cash Out (ATM) · Fees & Adjustments · Professional Services · Debt / Credit Payments ·
Finance & Investments · Automotive · Travel · Bills & Utilities · Health & Wellness · Gifts & Donations ·
Education · Uncertain

Many memos fit more than one category, so a rulebook keeps labeling decisions consistent:

- **Transaction type over merchant:** an "ATM fee" is a Fee, not Cash Out (ATM).
- **Credit/debit direction:** cash-advance apps count as Income when money comes in and as Debt / Credit Payments
  when a repayment goes out.
- **Edge cases:** buy-now-pay-later services are Debt / Credit Payments; gambling and sportsbook sites are
  Subscriptions & Entertainment.
- **Uncertain:** only when the memo does not contain enough information to support any category, such as a
  truncated merchant name or a bare reference number.

### Labels: AI suggestions, human review

Each memo first receives a proposed label from ordered regular-expression pre-tag rules (`src/rules.py`). A rule
matches a merchant name or keyword (for example "zelle" for Transfers) and returns the matched text and a high or
low confidence. Rules are checked in order and the first match wins, so transaction rules (fees, ATM
withdrawals, income) take priority over merchant rules. Where no rule matches, a fastText model trained on earlier
labels gives a second suggestion.

Reviewers label memos in an interactive loop that shows the raw memo, amount, credit/debit flag, date and
suggested label, and accept the suggestion, correct it with a short note, or mark it unresolved. Each decision is
timed. A random 25% sample of high-confidence suggestions is audited; the rest would be auto-accepted only if 5% or
fewer needed correcting. In our batch 14% needed correcting, so every memo was labeled by hand.

Training memos are chosen on purpose:

- **Rare categories:** memos the rules propose for categories with few examples, so every category has training data.
- **Common patterns:** memos from the most frequent memo patterns (first three meaningful words after removing
  bank boilerplate).
- **Random memos:** so the model also sees memos no rule recognizes.

Validation and test memos are random samples, so evaluation reflects real transactions.

### Split

Fixed before choosing memos and saved to `split_plan.json`: train, then validation, then test by date, with about
20% of accounts held out entirely. **Test A** = later transactions from accounts seen in training; **Test B** =
later transactions from held-out accounts. `outputs/split_summary.csv` lists the date window each split
actually covers.

### Cleaning and model

The raw memo is always kept unchanged. For modeling, memos are lowercased with extra spaces removed
(`text_lower`). A second version (`text_norm`) also replaces dates, masked card numbers and long ID strings with
placeholder tokens (`<date>`, `<mask>`, `<num>`), so reference codes such as "JPM99aa9jfks" are not treated as
meaningful words. Each labeled memo is written in fastText's training format, category token then text:

```
__label__groceries trader joes
```

Both versions are compared during tuning on validation macro-F1 (word and character n-grams, then epochs and
learning rate; the best three settings are rerun with other seeds). Test A and Test B are scored once with the
final model.

We also tried TF-IDF to filter memo tokens during cleaning; it removed useful merchant cues along with the noise,
so it is not part of the pipeline.

### Outputs (`outputs/`)

`class_counts.csv`, `split_summary.csv`, `experiments.csv`, `stability.csv`, `final_model.json`, `model.bin`,
`scores.csv` (with majority-class baseline), `per_class_val.csv`, `confusion.png`, `errors_val.csv`,
`class_mix.csv`.

## References

- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. EACL.
- Jørgensen, R. K., & Igel, C. (2021). Machine learning for financial transaction classification across companies
  using character-level word embeddings of text fields. *Intelligent Systems in Accounting, Finance and Management*.
- Toran, L., Van Der Walt, C., Sammarone, A., & Keller, A. (2023). Scalable and weakly supervised bank transaction
  classification. arXiv:2305.18430.

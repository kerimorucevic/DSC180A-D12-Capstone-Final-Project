"""Dataset profile with DuckDB (from Vanessa's PA4 notebook). Runs on DataHub against the raw parquet files.

Writes:
  data_summary.json        totals, accounts, date range, credit/debit share, empty memos, split sizes
  transactions_by_year.csv
  transactions_by_month.csv  with cumulative % and the months nearest 70% / 85% (basis for the split cutoffs)
  transactions_by_month.png, cumulative_transactions.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def profile(cfg, out):
    import duckdb
    src = str(Path(cfg["paths"]["data_dir"]) / cfg["data"]["parquet_glob"])
    vs, ts = cfg["split"]["val_start"], cfg["split"]["test_start"]
    con = duckdb.connect()
    q = lambda sql: con.execute(sql.replace("{src}", f"read_parquet('{src}')")).df()

    tot = q("""SELECT COUNT(*) AS transactions, COUNT(DISTINCT account_key) AS accounts,
                      MIN(posted_date) AS first_date, MAX(posted_date) AS last_date,
                      AVG(CASE WHEN credit_or_debit = 2 THEN 1.0 ELSE 0 END) AS debit_share,
                      SUM(CASE WHEN memo IS NULL THEN 1 ELSE 0 END) AS null_memos,
                      SUM(CASE WHEN TRIM(TRY_CAST(memo AS VARCHAR)) = '' THEN 1 ELSE 0 END) AS empty_memos
               FROM {src}""").iloc[0]
    year = q("""SELECT YEAR(posted_date) AS year, COUNT(*) AS transactions, COUNT(DISTINCT account_key) AS accounts
                FROM {src} GROUP BY 1 ORDER BY 1""")
    month = q("""SELECT DATE_TRUNC('month', posted_date) AS month, COUNT(*) AS transactions
                 FROM {src} GROUP BY 1 ORDER BY 1""")
    split = q(f"""SELECT CASE WHEN posted_date < DATE '{vs}' THEN 'train'
                              WHEN posted_date < DATE '{ts}' THEN 'val' ELSE 'test' END AS split,
                         COUNT(*) AS transactions, COUNT(DISTINCT account_key) AS accounts
                  FROM {{src}} GROUP BY 1""")
    con.close()

    month["month"] = pd.to_datetime(month["month"])
    month["cumulative_pct"] = month["transactions"].cumsum() / month["transactions"].sum() * 100
    nearest = {f"{t}%": str(month.iloc[(month.cumulative_pct - t).abs().argmin()]["month"].date()) for t in (70, 85)}
    split["pct"] = (split.transactions / split.transactions.sum() * 100).round(2)
    year.to_csv(out / "transactions_by_year.csv", index=False)
    month.to_csv(out / "transactions_by_month.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(month["month"], month["transactions"] / 1e6, width=20)
    ax.set_title("Transactions by month")
    ax.set_ylabel("transactions (millions)")
    plt.tight_layout()
    plt.savefig(out / "transactions_by_month.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(month["month"], month["cumulative_pct"], marker="o")
    for t in (70, 85):
        ax.axhline(t, ls="--", c="grey")
    for d in (vs, ts):
        ax.axvline(pd.Timestamp(d), ls=":", c="red")
    ax.set_title("Cumulative share of transactions (red: configured split cutoffs)")
    ax.set_ylabel("%")
    plt.tight_layout()
    plt.savefig(out / "cumulative_transactions.png", dpi=150)
    plt.close(fig)

    summary = dict(transactions=int(tot.transactions), accounts=int(tot.accounts),
                   tx_per_account=round(tot.transactions / tot.accounts, 1),
                   first_date=str(pd.Timestamp(tot.first_date).date()), last_date=str(pd.Timestamp(tot.last_date).date()),
                   debit_share=round(float(tot.debit_share), 4), null_memos=int(tot.null_memos),
                   empty_memos=int(tot.empty_memos), months_nearest=nearest,
                   split_sizes=split.set_index("split").to_dict("index"))
    summary["account_files"] = account_checks(cfg, src)
    json.dump(summary, open(out / "data_summary.json", "w"), indent=2, default=str)
    print(f"explore: {summary['transactions']:,} transactions | {summary['accounts']:,} accounts | "
          f"{summary['first_date']} to {summary['last_date']} | 70%/85% months {nearest}")
    return summary


def account_checks(cfg, txn_src):
    """Lecture 2 account + balance EDA: duplicates, null/zero/negative balances, currencies, link-file joins."""
    import duckdb
    p = cfg["paths"]
    res = {}
    con = duckdb.connect()
    bal = p.get("account_balances")
    if bal and Path(bal).exists():
        b = con.execute(f"""SELECT COUNT(*) AS rows, COUNT(DISTINCT account_key) AS accounts,
                                   SUM(CASE WHEN balance IS NULL THEN 1 ELSE 0 END) AS null_balance,
                                   SUM(CASE WHEN balance = 0 THEN 1 ELSE 0 END) AS zero_balance,
                                   SUM(CASE WHEN balance < 0 THEN 1 ELSE 0 END) AS negative_balance,
                                   MIN(balance_date) AS first_balance_date, MAX(balance_date) AS last_balance_date
                            FROM read_parquet('{bal}')""").df().iloc[0].to_dict()
        b["duplicate_account_keys"] = int(b["rows"] - b["accounts"])
        b["currencies"] = con.execute(f"""SELECT COALESCE(CAST(iso_currency_code AS VARCHAR), 'None') AS c, COUNT(*) AS n
                                          FROM read_parquet('{bal}') GROUP BY 1 ORDER BY 2 DESC""").df() \
            .set_index("c")["n"].to_dict()
        if cfg["explore"].get("check_accounts_without_transactions"):
            b["accounts_without_transactions"] = int(con.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{bal}') a
                WHERE a.account_key NOT IN (SELECT DISTINCT account_key FROM read_parquet('{txn_src}'))""").fetchone()[0])
        res["balances"] = b
    link = p.get("consumer_links")
    if link and Path(link).exists():
        l = con.execute(f"""SELECT COUNT(*) AS rows, COUNT(DISTINCT consumer_key) AS consumers,
                                   COUNT(DISTINCT account_key) AS accounts FROM read_parquet('{link}')""").df().iloc[0].to_dict()
        l["accounts_linked_to_multiple_consumers"] = int(con.execute(f"""
            SELECT COUNT(*) FROM (SELECT account_key FROM read_parquet('{link}') GROUP BY 1
                                  HAVING COUNT(DISTINCT consumer_key) > 1)""").fetchone()[0])
        l["note"] = "transactions have no consumer_key; joining through this file can duplicate activity"
        res["consumer_links"] = l
    con.close()
    return {k: {kk: (str(vv) if not isinstance(vv, (int, float, dict, str)) else vv) for kk, vv in v.items()}
            for k, v in res.items()}


def merchant_frequencies(memos, directions, out, top=25):
    """Lecture 2/3: measured memo-pattern frequencies per category the pre-tag rules propose, on a random sample.
    Verifies the merchant lists against real memos and shows what each category is made of."""
    from .clean import memo_pattern, text_norm
    from .rules import pretag
    d = pd.DataFrame({"memo_raw": memos, "direction": directions})
    tags = [pretag(m, r) for m, r in zip(d.memo_raw, d.direction)]
    d["category"] = [t[0] or "(no rule)" for t in tags]
    d["rule_id"] = [t[2] or "" for t in tags]
    d["pattern"] = d.memo_raw.map(lambda m: memo_pattern(text_norm(m)))
    cov = d.category.value_counts().rename("memos").to_frame()
    cov["share"] = (cov.memos / len(d)).round(4)
    cov.to_csv(out / "rule_coverage_by_category.csv")
    tops = (d.groupby(["category", "rule_id", "pattern"]).size().rename("memos").reset_index()
            .sort_values(["category", "memos"], ascending=[True, False]).groupby("category").head(top))
    tops.to_csv(out / "top_patterns_by_category.csv", index=False)
    print(f"merchant frequencies: rules matched {1 - cov.loc['(no rule)', 'share'] if '(no rule)' in cov.index else 1:.0%} "
          f"of {len(d):,} memos")
    return dict(memos=len(d), rule_coverage=cov.share.to_dict())

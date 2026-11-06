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
    json.dump(summary, open(out / "data_summary.json", "w"), indent=2, default=str)
    print(f"explore: {summary['transactions']:,} transactions | {summary['accounts']:,} accounts | "
          f"{summary['first_date']} to {summary['last_date']} | 70%/85% months {nearest}")
    return summary

"""Project A scaffold: turn predicted categories into monthly count + dollar series.

define eligible transactions -> aggregate monthly count + value -> check coverage -> compare to a FRED series.
Only accepted predictions (score >= review cutoff, not Uncertain) are counted; debits for spending categories,
credits for Income / Deposits. Caveat: the random sample is row groups, not an account panel. For the real
project, sample accounts that have the category of interest and follow them over time.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clean import text_lower, text_norm
from .model import predict


def monthly_series(df, model, text_col, cutoff, out, category=None):
    clean = text_norm if text_col == "text_norm" else text_lower
    d = df[["account_key", "date", "amount", "direction", "memo_raw"]].copy()
    d["predicted"], d["score"] = predict(model, d["memo_raw"].map(clean))
    d["accepted"] = (d.score >= cutoff) & (d.predicted != "Uncertain")
    want = np.where(d.predicted == "Income / Deposits", "credit", "debit")
    elig = d[d.accepted & ((d.direction == want) | (d.direction == "unknown")) & d.date.notna()].copy()
    elig["month"] = pd.to_datetime(elig.date).dt.to_period("M")
    monthly = (elig.groupby(["predicted", "month"])
               .agg(transactions=("amount", "size"), dollars=("amount", lambda x: x.abs().sum()),
                    accounts=("account_key", "nunique")).reset_index())
    monthly.to_csv(out / "monthly_category_series.csv", index=False)
    if category:
        m = monthly[monthly.predicted == category].set_index("month")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        m.transactions.plot(ax=ax, marker="o", label="transactions")
        m.dollars.plot(ax=ax, secondary_y=True, marker="s", label="dollars")
        ax.set_title(f"{category}: monthly count and dollars")
        plt.tight_layout()
        plt.savefig(out / "monthly_series.png", dpi=150)
        plt.close(fig)
    return dict(eligible=len(elig), total=len(d), share=round(len(elig) / max(len(d), 1), 3))

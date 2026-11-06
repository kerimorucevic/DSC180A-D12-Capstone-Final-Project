"""Project A: turn predicted categories into monthly count + dollar series and compare with FRED.

Theme: Subscriptions & Entertainment (set in config `project_a.category`).

define eligible transactions -> aggregate monthly count + value -> check coverage -> compare to FRED series.
Only accepted predictions (score >= review cutoff, not Uncertain) count; debits for spending categories,
credits for Income / Deposits. Within the theme category, memos are split into groups by the pre-tag rule that
matches them (streaming/subscriptions, events/games, gambling, other), because gambling is a large and volatile
share of this category and the FRED comparison should be read with and without it.

Caveat: the random sample is row groups, not an account panel. For the final project, sample accounts that have
the category and follow them over time (per-account columns below are the start of that).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clean import text_lower, text_norm
from .model import predict
from .rules import pretag

RULE_GROUP = {"streaming": "streaming & subscriptions", "subs_generic": "streaming & subscriptions",
              "games_events": "events & games", "gambling": "gambling"}


def theme_group(memo, direction):
    return RULE_GROUP.get(pretag(memo, direction)[2], "other")


def _agg(d, keys):
    return (d.groupby(keys).agg(transactions=("amount", "size"), dollars=("amount", lambda x: x.abs().sum()),
                                accounts=("account_key", "nunique")).reset_index())


def fetch_fred(series_ids, out):
    """Download FRED series as CSV (works where fred.stlouisfed.org is reachable; skipped otherwise)."""
    frames = {}
    for sid in series_ids:
        try:
            s = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
            s.columns = ["date", sid]
            s["date"] = pd.to_datetime(s["date"])
            s[sid] = pd.to_numeric(s[sid], errors="coerce")
            frames[sid] = s.set_index("date")[sid]
        except Exception as e:
            print(f"FRED {sid}: not downloaded ({type(e).__name__}); download it manually into outputs/")
    if frames:
        f = pd.concat(frames, axis=1)
        f.to_csv(out / "fred_series.csv")
        return f
    return None


def monthly_series(df, model, text_col, cutoff, out, category=None, fred_ids=()):
    clean = text_norm if text_col == "text_norm" else text_lower
    d = df[["account_key", "date", "amount", "direction", "memo_raw"]].copy()
    d["predicted"], d["score"] = predict(model, d["memo_raw"].map(clean))
    d["accepted"] = (d.score >= cutoff) & (d.predicted != "Uncertain")
    want = np.where(d.predicted == "Income / Deposits", "credit", "debit")
    elig = d[d.accepted & ((d.direction == want) | (d.direction == "unknown")) & d.date.notna()].copy()
    elig["month"] = pd.to_datetime(elig.date).dt.to_period("M")
    monthly = _agg(elig, ["predicted", "month"])
    monthly.to_csv(out / "monthly_category_series.csv", index=False)
    info = dict(eligible=len(elig), total=len(d), share=round(len(elig) / max(len(d), 1), 3))
    if not category:
        return info

    # coverage for the theme: how many memos the model called this category, and how many were accepted
    called = d[d.predicted == category]
    info["theme"] = dict(category=category, predicted=len(called), accepted=int(called.accepted.sum()),
                         accepted_share=round(float(called.accepted.mean()), 3) if len(called) else None)

    t = elig[elig.predicted == category].copy()
    t["group"] = [theme_group(m, r) for m, r in zip(t.memo_raw, t.direction)]
    by_group = _agg(t, ["group", "month"])
    by_group.to_csv(out / "theme_monthly_by_group.csv", index=False)
    total = _agg(t, ["month"]).assign(series="all")
    no_gamb = _agg(t[t.group != "gambling"], ["month"]).assign(series="excluding gambling")
    theme = pd.concat([total, no_gamb])
    active = elig.groupby("month")["account_key"].nunique().rename("active_accounts")
    theme = theme.join(active, on="month")
    theme["dollars_per_active_account"] = theme.dollars / theme.active_accounts
    theme["share_of_active_accounts"] = theme.accounts / theme.active_accounts
    theme.to_csv(out / "theme_monthly.csv", index=False)
    info["theme"]["groups"] = t.group.value_counts().to_dict()

    # quarterly totals line up with BEA's quarterly PCE recreation services
    # (partial quarters at either end of the sample are dropped: they would show as fake jumps)
    q = (theme.assign(quarter=theme.month.dt.asfreq("Q")).groupby(["series", "quarter"])
         .agg(transactions=("transactions", "sum"), dollars=("dollars", "sum"), months=("month", "nunique")))
    q = q[q.months == 3].drop(columns="months")
    q.reset_index().to_csv(out / "theme_quarterly.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, g in theme.groupby("series"):
        axes[0].plot(g.month.dt.to_timestamp(), g.dollars, marker="o", label=name)
    axes[0].set_title(f"{category}: monthly dollars")
    axes[0].legend()
    piv = by_group.pivot(index="month", columns="group", values="dollars").fillna(0)
    piv.index = piv.index.to_timestamp()
    axes[1].stackplot(piv.index, piv.T.values, labels=piv.columns)
    axes[1].set_title("by group")
    axes[1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "theme_monthly.png", dpi=150)
    plt.close(fig)

    if fred_ids:
        fred = fetch_fred(fred_ids, out)
        info["fred_downloaded"] = list(fred.columns) if fred is not None else []
    return info

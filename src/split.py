"""Combined time + account split, frozen before any sampling or tuning.

train < val_start <= validation < test_start <= test.  A share of accounts with test-window activity is held
out entirely: none of their rows are used in train or validation.
  test_A = future rows from accounts seen in training   (time drift)
  test_B = future rows from held-out accounts           (new-customer transfer)
The plan is written to split_plan.json and reused on every rerun.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd


def windows(dates, val_start, test_start):
    dates = pd.to_datetime(dates)
    return pd.Series(np.select([dates.isna(), dates >= test_start, dates >= val_start],
                               ["no_date", "test", "val"], "train"), index=dates.index)


def cutoffs_70_85(dates):
    """Month boundaries nearest to 70% and 85% of transactions (fallback when team cutoffs don't fit a pool)."""
    m = dates.dt.to_period("M").value_counts().sort_index()
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"), fill_value=0)
    cum = m.cumsum() / m.sum()
    vs = ((cum - .70).abs().idxmin() + 1).to_timestamp()
    ts = ((cum - .85).abs().idxmin() + 1).to_timestamp()
    if vs >= ts:
        q = dates.dropna().sort_values()
        vs, ts = q.iloc[int(len(q) * .70)].normalize(), q.iloc[int(len(q) * .85)].normalize()
    return vs, ts


def acct_hash(account, seed):
    return hashlib.md5(f"{seed}-{account}".encode()).hexdigest()


def freeze_plan(cfg, pool, prior_labels=None):
    """Create split_plan.json once; later calls load it unchanged."""
    path = cfg["paths"]["split_plan"]
    if path.exists():
        print("loaded frozen split plan", path)
        return json.load(open(path))
    s, targets, seed = cfg["split"], cfg["sampling"]["bucket_targets"], cfg["seed"]
    vs, ts = pd.Timestamp(s["val_start"]), pd.Timestamp(s["test_start"])
    per = windows(pool["date"], vs, ts)
    need = {"train": targets["train"], "val": targets["val"], "test": targets["test_A"] + targets["test_B"]}
    source = "team cutoffs"
    if not all((per == w).sum() >= 2 * n for w, n in need.items()):
        vs, ts = cutoffs_70_85(pool["date"])
        source = "70/85 rule on the pool's own dates"
        per = windows(pool["date"], vs, ts)
    early = set()
    if prior_labels is not None and len(prior_labels):
        pper = windows(prior_labels["date"], vs, ts)
        early = set(prior_labels.loc[pper.isin(["train", "val"]), "account_key"].astype(str))
    tcount = pool[per == "test"].groupby("account_key").size()
    # prefer accounts with no earlier labels, then a seeded hash order
    cands = sorted([a for a, n in tcount.items() if n >= 3], key=lambda a: (a in early, acct_hash(a, seed)))
    n_hold = max(3, round(s["heldout_account_share"] * len(tcount)))
    held, covered = [], 0
    for a in cands:
        if len(held) >= n_hold and covered >= 1.5 * targets["test_B"]:
            break
        held.append(a)
        covered += tcount[a]
    plan = dict(val_start=str(vs.date()), test_start=str(ts.date()), cutoff_source=source,
                held_out_accounts=held, frozen_at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(plan, open(path, "w"), indent=2)
    return plan


def assign_split(df, plan):
    """Split for already-labeled rows (used to re-split seed labels with the frozen plan)."""
    vs, ts = pd.Timestamp(plan["val_start"]), pd.Timestamp(plan["test_start"])
    per = windows(df["date"], vs, ts)
    held = df["account_key"].astype(str).isin(set(plan["held_out_accounts"]))
    return pd.Series(np.select(
        [per == "no_date", held & (per == "test"), held, per == "test", per == "val"],
        ["train", "test_B", "excluded", "test_A", "val"], "train"), index=df.index)

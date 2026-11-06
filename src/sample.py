"""Choose the next batch of memos to label.

- Evaluation buckets (val, test_A, test_B): plain random samples of their window, capped per account so one
  busy account can't dominate. They stay representative, so validation scores mean something.
- Training bucket is targeted:
  1. sparse categories first: rows the pre-tag rules propose for categories under `sparse_target` labels;
  2. Pareto head: up to 2 rows from each of the most common memo patterns;
  3. random remainder: memos no rule recognizes, so the model doesn't only see easy cases.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from .clean import add_text_cols, txn_key
from .rules import pretag
from .split import windows
from .taxonomy import CATEGORIES


def _pick(df, n, cap, taken, acct_counts, rng):
    """Random rows not yet taken, at most `cap` per account (the cap doubles if the pool runs short)."""
    df = df[~df.index.isin(taken)]
    df = df.iloc[rng.permutation(len(df))]
    out = []
    while len(out) < n and len(df):
        for i, a in zip(df.index, df["account_key"]):
            if len(out) >= n:
                break
            if i not in taken and acct_counts[a] < cap:
                out.append(i)
                taken.add(i)
                acct_counts[a] += 1
        df = df[~df.index.isin(taken)]
        cap *= 2
        if cap > 10 ** 6:
            break
    return out


def build_pool(raw, prior_labels):
    """Drop duplicates, empty memos and transactions that are already labeled."""
    pool = raw.drop_duplicates(["account_key", "date", "amount", "memo_raw"])
    if prior_labels is not None and len(prior_labels):
        dated = prior_labels[prior_labels["account_key"].astype(str) != ""]
        pool = pool[~txn_key(pool).isin(set(txn_key(dated)))]
        memo_amt = set(zip(prior_labels["memo_raw"], (prior_labels["amount"].abs() * 100).round()))
        pool = pool[~pd.Series(list(zip(pool["memo_raw"], (pool["amount"].abs() * 100).round())),
                               index=pool.index).isin(memo_amt)]
    pool = pool[pool["memo_raw"].str.strip() != ""].reset_index(drop=True)
    return add_text_cols(pool)


def choose_batch(cfg, pool, plan, prior_labels=None):
    rng = np.random.default_rng(cfg["seed"])
    targets, sparse_target = cfg["sampling"]["bucket_targets"], cfg["sampling"]["sparse_target"]
    vs, ts = pd.Timestamp(plan["val_start"]), pd.Timestamp(plan["test_start"])
    pool = pool.copy()
    pool["period"] = windows(pool["date"], vs, ts)
    pool["held_out"] = pool["account_key"].isin(set(plan["held_out_accounts"]))
    dev, hold = pool[~pool.held_out], pool[pool.held_out]

    taken, chosen = set(), {}
    cap_eval = lambda n, d: max(2, math.ceil(1.5 * n / max(1, d["account_key"].nunique())))
    for b, d in [("test_B", hold[hold.period == "test"]), ("test_A", dev[dev.period == "test"]),
                 ("val", dev[dev.period == "val"])]:
        chosen[b] = _pick(d, targets[b], cap_eval(targets[b], d), taken, Counter(), rng)

    tr = dev[dev.period == "train"].copy()
    T = targets["train"]
    cap_tr = max(10, math.ceil(0.08 * T))
    acct_counts, train_ids = Counter(), []
    tr["rule_cat"] = [pretag(m, d)[0] for m, d in zip(tr["memo_raw"], tr["direction"])]
    have = prior_labels["category"].value_counts() if prior_labels is not None else pd.Series(dtype=int)
    sparse = sorted([c for c in CATEGORIES if c not in ("Uncertain", "Miscellaneous")
                     and have.get(c, 0) < sparse_target], key=lambda c: have.get(c, 0))
    budget, boost = int(0.35 * T), {}
    for c in sparse:
        want = min(sparse_target - have.get(c, 0), budget - len(train_ids))
        cand = tr[(tr.rule_cat == c) & ~tr.index.isin(taken)]
        if want <= 0 or cand.empty:
            continue
        cand = cand.iloc[rng.permutation(len(cand))]
        cand = cand.assign(_r=cand.groupby("pattern").cumcount()).sort_values("_r", kind="stable")
        got = _pick(cand.head(want * 3), want, cap_tr, taken, acct_counts, rng)
        train_ids += got
        boost[c] = len(got)
    head_ids = []
    for p in tr["pattern"].value_counts().index:
        if len(head_ids) >= int(0.35 * T):
            break
        head_ids += _pick(tr[tr.pattern == p], 2, cap_tr, taken, acct_counts, rng)
    train_ids += head_ids
    train_ids += _pick(tr, T - len(train_ids), cap_tr, taken, acct_counts, rng)
    chosen["train"] = train_ids

    batch = pd.concat([pool.loc[ids].assign(bucket=b) for b, ids in chosen.items()])
    batch = batch.drop(columns=["rule_cat"], errors="ignore").reset_index(drop=True)
    short = {b: targets[b] - len(ids) for b, ids in chosen.items() if len(ids) < targets[b]}
    print(f"selected {len(batch)} memos | sparse boost {boost} | Pareto head {len(head_ids)}"
          + (f" | SHORT: {short}" if short else ""))
    leak = int((batch.bucket.isin(["train", "val"]) & batch.held_out).sum())
    assert leak == 0, "a held-out account landed in train/val"
    return batch

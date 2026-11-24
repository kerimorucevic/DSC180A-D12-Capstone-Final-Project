"""Charge labelling and the four recurrence definitions.

Unit of decision: an (account, merchant, month) "pair-month" = the sum of that account's charges
from that merchant in that calendar month. A method labels each pair-month

    confirmed      recurring at month t under the method
    potential      subscription-type merchant, but the observed history up to t is too short to
                   decide (new merchant inside the window, window months unobserved / partial)
    not_confirmed  enough history was observable and the rule failed (gap, pause, unstable amount)

M0  >= min_months distinct charged months anywhere in the account history (uses the future;
    retrospective comparison only; reproduces pa7)
M1  >= min_months distinct charged *covered* months up to and including t (partial edge months
    are not history)
M2  a charge in each of the `window` history months ending at t (consecutive calendar months
    under gap_policy='break'), every window month before t 'covered'
M3  M2 and every monthly total in the window within threshold of the window median
"""
import numpy as np
import pandas as pd

from .config import (STATUS_CONFIRMED, STATUS_NOT_CONFIRMED, STATUS_POTENTIAL, TIER_AMBIGUOUS,
                     RecurrenceConfig)
from .rules_baseline import IN_FEATURE as IN_FEATURE_BASELINE
from .rules_baseline import classify_baseline
from .rules_v2 import IN_FEATURE_V2, classify_v2

CLASSIFIERS = {
    "baseline": (classify_baseline, IN_FEATURE_BASELINE),
    "v2": (classify_v2, IN_FEATURE_V2),
}


def label_memos(texts, classifier, cache=None):
    """Classify each distinct memo once. Returns a DataFrame indexed by text."""
    fn, _ = CLASSIFIERS[classifier]
    cache = {} if cache is None else cache
    todo = [t for t in pd.unique(pd.Series(texts, dtype=object)) if t not in cache]
    for t in todo:
        cache[t] = fn(t)
    return cache


def label_charges(hist, classifier, cache=None):
    """All debits with (subtype, merchant, tier, cue, in_feature) from the chosen classifier."""
    _, in_feature = CLASSIFIERS[classifier]
    deb = hist[hist["is_debit"]].copy()
    lab = label_memos(deb["text"], classifier, cache)
    parts = deb["text"].map(lab)
    deb["subtype"] = parts.map(lambda x: x[0])
    deb["merchant"] = parts.map(lambda x: x[1])
    deb["tier"] = parts.map(lambda x: x[2])
    deb["cue"] = parts.map(lambda x: x[3])
    deb["in_feature"] = deb["subtype"].isin(in_feature)
    return deb


def pair_months(subs):
    """Collapse in-feature charges to (account, merchant, month) totals."""
    pm = (subs.groupby(["account_key", "merchant", "month"], observed=True)
              .agg(amount=("amount", "sum"), n_charges=("amount", "size"),
                   tier=("tier", "first"), subtype=("subtype", "first"))
              .reset_index())
    pm["m"] = pm["month"].map(lambda p: p.ordinal).astype(int)
    return pm.sort_values(["account_key", "merchant", "m"]).reset_index(drop=True)


def _lookup(pm, prev_col, value_col):
    """Value of `value_col` for the same pair at month ordinal pm[prev_col] (NaN if absent)."""
    key = pm[["account_key", "merchant", "m", value_col]].rename(columns={"m": prev_col, value_col: "_v"})
    left = pm[["account_key", "merchant", prev_col]].copy()
    left[prev_col] = left[prev_col].astype("float")
    key[prev_col] = key[prev_col].astype("float")
    return left.merge(key, on=["account_key", "merchant", prev_col], how="left")["_v"].values


def apply_methods(pm, grid_w, cfg: RecurrenceConfig):
    """Add status_<method> for every method plus diagnostic columns. grid_w from add_window_columns."""
    pm = pm.copy()
    W = cfg.window
    prev_cols = [f"prev{k}_m" for k in range(1, W)]
    pm = pm.merge(grid_w[["account_key", "m", "status", "window_ok", *prev_cols]],
                  on=["account_key", "m"], how="left", validate="many_to_one")
    pair = pm.groupby(["account_key", "merchant"])

    # M0 -- whole history (future included)
    total_months = pair["m"].transform("size")
    pm["status_M0"] = np.where(total_months >= cfg.min_months, STATUS_CONFIRMED, STATUS_NOT_CONFIRMED)

    # M1 -- distinct charged months up to and including t. Partial edge months are not history
    # (history_requires_covered=True); pm has one row per pair-month, sorted by m.
    counts = pm["status"].eq("covered") if cfg.history_requires_covered else pd.Series(True, index=pm.index)
    pm["months_so_far"] = counts.astype(int).groupby([pm["account_key"], pm["merchant"]]).cumsum()
    pm["status_M1"] = np.where(pm["months_so_far"] >= cfg.min_months, STATUS_CONFIRMED, STATUS_POTENTIAL)

    # M2 -- charge in each previous window month, window observable
    pm["pair_first_m"] = pair["m"].transform("min")
    amounts = [pm["amount"].values]
    charged_all = np.ones(len(pm), dtype=bool)
    for c in prev_cols:
        a = _lookup(pm, c, "amount")
        amounts.append(a)
        charged_all &= ~np.isnan(a)
    window_ok = pm["window_ok"].fillna(False).astype(bool).values
    m2 = window_ok & charged_all
    start = pm[prev_cols[-1]].values                       # earliest month of the window
    new_pair = ~(pm["pair_first_m"].values <= np.nan_to_num(start, nan=np.inf))
    potential = ~m2 & (~window_ok | new_pair)
    pm["status_M2"] = np.select([m2, potential], [STATUS_CONFIRMED, STATUS_POTENTIAL], STATUS_NOT_CONFIRMED)

    # M3 -- M2 + stable monthly totals
    A = np.column_stack(amounts)
    if len(pm):
        with np.errstate(invalid="ignore", divide="ignore"):
            med = np.nanmedian(A, axis=1)          # column 0 (month t) is never NaN
            dev = np.nanmax(np.abs(A - med[:, None]), axis=1) / med
    else:
        dev = np.array([], dtype=float)
    pm["window_rel_dev"] = np.where(charged_all, dev, np.nan)
    for name, thr in cfg.m3_thresholds.items():
        ok = m2 & (pm["window_rel_dev"].values <= thr + 1e-12)
        pm[f"status_{name}"] = np.select([ok, potential], [STATUS_CONFIRMED, STATUS_POTENTIAL],
                                         STATUS_NOT_CONFIRMED)

    # Diagnostic only (not a method), past only: the pair's previous charge was exactly 12 months
    # earlier (nothing in between) with a similar amount -> looks like an annual plan.
    prev_m = pair["m"].shift(1).values
    prev_a = pair["amount"].shift(1).values
    pm["annual_candidate"] = (prev_m == pm["m"].values - 12) & (
        np.abs(pm["amount"].values - prev_a) <= cfg.annual_tolerance * np.maximum(prev_a, 1e-9))
    pm["is_ambiguous"] = pm["tier"].eq(TIER_AMBIGUOUS)
    return pm

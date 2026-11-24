"""Human-review sampling and scoring for future validation of the recurrence definitions.

Procedure (see the notebook section "Human review"):
1. build_review_sample() draws a stratified sample of covered pair-months. Strata = tier x the
   pattern of method decisions (e.g. 'M0+ M1+ M2- M3_15-'), so disagreements between methods are
   over-represented on purpose; each row carries an inverse-probability weight.
2. Two reviewers fill `human_label` independently (recurring / not_recurring / unsure) using only
   the memo and the amount history shown. The file contains memo text: keep it on DSMLP.
3. evaluate_review() returns weighted precision / recall of each method against the human labels,
   and cohen_kappa() reports reviewer agreement. Recall is relative to the classifier-flagged
   universe only (charges the classifier never flagged are not sampled).
"""
import numpy as np
import pandas as pd

from .config import STATUS_CONFIRMED

REVIEW_METHODS = ["M0", "M1", "M2", "M3_15"]
LABELS = ("recurring", "not_recurring", "unsure")


def _pattern(df, methods):
    return df[[f"status_{m}" for m in methods]].apply(
        lambda r: " ".join(f"{m}{'+' if v == STATUS_CONFIRMED else '-'}" for m, v in zip(methods, r)), axis=1)


def _history_string(pm, acct, merch, m, before=12, after=6):
    s = pm[(pm.account_key == acct) & (pm.merchant == merch) & pm.m.between(m - before, m + after)]
    return "; ".join(("[" if r.m == m else "") + f"{r.month}:{r.amount:.2f}" + ("]" if r.m == m else "")
                     for r in s.itertuples())


def build_review_sample(pm, charges, n_per_stratum=15, seed=0, methods=REVIEW_METHODS, covered_only=True):
    df = pm[pm["status"].eq("covered")].copy() if covered_only else pm.copy()
    if df.empty:
        return df
    df["decision_pattern"] = _pattern(df, methods)
    df["stratum"] = df["tier"].fillna("none") + " | " + df["decision_pattern"]
    sizes = df.groupby("stratum").size()
    samp = (df.groupby("stratum", group_keys=False)
              .apply(lambda g: g.sample(min(n_per_stratum, len(g)), random_state=seed)))
    taken = samp.groupby("stratum").size()
    samp["stratum_size"] = samp["stratum"].map(sizes)
    samp["sample_weight"] = samp["stratum"].map(sizes / taken)
    memos = (charges[charges["in_feature"]]
             .groupby(["account_key", "merchant", "month"])["memo"]
             .agg(lambda s: " || ".join(pd.unique(s)[:3])))
    samp["example_memos"] = [memos.get((a, mer, mo), "") for a, mer, mo in
                             zip(samp.account_key, samp.merchant, samp.month)]
    samp["amount_history"] = [_history_string(pm, a, mer, m) for a, mer, m in
                              zip(samp.account_key, samp.merchant, samp.m)]
    samp = samp.sample(frac=1, random_state=seed).reset_index(drop=True)      # hide stratum order
    samp.insert(0, "review_id", np.arange(1, len(samp) + 1))
    keep = ["review_id", "account_key", "month", "merchant", "tier", "subtype", "amount", "n_charges",
            "example_memos", "amount_history", "stratum", "stratum_size", "sample_weight",
            *[f"status_{m}" for m in methods]]
    out = samp[keep].copy()
    out["month"] = out["month"].astype(str)
    for c in ["human_label", "human_cadence", "reviewer", "notes"]:
        out[c] = ""
    return out


def evaluate_review(rev, methods=REVIEW_METHODS, label_col="human_label"):
    """Weighted precision / recall of each method against human labels ('unsure' dropped)."""
    r = rev[rev[label_col].isin(["recurring", "not_recurring"])]
    yes = r[label_col].eq("recurring").values
    w = r["sample_weight"].astype(float).values
    rows = []
    for m in methods:
        conf = r[f"status_{m}"].eq(STATUS_CONFIRMED).values
        tp = (w * (conf & yes)).sum()
        rows.append({"method": m, "reviewed": int(len(r)), "reviewed_confirmed": int(conf.sum()),
                     "weighted_precision": tp / (w * conf).sum() if conf.any() else np.nan,
                     "weighted_recall": tp / (w * yes).sum() if yes.any() else np.nan})
    return pd.DataFrame(rows)


def cohen_kappa(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan

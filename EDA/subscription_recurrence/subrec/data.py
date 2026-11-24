"""Loading, preparing and repeated-record diagnostics for transaction histories."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEBIT_CODE, FAKE_ACCOUNT, KEY_COLS


def lower_text(s):
    """Same normalisation as the reference notebooks: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def load_history(path, columns=KEY_COLS):
    """Read the cached full-history parquet written by income_feature.ipynb.

    The cache is never rebuilt here: rebuilding scans the restricted share and belongs in
    income_feature.ipynb on DSMLP.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. On DSMLP, run income_feature.ipynb once (it writes "
            "full_histories_5000accts.parquet) or set SUBREC_HIST_CACHE to the cache path.")
    hist = pd.read_parquet(path, columns=columns)
    hist["memo"] = hist["memo"].map(lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else b)
    return hist


def prepare_history(hist, fake_account=FAKE_ACCOUNT, debit_code=DEBIT_CODE):
    """Exclude the synthetic account and add the derived columns used downstream.

    Rows are NOT removed or collapsed here (see exact_distinct for the separate sensitivity variant).
    """
    missing = set(KEY_COLS) - set(hist.columns)
    if missing:
        raise KeyError(f"history is missing columns {sorted(missing)}")
    out = hist[hist["account_key"] != fake_account].copy()
    out["amount"] = out["amount"].astype(float)          # parquet stores Decimal
    out["date"] = pd.to_datetime(out["posted_date"])
    out["month"] = out["date"].dt.to_period("M")
    out["is_debit"] = out["credit_or_debit"] == debit_code
    out["text"] = None
    out.loc[out["is_debit"], "text"] = out.loc[out["is_debit"], "memo"].map(lower_text)
    return out.reset_index(drop=True)


def exact_distinct(hist, keys=KEY_COLS):
    """One row per exact (account, date, amount, direction, memo) combination.

    This is a sensitivity variant, not a cleaning step: without a transaction id we cannot tell
    a duplicated export row from two genuine identical purchases on the same day.
    """
    return hist.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def repeated_record_diagnostics(hist, keys=KEY_COLS):
    """How often rows repeat an existing (account, date, amount, direction, memo) combination.

    Pure description. Without a transaction id, a repeat may be an export artefact or a genuine second identical
    purchase; nothing here classifies repeats as duplicates or errors.
    """
    n = len(hist)
    sizes = hist.groupby(keys, dropna=False, sort=False).size()
    n_distinct = len(sizes)
    per_acct = (hist.assign(_dup=hist.duplicated(subset=keys, keep="first"))
                .groupby("account_key")["_dup"].mean())
    deb = hist[hist["is_debit"]] if "is_debit" in hist else hist
    overall = {
        "rows": int(n),
        "distinct_key_combinations": int(n_distinct),
        "rows_repeating_an_existing_combination": int(n - n_distinct),
        "repeat_share": float((n - n_distinct) / n) if n else np.nan,
        "debit_repeat_share": float(deb.duplicated(subset=keys).mean()) if len(deb) else np.nan,
        "combinations_seen_once": int((sizes == 1).sum()),
        "max_copies_of_one_combination": int(sizes.max()) if n else 0,
        "median_copies_per_combination": float(sizes.median()) if n else np.nan,
        "accounts": int(hist["account_key"].nunique()),
        "accounts_with_any_repeat": int((per_acct > 0).sum()),
        "account_repeat_share_median": float(per_acct.median()) if len(per_acct) else np.nan,
    }
    copies = sizes.value_counts().sort_index().rename("combinations").to_frame()
    copies.index.name = "copies"
    return overall, copies, per_acct.rename("repeat_share")

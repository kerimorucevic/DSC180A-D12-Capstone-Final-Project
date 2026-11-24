"""Account-month grid with the coverage statuses used by income_feature.ipynb and pa7.

covered      full month with >= 1 transaction of any kind -> a $0 feature value is a real zero
no_data_gap  month inside the account's span with no transactions -> missing coverage, not zero
partial_edge the account's first or last month -> only partly observed

The status logic below is the reference implementation (pa7 section 6) unchanged; the window
columns are additions used by M2 / M3 and by the common comparison population.
"""
import numpy as np
import pandas as pd

from .config import RecurrenceConfig


def build_grid(hist):
    """Every calendar month from each account's first to last transaction, with status."""
    act = hist.groupby(["account_key", "month"]).size().rename("n_txn").reset_index()
    span = hist.groupby("account_key")["month"].agg(first="min", last="max").reset_index()
    grid = pd.concat([pd.DataFrame({"account_key": r.account_key,
                                    "month": pd.period_range(r.first, r.last, freq="M")})
                      for r in span.itertuples()], ignore_index=True)
    am = (grid.merge(span, on="account_key")
              .merge(act, on=["account_key", "month"], how="left"))
    am["n_txn"] = am["n_txn"].fillna(0).astype(int)
    am["status"] = np.select([(am.month == am["first"]) | (am.month == am["last"]), am.n_txn == 0],
                             ["partial_edge", "no_data_gap"], "covered")
    am = am.drop(columns=["first", "last"])
    am["m"] = am["month"].map(lambda p: p.ordinal).astype(int)
    return am.sort_values(["account_key", "m"]).reset_index(drop=True)


def add_window_columns(grid, cfg: RecurrenceConfig):
    """Attach the previous window-1 history months for each account-month (past only).

    prev{k}_m       month ordinal of the k-th previous history month (calendar month under
                    gap_policy='break', k-th previous non-gap month under 'bridge')
    window_ok       all window-1 previous months exist inside the account span and are
                    'covered' (or merely observed, if history_requires_covered=False)
    eligible_common the month is covered AND window_ok: the population on which every method,
                    including M2 / M3, could in principle confirm a subscription.
    """
    g = grid.copy()
    base = g if cfg.gap_policy == "break" else g[g["status"] != "no_data_gap"]
    base = base.sort_values(["account_key", "m"])
    ok = pd.Series(True, index=base.index)
    for k in range(1, cfg.window):
        pm = base.groupby("account_key")["m"].shift(k)
        ps = base.groupby("account_key")["status"].shift(k)
        g.loc[base.index, f"prev{k}_m"] = pm
        good = ps.eq("covered") if cfg.history_requires_covered else ps.isin(["covered", "partial_edge"])
        ok &= good
    g["window_ok"] = False
    g.loc[base.index, "window_ok"] = ok.values
    g["eligible_covered"] = g["status"].eq("covered")
    g["eligible_common"] = g["eligible_covered"] & g["window_ok"]
    return g

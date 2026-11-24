"""Account-month features: confirmed recurring spend per method, kept apart from potential spend."""
import pandas as pd

from .config import STATUS_CONFIRMED, STATUS_POTENTIAL


def account_month_features(grid_w, pm, methods):
    """One row per account-month of the grid with, for every method:

    rec_<M>   confirmed recurring subscription dollars (all tiers)
    amb_<M>   the part of rec_<M> from ambiguous-platform merchants
    pot_<M>   'potential' dollars: subscription-type charges not yet confirmable (never in rec_<M>)
    nsvc_<M>  number of confirmed merchants
    Months with no qualifying charge are 0; coverage status decides whether 0 is a real zero.
    """
    am = grid_w.copy()
    keys = ["account_key", "m"]
    for M in methods:
        st = pm[f"status_{M}"]
        conf = pm[st.eq(STATUS_CONFIRMED)]
        agg = conf.groupby(keys).agg(**{f"rec_{M}": ("amount", "sum"), f"nsvc_{M}": ("merchant", "nunique")})
        amb = conf[conf["is_ambiguous"]].groupby(keys)["amount"].sum().rename(f"amb_{M}")
        pot = pm[st.eq(STATUS_POTENTIAL)].groupby(keys)["amount"].sum().rename(f"pot_{M}")
        am = am.merge(agg, on=keys, how="left").merge(amb, on=keys, how="left").merge(pot, on=keys, how="left")
    fill = [c for c in am.columns if c.split("_")[0] in {"rec", "amb", "pot", "nsvc"}]
    am[fill] = am[fill].fillna(0.0)
    return am


# The "ambiguous platforms excluded" variant is account_month_features(grid_w, pm[~pm.is_ambiguous], ...):
# statuses are decided per (account, merchant), so filtering pair-months first is exact (rec, pot and
# nsvc all exclude platforms; no subtraction, so no floating-point residue in the zero test).

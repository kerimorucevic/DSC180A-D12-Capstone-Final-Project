"""Distribution summaries for each recurrence definition over an eligible population."""
import numpy as np
import pandas as pd

POPULATIONS = {
    "covered": "eligible_covered",   # all covered account-months (pa7 denominator)
    "common": "eligible_common",     # covered AND the window before t observable: same rows for every method
}


def _q(s, p):
    return float(s.quantile(p)) if len(s) else np.nan


def summarize(am, method, population="common"):
    """Evaluation metrics for rec_<method> over the chosen population."""
    pop = am[am[POPULATIONS[population]]]
    x = pop[f"rec_{method}"]
    pos = x[x > 0]
    per_acct = pop.groupby("account_key")[f"rec_{method}"].max()
    total = float(x.sum())
    amb = float(pop[f"amb_{method}"].sum())
    return {
        "method": method,
        "population": population,
        "eligible_account_months": int(len(pop)),
        "accounts_in_population": int(pop["account_key"].nunique()),
        "accounts_detected": int((per_acct > 0).sum()),
        "pct_accounts_detected": float((per_acct > 0).mean()) if len(per_acct) else np.nan,
        "zero_share": float((x == 0).mean()) if len(x) else np.nan,
        "median_all": _q(x, .5),
        "median_positive": _q(pos, .5),
        "p90_all": _q(x, .9), "p95_all": _q(x, .95), "p99_all": _q(x, .99),
        "p90_positive": _q(pos, .9), "p95_positive": _q(pos, .95), "p99_positive": _q(pos, .99),
        "total_recurring": total,
        "ambiguous_total": amb,
        "ambiguous_share": amb / total if total else np.nan,
        "potential_total": float(pop[f"pot_{method}"].sum()),
    }


def metrics_table(am, methods, populations=("common", "covered"), **labels):
    rows = [{**labels, **summarize(am, M, p)} for p in populations for M in methods]
    return pd.DataFrame(rows)


def compare_variants(table, by, left, right, metrics=("pct_accounts_detected", "zero_share", "median_all",
                                                     "median_positive", "p95_all", "total_recurring")):
    """Side-by-side of two variants (e.g. dataset original vs exact_distinct) per method/population."""
    idx = [c for c in ["classifier", "platforms", "population", "method"] if c in table and c != by]
    a = table[table[by] == left].set_index(idx)[list(metrics)]
    b = table[table[by] == right].set_index(idx)[list(metrics)]
    out = pd.concat({left: a, right: b}, axis=1)
    for m in metrics:
        out[("diff", m)] = out[(right, m)] - out[(left, m)]
    return out

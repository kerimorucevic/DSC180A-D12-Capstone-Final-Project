"""End-to-end runner: datasets x classifiers x platform-inclusion x methods."""
from dataclasses import dataclass, field

import pandas as pd

from .config import RecurrenceConfig
from .coverage import add_window_columns, build_grid
from .features import account_month_features
from .metrics import metrics_table
from .recurrence import apply_methods, label_charges, pair_months


@dataclass
class RunResult:
    dataset: str
    classifier: str
    cfg: RecurrenceConfig
    grid: pd.DataFrame
    charges: pd.DataFrame      # all debits with classifier labels
    pm: pd.DataFrame           # in-feature pair-months with method statuses
    am: pd.DataFrame           # account-month features (all tiers)
    am_high: pd.DataFrame      # account-month features, ambiguous platforms removed


@dataclass
class Analysis:
    cfg: RecurrenceConfig
    runs: dict = field(default_factory=dict)       # (dataset, classifier) -> RunResult
    metrics: pd.DataFrame = None


def run_one(hist, dataset, classifier, cfg, label_cache=None, grid_w=None):
    grid_w = add_window_columns(build_grid(hist), cfg) if grid_w is None else grid_w
    charges = label_charges(hist, classifier, label_cache)
    pm = apply_methods(pair_months(charges[charges["in_feature"]]), grid_w, cfg)
    am = account_month_features(grid_w, pm, cfg.methods)
    am_high = account_month_features(grid_w, pm[~pm["is_ambiguous"]], cfg.methods)
    return RunResult(dataset, classifier, cfg, grid_w, charges, pm, am, am_high)


def run_analysis(datasets, cfg=None, classifiers=("baseline", "v2"), populations=("common", "covered")):
    """datasets: {'original': hist, 'exact_distinct': hist_distinct} (both already prepared)."""
    cfg = cfg or RecurrenceConfig()
    out = Analysis(cfg)
    caches = {c: {} for c in classifiers}
    tables = []
    for ds_name, hist in datasets.items():
        grid_w = add_window_columns(build_grid(hist), cfg)
        for clf in classifiers:
            r = run_one(hist, ds_name, clf, cfg, caches[clf], grid_w)
            out.runs[(ds_name, clf)] = r
            for plat, am in [("included", r.am), ("excluded", r.am_high)]:
                tables.append(metrics_table(am, cfg.methods, populations,
                                            dataset=ds_name, classifier=clf, platforms=plat))
    out.metrics = pd.concat(tables, ignore_index=True)
    return out


def status_dollars(pm, methods, covered_only=True):
    """Subscription-merchant dollars by method status (covered months only by default)."""
    pm = pm[pm["status"].eq("covered")] if covered_only else pm
    rows = []
    for M in methods:
        g = pm.groupby(f"status_{M}")["amount"].sum()
        rows.append({"method": M, **g.to_dict()})
    return pd.DataFrame(rows).set_index("method").fillna(0.0)

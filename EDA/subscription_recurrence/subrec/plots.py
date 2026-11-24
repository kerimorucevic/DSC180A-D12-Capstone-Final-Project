"""Comparison figures (matplotlib, static PNG).

Colour roles: methods are ordered by strictness, so M1..M3_20 use one sequential blue ramp
(light = lenient, dark = strict); M0 is retrospective and drawn in neutral grey. Two-way
comparisons use categorical slots 1 (blue) and 2 (orange); status uses slots 1-3.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
METHOD_COLORS = {"M0": "#a8a69e", "M1": "#86b6ef", "M2": "#5598e7", "M3_20": "#2a78d6",
                 "M3_15": "#1c5cab", "M3_10": "#104281"}
METHOD_LABELS = {"M0": "M0\n(retro)", "M1": "M1", "M2": "M2", "M3_10": "M3\n10%", "M3_15": "M3\n15%\n(primary)",
                 "M3_20": "M3\n20%"}
LEGEND = {"baseline": "PA7 baseline classifier", "v2": "v2 classifier (provisional)",
          "original": "original records", "exact_distinct": "exact-distinct records"}
_money = FuncFormatter(lambda v, _: f"${v/1000:,.0f}k" if abs(v) >= 1000 else f"${v:,.0f}")
_pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def _style(ax, title=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _fig(ncols, w=5.2, h=3.8):
    fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def _sel(metrics, **kw):
    m = metrics
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.set_index("method")


def method_overview(metrics, methods, subtitle, **sel):
    m = _sel(metrics, **sel).loc[methods]
    fig, axes = _fig(3)
    panels = [("pct_accounts_detected", "Accounts with a detected subscription", _pct),
              ("zero_share", "Eligible months with $0 recurring spend", _pct),
              ("total_recurring", "Total confirmed recurring spend", _money)]
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = m[col].values
        ax.bar(range(len(methods)), vals, color=[METHOD_COLORS[k] for k in methods], width=0.62,
               edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[k] for k in methods])
        ax.yaxis.set_major_formatter(fmt)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt(v, None), ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_ylim(0, max(vals.max() * 1.15, 1e-9))
        _style(ax, title)
    fig.suptitle(subtitle, x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def spending_distribution(am, methods, population_col="eligible_common", subtitle=""):
    pop = am[am[population_col]]
    fig, axes = _fig(2, w=6.2)
    for k in methods:
        x = np.sort(pop[f"rec_{k}"].values)
        pos = x[x > 0]
        if len(pos):
            axes[0].plot(pos, np.arange(1, len(pos) + 1) / len(pos), color=METHOD_COLORS[k], lw=2,
                         label=METHOD_LABELS[k].replace("\n", " "), ls="--" if k == "M0" else "-")
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(_money)
    axes[0].yaxis.set_major_formatter(_pct)
    axes[0].set_xlabel("monthly confirmed recurring spend, positive months (log)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, title="method", title_fontsize=8)
    _style(axes[0], "Cumulative share of positive months", None)
    axes[0].grid(axis="x", color=GRID, lw=0.8)
    qs = [("median_positive", .5), ("p90", .9), ("p95", .95), ("p99", .99)]
    width = 0.8 / len(qs)
    shades = ["#b7d3f6", "#6da7ec", "#256abf", "#0d366b"]
    for j, (name, q) in enumerate(qs):
        vals = [np.quantile(pop[f"rec_{k}"][pop[f"rec_{k}"] > 0], q) if (pop[f"rec_{k}"] > 0).any() else 0
                for k in methods]
        axes[1].bar(np.arange(len(methods)) + (j - 1.5) * width, vals, width, color=shades[j],
                    edgecolor=SURFACE, linewidth=1.5, label=name.replace("_positive", ""))
    axes[1].set_xticks(range(len(methods)), [METHOD_LABELS[k] for k in methods])
    axes[1].yaxis.set_major_formatter(_money)
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.3)
    axes[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper left", title="quantile among positive months",
                   title_fontsize=8)
    _style(axes[1], "Quantiles of positive monthly spend")
    fig.suptitle(subtitle, x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def platform_sensitivity(metrics, methods, subtitle, **sel):
    fig, axes = _fig(2, w=6.0)
    m = _sel(metrics, classifier="v2", platforms="included", **sel).loc[methods]
    high = m["total_recurring"] - m["ambiguous_total"]
    x = np.arange(len(methods))
    axes[0].bar(x, high, 0.62, color=CAT[0], edgecolor=SURFACE, linewidth=2, label="high-confidence merchants")
    axes[0].bar(x, m["ambiguous_total"], 0.62, bottom=high, color=CAT[1], edgecolor=SURFACE, linewidth=2,
                label="ambiguous platforms")
    axes[0].set_xticks(x, [METHOD_LABELS[k] for k in methods])
    axes[0].yaxis.set_major_formatter(_money)
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0], "v2: confirmed spend by merchant tier")
    for i, clf in enumerate(["baseline", "v2"]):
        s = _sel(metrics, classifier=clf, platforms="included", **sel).loc[methods, "ambiguous_share"]
        axes[1].bar(x + (i - 0.5) * 0.36, s.values, 0.36, color=CAT[i], edgecolor=SURFACE, linewidth=1.5,
                    label=LEGEND.get(clf, clf))
    axes[1].set_xticks(x, [METHOD_LABELS[k] for k in methods])
    axes[1].yaxis.set_major_formatter(_pct)
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1], "Share of confirmed spend from ambiguous platforms")
    fig.suptitle(subtitle, x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def two_way(metrics, methods, by, levels, cols, subtitle, **sel):
    """Grouped bars comparing two variants (dataset, classifier, ...) on several metrics."""
    fig, axes = _fig(len(cols), w=5.0)
    x = np.arange(len(methods))
    for ax, (col, title, fmt) in zip(axes, cols):
        for i, lv in enumerate(levels):
            s = _sel(metrics, **{by: lv}, **sel).loc[methods, col]
            ax.bar(x + (i - 0.5) * 0.36, s.values, 0.36, color=CAT[i], edgecolor=SURFACE, linewidth=1.5, label=LEGEND.get(lv, lv))
        ax.set_xticks(x, [METHOD_LABELS[k] for k in methods])
        ax.yaxis.set_major_formatter(fmt)
        ax.legend(frameon=False, fontsize=8)
        _style(ax, title)
    fig.suptitle(subtitle, x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def status_composition(status_df, methods, subtitle):
    """Stacked dollars by status per method (confirmed / potential / not_confirmed)."""
    fig, axes = _fig(1, w=7.5)
    ax = axes[0]
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for i, st in enumerate(["confirmed", "potential", "not_confirmed"]):
        v = status_df.reindex(methods)[st].fillna(0).values if st in status_df else np.zeros(len(methods))
        ax.bar(x, v, 0.62, bottom=bottom, color=CAT[i], edgecolor=SURFACE, linewidth=2, label=st)
        bottom += v
    ax.set_xticks(x, [METHOD_LABELS[k] for k in methods])
    ax.yaxis.set_major_formatter(_money)
    ax.set_ylim(0, max(bottom.max(), 1e-9) * 1.18)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _style(ax, "Subscription-merchant dollars in covered months, by method status")
    fig.suptitle(subtitle, x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


PCT, MONEY = _pct, _money

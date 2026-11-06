"""Labeler agreement on the shared, multiply-labeled memos (from Vanessa's PA4 100-label file).

Input: one row per memo, one column per labeler (names in config `agreement.labelers`), optionally the final
agreed label. Labels are mapped to the 20-category guide first (e.g. "Entertainment" -> Subscriptions &
Entertainment), so differences in naming don't count as disagreement.

Writes agreement.json (all-agree rate, majority rate, pairwise agreement, Fleiss' kappa, each labeler vs the final
label) and agreement_by_category.csv (how often memos whose final label is X were unanimous).
"""
import itertools
import json

import numpy as np
import pandas as pd

from .taxonomy import parse_category


def fleiss_kappa(table):
    """table: memos x categories counts, same number of raters per memo."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def agreement(cfg, out):
    a = cfg["agreement"]
    df = pd.read_csv(a["file"])
    raters = a["labelers"]
    lab = df[raters].apply(lambda col: col.map(lambda x: parse_category(x) if pd.notna(x) else None))
    unmapped = {r: sorted(set(df.loc[lab[r].isna() & df[r].notna(), r])) for r in raters}
    full = lab.dropna()
    cats = sorted(set(full.values.ravel()))
    table = np.array([[row.tolist().count(c) for c in cats] for _, row in full.iterrows()])
    top = table.max(axis=1)
    res = dict(memos=len(df), memos_all_rated=len(full), labelers=raters,
               all_agree=round(float((top == len(raters)).mean()), 3),
               majority_agree=round(float((top > len(raters) / 2).mean()), 3),
               fleiss_kappa=round(fleiss_kappa(table), 3),
               pairwise={f"{x} vs {y}": round(float((full[x] == full[y]).mean()), 3)
                         for x, y in itertools.combinations(raters, 2)},
               unmapped_labels={r: v for r, v in unmapped.items() if v})
    final = a.get("final_column")
    if final and final in df:
        f = df[final].map(lambda x: parse_category(x) if pd.notna(x) else None)
        res["vs_final"] = {r: round(float((lab[r] == f)[f.notna() & lab[r].notna()].mean()), 3) for r in raters}
        by = pd.DataFrame({"final": f.loc[full.index], "unanimous": top == len(raters)})
        by.groupby("final")["unanimous"].agg(["size", "mean"]).rename(
            columns={"size": "memos", "mean": "unanimous_share"}).sort_values("memos", ascending=False).to_csv(
            out / "agreement_by_category.csv")
    json.dump(res, open(out / "agreement.json", "w"), indent=2)
    print(f"agreement: all {len(raters)} agree on {res['all_agree']:.0%} | majority {res['majority_agree']:.0%} | "
          f"Fleiss kappa {res['fleiss_kappa']}")
    return res

"""Validation-only tuning, one group of settings at a time (Lecture 5 experiment plan). Test sets are untouched.

stage 1  features      : text_col x wordNgrams x char n-grams (off / 3-5 / 2-5), then minCount
stage 2  effort        : epoch x lr
stage 3  vector size   : dim
stage 4  recheck       : char n-grams and word bigrams again at the tuned epoch/lr/dim. They barely train at
                         lr=0.1, so stage 1 alone under-rates them (this recheck found the best PA5 setting).
stability              : top 3 distinct settings x seeds; final = best mean validation macro-F1.
Ties go to the faster run.
"""
import gc
import json
import time

import numpy as np
import pandas as pd

from .model import fit, model_mb, predict, scores


def tune(cfg, sets, work_dir):
    ft = cfg["fasttext"]
    grid = ft["grid"]
    base = dict(thread=ft["thread"], bucket=ft["bucket"], seed=cfg["seed"], **ft["defaults"])
    rows = []

    def run(stage, text_col, **params):
        m, secs, err = fit(sets["train"], text_col, work_dir, base, **params)
        row = dict(stage=stage, text_col=text_col, params=json.dumps(params, sort_keys=True), fit_s=round(secs, 3))
        if m is None:
            row["error"] = err[:80]
        else:
            row.update(scores(sets["val"].category, predict(m, sets["val"][text_col])[0]),
                       train_acc=scores(sets["train"].category, predict(m, sets["train"][text_col])[0])["accuracy"],
                       model_mb=model_mb(m))
            del m
            gc.collect()
        rows.append(row)

    def best(stage):
        d = pd.DataFrame([r for r in rows if r["stage"] == stage and "macro_f1" in r])
        r = d.sort_values(["macro_f1", "accuracy", "fit_s"], ascending=[False, False, True]).iloc[0]
        return r["text_col"], json.loads(r["params"])

    t0 = time.time()
    text_cols = grid["text_col"] if cfg["text_col"] == "auto" else [cfg["text_col"]]
    s1_effort = dict(epoch=grid["stage1_epoch"], lr=grid["stage1_lr"])
    for tc in text_cols:
        for wn in grid["wordNgrams"]:
            for mn, mx in grid["char_ngrams"]:
                run(1, tc, wordNgrams=wn, minn=mn, maxn=mx, **s1_effort)
    tc, b = best(1)
    for mc in grid["minCount"]:
        if mc != b.get("minCount", 1):
            run(1, tc, **{**b, "minCount": mc})
    tc, b = best(1)
    for ep in grid["epoch"]:
        for lr in grid["lr"]:
            run(2, tc, **{**b, "epoch": ep, "lr": lr})
    tc, b = best(2)
    for dim in grid["dim"]:
        run(3, tc, **{**b, "dim": dim})
    tc, b = best(3)
    for wn in [1, 2]:
        for mn, mx in grid["char_ngrams"]:
            run(4, tc, **{**b, "wordNgrams": wn, "minn": mn, "maxn": mx})

    done = pd.DataFrame([r for r in rows if "macro_f1" in r]).drop_duplicates(["text_col", "params"])
    top = done.sort_values(["macro_f1", "fit_s"], ascending=[False, True]).head(3)
    for _, r in top.iterrows():
        for s in ft["stability_seeds"]:
            if s != cfg["seed"]:
                run("seed", r["text_col"], **{**json.loads(r["params"]), "seed": s})
    tuning_s = time.time() - t0

    exp = pd.DataFrame(rows)
    stab = []
    for _, r in top.iterrows():
        p = json.loads(r["params"])
        vals = [x["macro_f1"] for x in rows if "macro_f1" in x and x["text_col"] == r["text_col"]
                and {k: v for k, v in json.loads(x["params"]).items() if k != "seed"} == p]
        stab.append(dict(text_col=r["text_col"], params=r["params"], runs=len(vals),
                         mean_macro_f1=float(np.mean(vals)), sd=float(np.std(vals)), fit_s=r["fit_s"]))
    stab = pd.DataFrame(stab).sort_values(["mean_macro_f1", "fit_s"], ascending=[False, True])
    win = stab.iloc[0]
    final = dict(text_col=win["text_col"], **json.loads(win["params"]))
    return final, exp, stab, tuning_s, base


def baseline_runs(cfg, sets, work_dir, base):
    """Majority class and last week's settings on this week's labels, scored on the same validation rows."""
    v = sets["val"]
    maj = sets["train"]["category"].value_counts().idxmax()
    out = [dict(model=f"majority class ({maj})", **scores(v.category, [maj] * len(v)))]
    for tc in ["text_lower", "text_norm"]:
        m, secs, err = fit(sets["train"], tc, work_dir, base, **cfg["fasttext"]["baseline"])
        if m is not None:
            out.append(dict(model=f"previous settings {cfg['fasttext']['baseline']}, {tc}",
                            **scores(v.category, predict(m, v[tc])[0]), fit_s=round(secs, 3)))
    return pd.DataFrame(out)

"""fastText classifier: training files, tuning on validation, evaluation on Test A / Test B.

Each labeled memo is written in fastText's format, category token then text:
    __label__groceries trader joes
Both cleaned versions (text_lower, text_norm) are compared during tuning. Tuning uses validation macro-F1 only:
features first (text version, word n-grams, character n-grams), then epochs x learning rate; the best three
settings are rerun with other seeds and the best mean wins. The test sets are scored once with the final model.
"""
import gc
import json
import time

import fasttext
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .rules import CATEGORIES, LABEL_TO_CATEGORY, to_label

fasttext.FastText.eprint = lambda *a, **k: None


def write_ft(df, path, text_col):
    with open(path, "w") as f:
        for c, t in zip(df["category"], df[text_col]):
            f.write(f"{to_label(c)} {t or 'empty_memo'}\n")


def fit(train, text_col, work_dir, params, retries=2):
    """fastText 0.9.x sometimes raises 'Encountered NaN' depending on memory state, so failures are retried."""
    path = work_dir / f"_train_{text_col}.txt"
    write_ft(train, path, text_col)
    err = ""
    for _ in range(retries + 1):
        try:
            t0 = time.perf_counter()
            m = fasttext.train_supervised(input=str(path), verbose=0, **params)
            return m, time.perf_counter() - t0, ""
        except RuntimeError as e:
            err = str(e)
            gc.collect()
    return None, 0.0, err


def predict(model, texts):
    out = [model.f.predict((t or "empty_memo") + "\n", 1, 0.0, "strict") for t in texts]   # numpy-2 safe
    return [LABEL_TO_CATEGORY.get(x[0][1], "?") if x else "?" for x in out], \
        np.clip([x[0][0] if x else 0.0 for x in out], 0, 1)


def load(path):
    return fasttext.load_model(str(path))


def scores(y, p):
    y, p = list(y), list(p)
    return dict(n=len(y), accuracy=float(np.mean(np.array(y) == np.array(p))) if y else np.nan,
                macro_f1=float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)) if y else np.nan)


def tune(cfg, sets, out):
    ft, g = cfg["fasttext"], cfg["fasttext"]["grid"]
    base = dict(thread=ft["thread"], bucket=ft["bucket"], seed=cfg["seed"], **ft["defaults"])
    rows = []

    def run(stage, tc, **p):
        m, secs, err = fit(sets["train"], tc, out, {**base, **p})
        row = dict(stage=stage, text_col=tc, params=json.dumps(p, sort_keys=True), fit_s=round(secs, 3), error=err)
        if m is not None:
            row.update(scores(sets["val"].category, predict(m, sets["val"][tc])[0]))
        rows.append(row)

    def best(stage):
        d = pd.DataFrame([r for r in rows if r["stage"] == stage and "macro_f1" in r])
        r = d.sort_values(["macro_f1", "fit_s"], ascending=[False, True]).iloc[0]
        return r.text_col, json.loads(r.params)

    for tc in g["text_col"]:                                       # 1. features
        for wn in g["wordNgrams"]:
            for mn, mx in g["char_ngrams"]:
                run(1, tc, wordNgrams=wn, minn=mn, maxn=mx, epoch=g["stage1_epoch"], lr=g["stage1_lr"])
    tc, p = best(1)
    for ep in g["epoch"]:                                          # 2. epochs x learning rate
        for lr in g["lr"]:
            run(2, tc, **{**p, "epoch": ep, "lr": lr})
    done = pd.DataFrame([r for r in rows if "macro_f1" in r]).drop_duplicates(["text_col", "params"])
    top = done.sort_values(["macro_f1", "fit_s"], ascending=[False, True]).head(3)
    stab = []
    for r in top.itertuples():                                     # 3. top three x seeds
        vals = [r.macro_f1]
        for s in ft["seeds"][1:]:
            run("seed", r.text_col, **{**json.loads(r.params), "seed": s})
            if "macro_f1" in rows[-1]:
                vals.append(rows[-1]["macro_f1"])
        stab.append(dict(text_col=r.text_col, params=r.params, mean_macro_f1=float(np.mean(vals)),
                         sd=float(np.std(vals)), runs=len(vals)))
    pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    stab = pd.DataFrame(stab).sort_values("mean_macro_f1", ascending=False)
    stab.to_csv(out / "stability.csv", index=False)

    for r in stab.itertuples():                                    # final model (next seed if fastText NaN)
        for s in ft["seeds"]:
            params = {**base, **json.loads(r.params), "seed": s}
            m, secs, _ = fit(sets["train"], r.text_col, out, params)
            if m is not None:
                m.save_model(str(out / "model.bin"))
                info = dict(text_col=r.text_col, params={k: v for k, v in params.items() if k not in ("thread",)},
                            val_macro_f1_mean=r.mean_macro_f1, val_macro_f1_sd=r.sd, fit_s=round(secs, 3),
                            train_rows=len(sets["train"]), runs=len(rows))
                json.dump(info, open(out / "final_model.json", "w"), indent=2)
                return m, info
    raise RuntimeError("every final fit failed")


def plot_confusion(d, path):
    labs = [c for c in CATEGORIES if c in set(d.category) | set(d.predicted)]
    cm = confusion_matrix(d.category, d.predicted, labels=labs)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labs)), labs, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labs)), labs, fontsize=8)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("reviewed label")
    ax.set_title("Validation confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model, sets, text_col, out):
    maj = sets["train"].category.value_counts().idxmax()
    res, preds = [], {}
    for name in ["val", "test_A", "test_B"]:
        d = sets[name].copy()
        d["predicted"], d["score"] = predict(model, d[text_col])
        preds[name] = d
        res.append(dict(set=name, accounts=d.account_key.nunique(), **scores(d.category, d.predicted),
                        majority_baseline_acc=float((d.category == maj).mean())))
    res = pd.DataFrame(res)
    res.to_csv(out / "scores.csv", index=False)
    v = preds["val"]
    rep = pd.DataFrame(classification_report(v.category, v.predicted, labels=sorted(set(v.category)),
                                             output_dict=True, zero_division=0)).T
    rep = rep.loc[[c for c in rep.index if c in CATEGORIES]].sort_values("support", ascending=False)
    rep["train_rows"] = rep.index.map(sets["train"].category.value_counts()).fillna(0).astype(int)
    rep.to_csv(out / "per_class_val.csv")
    plot_confusion(v, out / "confusion.png")
    v[v.predicted != v.category][["memo_raw", "category", "predicted", "score"]].to_csv(out / "errors_val.csv", index=False)
    mix = pd.concat({s: sets[s].category.value_counts(normalize=True) for s in sets}, axis=1).fillna(0)
    (mix * 100).round(1).to_csv(out / "class_mix.csv")
    return res

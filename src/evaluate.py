"""Class quality and review coverage (Lecture 5 "Separate class quality & review coverage").

validation : per-class precision/recall/F1, confusion matrix, error table, seen vs unseen memo patterns
coverage   : accept a prediction when score >= cutoff and it isn't Uncertain; otherwise send to review.
             row coverage = accepted / eligible, dollar coverage = accepted $ / eligible $.
             Cutoff rule (chosen on validation): highest coverage with accepted error <= max_accepted_error.
tests      : Test A and Test B scored once, next to the class mix and the gambling slice.
"""
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .model import predict, scores, wilson
from .taxonomy import CATEGORIES


def add_predictions(model, df, text_col, train_patterns):
    d = df.copy()
    d["predicted"], d["score"] = predict(model, d[text_col])
    d["correct"] = d["predicted"] == d["category"]
    d["pattern_in_train"] = d["pattern"].isin(train_patterns)
    return d


def per_class(d, train_counts):
    rep = pd.DataFrame(classification_report(d.category, d.predicted, labels=sorted(set(d.category)),
                                             output_dict=True, zero_division=0)).T
    rep = rep.loc[[c for c in rep.index if c in CATEGORIES]].sort_values("support", ascending=False)
    rep["train_rows"] = rep.index.map(train_counts).fillna(0).astype(int)
    return rep


def plot_confusion(d, path, title):
    labs = [c for c in CATEGORIES if c in set(d.category) | set(d.predicted)]
    cm = confusion_matrix(d.category, d.predicted, labels=labs)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("reviewed label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def coverage(d, cut):
    acc = (d.score >= cut) & (d.predicted != "Uncertain")
    a = d[acc]
    total = d.amount.abs().sum()
    return dict(cutoff=cut, accepted=int(acc.sum()), eligible=len(d), row_coverage=float(acc.mean()),
                dollar_coverage=float(a.amount.abs().sum() / total) if total else np.nan,
                accepted_error=float((~a.correct).mean()) if len(a) else np.nan, to_review=int((~acc).sum()))


def choose_cutoff(val, cutoffs, max_err, min_accepted=10):
    cov = pd.DataFrame([coverage(val, c) for c in cutoffs])
    ok = cov[(cov.accepted_error <= max_err) & (cov.accepted >= min_accepted)]
    if len(ok):
        cut = float(ok.sort_values(["row_coverage", "cutoff"], ascending=[False, True]).iloc[0].cutoff)
        why = f"highest coverage with accepted error <= {max_err:.0%}"
    else:
        c2 = cov[cov.accepted >= min_accepted].sort_values(["accepted_error", "row_coverage"], ascending=[True, False])
        cut = float(c2.iloc[0].cutoff) if len(c2) else max(cutoffs)
        why = f"no cutoff reached <= {max_err:.0%}; lowest accepted error"
    return cut, why, cov


def evaluate(cfg, model, sets, text_col, out):
    train_patterns = set(sets["train"]["pattern"])
    train_counts = sets["train"].category.value_counts()
    val = add_predictions(model, sets["val"], text_col, train_patterns)
    vs = scores(val.category, val.predicted)
    lo, hi = wilson(int(val.correct.sum()), len(val))
    rep = per_class(val, train_counts)
    rep.to_csv(out / "per_class_val.csv")
    plot_confusion(val, out / "confusion.png", "Validation confusion matrix")

    train_tok = Counter(t for x in sets["train"][text_col] for t in set(x.split()) if not t.startswith("<"))
    errors = val[~val.correct].copy()
    errors["unseen_words"] = errors[text_col].map(
        lambda x: all(train_tok.get(t, 0) == 0 for t in x.split() if not t.startswith("<")))
    keep = [c for c in ["item_id", "memo_raw", "category", "predicted", "score", "label_source", "pattern_in_train",
                        "unseen_words", "account_key", "amount"] if c in errors]
    errors[keep].sort_values("score", ascending=False).to_csv(out / "errors_val.csv", index=False)

    cut, why, cov = choose_cutoff(val, cfg["evaluate"]["cutoffs"], cfg["evaluate"]["max_accepted_error"])
    cov.to_csv(out / "coverage_val.csv", index=False)

    tests = {s: add_predictions(model, sets[s], text_col, train_patterns) for s in ["test_A", "test_B"]}
    shared = set(tests["test_A"].category) & set(tests["test_B"].category)
    train_words = set(" ".join(sets["train"][text_col]).split())

    def unseen_share(texts):   # average share of a memo's words never seen in training (Vanessa's PA4 check)
        sh = [sum(w not in train_words for w in t.split()) / len(t.split()) for t in texts if t.split()]
        return float(np.mean(sh)) if sh else np.nan

    rows = []
    for name, d in [("validation", val), *tests.items()]:
        c = coverage(d, cut)
        rows.append(dict(set=name, **scores(d.category, d.predicted), accounts=d.account_key.nunique(),
                         unseen_word_share=unseen_share(d[text_col]),
                         pattern_in_train=float(d.pattern_in_train.mean()),
                         acc_seen_patterns=float(d[d.pattern_in_train].correct.mean()) if d.pattern_in_train.any() else np.nan,
                         acc_unseen_patterns=float(d[~d.pattern_in_train].correct.mean()) if (~d.pattern_in_train).any() else np.nan,
                         acc_on_shared_classes=float(d[d.category.isin(shared)].correct.mean()) if shared else np.nan,
                         row_coverage=c["row_coverage"], dollar_coverage=c["dollar_coverage"],
                         accepted_error=c["accepted_error"]))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "scores.csv", index=False)

    mix = pd.concat({s: d.category.value_counts(normalize=True) for s, d in
                     [("train", sets["train"]), ("val", val), *tests.items()]}, axis=1).fillna(0)
    (mix * 100).round(1).to_csv(out / "class_mix.csv")

    gamb = []
    for name, d in tests.items():
        g = d["subcategory"].eq("gambling") if "subcategory" in d else pd.Series(False, index=d.index)
        for flag, part in d.groupby(g):
            gamb.append(dict(set=name, gambling=bool(flag), rows=len(part), accuracy=float(part.correct.mean())))

    for name, d in tests.items():
        rep_t = per_class(d, train_counts)
        rep_t.to_csv(out / f"per_class_{name}.csv")
    pd.concat([val.assign(set="val")] + [d.assign(set=s) for s, d in tests.items()]).to_csv(
        out / "predictions.csv", index=False)

    return dict(validation=dict(**vs, ci95=[lo, hi], one_row_points=100 / max(len(val), 1)),
                cutoff=dict(value=cut, why=why, val=coverage(val, cut)),
                scores=summary.to_dict("records"), gambling_slice=gamb,
                most_confused_val=[f"{a} -> {b}: {n}" for (a, b), n in
                                   Counter(zip(errors.category, errors.predicted)).most_common(8)])

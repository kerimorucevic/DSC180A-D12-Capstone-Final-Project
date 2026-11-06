"""Entrypoint:  python -m src.main --config config.yaml --steps <steps or stage>

The steps follow the quarter (Lectures 2-5, then Project A). A stage name runs its steps in order:

  l2         explore, merchants        data exploration: parquet + account files, measured merchant frequencies
  l3         tfidf, agreement          TF-IDF cleaning thresholds vs case/spacing baseline; agreement on shared 100
  l4         prepare, baseline         reviewed labels + frozen split; majority, TF-IDF + logistic regression and
                                       first fastText baselines on the same validation rows
  l5         prepare, train, evaluate, runtime
                                       tuned fastText, class errors, review cutoff + coverage, Test A / B, runtime
  project-a  monthly                   Subscriptions & Entertainment monthly series + FRED comparison
  all        prepare, baseline, train, evaluate, runtime   (default)

Steps:
  explore    DuckDB profile of the raw parquet files and account/balance/link files
  merchants  pre-tag rule coverage and top memo patterns per category on a random sample
  tfidf      Lecture 3 PQ1: three TF-IDF token-removal settings vs the case/spacing-only baseline
  agreement  labeler agreement on the shared multiply-labeled memos (pairwise, unanimous, Fleiss' kappa)
  prepare    validate labels, add cleaned text columns, leakage checks, class counts, Pareto check
  baseline   majority class, TF-IDF + logistic regression, earlier fastText settings
  train      staged fastText tuning on validation, then fit and save the final model
  evaluate   per-class report, confusion matrix, errors, review cutoff, Test A / Test B (scored once)
  runtime    fit / load / inference timing and human review effort
  monthly    Project A series by predicted category (needs parquet access)

Labeling steps (Lecture 5, DataHub, run in order when growing the label set):
  sample        scan the seed-label accounts, freeze the split, choose the batch, write review_batch.csv
                -> then review interactively:  python -m src.review --config config.yaml
  audit         score the audit sample; accept deferred AI labels only if <= 5% needed correcting
  build-labels  seed labels + reviewed batch -> labels_v2.csv
"""
import argparse
import json
import time

import fasttext
import numpy as np
import pandas as pd

from .clean import add_text_cols, to_direction, txn_key
from .config import load_config
from .rules import subcategory
from .taxonomy import CATEGORIES, parse_category

SPLITS = ["train", "val", "test_A", "test_B"]
STAGES = {"l2": "explore,merchants", "l3": "tfidf,agreement", "l4": "prepare,baseline",
          "l5": "prepare,train,evaluate,runtime", "project-a": "monthly",
          "all": "prepare,baseline,train,evaluate,runtime"}


def read_labels(path):
    df = pd.read_csv(path, dtype={"item_id": str, "account_key": str}).fillna({"account_key": "", "note": ""})
    if "posted_date" in df and "date" not in df:
        df["date"] = df["posted_date"]
    if "credit_or_debit" in df and "direction" not in df:
        df["direction"] = df["credit_or_debit"].map(to_direction)
    df["account_key"] = df["account_key"].astype(str).str.replace(r"\.0$", "", regex=True).replace("nan", "")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["category"] = df["category"].map(lambda c: parse_category(c) if pd.notna(c) and str(c).strip() else None)
    return df


def step_prepare(cfg, out):
    labels = read_labels(cfg["paths"]["labels"])
    n0 = len(labels)
    unlabeled = labels["category"].isna().sum()
    labels = labels[labels["category"].notna()]                  # train on reviewed, nonempty labels only
    labels = add_text_cols(labels)
    labels["subcategory"] = [subcategory(m, d) for m, d in zip(labels["memo_raw"], labels.get("direction", None))]
    use = labels[labels["split"].isin(SPLITS)].copy()

    counts = pd.crosstab(use["category"], use["split"]).reindex(index=CATEGORIES, columns=SPLITS, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["flag"] = np.select([counts.total == 0, counts.train == 0, counts.train < 5],
                               ["missing", "no train rows", "<5 train"], "")
    counts.to_csv(out / "class_counts.csv")

    # leakage checks: held-out (test_B) accounts never in train/val, no transaction in two splits
    tb = set(use.loc[use.split == "test_B", "account_key"]) - {""}
    tv = set(use.loc[use.split.isin(["train", "val"]), "account_key"]) - {""}
    assert not (tb & tv), f"test_B accounts also in train/val: {sorted(tb & tv)[:5]}"
    kd = use[use.date.notna() & (use.account_key != "")]
    dup = txn_key(kd).duplicated().sum()
    assert dup == 0, f"{dup} transactions appear more than once"

    pat = use.assign(dollars=use.amount.abs()).groupby("pattern").agg(rows=("pattern", "size"), dollars=("dollars", "sum"))
    pat = pat.sort_values("rows", ascending=False)
    rc, uc = pat.rows.cumsum() / pat.rows.sum(), pat.dollars.cumsum() / pat.dollars.sum()
    pareto = {f"top_{k}": dict(rows=round(float(rc.iloc[k - 1]), 3), dollars=round(float(uc.iloc[k - 1]), 3))
              for k in [10, 50, 100] if k <= len(pat)}
    pareto.update(patterns=len(pat), singleton_share=round(float((pat.rows == 1).mean()), 3))

    use.to_csv(out / "labels_prepared.csv", index=False)
    info = dict(rows_in_file=n0, unlabeled_dropped=int(unlabeled), excluded=int((~labels.split.isin(SPLITS)).sum()),
                by_split=use.split.value_counts().to_dict(),
                by_source=use["label_source"].value_counts().to_dict() if "label_source" in use else None,
                classes_lt5_train=list(counts.index[(counts.train < 5) & (counts.total > 0)]),
                classes_missing=list(counts.index[counts.total == 0]), gambling_rows=int((use.subcategory == "gambling").sum()),
                pareto=pareto)
    print(f"prepare: {len(use)} labeled rows {info['by_split']} | <5 train rows: {info['classes_lt5_train']}")
    return use, info


def load_sets(out):
    use = pd.read_csv(out / "labels_prepared.csv", dtype={"item_id": str, "account_key": str}).fillna(
        {"account_key": "", "text_lower": "", "text_norm": "", "subcategory": ""})
    return {s: use[use.split == s].reset_index(drop=True) for s in SPLITS}


def step_train(cfg, out):
    from .model import fit, model_mb
    from .tune import tune
    sets = load_sets(out)
    final, exp, stab, tuning_s, base = tune(cfg, sets, out)
    exp.to_csv(out / "experiments.csv", index=False)
    stab.to_csv(out / "stability.csv", index=False)
    # final fit: tuned winner first; on fastText's NaN bug fall back through other seeds, then the next candidate
    model, note = None, ""
    for rank, cand in enumerate(stab.itertuples()):
        for s in cfg["fasttext"]["stability_seeds"]:
            params = {**json.loads(cand.params), "seed": s}
            model, fit_s, err = fit(sets["train"], cand.text_col, out, base, **params)
            if model is not None:
                text_col, final = cand.text_col, params
                if rank or s != cfg["seed"]:
                    note = f"winner hit fastText NaN; used stability candidate #{rank + 1} with seed {s}"
                break
        if model is not None:
            break
    assert model is not None, f"every final fit failed: {err}"
    model.save_model(str(out / "model.bin"))
    info = dict(text_col=text_col, params=final, fit_note=note, fit_s=round(fit_s, 3), train_rows=len(sets["train"]),
                model_mb=model_mb(model), tuning_s=round(tuning_s, 1), n_runs=len(exp),
                stability=stab.to_dict("records"))
    json.dump(info, open(out / "final_model.json", "w"), indent=2, default=float)
    print(f"train: {text_col} {final} | fit {fit_s:.2f}s | best mean val macro-F1 "
          f"{stab.iloc[0].mean_macro_f1:.3f} ± {stab.iloc[0].sd:.3f}")
    return info


def step_baseline(cfg, out):
    from .tune import baseline_runs
    ft = cfg["fasttext"]
    base = dict(thread=ft["thread"], bucket=ft["bucket"], seed=cfg["seed"], **ft["defaults"])
    tbl = baseline_runs(cfg, load_sets(out), out, base)
    tbl.to_csv(out / "baseline.csv", index=False)
    for r in tbl.itertuples():
        print(f"baseline: {r.model:60s} acc={r.accuracy:.3f} macro-F1={r.macro_f1:.3f}")
    return tbl.to_dict("records")


def sample_memos(cfg, out, n):
    """Random parquet memos when reachable (DataHub), otherwise the labeled memos."""
    try:
        from .data import general_sample
        g = general_sample(cfg)
        g = g[g.memo_raw.str.strip() != ""].head(n)
        return g.memo_raw.tolist(), g.direction.tolist(), "random parquet sample"
    except Exception as e:
        lab = read_labels(cfg["paths"]["labels"])
        return (lab.memo_raw.fillna("").tolist(), lab.get("direction", pd.Series([None] * len(lab))).tolist(),
                f"labeled memos (parquet not reachable: {type(e).__name__})")


def step_evaluate(cfg, out):
    from .evaluate import evaluate
    from .model import load
    sets = load_sets(out)
    info = json.load(open(out / "final_model.json"))
    res = evaluate(cfg, load(out / "model.bin"), sets, info["text_col"], out)
    for r in res["scores"]:
        print(f"evaluate: {r['set']:10s} n={r['n']:4d} acc={r['accuracy']:.3f} macro-F1={r['macro_f1']:.3f}")
    print(f"review cutoff {res['cutoff']['value']} ({res['cutoff']['why']})")
    return res


def step_runtime(cfg, out):
    from .runtime import human_effort, measure
    info = json.load(open(out / "final_model.json"))
    memos = None
    try:
        from .data import general_sample
        memos = general_sample(cfg, n=cfg["runtime"]["n_rows"])["memo_raw"].tolist()
        source = "random parquet sample"
    except Exception as e:
        memos = pd.read_csv(out / "labels_prepared.csv")["memo_raw"].fillna("").tolist()
        memos = (memos * (cfg["runtime"]["n_rows"] // max(len(memos), 1) + 1))[:cfg["runtime"]["n_rows"]]
        source = f"labeled memos repeated (parquet not reachable: {type(e).__name__})"
    rt = measure(cfg, out / "model.bin", info["text_col"], memos)
    rt["memo_source"] = source
    rt["fit_s"], rt["train_rows"] = info["fit_s"], info["train_rows"]
    rt["human"] = human_effort(cfg["paths"].get("review_file"))
    print(f"runtime: {rt['rows_per_s']:,} rows/s | 1M memos ~{rt['seconds_per_1m']}s | "
          f"full dataset ~{rt['hours_full_dataset']} h on one core ({source})")
    return rt


def step_monthly(cfg, out, metrics):
    from .data import general_sample
    from .model import load
    from .monthly import monthly_series
    info = json.load(open(out / "final_model.json"))
    cut = metrics.get("evaluate", {}).get("cutoff", {}).get("value", 0.9)
    pa = cfg["project_a"]
    return monthly_series(general_sample(cfg), load(out / "model.bin"), info["text_col"], cut, out,
                          pa["category"], [f["id"] for f in pa.get("fred_series", [])])


def step_sample(cfg):
    from .data import scan_accounts
    from .review import make_review_file
    from .sample import build_pool, choose_batch
    from .split import freeze_plan
    seed = read_labels(cfg["paths"]["seed_labels"])
    accounts = sorted(set(seed.loc[seed.account_key != "", "account_key"]))
    pool = build_pool(scan_accounts(cfg, accounts), seed)
    plan = freeze_plan(cfg, pool, seed)
    make_review_file(cfg, choose_batch(cfg, pool, plan, seed))
    print("next: python -m src.review --config config.yaml")


def run(config="config.yaml", steps="all"):
    """Run steps and/or stages; returns the updated metrics dict (also saved to outputs/metrics.json)."""
    cfg = load_config(config)
    out = cfg["paths"]["out_dir"]
    todo = []
    for s in [x.strip() for x in steps.split(",") if x.strip()]:
        for t in STAGES.get(s, s).split(","):
            if t not in todo:
                todo.append(t)
    mpath = out / "metrics.json"
    metrics = json.load(open(mpath)) if mpath.exists() else {}
    for s in todo:
        if s == "sample":
            step_sample(cfg)
        elif s == "audit":
            from .review import audit_decision
            _, err = audit_decision(cfg)
            metrics["audit_error"] = err
        elif s == "build-labels":
            from .review import build_labels
            plan = json.load(open(cfg["paths"]["split_plan"]))
            build_labels(cfg, read_labels(cfg["paths"]["seed_labels"]), plan)
        elif s == "prepare":
            _, metrics["labels"] = step_prepare(cfg, out)
        elif s == "baseline":
            metrics["baseline"] = step_baseline(cfg, out)
        elif s == "train":
            metrics["model"] = step_train(cfg, out)
        elif s == "evaluate":
            metrics["evaluate"] = step_evaluate(cfg, out)
        elif s == "runtime":
            metrics["runtime"] = step_runtime(cfg, out)
        elif s == "monthly":
            metrics["monthly"] = step_monthly(cfg, out, metrics)
        elif s == "explore":
            from .explore import profile
            metrics["data"] = profile(cfg, out)
        elif s == "merchants":
            from .explore import merchant_frequencies
            memos, dirs, src = sample_memos(cfg, out, cfg["explore"]["merchant_sample"])
            metrics["merchants"] = dict(source=src, **merchant_frequencies(memos, dirs, out))
        elif s == "tfidf":
            from .tfidf_study import study
            memos, dirs, src = sample_memos(cfg, out, 10 ** 6)
            cf = cfg["tfidf_study"].get("cue_file")
            manual = pd.read_csv(cf) if cf else None
            metrics["tfidf_study"] = dict(source=src, **study(cfg, memos, dirs, out, manual))
        elif s == "agreement":
            from .agreement import agreement
            metrics["agreement"] = agreement(cfg, out)
        else:
            raise SystemExit(f"unknown step {s!r}")
        json.dump(metrics, open(mpath, "w"), indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M"), steps=todo,
                    fasttext=getattr(fasttext, "__version__", "?"), pandas=pd.__version__,
                    files=sorted(p.name for p in out.iterdir() if p.is_file() and not p.name.startswith("_")))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", default="all", help="comma-separated steps and/or stages (l2, l3, l4, l5, project-a, all)")
    a = ap.parse_args()
    run(a.config, a.steps)


if __name__ == "__main__":
    main()

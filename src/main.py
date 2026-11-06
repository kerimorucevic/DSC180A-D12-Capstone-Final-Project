"""Entrypoint:  python -m src.main --config config.yaml [--steps train,evaluate]

Labeling (DataHub, in order):
  sample        freeze split, choose memos (rare categories, common patterns, random), write review_batch.csv
  (review)      python -m src.labeling --config config.yaml
  audit         check the 25% audit sample; auto-accept the rest only if <= 5% needed correcting
  build-labels  earlier labels + reviewed batch -> labels_v2.csv

Model (rerun any time from labels_v2.csv):
  train         clean both text versions, check for leakage, tune fastText on validation, save the final model
  evaluate      validation, Test A and Test B scores, per-class report, confusion matrix, errors
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .labeling import (add_text_cols, audit, build_labels, choose_batch, freeze_split, make_review_file,
                       read_labels, scan_accounts, txn_key)
from .rules import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "val", "test_A", "test_B"]


def load_config(path="config.yaml"):
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    cfg = yaml.safe_load(open(path))
    for k, v in cfg["paths"].items():
        if v:
            p = Path(v).expanduser()
            cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    cfg["paths"]["out_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def load_sets(cfg):
    """Reviewed labels -> cleaned text, leakage checks, class counts; returns one frame per split."""
    out = cfg["paths"]["out_dir"]
    lab = read_labels(cfg["paths"]["labels"])
    lab = add_text_cols(lab[lab["category"].notna() & lab["split"].isin(SPLITS)])
    tb = set(lab.loc[lab.split == "test_B", "account_key"]) - {""}
    tv = set(lab.loc[lab.split.isin(["train", "val"]), "account_key"]) - {""}
    assert not (tb & tv), "a Test B account also appears in train/val"
    keyed = lab[lab.date.notna() & (lab.account_key != "")]
    assert not txn_key(keyed).duplicated().any(), "a transaction appears more than once"
    counts = pd.crosstab(lab.category, lab.split).reindex(index=CATEGORIES, columns=SPLITS, fill_value=0)
    counts.to_csv(out / "class_counts.csv")
    windows = lab.groupby("split").agg(rows=("category", "size"), accounts=("account_key", "nunique"),
                                       first_date=("date", "min"), last_date=("date", "max"))
    windows.to_csv(out / "split_summary.csv")
    print(windows.to_string(), "\n")
    return {s: lab[lab.split == s].reset_index(drop=True) for s in SPLITS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", default="train,evaluate")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["out_dir"]
    for step in [s.strip() for s in a.steps.split(",") if s.strip()]:
        if step == "sample":
            seed = read_labels(cfg["paths"]["seed_labels"])
            raw = scan_accounts(cfg, sorted(set(seed.account_key) - {""}))
            plan = freeze_split(cfg, raw, seed)
            make_review_file(cfg, choose_batch(cfg, raw, plan, seed))
            print("next: python -m src.labeling --config", a.config)
        elif step == "audit":
            audit(cfg)
        elif step == "build-labels":
            build_labels(cfg, read_labels(cfg["paths"]["seed_labels"]), json.load(open(cfg["paths"]["split_plan"])))
        elif step == "train":
            from .model import tune
            _, info = tune(cfg, load_sets(cfg), out)
            print(f"final model: {info['text_col']} {info['params']} | val macro-F1 "
                  f"{info['val_macro_f1_mean']:.3f} ± {info['val_macro_f1_sd']:.3f}")
        elif step == "evaluate":
            from .model import evaluate, load
            info = json.load(open(out / "final_model.json"))
            print(evaluate(load(out / "model.bin"), load_sets(cfg), info["text_col"], out).to_string(index=False))
        else:
            raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()

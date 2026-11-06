"""AI suggests, people adjudicate.

1. make_review_file : pre-tag rule suggestion (label + quoted cue + confidence); where no rule matches, an
                      earlier fastText model fills in if its score >= 0.5.
2. review_loop      : interactive, timed review in a terminal/notebook (accept / correct / unresolved).
3. audit_decision   : in the training bucket a random 25% of high-confidence suggestions is reviewed; the rest
                      is accepted only if <= 5% of the audit needed correcting (ours was 14%, so all rows were
                      reviewed by hand). Validation and test rows are always reviewed individually.
4. build_labels     : prior labels + reviewed batch -> labels_v2.csv (reviewed, nonempty labels only).

Run interactively:  python -m src.review --config config.yaml
"""
import argparse
import math
import time

import numpy as np
import pandas as pd

from .clean import add_text_cols
from .config import load_config
from .rules import pretag
from .split import assign_split
from .taxonomy import CATEGORIES, code_table, parse_category

REVIEW_COLS = ["review_id", "bucket", "source", "account_key", "date", "amount", "direction", "memo_raw",
               "text_norm", "pattern", "pattern_n", "ai_label", "ai_conf", "ai_cue", "ai_source", "ft_label",
               "ft_prob", "review_required", "audit_sample", "final_label", "review_status", "note",
               "review_seconds", "reviewed_at"]


def make_review_file(cfg, batch, overwrite=False):
    path, rv = cfg["paths"]["review_file"], cfg["review"]
    if path.exists() and not overwrite:
        print(f"{path} exists; keeping review progress")
        return pd.read_csv(path, dtype=str).fillna("")
    rng = np.random.default_rng(cfg["seed"])
    b = batch.copy()
    r = [pretag(m, d) for m, d in zip(b["memo_raw"], b["direction"])]
    b["rule_cat"], b["rule_conf"], b["rule_id"], b["rule_cue"] = zip(*r) if r else ([], [], [], [])
    b["ft_label"], b["ft_prob"] = "", np.nan
    if rv.get("suggest_model") and cfg["paths"].get("suggest_model") and cfg["paths"]["suggest_model"].exists():
        from .model import load, predict
        m = load(cfg["paths"]["suggest_model"])
        b["ft_label"], b["ft_prob"] = predict(m, b["text_lower"])
    use_rule = b["rule_cat"].notna()
    use_ft = ~use_rule & (b["ft_prob"] >= 0.5)
    b["ai_label"] = np.where(use_rule, b["rule_cat"], np.where(use_ft, b["ft_label"], ""))
    b["ai_conf"] = np.where(use_rule, b["rule_conf"], np.where(use_ft, "low", ""))
    b["ai_source"] = np.where(use_rule, "rule:" + b["rule_id"].astype(str), np.where(use_ft, "fasttext", ""))
    b["ai_cue"] = np.where(use_rule, b["rule_cue"].astype(str),
                           np.where(use_ft, "fasttext score " + b["ft_prob"].round(2).astype(str), ""))
    hi = b.index[(b.bucket == "train") & (b.ai_conf == "high")]
    n_audit = min(len(hi), max(rv["audit_min"], math.ceil(rv["audit_fraction"] * len(hi))))
    audit_ids = set(rng.choice(hi, size=n_audit, replace=False)) if len(hi) else set()
    b["review_required"] = ~b.index.isin(hi) | b.index.isin(audit_ids)
    b["audit_sample"] = b.index.isin(audit_ids)
    b["pattern_n"] = b.groupby("pattern")["pattern"].transform("size")   # review repeated memos together
    b = b.sort_values(["pattern_n", "pattern", "date"], ascending=[False, True, True]).reset_index(drop=True)
    b.insert(0, "review_id", np.arange(1, len(b) + 1))
    b["source"] = b.get("source", "same_accounts")
    for c in ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]:
        b[c] = ""
    b[REVIEW_COLS].to_csv(path, index=False)
    print(f"wrote {path}: {int(b.review_required.sum())} to review, {len(audit_ids)} in the audit sample")
    return pd.read_csv(path, dtype=str).fillna("")


def review_loop(path, include_deferred=False):
    """Enter=accept | a note | <number or name> [note]=correct | u note=unresolved | b back | s skip | q quit."""
    df = pd.read_csv(path, dtype=str).fillna("")
    todo = [i for i in df.index if df.at[i, "review_status"] == ""
            and (include_deferred or df.at[i, "review_required"] == "True")]
    hist, k, session = [], 0, {}
    code_table()
    while k < len(todo):
        i = todo[k]
        r = df.loc[i]
        done = int(((df.review_status != "") & (df.review_required == "True")).sum())
        print("\n" + "-" * 90)
        print(f"[{done}/{int((df.review_required == 'True').sum())} reviewed]  pattern: {r.pattern!r} ({r.pattern_n} in batch)\n")
        print(f"  MEMO   {r.memo_raw}")
        print(f"         ${float(r.amount or 0):,.2f}  {r.direction}  {r.date[:10]}\n")
        print(f"  AI     {r.ai_label}  [{r.ai_conf}]  cue: {r.ai_cue!r}  ({r.ai_source})" if r.ai_label
              else "  AI     (no suggestion: pick a category)")
        if r.pattern in session:
            print(f"  earlier this session, {r.pattern!r} -> {session[r.pattern]}")
        t0 = time.time()
        ans = input("Enter=accept | a/number/u [+note] | b back | s skip | ? codes | q quit > ").strip()
        secs = round(time.time() - t0, 1)
        cmd, _, note = ans.partition(" ")
        low = cmd.lower()
        if low == "q":
            break
        if low == "?":
            code_table()
            continue
        if low == "s":
            k += 1
            continue
        if low == "b":
            if hist:
                j = hist.pop()
                df.loc[j, ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]] = ""
                df.to_csv(path, index=False)
                k = todo.index(j)
            continue
        if low in ("", "a"):
            if not r.ai_label:
                continue
            lab, status = r.ai_label, "accepted"
        elif low == "u":
            lab, status = "", "unresolved"
        else:
            lab = parse_category(ans) or parse_category(cmd)
            if lab is None:
                continue
            if parse_category(ans):
                note = ""
            status = "accepted" if lab == r.ai_label else "corrected"
        df.loc[i, ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]] = \
            [lab, status, note, str(secs), time.strftime("%Y-%m-%d %H:%M:%S")]
        df.to_csv(path, index=False)
        if lab:
            session[r.pattern] = lab
        hist.append(i)
        k += 1
    left = int(((df.review_status == "") & (df.review_required == "True")).sum())
    print(f"saved. reviewed: {(df.review_status != '').sum()} | still required: {left}")
    return df


def audit_decision(cfg):
    path, max_err = cfg["paths"]["review_file"], cfg["review"]["audit_max_error"]
    rv = pd.read_csv(path, dtype=str).fillna("")
    for i in rv.index[(rv.final_label != "") & ~rv.final_label.isin(CATEGORIES)]:     # CSV edits: numbers/names
        rv.at[i, "final_label"] = parse_category(rv.at[i, "final_label"]) or rv.at[i, "final_label"]
    rv["review_status"] = rv["review_status"].str.lower().str.strip().replace(
        {"a": "accepted", "c": "corrected", "u": "unresolved"})
    m = (rv.review_status == "accepted") & (rv.final_label == "")
    rv.loc[m, "final_label"] = rv.loc[m, "ai_label"]
    aud = rv[(rv.audit_sample == "True") & rv.review_status.isin(["accepted", "corrected"])]
    err = (aud.final_label != aud.ai_label).mean() if len(aud) else np.nan
    bad_rules = set(aud.loc[aud.final_label != aud.ai_label, "ai_source"])
    deferred = rv.index[(rv.review_required != "True") & (rv.review_status == "")]
    if ((rv.audit_sample == "True") & (rv.review_status == "")).any():
        print("finish reviewing the audit sample first")
    elif err <= max_err:
        ok = [i for i in deferred if rv.at[i, "ai_source"] not in bad_rules]
        redo = [i for i in deferred if rv.at[i, "ai_source"] in bad_rules]
        rv.loc[ok, "final_label"] = rv.loc[ok, "ai_label"]
        rv.loc[ok, "review_status"] = "ai_accepted_via_audit"
        rv.loc[redo, "review_required"] = "True"
        print(f"audit passed ({err:.1%}): {len(ok)} accepted via audit, {len(redo)} sent to review")
    else:
        rv.loc[deferred, "review_required"] = "True"
        print(f"audit failed ({err:.1%} > {max_err:.0%}): all {len(deferred)} deferred rows need review")
    rv.to_csv(path, index=False)
    return rv, err


def build_labels(cfg, prior_labels, plan):
    """prior (seed) labels re-split with the frozen plan + reviewed batch -> labels_v2.csv."""
    rv = pd.read_csv(cfg["paths"]["review_file"], dtype=str).fillna("")
    assert not ((rv.review_required == "True") & (rv.review_status == "")).any(), "review not finished"
    new = rv[rv.final_label.isin(CATEGORIES)].copy()
    new["label_source"] = np.select(
        [new.review_status == "ai_accepted_via_audit", new.ai_label == "", new.review_status == "corrected"],
        ["ai_accepted_via_audit", "human_no_suggestion", "human_corrected_ai"], "human_accepted_ai")
    new = new.rename(columns={"final_label": "category", "bucket": "split"})
    new["item_id"] = "n" + new["review_id"]
    cols = ["item_id", "split", "label_source", "review_status", "account_key", "date", "amount", "direction",
            "memo_raw", "category", "note"]
    frames = [new[cols]]
    if prior_labels is not None and len(prior_labels):
        o = prior_labels.copy()
        o["split"] = assign_split(o, plan)
        o["label_source"] = o.get("label_source", "prior_human")
        o["review_status"] = o.get("review_status", "prior")
        frames.insert(0, o.reindex(columns=cols))
    labels = add_text_cols(pd.concat(frames, ignore_index=True))
    labels.to_csv(cfg["paths"]["labels"], index=False)
    print(f"wrote {cfg['paths']['labels']}: {len(labels)} labels | unresolved left out: "
          f"{(rv.review_status == 'unresolved').sum()}")
    print(pd.crosstab(labels["label_source"], labels["split"], margins=True))
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--include-deferred", action="store_true")
    a = ap.parse_args()
    review_loop(load_config(a.config)["paths"]["review_file"], a.include_deferred)

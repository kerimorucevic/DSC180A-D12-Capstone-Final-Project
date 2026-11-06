"""Building the reviewed label set: cleaning, split, choosing memos, AI suggestions, human review, audit.

Workflow (DataHub):
  1. sample        scan the accounts behind our earlier labels, freeze the split, choose the batch and write
                   review_batch.csv with a suggested label for every memo
  2. review        python -m src.labeling   (interactive, timed: accept / correct with a note / unresolved)
  3. audit         25% of high-confidence suggestions are checked; the rest are auto-accepted only if <= 5%
                   needed correcting (ours: 14%, so every memo was labeled by hand)
  4. build-labels  earlier labels + reviewed batch -> labels_v2.csv
"""
import argparse
import gc
import glob
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .rules import CATEGORIES, CODE, parse_category, pretag

# ----------------------------------------------------------------------------------------------------------------
# Cleaning. memo_raw is never changed.
#   text_lower: lowercase + extra spaces removed
#   text_norm : also replaces dates, masked card numbers and long IDs with <date>, <mask>, <num>
#   pattern   : first three meaningful words after bank boilerplate (used to find common memo patterns)
# ----------------------------------------------------------------------------------------------------------------


def text_lower(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _mask_token(m):
    tok = m.group(0)
    digits = sum(ch.isdigit() for ch in tok)
    return " <num> " if digits >= 4 or (len(tok) >= 8 and digits >= 2) else tok


def text_norm(s):
    s = str(s).lower().replace("[name]", " <name> ")
    s = re.sub(r"https?://\S+|www\.\S+", " <url> ", s)
    s = re.sub(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", " <date> ", s)
    s = re.sub(r"\bx{3,}[0-9x]*", " <mask> ", s)
    s = re.sub(r"[a-z0-9]*\d[a-z0-9]*", _mask_token, s)
    s = re.sub(r"[^a-z0-9<>&' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_BOILER = re.compile(
    r"^(?:(?:purchase authorized on|money transfer authorized on|recurring payment authorized on|"
    r"visa debit purchase|debit card purchase|debit purchase|pos purchase|pos db|pos debit|checkcard|purchase|"
    r"withdrawal pos|withdrawal|mobile purchase|business to business ach|pos adjustment|point of sale)\s*)+")
_DROP = {"card", "on", "the", "to", "from", "for", "of", "payment", "<date>", "<num>", "<mask>", "<url>",
         "<name>", "ach", "web", "ppd", "id", "co"}


def memo_pattern(norm):
    toks = [w for w in _BOILER.sub("", norm).split() if w not in _DROP and not w.isdigit()][:3]
    return " ".join(toks) if toks else "(empty)"


def add_text_cols(df):
    df = df.copy()
    df["memo_raw"] = df["memo_raw"].fillna("").astype(str)
    df["text_lower"] = df["memo_raw"].map(text_lower)
    df["text_norm"] = df["memo_raw"].map(text_norm)
    df["pattern"] = df["text_norm"].map(memo_pattern)
    return df


def to_direction(v):
    """credit_or_debit: 1 = money in (credit), 2 = money out (debit)."""
    return {"1": "credit", "2": "debit", "credit": "credit", "debit": "debit"}.get(
        re.sub(r"\.0$", "", str(v).strip().lower()), "unknown")


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


def txn_key(df):
    return (df["account_key"].astype(str) + "|" + pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").fillna("")
            + "|" + (df["amount"].abs() * 100).round().astype("Int64").astype(str) + "|" + df["memo_raw"])


# ----------------------------------------------------------------------------------------------------------------
# Reading the Prism parquet files: every transaction for a set of accounts (cached per file)
# ----------------------------------------------------------------------------------------------------------------


def scan_accounts(cfg, accounts):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(str(Path(cfg["paths"]["data_dir"]) / cfg["data"]["parquet_glob"])))
    if not parts:
        raise FileNotFoundError(f"no parquet files under {cfg['paths']['data_dir']}")
    names = pq.ParquetFile(parts[0]).schema_arrow.names
    cols = [c for c in ["account_key", "posted_date", "amount", "credit_or_debit", "memo"] if c in names]
    cache = cfg["paths"]["out_dir"] / "account_scan"
    cache.mkdir(exist_ok=True)
    for path in parts:
        out = cache / (Path(path).stem + ".parquet")
        if out.exists():
            continue
        pf = pq.ParquetFile(path)
        ftype = pf.schema_arrow.field("account_key").type
        vs = pa.array([int(a) for a in accounts]).cast(ftype) if pa.types.is_integer(ftype) else \
            pa.array([str(a) for a in accounts]).cast(ftype)
        kept = []
        for g in range(pf.num_row_groups):        # read account_key first, full columns only where it matches
            mask = pc.is_in(pf.read_row_group(g, columns=["account_key"]).column(0), value_set=vs)
            if pc.any(mask).as_py():
                kept.append(pf.read_row_group(g, columns=cols).filter(mask))
        pq.write_table(pa.concat_tables(kept) if kept else pf.schema_arrow.empty_table().select(cols), out)
        del kept
        gc.collect()
    raw = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("*.parquet"))], ignore_index=True)
    raw["memo_raw"] = raw.pop("memo").map(
        lambda b: b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else ("" if b is None else str(b)))
    raw["account_key"] = raw["account_key"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["date"] = pd.to_datetime(raw["posted_date"], errors="coerce")
    raw["direction"] = raw["credit_or_debit"].map(to_direction) if "credit_or_debit" in raw else "unknown"
    return raw[["account_key", "date", "amount", "direction", "memo_raw"]]


# ----------------------------------------------------------------------------------------------------------------
# Split: time + held-out accounts, frozen before choosing memos and saved to split_plan.json
#   train < val_start <= validation < test_start <= test
#   test_A = later rows from accounts seen in training, test_B = later rows from held-out accounts
# ----------------------------------------------------------------------------------------------------------------


def windows(dates, vs, ts):
    d = pd.to_datetime(dates)
    return pd.Series(np.select([d.isna(), d >= ts, d >= vs], ["no_date", "test", "val"], "train"), index=d.index)


def freeze_split(cfg, pool, seed_labels):
    path = cfg["paths"]["split_plan"]
    if path.exists():
        return json.load(open(path))
    s, t = cfg["split"], cfg["sampling"]["bucket_targets"]
    vs, ts = pd.Timestamp(s["val_start"]), pd.Timestamp(s["test_start"])
    per = windows(pool["date"], vs, ts)
    need = {"train": t["train"], "val": t["val"], "test": t["test_A"] + t["test_B"]}
    source = "team cutoffs"
    if not all((per == w).sum() >= 2 * n for w, n in need.items()):
        # not enough rows in a window: use the months nearest 70% / 85% of this pool's own transactions
        m = pool["date"].dt.to_period("M").value_counts().sort_index()
        cum = m.cumsum() / m.sum()
        vs = ((cum - .70).abs().idxmin() + 1).to_timestamp()
        ts = ((cum - .85).abs().idxmin() + 1).to_timestamp()
        source = "70/85 rule on the pool's dates"
        per = windows(pool["date"], vs, ts)
    early = set(seed_labels.loc[windows(seed_labels["date"], vs, ts).isin(["train", "val"]), "account_key"])
    tcount = pool[per == "test"].groupby("account_key").size()
    order = sorted([a for a, n in tcount.items() if n >= 3],
                   key=lambda a: (a in early, hashlib.md5(f"{cfg['seed']}-{a}".encode()).hexdigest()))
    held, covered = [], 0
    for a in order:
        if len(held) >= max(3, round(s["heldout_account_share"] * len(tcount))) and covered >= 1.5 * t["test_B"]:
            break
        held.append(a)
        covered += tcount[a]
    plan = dict(val_start=str(vs.date()), test_start=str(ts.date()), cutoff_source=source,
                held_out_accounts=held, frozen_at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(plan, open(path, "w"), indent=2)
    return plan


def assign_split(df, plan):
    vs, ts = pd.Timestamp(plan["val_start"]), pd.Timestamp(plan["test_start"])
    per = windows(df["date"], vs, ts)
    held = df["account_key"].astype(str).isin(set(plan["held_out_accounts"]))
    return pd.Series(np.select([per == "no_date", held & (per == "test"), held, per == "test", per == "val"],
                               ["train", "test_B", "excluded", "test_A", "val"], "train"), index=df.index)


# ----------------------------------------------------------------------------------------------------------------
# Choosing memos. Validation and test memos are random samples (evaluation reflects real transactions).
# Training memos are chosen on purpose: rare categories, common memo patterns, random memos.
# ----------------------------------------------------------------------------------------------------------------


def _pick(df, n, cap, taken, per_acct, rng):
    """n random rows not taken yet, at most `cap` per account so one busy account can't dominate."""
    df = df[~df.index.isin(taken)]
    df = df.iloc[rng.permutation(len(df))]
    out = []
    while len(out) < n and len(df) and cap < 10 ** 6:
        for i, a in zip(df.index, df["account_key"]):
            if len(out) >= n:
                break
            if per_acct[a] < cap:
                out.append(i)
                taken.add(i)
                per_acct[a] += 1
        df = df[~df.index.isin(taken)]
        cap *= 2
    return out


def choose_batch(cfg, raw, plan, seed_labels):
    rng = np.random.default_rng(cfg["seed"])
    t, rare_target = cfg["sampling"]["bucket_targets"], cfg["sampling"]["rare_target"]
    pool = raw.drop_duplicates(["account_key", "date", "amount", "memo_raw"])
    pool = pool[~txn_key(pool).isin(set(txn_key(seed_labels)))]
    pool = add_text_cols(pool[pool["memo_raw"].str.strip() != ""].reset_index(drop=True))
    pool["period"] = windows(pool["date"], pd.Timestamp(plan["val_start"]), pd.Timestamp(plan["test_start"]))
    pool["held_out"] = pool["account_key"].isin(set(plan["held_out_accounts"]))
    dev, hold = pool[~pool.held_out], pool[pool.held_out]

    taken, chosen = set(), {}
    for b, d in [("test_B", hold[hold.period == "test"]), ("test_A", dev[dev.period == "test"]),
                 ("val", dev[dev.period == "val"])]:
        cap = max(2, math.ceil(1.5 * t[b] / max(1, d["account_key"].nunique())))
        chosen[b] = _pick(d, t[b], cap, taken, Counter(), rng)

    tr = dev[dev.period == "train"].copy()
    T, cap, per_acct, ids = t["train"], max(10, math.ceil(0.08 * t["train"])), Counter(), []
    tr["rule_cat"] = [pretag(m, d)[0] for m, d in zip(tr["memo_raw"], tr["direction"])]
    have = seed_labels["category"].value_counts()
    # 1. rare categories: memos the rules propose for categories with few labels
    rare = sorted([c for c in CATEGORIES if c not in ("Uncertain", "Miscellaneous") and have.get(c, 0) < rare_target],
                  key=lambda c: have.get(c, 0))
    for c in rare:
        want = min(rare_target - have.get(c, 0), int(0.35 * T) - len(ids))
        if want > 0:
            ids += _pick(tr[tr.rule_cat == c], want, cap, taken, per_acct, rng)
    n_rare = len(ids)
    # 2. common patterns: up to 2 memos from each of the most frequent memo patterns
    for p in tr["pattern"].value_counts().index:
        if len(ids) - n_rare >= int(0.35 * T):
            break
        ids += _pick(tr[tr.pattern == p], 2, cap, taken, per_acct, rng)
    n_common = len(ids) - n_rare
    # 3. random memos, including ones no rule recognizes
    ids += _pick(tr, T - len(ids), cap, taken, per_acct, rng)
    chosen["train"] = ids

    batch = pd.concat([pool.loc[i].assign(bucket=b) for b, i in chosen.items()]).reset_index(drop=True)
    print(f"chose {len(batch)} memos: train {len(ids)} (rare {n_rare}, common patterns {n_common}, "
          f"random {len(ids) - n_rare - n_common}), " + ", ".join(f"{b} {len(chosen[b])}" for b in ["val", "test_A", "test_B"]))
    assert not (batch.bucket.isin(["train", "val"]) & batch.held_out).any(), "held-out account in train/val"
    return batch.drop(columns=["rule_cat"], errors="ignore")


# ----------------------------------------------------------------------------------------------------------------
# AI suggestions + human review + audit
# ----------------------------------------------------------------------------------------------------------------

REVIEW_COLS = ["review_id", "bucket", "account_key", "date", "amount", "direction", "memo_raw", "pattern",
               "pattern_n", "ai_label", "ai_conf", "ai_cue", "ai_source", "review_required", "audit_sample",
               "final_label", "review_status", "note", "review_seconds", "reviewed_at"]


def make_review_file(cfg, batch):
    """Suggested label for every memo: the first matching pre-tag rule, else an earlier fastText model."""
    path, rv = cfg["paths"]["review_file"], cfg["review"]
    if path.exists():
        print(f"{path} exists; keeping review progress")
        return
    rng = np.random.default_rng(cfg["seed"])
    b = batch.copy()
    r = [pretag(m, d) for m, d in zip(b["memo_raw"], b["direction"])]
    b["ai_label"] = [x[0] or "" for x in r]
    b["ai_conf"] = [x[1] or "" for x in r]
    b["ai_source"] = ["rule:" + x[2] if x[2] else "" for x in r]
    b["ai_cue"] = [x[3] or "" for x in r]
    sm = cfg["paths"].get("suggest_model")
    if sm and Path(sm).exists():
        from .model import load, predict
        labs, probs = predict(load(sm), b["text_lower"])
        fill = (b["ai_label"] == "") & (probs >= 0.5)
        b.loc[fill, "ai_label"] = np.array(labs)[fill]
        b.loc[fill, "ai_conf"] = "low"
        b.loc[fill, "ai_source"] = "fasttext"
        b.loc[fill, "ai_cue"] = ["fasttext score %.2f" % p for p in probs[fill]]
    hi = b.index[(b.bucket == "train") & (b.ai_conf == "high")]
    n_audit = min(len(hi), max(rv["audit_min"], math.ceil(rv["audit_fraction"] * len(hi))))
    audit = set(rng.choice(hi, size=n_audit, replace=False)) if len(hi) else set()
    b["review_required"] = ~b.index.isin(hi) | b.index.isin(audit)     # val/test always reviewed
    b["audit_sample"] = b.index.isin(audit)
    b["pattern_n"] = b.groupby("pattern")["pattern"].transform("size")
    b = b.sort_values(["pattern_n", "pattern"], ascending=[False, True]).reset_index(drop=True)   # repeats together
    b["review_id"] = np.arange(1, len(b) + 1)
    for c in ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]:
        b[c] = ""
    b[REVIEW_COLS].to_csv(path, index=False)
    print(f"wrote {path}: {int(b.review_required.sum())} to review ({len(audit)} audit)")


def review_loop(path):
    """Enter = accept | <number or name> [note] = correct | u <note> = unresolved | b = undo | q = save and quit."""
    df = pd.read_csv(path, dtype=str).fillna("")
    todo = [i for i in df.index if df.at[i, "review_status"] == "" and df.at[i, "review_required"] == "True"]
    print("\n".join(f"  {k:>2} {v}" for k, v in CODE.items()))
    hist, k = [], 0
    while k < len(todo):
        i, r = todo[k], df.loc[todo[k]]
        print(f"\n[{k + 1}/{len(todo)}]  {r.memo_raw}\n  ${float(r.amount or 0):,.2f}  {r.direction}  {r.date[:10]}")
        print(f"  suggested: {r.ai_label} [{r.ai_conf}]  cue: {r.ai_cue!r}" if r.ai_label else "  suggested: (none)")
        t0 = time.time()
        ans = input("Enter accept | number/name [note] | u note | b | q > ").strip()
        secs = round(time.time() - t0, 1)
        cmd, _, note = ans.partition(" ")
        if cmd.lower() == "q":
            break
        if cmd.lower() == "b":
            if hist:
                j = hist.pop()
                df.loc[j, ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]] = ""
                k = todo.index(j)
            continue
        if cmd == "":
            if not r.ai_label:
                continue
            lab, status = r.ai_label, "accepted"
        elif cmd.lower() == "u":
            lab, status = "", "unresolved"
        else:
            lab = parse_category(ans) or parse_category(cmd)
            if lab is None:
                continue
            note = "" if parse_category(ans) else note
            status = "accepted" if lab == r.ai_label else "corrected"
        df.loc[i, ["final_label", "review_status", "note", "review_seconds", "reviewed_at"]] = \
            [lab, status, note, str(secs), time.strftime("%Y-%m-%d %H:%M:%S")]
        df.to_csv(path, index=False)            # saved after every decision
        hist.append(i)
        k += 1
    df.to_csv(path, index=False)
    print(f"saved. still to review: {int(((df.review_status == '') & (df.review_required == 'True')).sum())}")


def audit(cfg):
    path, max_err = cfg["paths"]["review_file"], cfg["review"]["audit_max_error"]
    rv = pd.read_csv(path, dtype=str).fillna("")
    a = rv[(rv.audit_sample == "True") & rv.review_status.isin(["accepted", "corrected"])]
    err = float((a.final_label != a.ai_label).mean()) if len(a) else float("nan")
    deferred = rv.index[(rv.review_required != "True") & (rv.review_status == "")]
    if err <= max_err:
        rv.loc[deferred, "final_label"] = rv.loc[deferred, "ai_label"]
        rv.loc[deferred, "review_status"] = "ai_accepted_via_audit"
        print(f"audit passed ({err:.0%} corrected): {len(deferred)} suggestions accepted without review")
    else:
        rv.loc[deferred, "review_required"] = "True"
        print(f"audit failed ({err:.0%} corrected > {max_err:.0%}): all {len(deferred)} remaining memos need review")
    rv.to_csv(path, index=False)
    return err


def build_labels(cfg, seed_labels, plan):
    rv = pd.read_csv(cfg["paths"]["review_file"], dtype=str).fillna("")
    assert not ((rv.review_required == "True") & (rv.review_status == "")).any(), "review not finished"
    new = rv[rv.final_label.isin(CATEGORIES)].rename(columns={"final_label": "category", "bucket": "split"})
    new["item_id"] = "n" + new["review_id"]
    new["label_source"] = np.select([new.review_status == "ai_accepted_via_audit", new.ai_label == "",
                                     new.review_status == "corrected"],
                                    ["ai_accepted_via_audit", "human_no_suggestion", "human_corrected_ai"],
                                    "human_accepted_ai")
    old = seed_labels.copy()
    old["split"], old["label_source"] = assign_split(old, plan), "earlier_human_label"
    cols = ["item_id", "split", "label_source", "account_key", "date", "amount", "direction", "memo_raw",
            "category", "note"]
    labels = add_text_cols(pd.concat([old.reindex(columns=cols), new[cols]], ignore_index=True))
    labels.to_csv(cfg["paths"]["labels"], index=False)
    print(f"wrote {cfg['paths']['labels']}: {len(labels)} labels")
    print(pd.crosstab(labels["label_source"], labels["split"], margins=True))


if __name__ == "__main__":
    from .main import load_config
    ap = argparse.ArgumentParser(description="interactive review loop")
    ap.add_argument("--config", default="config.yaml")
    review_loop(load_config(ap.parse_args().config)["paths"]["review_file"])

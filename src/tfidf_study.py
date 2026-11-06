"""Lecture 3, participation question 1: does TF-IDF help memo cleaning?

Compares three TF-IDF token-removal settings with the case/spacing-only baseline (Lecture 2's conservative
cleaning) on a sample of memos. Document = one memo, corpus = the sample, tokenizer = whitespace split of
text_lower. Defaults are Group 3's settings from the Lecture 4 review: max_df = 0.10, min_df = 2, TF-IDF >= 0.3.

Each token is tagged:
  useful cue : part of a pre-tag rule match (merchant / transaction type) or a transaction-type word
  noise      : ID-like (4+ digits, long mixed letter-digit strings, card masks) or punctuation only
  other      : everything else (places, store numbers with few digits, ...)
Optionally, a hand-marked file (`tfidf_study.cue_file`: memo_raw, token, kind) overrides the automatic tags.

Reports per setting: tokens kept, noise removed %, useful cues lost %, empty outputs, plus one helpful and one
harmful before/after example. Our finding (report section 4.1): removal lost useful cues, so TF-IDF is kept for
model features only (the TF-IDF + logistic regression baseline), not for cleaning.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .clean import text_lower
from .rules import pretag

TYPE_WORDS = {"purchase", "payment", "pmt", "transfer", "xfer", "atm", "withdrawal", "deposit", "fee", "refund",
              "debit", "credit", "card", "pos", "ach", "check", "cash", "bill", "autopay", "zelle", "venmo",
              "paypal", "loan", "interest", "overdraft", "reversal", "adjustment", "payroll"}


def tag_token(tok, cue_tokens):
    digits = sum(c.isdigit() for c in tok)
    if tok in cue_tokens or tok.strip(".,*#:/-") in TYPE_WORDS:
        return "useful"
    if digits >= 4 or (len(tok) >= 8 and digits >= 2) or re.fullmatch(r"x{3,}[0-9x]*", tok) or \
            re.fullmatch(r"[^a-z0-9]+", tok):
        return "noise"
    return "other"


def _kept_tokens(docs, setting):
    kind, value = setting["kind"], setting.get("value")
    if kind == "baseline":
        return [d.split() for d in docs]
    if kind in ("max_df", "min_df"):
        v = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, **{kind: value}).fit(docs)
        vocab = set(v.vocabulary_)
        return [[t for t in d.split() if t in vocab] for d in docs]
    if kind == "score":
        v = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
        X = v.fit_transform(docs)
        inv = {i: t for t, i in v.vocabulary_.items()}
        out = []
        for r, d in enumerate(docs):
            row = X.getrow(r)
            keep = {inv[i] for i, w in zip(row.indices, row.data) if w >= value}
            out.append([t for t in d.split() if t in keep])
        return out
    raise ValueError(kind)


def study(cfg, memos, directions, out, manual=None):
    s = cfg["tfidf_study"]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(memos), size=min(s["n_memos"], len(memos)), replace=False)
    raw = [memos[i] for i in idx]
    dirs = [directions[i] for i in idx]
    docs = [text_lower(m) for m in raw]
    cues = []
    for m, d in zip(raw, dirs):
        cue = pretag(m, d)[3]
        cues.append(set(text_lower(cue).split()) if cue else set())
    tags = [{t: tag_token(t, c) for t in doc.split()} for doc, c in zip(docs, cues)]
    if manual is not None:
        for _, r in manual.iterrows():
            for i, m in enumerate(raw):
                if m == r.memo_raw:
                    tags[i][str(r.token).lower()] = r.kind
    base_counts = pd.Series([k for t in tags for k in t.values()]).value_counts()

    rows, examples = [], {}
    for setting in [dict(name="baseline (case + spacing)", kind="baseline")] + s["settings"]:
        kept = _kept_tokens(docs, setting)
        lost = {"useful": 0, "noise": 0, "other": 0}
        per_memo = []
        for doc, k, tg in zip(docs, kept, tags):
            ks = set(k)
            removed = [t for t in doc.split() if t not in ks]
            for t in removed:
                lost[tg[t]] += 1
            per_memo.append((sum(tg[t] == "noise" for t in removed), sum(tg[t] == "useful" for t in removed)))
        rows.append(dict(setting=setting["name"], tokens_before=sum(len(d.split()) for d in docs),
                         tokens_kept=sum(len(k) for k in kept),
                         noise_removed_pct=round(100 * lost["noise"] / max(base_counts.get("noise", 0), 1), 1),
                         useful_cues_lost_pct=round(100 * lost["useful"] / max(base_counts.get("useful", 0), 1), 1),
                         empty_outputs=int(sum(len(k) == 0 for k in kept))))
        if setting["kind"] != "baseline":
            helpful = max(range(len(docs)), key=lambda i: (per_memo[i][0] - 10 * per_memo[i][1]))
            harmful = max(range(len(docs)), key=lambda i: per_memo[i][1])
            examples[setting["name"]] = {
                "helpful": dict(before=docs[helpful], after=" ".join(kept[helpful])),
                "harmful": dict(before=docs[harmful], after=" ".join(kept[harmful]))}
    res = pd.DataFrame(rows)
    res.to_csv(out / "tfidf_cleaning_study.csv", index=False)
    json.dump(dict(n_memos=len(docs), token_tags=base_counts.to_dict(), results=rows, examples=examples,
                   note="examples contain memo text: keep this file on DataHub"),
              open(out / "tfidf_cleaning_examples.json", "w"), indent=2)
    print(res.to_string(index=False))
    return dict(n_memos=len(docs), token_tags=base_counts.to_dict(), results=rows)

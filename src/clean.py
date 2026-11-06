"""Memo cleaning. memo_raw is never changed; derived columns are added next to it.

text_lower : lowercase + collapsed spaces
text_norm  : also masks dates, card masks, URLs, names and long IDs (<date>, <mask>, <num>, <url>, <name>)
pattern    : first three meaningful words of text_norm after bank boilerplate (used for Pareto ranking)
"""
import re

import pandas as pd


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
    r"visa debit purchase|debit card purchase|debit purchase|pos purchase|pos db|pos debit|"
    r"checkcard|purchase|withdrawal pos|withdrawal|mobile purchase|business to business ach|"
    r"pos adjustment|point of sale)\s*)+")
_DROP = {"card", "on", "the", "to", "from", "for", "of", "payment", "<date>", "<num>", "<mask>", "<url>",
         "<name>", "ach", "web", "ppd", "id", "co"}


def memo_pattern(norm):
    t = _BOILER.sub("", norm)
    toks = [w for w in t.split() if w not in _DROP and not w.isdigit()][:3]
    return " ".join(toks) if toks else "(empty)"


_DIR = {"1": "credit", "2": "debit", "c": "credit", "d": "debit", "cr": "credit", "dr": "debit",
        "credit": "credit", "debit": "debit"}


def to_direction(v):
    """credit_or_debit: 1 = money in (credit), 2 = money out (debit)."""
    return _DIR.get(re.sub(r"\.0$", "", str(v).strip().lower()), "unknown")


def add_text_cols(df):
    df = df.copy()
    df["memo_raw"] = df["memo_raw"].fillna("").astype(str)
    df["text_lower"] = df["memo_raw"].map(text_lower)
    df["text_norm"] = df["memo_raw"].map(text_norm)
    df["pattern"] = df["text_norm"].map(memo_pattern)
    return df


def std_frame(df, dir_col="credit_or_debit"):
    """Raw parquet rows -> account_key, date, amount, direction, memo_raw."""
    df = df.copy()
    df["memo_raw"] = df.pop("memo").map(
        lambda b: b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else ("" if b is None else str(b)))
    df["account_key"] = df["account_key"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["direction"] = df[dir_col].map(to_direction) if dir_col in df else "unknown"
    return df[["account_key", "date", "amount", "direction", "memo_raw"]]


def txn_key(df):
    """account|date|cents|memo, used to stop the same transaction landing in two splits."""
    return (df["account_key"].astype(str) + "|" + pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").fillna("")
            + "|" + (df["amount"].abs() * 100).round().astype("Int64").astype(str) + "|" + df["memo_raw"])

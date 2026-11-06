"""Measure the effort to scale: fit time, model load, inference rows/second, and human review time.

Inference is timed on the 1M random memos when the parquet files are reachable; otherwise the labeled memos are
repeated to `runtime.n_rows` (fine for a speed estimate, says nothing about accuracy).
"""
import os
import platform
import time

import pandas as pd

from .clean import text_lower, text_norm
from .model import load


def _ram_gb():
    try:
        kb = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1])
        return round(kb / 1e6, 1)
    except Exception:
        return None


def measure(cfg, model_path, text_col, memos):
    rt = cfg["runtime"]
    hw = dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
              cpu_count=os.cpu_count(), ram_gb=_ram_gb(), python=platform.python_version())
    t0 = time.perf_counter()
    m = load(model_path)
    load_s = time.perf_counter() - t0
    clean = text_norm if text_col == "text_norm" else text_lower
    t0 = time.perf_counter()
    texts = [clean(x) or "empty_memo" for x in memos]
    clean_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(0, len(texts), rt["batch"]):
        m.predict(texts[i:i + rt["batch"]], k=1)
    infer_s = time.perf_counter() - t0
    n = len(texts)
    rps = n / infer_s
    return dict(hardware=hw, rows=n, batch=rt["batch"], load_s=round(load_s, 3), clean_s=round(clean_s, 2),
                infer_s=round(infer_s, 2), rows_per_s=round(rps),
                seconds_per_1m=round(load_s + (clean_s + infer_s) * 1e6 / n, 1),
                hours_full_dataset=round((clean_s + infer_s) * rt["full_rows"] / n / 3600, 2),
                full_rows=rt["full_rows"])


def human_effort(review_file):
    """Seconds per memo from the timed review loop (pauses over 5 minutes capped)."""
    if review_file is None or not review_file.exists():
        return None
    rv = pd.read_csv(review_file, dtype=str).fillna("")
    ind = rv[rv.review_status.isin(["accepted", "corrected", "unresolved"])]
    secs = pd.to_numeric(ind.review_seconds, errors="coerce").dropna()
    if secs.empty:
        return None
    minutes = secs.clip(upper=300).sum() / 60
    return dict(memos_reviewed=len(ind), minutes=round(minutes, 1), median_seconds=float(secs.median()),
                minutes_per_1k=round(minutes / len(ind) * 1000, 1),
                accepted_via_audit=int((rv.review_status == "ai_accepted_via_audit").sum()),
                accept_rate=float((ind.review_status == "accepted").mean()),
                correct_rate=float((ind.review_status == "corrected").mean()),
                unresolved_rate=float((ind.review_status == "unresolved").mean()))

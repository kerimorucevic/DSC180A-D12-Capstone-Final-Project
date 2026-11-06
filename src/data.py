"""Reading the Prism parquet files (DataHub only; transaction data never leaves the server).

Two readers:
- scan_accounts: every transaction for a set of account keys, reading only account_key per row group first
  and caching one output file per input file, so a crash loses at most one file.
- general_sample: a random sample of memos from all files (Pareto check, 1M-memo timing, Project A scaffold).
"""
import gc
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .clean import std_frame

DIR_CANDIDATES = ["credit_or_debit", "direction", "debit_credit", "cr_dr", "transaction_type"]


def parquet_parts(cfg):
    parts = sorted(glob.glob(str(Path(cfg["paths"]["data_dir"]) / cfg["data"]["parquet_glob"])))
    if not parts:
        raise FileNotFoundError(f"no parquet files under {cfg['paths']['data_dir']}")
    schema = pq.ParquetFile(parts[0]).schema_arrow
    dir_col = next((c for c in DIR_CANDIDATES if c in schema.names), None)
    cols = [c for c in ["account_key", "posted_date", "amount", dir_col, "memo"] if c]
    return parts, cols, dir_col


def _value_set(values, ftype):
    if pa.types.is_integer(ftype):
        return pa.array([int(v) for v in values], type=pa.int64()).cast(ftype)
    if pa.types.is_binary(ftype) or pa.types.is_large_binary(ftype):
        return pa.array([str(v).encode() for v in values], type=pa.binary()).cast(ftype)
    return pa.array([str(v) for v in values]).cast(ftype)


def scan_accounts(cfg, accounts):
    parts, cols, dir_col = parquet_parts(cfg)
    cache = cfg["paths"]["out_dir"] / "account_scan"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    for path in parts:
        out = cache / (Path(path).stem + ".parquet")
        if out.exists():
            continue
        pf = pq.ParquetFile(path)
        vs = _value_set(accounts, pf.schema_arrow.field("account_key").type)
        kept = []
        for g in range(pf.num_row_groups):
            mask = pc.is_in(pf.read_row_group(g, columns=["account_key"]).column(0), value_set=vs)
            if pc.any(mask).as_py():
                kept.append(pf.read_row_group(g, columns=cols).filter(mask))
        tbl = pa.concat_tables(kept) if kept else pf.schema_arrow.empty_table().select(cols)
        pq.write_table(tbl, out)
        del kept, tbl
        gc.collect()
        print(f"{Path(path).name}: {pq.ParquetFile(out).metadata.num_rows} rows")
    print(f"account scan: {time.time() - t0:.0f}s (0 if cached)")
    raw = pd.concat([pq.read_table(p).to_pandas() for p in sorted(cache.glob("*.parquet"))], ignore_index=True)
    return std_frame(raw, dir_col)


def general_sample(cfg, n=None, seed=None):
    n = n or cfg["sampling"]["n_general"]
    seed = cfg["seed"] if seed is None else seed
    path = cfg["paths"]["out_dir"] / f"general_sample_{n}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts, cols, dir_col = parquet_parts(cfg)
    rng = np.random.default_rng(seed)
    groups = [(p, g) for p in parts for g in range(pq.ParquetFile(p).num_row_groups)]
    chunks, total = [], 0
    for k in rng.permutation(len(groups)):
        p, g = groups[k]
        t = pq.ParquetFile(p).read_row_group(g, columns=cols)
        take = np.sort(rng.choice(t.num_rows, size=max(1, t.num_rows // 4), replace=False))
        chunks.append(t.take(pa.array(take)))
        total += len(take)
        if total >= n:
            break
    df = std_frame(pa.concat_tables(chunks).to_pandas(), dir_col).head(n).reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df

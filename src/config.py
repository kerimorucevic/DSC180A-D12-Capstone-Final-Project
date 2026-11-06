"""Load config.yaml and resolve paths relative to the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="config.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        if value is None:
            continue
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    cfg["paths"]["out_dir"].mkdir(parents=True, exist_ok=True)
    return cfg

"""The preserved baseline reproduces pa7_subscriptions.ipynb exactly (rules and feature)."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subrec import rules_baseline
from subrec.data import lower_text

NAME = "pa7_subscriptions.ipynb"


def find_pa7():
    """Locate the reference notebook from wherever this package sits in the repository.

    Order: $SUBREC_PA7_NOTEBOOK; the notebook in any ancestor folder (e.g. repo root); then a search of the repository
    (the nearest ancestor with a .git folder, else three levels above tests/) that ignores this package.
    """
    env = os.environ.get("SUBREC_PA7_NOTEBOOK")
    if env:
        return Path(env) if Path(env).is_file() else None
    here = Path(__file__).resolve()
    ancestors = list(here.parents)[:7]
    for a in ancestors:
        if (a / NAME).is_file():
            return a / NAME
    root = next((a for a in ancestors if (a / ".git").exists()), here.parents[3] if len(here.parents) > 3 else None)
    if root is not None:
        pkg = here.parents[1]
        for hit in sorted(root.rglob(NAME)):
            if pkg not in hit.parents and ".git" not in hit.parts:
                return hit
    return None


PA7 = find_pa7()
pytestmark = pytest.mark.skipif(PA7 is None, reason=f"{NAME} not found; set SUBREC_PA7_NOTEBOOK=/path/to/{NAME}")


def _cells():
    return ["".join(c["source"]) for c in json.load(open(PA7))["cells"]]


def _exec_pa7(hist):
    """Run pa7 cells 3 (rules), 7 (labelling), 11 (recurrence), 13 (grid) on a prepared history."""
    cells = _cells()
    min_months = int(re.search(r"^MIN_MONTHS\s*=\s*(\d+)", cells[1], re.M).group(1))   # pa7 config cell
    g = {"re": re, "np": np, "pd": pd, "display": lambda *a, **k: None, "print": lambda *a, **k: None,
         "MIN_MONTHS": min_months, "hist": hist.drop(columns=["text"]).copy()}
    for i in (3, 7, 11, 13):
        exec(cells[i], g)
    return g


def test_rules_copy_is_verbatim():
    assert _cells()[3] in Path(rules_baseline.__file__).read_text()


def test_classify_matches_pa7_on_all_synthetic_memos(raw):
    g = {"re": re}
    exec(_cells()[3], g)
    for memo in raw["memo"].unique():
        t = lower_text(memo)
        assert rules_baseline.classify(t) == g["classify"](t)


def test_baseline_M0_reproduces_pa7_feature(hist, analysis):
    g = _exec_pa7(hist)
    ref = g["am"][["account_key", "month", "status", "recurring_sub_spend"]]
    ours = analysis.runs[("original", "baseline")].am[["account_key", "month", "status", "rec_M0"]]
    both = ref.merge(ours, on=["account_key", "month"], how="outer", indicator=True, suffixes=("_pa7", ""))
    assert (both["_merge"] == "both").all()
    assert (both["status_pa7"] == both["status"]).all()
    assert np.allclose(both["recurring_sub_spend"], both["rec_M0"])
    assert both["rec_M0"].sum() > 0

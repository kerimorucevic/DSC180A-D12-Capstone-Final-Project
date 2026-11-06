"""fastText supervised classifier: write training files, fit, predict, score."""
import gc
import math
import time

import fasttext
import numpy as np
from sklearn.metrics import f1_score

from .taxonomy import LABEL_TO_CATEGORY, to_label

fasttext.FastText.eprint = lambda *a, **k: None   # silence the load_model warning


def write_ft(df, path, text_col):
    with open(path, "w") as f:
        for c, t in zip(df["category"], df[text_col]):
            f.write(f"{to_label(c)} {t or 'empty_memo'}\n")


def fit(train_df, text_col, work_dir, base, retries=2, **params):
    """Returns (model or None, seconds, error).

    fastText 0.9.x sometimes raises 'Encountered NaN' depending on memory state inside the process, not only on
    the setting, so a failed fit is retried before it is recorded as an error. Successful fits are reproducible.
    """
    p = {**base, **params}
    path = work_dir / f"_train_{text_col}.txt"
    write_ft(train_df, path, text_col)
    err = ""
    for _ in range(retries + 1):
        t0 = time.perf_counter()
        try:
            m = fasttext.train_supervised(input=str(path), verbose=0, **p)
            return m, time.perf_counter() - t0, ""
        except RuntimeError as e:
            err = str(e)
            gc.collect()
    return None, time.perf_counter() - t0, err


def predict(model, texts):
    """Low-level predict: avoids fastText 0.9's np.array(copy=False) error under numpy >= 2."""
    out = [model.f.predict((t or "empty_memo") + "\n", 1, 0.0, "strict") for t in texts]
    labels = [LABEL_TO_CATEGORY.get(x[0][1], "?") if x else "?" for x in out]
    probs = np.clip([x[0][0] if x else 0.0 for x in out], 0, 1)
    return labels, probs


def load(path):
    return fasttext.load_model(str(path))


def scores(y, p):
    """Accuracy and macro-F1 over the classes present in y (rare classes count as much as common ones)."""
    y, p = list(y), list(p)
    if not y:
        return dict(n=0, accuracy=np.nan, macro_f1=np.nan)
    return dict(n=len(y), accuracy=float(np.mean(np.array(y) == np.array(p))),
                macro_f1=float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)))


def model_mb(m):
    a = m.f.getArgs()
    rows = len(m.words) + (a.bucket if (a.wordNgrams > 1 or a.maxn > 0) else 0)
    return round((rows + len(m.labels)) * a.dim * 4 / 1e6, 1)


def wilson(k, n, z=1.96):
    """95% interval on an accuracy of k/n (small validation sets swing a lot)."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

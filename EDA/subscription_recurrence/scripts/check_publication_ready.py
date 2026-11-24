"""Pre-commit hygiene check for EDA/subscription_recurrence/.

Run from this folder:  python scripts/check_publication_ready.py
Exit status 1 if any check fails. It checks files on disk only; it does not use Git or the network.

Checks
  1. every notebook has no outputs and no execution counts
  2. no data / cache / output files or private folders are present
  3. no text file contains an absolute home path, an e-mail address, or the restricted account key
     (the key literal is allowed only in subrec/config.py)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subrec.config import FAKE_ACCOUNT  # noqa: E402

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "venv", "env"}
FORBIDDEN_SUFFIX = {".csv", ".tsv", ".parquet", ".feather", ".pkl", ".pickle", ".db", ".sqlite", ".h5", ".xlsx",
                    ".bin", ".zip", ".png"}
FORBIDDEN_DIRS = ("outputs_", "private")
PATTERNS = {
    "absolute home path": re.compile(r"(?:/Users|/home)/[A-Za-z0-9_.\-]+"),
    "e-mail address": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "restricted account key": re.compile(rf"\b{FAKE_ACCOUNT}\b"),
}
SELF = Path(__file__).resolve()


def files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
            yield p


def main():
    problems = []
    for p in files():
        rel = p.relative_to(ROOT)
        if p.suffix.lower() in FORBIDDEN_SUFFIX:
            problems.append(f"forbidden file type: {rel}")
        if any(part.startswith(FORBIDDEN_DIRS[0]) or part == FORBIDDEN_DIRS[1] for part in rel.parts[:-1]):
            problems.append(f"output / private folder content: {rel}")
        if p.suffix == ".ipynb":
            nb = json.loads(p.read_text())
            n_out = sum(1 for c in nb["cells"] if c.get("outputs"))
            n_exec = sum(1 for c in nb["cells"] if c.get("execution_count"))
            if n_out or n_exec:
                problems.append(f"notebook not clear: {rel} ({n_out} cells with outputs, {n_exec} execution counts)")
        if p.suffix in {".py", ".md", ".ipynb", ".txt", ".gitignore", ""} and p.resolve() != SELF:
            text = p.read_text(errors="replace")
            for name, rx in PATTERNS.items():
                if name == "restricted account key" and rel.as_posix() == "subrec/config.py":
                    continue
                m = rx.search(text)
                if m:
                    problems.append(f"{name} in {rel}: {m.group(0)[:40]}")
    if problems:
        print("NOT READY:\n  " + "\n  ".join(problems))
        return 1
    print(f"OK: {sum(1 for _ in files())} files checked; notebooks clear; no data, outputs, paths, e-mails or account key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

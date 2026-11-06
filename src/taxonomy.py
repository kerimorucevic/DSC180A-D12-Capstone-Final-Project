"""The group's 20-category guide (see README "Category guide" and "Labeling rulebook")."""
import re

CATEGORIES = [
    "Transfers", "Food & Drink", "Shopping", "Groceries", "Subscriptions & Entertainment",
    "Miscellaneous", "Gas", "Income / Deposits", "Cash Out (ATM)", "Fees & Adjustments",
    "Professional Services", "Debt / Credit Payments", "Finance & Investments", "Automotive",
    "Travel", "Bills & Utilities", "Health & Wellness", "Gifts & Donations", "Education", "Uncertain",
]
CODE = {i + 1: c for i, c in enumerate(CATEGORIES)}   # numbers used by the review loop


def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_BY_KEY = {_key(c): c for c in CATEGORIES}
_BY_KEY.update({_key(a): c for a, c in [
    ("food and drink", "Food & Drink"), ("atm", "Cash Out (ATM)"), ("cash out", "Cash Out (ATM)"),
    ("income", "Income / Deposits"), ("income/deposits", "Income / Deposits"), ("fees", "Fees & Adjustments"),
    ("debt", "Debt / Credit Payments"), ("debit / credit payments", "Debt / Credit Payments"),
    ("subscriptions", "Subscriptions & Entertainment"), ("entertainment", "Subscriptions & Entertainment"),
    ("bills", "Bills & Utilities"), ("utilities", "Bills & Utilities"), ("health", "Health & Wellness"),
    ("misc", "Miscellaneous"),
    # Aiden's PA5 notebook split Uncertain in two; both map back to the guide's single class
    ("uncertain - ambiguous merchant", "Uncertain"), ("uncertain - unidentifiable", "Uncertain"),
]})


def parse_category(s):
    """'3', 'shopping', 'Food and Drink' -> canonical category name, or None."""
    s = str(s).strip()
    if s.isdigit():
        return CODE.get(int(s))
    return _BY_KEY.get(_key(s))


def to_label(category):
    """Category -> fastText label token, e.g. 'Food & Drink' -> '__label__food_drink'."""
    return "__label__" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


LABEL_TO_CATEGORY = {to_label(c): c for c in CATEGORIES}


def code_table():
    half = (len(CATEGORIES) + 1) // 2
    for i in range(half):
        a = f"{i + 1:>2} {CATEGORIES[i]}"
        b = f"{i + 1 + half:>2} {CATEGORIES[i + half]}" if i + half < len(CATEGORIES) else ""
        print(f"  {a:<36}{b}")

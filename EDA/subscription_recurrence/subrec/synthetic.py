"""Synthetic transaction histories for local testing. Contains no real data.

Same columns as the cached UCSD history (account_key, posted_date, amount, credit_or_debit, memo)
plus ground-truth columns that only exist here:
    scenario        which scripted case produced the row
    truth           what the row really is (we constructed it)
    injected_copy   True for rows added as exact copies of another row
"""
import numpy as np
import pandas as pd

from .config import FAKE_ACCOUNT

START, END = "2023-01", "2025-12"
MONTHS = pd.period_range(START, END, freq="M")

BACKGROUND = [("TRADER JOE'S #552 SAN DIEGO CA", 20, 120), ("SHELL OIL 57442 SAN DIEGO CA", 25, 70),
              ("CHIPOTLE 1932 LA JOLLA CA", 9, 18), ("TARGET T-2771 SAN DIEGO CA", 10, 90),
              ("SDG&E WEB PAYMENT", 40, 140), ("STARBUCKS STORE 08821", 4, 9),
              ("RALPHS #0034 SAN DIEGO CA", 15, 110), ("CVS/PHARMACY #09921", 5, 40)]

MONTHLY_CATALOG = [("NETFLIX.COM 866-579-7172 CA", 15.49), ("SPOTIFY USA 877-778-1161 NY", 11.99),
                   ("HULU 877-8244858 CA", 17.99), ("DISNEY PLUS 888-905-7888 CA", 13.99),
                   ("PLANET FITNESS CLUB FEES", 24.99), ("ADOBE *CREATIVE CLD 408-536-6000", 54.99),
                   ("YOUTUBE PREMIUM 650-253-0000", 13.99), ("PEACOCK 9C3A7 PREMIUM", 7.99),
                   ("CRUNCH FITNESS #221", 29.99), ("APPLE.COM/BILL 866-712-7753 CA", 2.99),
                   ("SIRIUSXM RADIO 888-635-5144", 10.99), ("OPENAI *CHATGPT SUBSCR", 20.00)]
PLATFORM_ONE_OFF = ["APPLE.COM/BILL 866-712-7753 CA", "GOOGLE *GOOGLE PLAY APP", "MICROSOFT*STORE PURCHASE",
                    "XBOX DIGITAL PURCHASE", "AMZN DIGITAL*2K81 WA"]
EVENTS = ["TICKETMASTER 800-653-8000", "AMC THEATRES 2231", "STUBHUB INC", "ROUND1 BOWLING SAN DIEGO"]
GAMBLING = ["DRAFTKINGS SPORTSBOOK", "FANDUEL SPORTSBOOK", "PRIZEPICKS"]


class _Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.rows = []

    def add(self, acct, period, day, amount, memo, cod=2, truth="background", scenario="", copies=0):
        last_day = period.days_in_month
        date = pd.Timestamp(year=period.year, month=period.month, day=int(min(day, last_day)))
        row = dict(account_key=acct, posted_date=date, amount=round(float(amount), 2), credit_or_debit=cod,
                   memo=memo, scenario=scenario, truth=truth, injected_copy=False)
        self.rows.append(row)
        for _ in range(copies):
            self.rows.append({**row, "injected_copy": True})

    def background(self, acct, months, scenario, gaps=(), n_range=(6, 12)):
        for i, p in enumerate(months):
            if p in gaps:
                continue
            self.add(acct, p, 1 if i else 20, 2400, "PAYROLL ACME CORP PPD", cod=1, truth="income", scenario=scenario)
            for _ in range(self.rng.integers(*n_range)):
                memo, lo, hi = BACKGROUND[self.rng.integers(len(BACKGROUND))]
                self.add(acct, p, self.rng.integers(21, 28) if i == 0 else self.rng.integers(1, 28),
                         self.rng.uniform(lo, hi), memo, scenario=scenario)

    def monthly(self, acct, months, memo, amount, scenario, day=7, truth="subscription_monthly", copies=None):
        for i, p in enumerate(months):
            amt = amount[i] if np.ndim(amount) else amount
            c = 0 if copies is None else copies.get(i, 0)
            self.add(acct, p, day if i else 25, amt, memo, truth=truth, scenario=scenario, copies=c)

    def frame(self):
        df = pd.DataFrame(self.rows)
        df["account_key"] = df["account_key"].astype("int64")
        return df.sort_values(["account_key", "posted_date"], kind="stable").reset_index(drop=True)


def make_scripted(b: _Builder):
    M = MONTHS
    sc = "1001_monthly"; b.background(1001, M, sc); b.monthly(1001, M, "NETFLIX.COM 866-579-7172 CA", 15.49, sc)

    sc = "1002_irregular"; b.background(1002, M, sc)
    for i, amt in zip([2, 5, 6, 11, 19, 27], [4.99, 29.99, 0.99, 49.99, 9.99, 1.99]):
        b.add(1002, M[i], 12, amt, "APPLE.COM/BILL 866-712-7753 CA", truth="one_off_purchase", scenario=sc)
    for i, amt in zip([3, 9, 10, 15], [19.99, 4.99, 2.99, 99.99]):
        b.add(1002, M[i], 18, amt, "GOOGLE *GOOGLE PLAY APP", truth="one_off_purchase", scenario=sc)

    sc = "1003_annual"; b.background(1003, M, sc)
    for i in [1, 13, 25]:
        b.add(1003, M[i], 3, 139.00, "AMAZON PRIME*2K4LM8", truth="subscription_annual", scenario=sc)

    sc = "1004_price_change"; b.background(1004, M, sc)
    b.monthly(1004, M, "SPOTIFY USA 877-778-1161 NY", [11.99 if i < 18 else 13.99 for i in range(len(M))], sc)

    sc = "1005_paused"; b.background(1005, M, sc)
    active = [p for i, p in enumerate(M) if not 11 <= i <= 15]
    b.monthly(1005, active, "PLANET FITNESS CLUB FEES", 24.99, sc)

    sc = "1006_missing_month"; gap = {M[12]}; b.background(1006, M, sc, gaps=gap)
    b.monthly(1006, [p for p in M if p not in gap], "HULU 877-8244858 CA", 17.99, sc)

    sc = "1007_repeated_records"; b.background(1007, M, sc)
    b.monthly(1007, M, "DISNEY PLUS 888-905-7888 CA", 13.99, sc, copies={8: 2, 20: 1})

    sc = "1008_gambling_events"; b.background(1008, M, sc)
    b.monthly(1008, M, "DRAFTKINGS SPORTSBOOK", 50.00, sc, truth="gambling")
    for i in [3, 4, 5]:
        b.add(1008, M[i], 9, 89.50, "TICKETMASTER 800-653-8000", truth="event", scenario=sc)
    for i in [6, 7, 8]:
        b.add(1008, M[i], 9, 14.00, "AMC THEATRES 2231", truth="event", scenario=sc)
    for i in [10, 11, 12]:
        b.add(1008, M[i], 9, 32.00, "ROUND1 BOWLING SAN DIEGO", truth="event", scenario=sc)

    sc = "1009_apple_mixed"; b.background(1009, M, sc)
    b.monthly(1009, M, "APPLE.COM/BILL 866-712-7753 CA", 2.99, sc, truth="platform_subscription")
    for i, amt in zip([4, 14, 22, 30], [9.99, 24.99, 4.99, 14.99]):
        b.add(1009, M[i], 19, amt, "APPLE.COM/BILL 866-712-7753 CA", truth="one_off_purchase", scenario=sc)

    sc = "1010_whole_account_copies"; before = len(b.rows); b.background(1010, M, sc)
    b.monthly(1010, M, "SPOTIFY USA 877-778-1161 NY", 11.99, sc)
    for r in b.rows[before:]:
        if not r["injected_copy"]:
            b.rows.append({**r, "injected_copy": True})

    sc = "1011_new_subscription"; b.background(1011, M, sc)
    b.monthly(1011, M[33:], "OPENAI *CHATGPT SUBSCR", 20.00, sc)

    sc = "1012_false_positives"; b.background(1012, M, sc)
    b.monthly(1012, M, "GOOGLE PAY TRANSFER 4821", 100.00, sc, truth="transfer")
    for i in [4, 5, 6]:
        b.add(1012, M[i], 14, 154.00, "DISNEYLAND RESORT TICKETS", truth="event", scenario=sc)

    sc = "1013_missing_keywords"; b.background(1013, M, sc)
    b.monthly(1013, M, "ADOBE *CREATIVE CLD 408-536-6000", 54.99, sc, day=5)
    b.monthly(1013, M, "CRUNCH FITNESS #221", 29.99, sc, day=6)
    b.monthly(1013, M, "YOUTUBE PREMIUM 650-253-0000", 13.99, sc, day=8)

    sc = "1014_fee_memos"; b.background(1014, M, sc)
    b.monthly(1014, M, "FOREIGN TRANSACTION FEE NETFLIX", 0.46, sc, truth="bank_fee")
    b.monthly(1014, M, "PARAMOUNT+ MONTHLY FEE", 11.99, sc, day=9)

    sc = "1015_short_account"; b.background(1015, M[30:], sc)
    b.monthly(1015, M[30:], "NETFLIX.COM 866-579-7172 CA", 15.49, sc)

    sc = "fake_account"; b.background(FAKE_ACCOUNT, M, sc)
    b.monthly(FAKE_ACCOUNT, M, "NETFLIX.COM 866-579-7172 CA", 15.49, sc)


def make_random(b: _Builder, n_accounts, first_key=2001):
    rng = b.rng
    for k in range(n_accounts):
        acct, sc = first_key + k, "random"
        s = int(rng.integers(0, 12)); e = int(rng.integers(max(s + 6, 24), len(MONTHS)))
        months = MONTHS[s:e + 1]
        gaps = {p for p in months[1:-1] if rng.random() < 0.03}
        dup_rate = float(rng.choice([0.0, 0.0, 0.05, 0.3, 0.9]))
        start_n = len(b.rows)
        b.background(acct, months, sc, gaps=gaps)
        live = [p for p in months if p not in gaps]
        for j in rng.choice(len(MONTHLY_CATALOG), size=int(rng.integers(0, 5)), replace=False):
            memo, amt = MONTHLY_CATALOG[j]
            a = int(rng.integers(0, len(live) - 3)); z = int(rng.integers(a + 3, len(live) + 1))
            sub = live[a:z]
            if rng.random() < 0.15 and len(sub) > 8:                     # pause
                p0 = int(rng.integers(2, len(sub) - 5)); sub = sub[:p0] + sub[p0 + int(rng.integers(2, 5)):]
            amts = [amt * (1.15 if (rng.random() < 0.15 and i >= len(sub) // 2) else 1.0) for i in range(len(sub))]
            truth = "platform_subscription" if memo.startswith("APPLE") else "subscription_monthly"
            b.monthly(acct, sub, memo, amts, sc, day=int(rng.integers(1, 28)), truth=truth)
        p_plat = float(rng.choice([0.0, 0.1, 0.3]))
        for p in live:
            if rng.random() < p_plat:
                b.add(acct, p, int(rng.integers(1, 28)), rng.choice([0.99, 2.99, 4.99, 9.99, 19.99]),
                      PLATFORM_ONE_OFF[rng.integers(len(PLATFORM_ONE_OFF))], truth="one_off_purchase", scenario=sc)
            if rng.random() < 0.05:
                b.add(acct, p, int(rng.integers(1, 28)), rng.uniform(15, 150), EVENTS[rng.integers(len(EVENTS))],
                      truth="event", scenario=sc)
        if rng.random() < 0.15:
            g = GAMBLING[rng.integers(len(GAMBLING))]
            for p in live:
                if rng.random() < 0.5:
                    b.add(acct, p, int(rng.integers(1, 28)), rng.choice([10, 25, 50, 100]), g,
                          truth="gambling", scenario=sc)
        if dup_rate:
            for r in list(b.rows[start_n:]):
                if rng.random() < dup_rate:
                    for _ in range(int(rng.integers(1, 4))):
                        b.rows.append({**r, "injected_copy": True})


def make_synthetic_history(seed=0, n_random_accounts=40):
    """Scripted edge cases (accounts 1001-1015, plus the known fake account) + random accounts."""
    b = _Builder(seed)
    make_scripted(b)
    make_random(b, n_random_accounts)
    return b.frame()

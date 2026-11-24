"""Shared constants and the run configuration for the recurrence analysis."""
from dataclasses import dataclass, field

# Account key of the known synthetic (fake) account in the shared transaction table; excluded everywhere.
# This is the only place the literal appears; code, tests and docs refer to it as FAKE_ACCOUNT.
FAKE_ACCOUNT = 207405
CREDIT_CODE, DEBIT_CODE = 1, 2  # credit_or_debit codes, verified in income_feature.ipynb section 3
KEY_COLS = ["account_key", "posted_date", "amount", "credit_or_debit", "memo"]  # exact-repeat key (not a transaction id)

METHODS = ["M0", "M1", "M2", "M3_10", "M3_15", "M3_20"]
PROSPECTIVE_METHODS = ["M1", "M2", "M3_10", "M3_15", "M3_20"]   # must not use future information
M3_THRESHOLDS = {"M3_10": 0.10, "M3_15": 0.15, "M3_20": 0.20}

STATUS_CONFIRMED = "confirmed"          # meets the method's recurrence rule at month t
STATUS_POTENTIAL = "potential"          # subscription-merchant charge, too little observed history to decide
STATUS_NOT_CONFIRMED = "not_confirmed"  # enough history observed, rule not met

TIER_HIGH = "high_confidence"
TIER_AMBIGUOUS = "ambiguous_platform"


@dataclass
class RecurrenceConfig:
    """Parameters of the recurrence definitions.

    min_months       M0 / M1: distinct charged months required.
    window           M2 / M3: consecutive months (ending at t) that must each have a charge.
    m3_thresholds    M3: max |monthly total - window median| / window median, per variant.
    gap_policy       'break'  -> the window is consecutive *calendar* months; a no_data_gap month
                                 inside it makes the window unobservable (literal reading of the spec).
                     'bridge' -> the window is the last `window` non-gap months of the account, so a
                                 month with no data at all is skipped rather than treated as a break.
    history_requires_covered  M1 / M2 / M3 history months must have status 'covered': partial edge months
                    are not recurrence history (research-lead decision). M0 is unaffected.
    annual_tolerance  relative amount tolerance for the (diagnostic-only) annual-candidate flag.
    """
    min_months: int = 3
    window: int = 3
    m3_thresholds: dict = field(default_factory=lambda: dict(M3_THRESHOLDS))
    gap_policy: str = "break"
    history_requires_covered: bool = True
    annual_tolerance: float = 0.20

    def __post_init__(self):
        if self.gap_policy not in ("break", "bridge"):
            raise ValueError("gap_policy must be 'break' or 'bridge'")
        if self.window < 2 or self.min_months < 1:
            raise ValueError("window must be >= 2 and min_months >= 1")

    @property
    def methods(self):
        return ["M0", "M1", "M2", *self.m3_thresholds]

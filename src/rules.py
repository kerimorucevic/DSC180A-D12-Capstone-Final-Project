"""The 20-category guide and the rulebook, written as ordered pre-tag rules.

Categories are modeled on the spending categories used by Chase, Bank of America and American Express.
Many memos fit more than one category, so the rulebook makes labeling decisions consistent:
  - transaction type over merchant: an "ATM fee" is a Fee, not Cash Out (ATM)
  - credit/debit direction: cash-advance apps are Income when money comes in, Debt / Credit Payments when a
    repayment goes out
  - edge cases: buy-now-pay-later is Debt / Credit Payments; gambling and sportsbooks are Subscriptions &
    Entertainment
  - Uncertain only when the memo cannot support any category (truncated merchant name, bare reference number)

Each pre-tag rule is (rule_id, category, regex, confidence, required_direction) and returns the matched text and a
high or low confidence. Rules are checked in order and the first match wins, so transaction rules (fees, income,
ATM withdrawals) come before merchant rules.
"""
import re

CATEGORIES = [
    "Transfers", "Food & Drink", "Shopping", "Groceries", "Subscriptions & Entertainment",
    "Miscellaneous", "Gas", "Income / Deposits", "Cash Out (ATM)", "Fees & Adjustments",
    "Professional Services", "Debt / Credit Payments", "Finance & Investments", "Automotive",
    "Travel", "Bills & Utilities", "Health & Wellness", "Gifts & Donations", "Education", "Uncertain",
]
CODE = {i + 1: c for i, c in enumerate(CATEGORIES)}      # numbers typed in the review loop


def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_BY_KEY = {_key(c): c for c in CATEGORIES}
_BY_KEY.update({_key(a): c for a, c in [
    ("food and drink", "Food & Drink"), ("atm", "Cash Out (ATM)"), ("income", "Income / Deposits"),
    ("fees", "Fees & Adjustments"), ("debt", "Debt / Credit Payments"), ("entertainment", "Subscriptions & Entertainment"),
    ("debit / credit payments", "Debt / Credit Payments"), ("misc", "Miscellaneous"),
    ("uncertain - ambiguous merchant", "Uncertain"), ("uncertain - unidentifiable", "Uncertain"),
]})


def parse_category(s):
    """'3', 'shopping', 'Food and Drink' -> canonical category name, or None."""
    s = str(s).strip()
    return CODE.get(int(s)) if s.isdigit() else _BY_KEY.get(_key(s))


def to_label(category):
    """'Food & Drink' -> '__label__food_drink' (fastText label token)."""
    return "__label__" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


LABEL_TO_CATEGORY = {to_label(c): c for c in CATEGORIES}


PRETAG_RULES = [
    # ---- 1. Bank fees first: "ATM fee" is a fee, not cash out ----------------------------
    ("fee",          "Fees & Adjustments", r"\b(overdraft|nsf|insufficient funds|returned item|service charge|"
                                           r"maintenance fee|monthly fee|late fee|atm fee|intl txn fee|foreign transaction|"
                                           r"interest charge|finance charge|fees?)\b", "high", None),


    # ---- 1b. Gambling / sportsbooks -> Subscriptions & Entertainment (rulebook edge case)
    ("gambling",     "Subscriptions & Entertainment", r"(draftkings|fanduel|fd sptsbk|fanatics sp|sportsbo|prizepicks|underdog|"
                                                      r"lottery|valottery|\bolg\b|betmgm|hard rock bet|crowncoins|chumba|"
                                                      r"wowvegas|fliff|sidepot|casino|nflallday|meetdapper|jbox game|baca game)", "high", None),

    # ---- 2. Income before ATM/transfer ("ATM deposit", "Zelle from employer") ------------
    ("payroll",      "Income / Deposits",  r"\b(payroll|direct dep|dir dep|salary|paycheck|ppd payroll|adp|gusto|paychex)\b", "high", None),
    ("gov_income",   "Income / Deposits",  r"\b(irs treas|tax ref|tax refund|ssa treas|unemployment|edd\b|state of \w+ refund)", "high", None),
    ("interest",     "Income / Deposits",  r"\b(interest (paid|earned|payment)|dividend)\b", "high", None),
    ("deposit",      "Income / Deposits",  r"\b(cash deposit|atm deposit|mobile deposit|check deposit|remote deposit|branch deposit)\b", "high", None),
    ("ewa_credit",   "Income / Deposits",  r"\b(dave|earnin|brigit|cleo|klover|moneylion|floatme|float me|empower|mypay|instacash|grant cash|tilt|beem|vola|credit ?genie)\b", "low", "credit"),
    ("deposit_gen",  "Income / Deposits",  r"\bdeposit\b", "low", "credit"),

    # ---- 3. Cash out ------------------------------------------------------------------------
    ("atm",          "Cash Out (ATM)",     r"\b(atm|cash withdrawal|cash back|cashback|withdrawal at)\b", "high", None),

    # ---- 4. Debt / credit payments (before Education: "student loan"; before Shopping: "amazon card") ----
    ("card_pmt",     "Debt / Credit Payments", r"\b(card payment|credit card|credit crd|crd pmt|crd autopay|crd epay|card autopay|cc pmt|payment thank you|"
                                               r"capital one|discover|amex|american express|credit one|citi card|"
                                               r"chase card|synchrony|barclaycard|amazon (store )?card|applecard|apple card)\b", "high", None),
    ("loans",        "Debt / Credit Payments", r"\b(loan|navient|nelnet|mohela|aidvantage|sallie mae|great lakes|"
                                               r"upstart|lendingclub|avant|oportun|onemain)\b", "high", None),
    ("bnpl",         "Debt / Credit Payments", r"\b(affirm|klarna|afterpay|sezzle|zip pay|quadpay|deferit|progressive leas\w*)\b", "high", None),
    ("ewa_debit",    "Debt / Credit Payments", r"\b(dave|earnin|brigit|cleo|klover|moneylion|floatme|float me|empower|mypay|instacash|grant cash|tilt|beem|vola|credit ?genie)\b", "low", "debit"),

    # ---- 5. Investments ----------------------------------------------------------------------
    ("invest",       "Finance & Investments", r"\b(robinhood|webull|fidelity|vanguard|schwab|etrade|e\*trade|acorns|stash|"
                                              r"betterment|wealthfront|m1 finance|public\.com|coinbase|binance|kraken|"
                                              r"gemini|crypto|brokerage)\b", "high", None),

    # ---- 6. Transfers (P2P + own accounts) ---------------------------------------------------
    ("p2p",          "Transfers",          r"\b(zelle|venmo|cash app|cashapp|square cash|apple cash|e-?transfer|interac)\b", "high", None),
    ("own_xfer",     "Transfers",          r"\b(online transfer|transfer (to|from)|internal transfer|inst xfer|xfer|"
                                           r"round ?up|to savings|from savings|to checking|from checking)\b", "high", None),
    ("paypal",       "Transfers",          r"\bpaypal\b", "low", None),   # PayPal is also used to pay merchants

    # ---- 7. Bills & Utilities (before Gas: "SoCalGas"; before Travel: "Metro by T-Mobile") ----
    ("utilities",    "Bills & Utilities",  r"\b(electric|energy|southwest gas|power co|pg&e|pge|sdg&e|sdge|edison|duke energy|"
                                           r"socalgas|gas co\b|national grid|water (bill|dept|district)|city water|"
                                           r"county water|sewer|waste management|republic services)", "high", None),
    ("telecom",      "Bills & Utilities",  r"(\bcomcast|\bxfinity|\bspectrum|\bverizon|\bat&t|\batt\*|\bt-mobile|\bsprint|"
                                           r"\bcricket wireless|\bmetro by t|\bboost mobile|\bmetro ?pcs|\bmint mobile|\bvisible|\bcox comm|\bfrontier comm)", "high", None),
    ("rent",         "Bills & Utilities",  r"\b(rent(?![- ]?a[- ]?car)|property management|apartments|leasing office|landlord)\b", "low", None),
    ("insurance",    "Bills & Utilities",  r"\b(insurance|lemonade|state farm|allstate|liberty mutual)\b", "low", None),

    # ---- 8. Food & Drink before Travel ("uber eats") and Shopping ----------------------------
    ("delivery",     "Food & Drink",       r"\b(uber\s*\*?\s*eats|doordash|grubhub|postmates|caviar|seamless)\b", "high", None),
    ("restaurants",  "Food & Drink",       r"(\b(starbucks|dunkin|peet'?s?|dutch bros|mcdonald'?s?|chipotle|taco bell|chick-?fil|"
                                           r"wendy'?s?|burger king|in-n-out|in n out|panda express|domino'?s?|pizza|papa john'?s?|kfc|"
                                           r"popeyes|jack in the box|panera|subway|sonic drive|five guys|shake shack|"
                                           r"boba|tea house|cafe|coffee|restaurant|grill|bistro|kitchen|sushi|ramen|"
                                           r"taqueria|brewery|pub|bar|365 market)\b|tst\*|sq \*.*(cafe|coffee|kitchen))", "high", None),

    # ---- 9. Subscriptions & Entertainment (before Shopping: "amazon prime video") ------------
    ("streaming",    "Subscriptions & Entertainment", r"\b(netflix|spotify|hulu|disney ?(\+|plus)|hbo|max\.com|peacock|"
                                                      r"paramount|youtube|apple\.com/bill|itunes|google \*?(play|youtube)|"
                                                      r"prime video|amazon prime|audible|sirius|pandora|tiktok|patreon)", "high", None),
    ("games_events", "Subscriptions & Entertainment", r"\b(xbox|playstation|nintendo|steam|twitch|amc|regal|cinemark|"
                                                      r"fandango|ticketmaster|stubhub|eventbrite|live nation|six flags|"
                                                      r"disneyland|universal studios|seaworld|dave & buster|bowl)", "high", None),

    # ---- 10. Gas (station brand alone does not prove fuel -> low) ------------------------------
    ("gas",          "Gas",                r"\b(shell|chevron|exxon\w*|mobil|arco|valero|circle k|union 76|76 (fuel|gas)|"
                                           r"speedway|marathon|sunoco|wawa|texaco|citgo|bp#?|sinclair|phillips 66|conoco|"
                                           r"costco gas|gas station|fuel)\b", "low", None),

    # ---- 11. Groceries --------------------------------------------------------------------------
    ("grocer",       "Groceries",          r"\b(trader joe|whole foods|wholefds|ralphs|vons|albertsons|safeway|kroger|aldi|"
                                           r"publix|h-?e-?b|food ?4 ?less|sprouts|smart & final|stater bros|winco|wegmans|"
                                           r"meijer|food lion|stop & shop|hy-?vee|piggly|freshco|99 ranch|h mart|"
                                           r"supermarket|grocery|groceries|farmers market)", "high", None),
    ("warehouse",    "Groceries",          r"\b(costco|sam'?s club|wal-?mart super|walmart super|wm supercenter)", "low", None),

    # ---- 12. Health & Wellness (pharmacies also sell general goods -> low) -----------------
    ("medical",      "Health & Wellness",  r"\b(dental|dds|dentist|medical|clinic|hospital|kaiser|urgent care|"
                                           r"optometr|vision|therapy|counseling|labcorp|quest diag|physician|pediatric)", "high", None),
    ("gym",          "Health & Wellness",  r"\b(planet fitness|24 hour fitness|la fitness|equinox|crunch|orangetheory|"
                                           r"gym|fitness|yoga|pilates)\b", "high", None),
    ("pharmacy",     "Health & Wellness",  r"\b(cvs|walgreens|rite aid|pharmacy)\b", "low", None),

    # ---- 13. Education (after Debt so "student loan" is a payment) --------------------------
    ("education",    "Education",          r"\b(tuition|university|college|ucsd|uc san diego|school|bookstore|"
                                           r"chegg|coursera|udemy|pearson|mcgraw|textbook|student fee)", "high", None),

    # ---- 14. Professional Services -------------------------------------------------------------
    ("professional", "Professional Services", r"\b(cpa|accounting|accountant|tax prep|h&r block|hr block|jackson hewitt|"
                                              r"turbotax|law office|attorney|legal|notary|consulting)", "high", None),

    # ---- 15. Gifts & Donations ---------------------------------------------------------------
    ("donations",    "Gifts & Donations",  r"\b(gofundme|donation|donate|charity|red cross|unicef|church|tithe|"
                                           r"salvation army|st jude|nonprofit|foundation)", "high", None),

    # ---- 16. Automotive ------------------------------------------------------------------------
    ("auto",         "Automotive",         r"\b(autozone|o'?reilly|advance auto|napa auto|jiffy lube|valvoline|firestone|"
                                           r"pep boys|discount tire|les schwab|midas|meineke|car ?wash|dmv|"
                                           r"vehicle registration|smog|auto repair|tire|parking|toll|fastrak|e-?zpass|"
                                           r"sunpass|geico|progressive)", "high", None),

    # ---- 17. Travel (after Food & Drink so "uber eats" is already taken) --------------------
    ("rideshare",    "Travel",             r"\b(uber|lyft)\b", "high", None),
    ("travel",       "Travel",             r"\b(delta air|united air|american air|southwest|jetblue|alaska air|spirit air|"
                                           r"frontier air|marriott|hilton|hyatt|holiday inn|airbnb|vrbo|expedia|"
                                           r"booking\.com|hotels\.com|priceline|hertz|avis|enterprise rent|budget rent|"
                                           r"turo|amtrak|greyhound|airline|hotel|transit|metro|mts|bart|mta)", "high", None),

    ("subs_generic", "Subscriptions & Entertainment", r"\b(subscription|membership|premium|recurring|microsoft|msft|rocket money)\b", "low", None),

    # ---- 18. Shopping last among merchants (broad) ----------------------------------------
    ("shopping",     "Shopping",           r"\b(amazon|amzn|target|zara|sephora|ulta|nike|best buy|apple store|ebay|etsy|"
                                           r"shein|temu|h&m|old navy|gap|tj ?maxx|marshalls|ross|kohl|macy|nordstrom|"
                                           r"home depot|lowe'?s|ikea|dollar tree|dollar general|family dollar|five below|"
                                           r"michaels|wal-?mart|walmart)", "high", None),

    # ---- 19. Adjustments last: generic words that also appear on transfers --------------
    ("adjustment",   "Fees & Adjustments", r"\b(adjustment|correction|reversal|provisional credit|chargeback)\b", "low", None),
]


_COMPILED = [(rid, cat, re.compile(p, re.I), conf, d) for rid, cat, p, conf, d in PRETAG_RULES]


def pretag(memo, direction=None):
    """First matching rule -> (category, confidence, rule_id, matched_text); all None if nothing matches."""
    direction = str(direction).lower().strip() if direction is not None else None
    for rid, cat, rx, conf, need_dir in _COMPILED:
        if need_dir and direction != need_dir:
            continue
        m = rx.search(str(memo))
        if m:
            return cat, conf, rid, m.group(0)
    return None, None, None, None

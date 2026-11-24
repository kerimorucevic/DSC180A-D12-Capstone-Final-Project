"""Baseline subscription classifier -- VERBATIM copy of pa7_subscriptions.ipynb, cell 3.

Privacy review (2026-11-24): the keyword tokens are unchanged and some derive from real memo text. A scan found no
personal names, account numbers, phone numbers, e-mail addresses or card masks; merchant-specific and locale-revealing
tokens were flagged for a team decision before publication (nothing was altered).

Do not edit the block between the BEGIN/END markers: it is the reference definition the
alternative classifier is compared against. tests/test_baseline_fidelity.py checks that this
copy still matches the notebook cell. Additions for this project live below the END marker.
"""
import re

# ---- BEGIN VERBATIM pa7_subscriptions.ipynb cell 3 ----
# ---- week 5 category rules (copied from label_rules.py) ----
GAMBLING = (r"draftkings|fanduel|fd sptsbk|fanatics sp|sportsbo|prizepicks|underdog|lottery|valottery"
            r"|\bolg\b|betmgm|hard rock bet|crowncoins|chumba|wowvegas|fliff|sidepot|casino"
            r"|nflallday|meetdapper|powerplay|jbox game|baca game")

# (category, regex, subcategory) in priority order
RULES = [
    # --- specific merchants whose memo also contains a misleading word (fee, bar, paypal ...)
    ("Bills & Utilities", r"agi\*tmo|magicja", None),
    ("Health & Wellness", r"iclub fees|planet fitness|nail bar|hair design", None),
    ("Automotive", r"dmv", None),
    ("Shopping", r"stevenmadde", None),
    ("Uncertain - Ambiguous Merchant", r"get go", None),

    # --- bank fees win over whatever merchant they mention
    ("Fees & Adjustments", r"\bfee\b|fee-|fees\b|overdraft|\bnsf\b|returned item|service charge|over limit"
                           r"|atm rebate|courtesy pay|uncollected protection|return item|reverse check", None),

    # --- gambling (guide has no gambling class -> Subscriptions & Entertainment, tagged)
    ("Subscriptions & Entertainment", GAMBLING, "gambling"),

    # --- cash-advance apps, BNPL, loans, card payments
    ("Transfers", r"dave\.com p2p", None),
    ("Debt / Credit Payments",
     r"\bdave\b|davesubfee|brigit|earnin|moneylion|instacash|cleo|empower|\btilt\b|klover|floatme|float me"
     r"|mypay|grant cash|grant repayment|credit ?genie|oneblinc|beem app|\bvola\b|atlasfinancial"
     r"|advance financia|true fin|flex finance|creditninja|onemain|avant llc|capital one|credit one"
     r"|afterpay|affirm|klarna|quadpay|\bzip\b|sezzle|deferit|progressive leas|progressivelease"
     r"|rent a center|mortgage|\bgrid\b|getgrid|pago de tarjeta|deposit advance", None),

    # --- P2P apps / own-account sweeps (beat the person or business on the other end)
    ("Transfers", r"zelle|\bzel to\b|venmo|cash\s*/?\s*app|apple cash|visa direc|visa money transfer|third prty"
                  r"|e-transfer|moneygram|round ?up|keep the change|save as you go|save your change"
                  r"|albert savings|inst xfer", None),

    # --- early income (memos that also contain the word "transfer"/"deposit")
    ("Income / Deposits", r"payroll|province of b\.c|child benefit|treas 310|\bssi\b|gusto pay|stripe"
                          r"|uber instantpay|uber pro card|gopuff|ccmsi|universal city d batch"
                          r"|interest earned|dividend|fidelity investm", None),

    # --- merchant categories
    ("Subscriptions & Entertainment",
     r"netflix|spotify|hulu|disney|peacock|paramount|prime video|amazon prime|apple\.com/bill|^apple$"
     r"|google|microsoft|audiobooks|audible|roku|philo|tiktok|patreon|chaturbill|of london|abcmouse"
     r"|riot fest|round1|\bbowl\b|stubhub|treehouse indoor|amzn apps|arlo tech|rocket money|creditreview"
     r"|hotmart|candybar", None),
    ("Food & Drink",
     r"mcdonald|burger|chipotle|starbucks|doordash|\bdd \*|uber \*?eats|chick-fil-a|dunkin|popeye"
     r"|church'?s chicken|zaxby|^sonic$|jack in the box|raising cane|waffle house|jersey mike|taco"
     r"|restaura|\bcafe\b|pizza|chicken|\bgrill|\bbar\b|\bpint\b|ice crea|bakery|dining|tst\*|aramark"
     r"|canteen|vending|\bvend\b|payrange|365 market|market@work|seaworld-food|food svc|halal food"
     r"|filibertos|mariscos|pollo campero|tim hortons|cookie|snack shop|cpa clb conc|lounge"
     r"|liquor|liqu\b|\bwine\b|charlie's|nslc", None),
    ("Groceries",
     r"walmart|wal-mart|wm supercenter|safeway|food4less|food lion|kroger|publix|\baldi\b|\blidl\b"
     r"|\bgiant\b|harris te|trader joe|whole foods|ralphs|costco|bjs wholesale|key foods|grocer"
     r"|instacart|maplebear|mariano|westtown foo|carniceria|danny's super f|nesters market|pete's fr", None),
    ("Health & Wellness",
     r"cvs|walgreens|rite aid|pharm|urgent care|dental|la fitness|central florida ym|nail bar"
     r"|hair design|ucsdh|medical|clinic|blue cross", None),
    ("Shopping",
     r"amazon|amzn|etsy|target|aliexpress|tjmaxx|ross store|burlington|winners|jcpenney|five belo"
     r"|dollar general|family dollar|dollar ?tree|clearance outlet|platos closet|the closet|galaxy comics"
     r"|beauty|smokers paradise|vape|exotic shop|lowe's|home depot|stevenmadde|shipmentfree"
     r"|new wave promos|farmersdog|pig & cat|carpe\b", None),
    ("Bills & Utilities",
     r"t-mobile|verizon|at&t|att\*|spectrum|comcast|frontier comm|boost mobile|cable svc|reliant energy"
     r"|georgia power|scana energy|electric|utility|insurance|\bins\b|geico|allstate|state farm"
     r"|natl gen ins|chubb|magicja|laundry|textm", None),
    ("Travel",
     r"uber|ubr\*|lyft|airbnb|marriott|hilton|hotel|meridien|homewood suites|jetblue|airline|greyhound"
     r"|masabi|parking|parkup|toll|e-z\*pass|rental car", None),
    ("Automotive",
     r"car ?wash|laser wash|autozone|o'reilly|advance auto|tire|motors|dmv|auto parts", None),
    ("Professional Services",
     r"law offices|legalnature|turbotax|intuit|accounting|notary|numbers team|got-ju|mowsnowpros", None),
    ("Finance & Investments", r"coinbase|crypto|robinhood|webull|acorns|stash", None),
    ("Gifts & Donations", r"gofundme|donation|world vision|scouting", None),
    ("Education", r"college|university|tuition|school", None),
    ("Gas", r"shell|chevron|exxon|mobil\b|\barco\b|\bbp\b|bp#|citgo|sunoco|amoco|pemex|am gas", None),

    # --- known store, unknown basket: gas-station convenience stores, corner markets, processors
    ("Uncertain - Ambiguous Merchant",
     r"7-?eleven|wawa|racetrac|\bqt\b|quiktrip|kwik trip|get go|enmarket|circle k|speedway|sheetz"
     r"|pic n pump|travel center|^parkers$|hollywood mini|market|\bmart\b|\bconv\b|pantry|\bsq \*|payper|c-rewards"
     r"|ilixium|flexepayments|or technologies|baytree", None),

    # --- generic money movement after all merchant checks
    ("Transfers", r"transfer|\btfr|xfer|from savings|to shares?\b|credit builder|savings|checking"
                  r"|way2save|from wallet", None),
    ("Income / Deposits", r"deposit", None),
    ("Cash Out (ATM)", r"\batm\b|cash withdrawal|withdrwl", None),
    ("Fees & Adjustments", r"return|refund|adjustment|correction", None),
]

_COMPILED = [(cat, re.compile(pat), sub) for cat, pat, sub in RULES]

def suggest_label(text_lower):
    for cat, rx, sub in _COMPILED:
        m = rx.search(text_lower)
        if m:
            return cat, m.group(0), sub
    return "Uncertain - Unidentifiable", "", None

# ---- subtypes inside the category (checked in this order) ----
SUBTYPES = [
    ("gambling", GAMBLING),
    ("gym", r"planet fitness|la fitness|central florida ym|\bymca\b|iclub|fitness|\bgym\b|crunch|anytime fit|orangetheory"),
    ("events / recreation", r"riot fest|round1|\bbowl|stubhub|ticketmaster|treehouse indoor|cinemark|\bamc\b|regal"
                            r"|theat|candybar|seaworld|six flags|dave & buster|topgolf|eventbrite"),
    ("streaming", r"netflix|spotify|hulu|disney|peacock|paramount|prime video|amazon prime|roku|philo|youtube"
                  r"|hbo|\bmax\b|sling|siriusxm|pandora|audible|audiobooks|crunchyroll|fubo"),
    ("software / apps", r"apple\.com/bill|^apple$|google|microsoft|amzn apps|patreon|tiktok|rocket money|creditreview"
                        r"|arlo tech|abcmouse|hotmart|adobe|dropbox|icloud|openai|chatgpt|xbox|playstation|nintendo"),
]
_SUB_RX = [(name, re.compile(p)) for name, p in SUBTYPES]
IN_FEATURE = {"streaming", "software / apps", "gym", "other subscription"}

def lower_text(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def classify(text):
    # returns (category, subtype, merchant_key); subtype None if not in our slice
    cat, cue, _ = suggest_label(text)
    for name, rx in _SUB_RX:
        m = rx.search(text)
        if m and (cat == "Subscriptions & Entertainment" or (name == "gym" and cat == "Health & Wellness")):
            return cat, name, m.group(0)
    if cat == "Subscriptions & Entertainment":
        return cat, "other subscription", cue
    return cat, None, None

# ---- END VERBATIM ----

# Broad app-store / platform cues, exactly the list pa7 section 8(c) used for its caveat.
BASELINE_AMBIGUOUS_CUES = {"google", "apple.com/bill", "apple", "microsoft"}


def classify_baseline(text):
    """Return (subtype, merchant, tier, cue) for a lower_text() memo using the pa7 rules.

    subtype is None outside the Subscriptions & Entertainment (+ gym) slice; in_feature subtypes
    are pa7's IN_FEATURE. tier is 'ambiguous_platform' for pa7's broad cues, else 'high_confidence'.
    """
    cat, subtype, merchant = classify(text)
    if subtype is None:
        return None, None, None, None
    tier = "ambiguous_platform" if merchant in BASELINE_AMBIGUOUS_CUES else "high_confidence"
    return subtype, merchant, tier, merchant

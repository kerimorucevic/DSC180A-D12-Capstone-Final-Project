"""Alternative (v2) subscription classifier.

Differences from the pa7 baseline (see the codebase review):
* Independent of the 21-class category rules, so subtype keywords are no longer dead: a memo is
  matched directly against a subscription-merchant dictionary.
* Two tiers. ``high_confidence``: a named service whose charges are normally a plan or membership
  (Netflix, Adobe, Planet Fitness ...). ``ambiguous_platform``: a billing platform that bills both
  subscriptions and one-off purchases under the same memo (Apple, Google Play, Microsoft, Xbox,
  PlayStation, Nintendo, Amazon digital, Steam, Roku, generic YouTube / Disney).
* Explicit exclusions, checked first: bank-fee memos, gambling, advertising, one-time events,
  P2P / wallet transfers, retail-store purchases from a platform company.
* A memo match only says "subscription-type merchant". Whether a charge is *recurring* is decided
  later by the recurrence methods, never by the classifier.

Every pattern is applied to lower_text(memo). Order matters: first match wins.
"""
import re

from .config import TIER_AMBIGUOUS, TIER_HIGH
from .rules_baseline import GAMBLING as BASELINE_GAMBLING

EXCLUDED = "excluded"

GAMBLING_V2 = (BASELINE_GAMBLING +
               r"|bovada|pointsbet|bet365|caesars sportsb|betrivers|stake\.us|pulsz|mcluck|high 5 casino"
               r"|luckyland|wow vegas|sportsbook")

# (reason, regex) -> never a subscription in this feature
EXCLUSIONS = [
    ("bank_fee", r"foreign (transaction|txn|trans) fee|int'?l (transaction|txn|trans)? ?fee|overdraft"
                 r"|\bnsf\b|insufficient funds|non-?sufficient|returned item|service charge"),
    ("gambling", GAMBLING_V2),
    ("advertising", r"google ?\*? ?ads|googleads|facebook ads|\bfb ads\b|meta ads"),
]

# (subtype, canonical merchant, regex) -- named services, high confidence
HIGH_CONFIDENCE = [
    # streaming / media
    ("streaming", "netflix", r"netflix"),
    ("streaming", "spotify", r"spotify"),
    ("streaming", "hulu", r"hulu"),
    ("streaming", "disney_plus", r"disney ?(\+|plus)|disneyplus"),
    ("streaming", "peacock", r"peacock"),
    ("streaming", "paramount_plus", r"paramount ?(\+|plus)|paramountplus|cbs all access"),
    ("streaming", "max_hbo", r"\bhbo\b|hbo ?max|hbomax|max\.com|^max$"),
    ("streaming", "amazon_prime", r"(amazon|amzn) prime(?! ?now)|prime ?video|prime membership"),
    ("streaming", "amazon_music_kindle", r"amazon music|kindle unlimited|kindle unltd"),
    ("streaming", "youtube_premium_tv", r"youtube ?(premium|tv|music)|youtubepremium|youtubetv"),
    ("streaming", "sling", r"sling\.com|sling ?tv|^sling\b"),
    ("streaming", "siriusxm", r"sirius ?xm"),
    ("streaming", "pandora", r"pandora (media|music|premium|plus)|pandora\.com"),
    ("streaming", "audible", r"audible|audiobooks"),
    ("streaming", "crunchyroll", r"crunchyroll"),
    ("streaming", "fubo", r"fubo"),
    ("streaming", "philo", r"\bphilo\b"),
    ("streaming", "apple_media", r"apple ?music|apple ?tv ?(\+|plus)|appletv\+"),
    ("streaming", "starz_showtime", r"\bstarz\b|showtime"),
    ("streaming", "espn_plus", r"espn ?(\+|plus)"),
    ("streaming", "amc_plus", r"amc ?(\+|plus)"),
    ("streaming", "tidal", r"tidal\.com|tidal music"),
    ("streaming", "roku_channel", r"roku ?(premium|channel)"),
    # software / apps / memberships billed by the service itself
    ("software / apps", "adobe", r"adobe"),
    ("software / apps", "dropbox", r"dropbox"),
    ("software / apps", "openai", r"openai|chatgpt"),
    ("software / apps", "microsoft_365", r"microsoft ?365|msft ?\*? ?365|office ?365"),
    ("software / apps", "xbox_game_pass", r"xbox ?game ?pass|game ?pass|xbox live"),
    ("software / apps", "playstation_plus", r"playstation ?plus|\bps ?plus\b"),
    ("software / apps", "nintendo_online", r"nintendo switch online"),
    ("software / apps", "google_one", r"google ?one|google storage|google ?workspace|\bg ?suite\b"),
    ("software / apps", "icloud", r"icloud"),
    ("software / apps", "canva", r"\bcanva\b"),
    ("software / apps", "notion", r"notion\.so|notion labs"),
    ("software / apps", "github", r"github"),
    ("software / apps", "zoom", r"zoom\.us|zoom video"),
    ("software / apps", "grammarly", r"grammarly"),
    ("software / apps", "antivirus_vpn", r"norton|mcafee|nordvpn|expressvpn|surfshark"),
    ("software / apps", "patreon", r"patreon"),
    ("software / apps", "duolingo", r"duolingo"),
    ("software / apps", "rocket_money", r"rocket money"),
    ("software / apps", "abcmouse", r"abcmouse"),
    ("software / apps", "arlo", r"arlo tech"),
    ("software / apps", "linkedin", r"linkedin"),
    ("software / apps", "dating", r"tinder|bumble|hinge|match\.com"),
    ("software / apps", "wellness_apps", r"headspace|calm\.com|peloton"),
    # gym / fitness memberships
    ("gym", "planet_fitness", r"planet fitness|pf club"),
    ("gym", "la_fitness", r"la fitness"),
    ("gym", "crunch", r"crunch fitness|crunch gym"),
    ("gym", "anytime_fitness", r"anytime fit"),
    ("gym", "24_hour_fitness", r"24 hour fit"),
    ("gym", "orangetheory", r"orangetheory|orange theory"),
    ("gym", "ymca", r"\bymca\b|central florida ym"),
    ("gym", "equinox", r"equinox"),
    ("gym", "golds_gym", r"gold'?s gym"),
    ("gym", "life_time", r"life ?time fitness"),
    ("gym", "eos_blink", r"eos fitness|blink fitness"),
    ("gym", "iclub", r"iclub"),
    ("gym", "gym_generic", r"\bgym\b|fitness club|fitness center"),
]

# (reason, regex) -- checked after the named services so "amc+" / "disney+" are not caught here
LATE_EXCLUSIONS = [
    ("events / recreation",
     r"ticketmaster|live ?nation|stubhub|seatgeek|vivid seats|axs\.com|eventbrite|riot fest|\bamc\b"
     r"|\bregal\b|cinemark|fandango|theat(er|re)|cinema|six flags|seaworld|knott'?s|universal studios"
     r"|disneyland|disney ?world|disney resort|disney parks?|topgolf|dave (&|and) buster|round1|\bbowl"
     r"|treehouse indoor|candybar|museum|\bzoo\b|concert"),
    ("transfer_or_p2p",
     r"zelle|venmo|cash\s*/?\s*app|apple cash|google pay|\bgpay\b|paypal inst|transfer|\bxfer\b|\btfr\b"),
    ("retail_store", r"apple store|apple online store|apple\.com/us|microsoft store|best buy"),
]

# (canonical merchant, regex) -- billing platforms, subscriptions and one-off purchases mixed
AMBIGUOUS_PLATFORMS = [
    ("apple_platform", r"apple\.com/bill|itunes|^apple$|apple services"),
    ("google_platform", r"google ?play|google ?\*|^google$|google services"),
    ("youtube_generic", r"youtube"),
    ("microsoft_platform", r"microsoft|\bmsft\b"),
    ("xbox_platform", r"\bxbox\b"),
    ("playstation_platform", r"playstation|\bpsn\b|sony interactive"),
    ("nintendo_platform", r"nintendo"),
    ("amazon_digital", r"amzn ?digital|amazon ?digital|amzn apps|amazon apps|kindle svcs"),
    ("steam", r"steampowered|steam games|valve corp"),
    ("roku_platform", r"\broku\b"),
    ("disney_generic", r"disney"),
]

_EXCL = [(r, re.compile(p)) for r, p in EXCLUSIONS]
_HIGH = [(s, m, re.compile(p)) for s, m, p in HIGH_CONFIDENCE]
_LATE = [(r, re.compile(p)) for r, p in LATE_EXCLUSIONS]
_AMB = [(m, re.compile(p)) for m, p in AMBIGUOUS_PLATFORMS]

IN_FEATURE_V2 = {"streaming", "software / apps", "gym", "platform"}


def classify_v2(text):
    """Return (subtype, merchant, tier, cue) for a lower_text() memo.

    subtype: one of IN_FEATURE_V2, or 'excluded:<reason>' for matched exclusions, or None.
    """
    for reason, rx in _EXCL:
        m = rx.search(text)
        if m:
            return f"{EXCLUDED}:{reason}", None, None, m.group(0)
    for subtype, merchant, rx in _HIGH:
        m = rx.search(text)
        if m:
            return subtype, merchant, TIER_HIGH, m.group(0)
    for reason, rx in _LATE:
        m = rx.search(text)
        if m:
            return f"{EXCLUDED}:{reason}", None, None, m.group(0)
    for merchant, rx in _AMB:
        m = rx.search(text)
        if m:
            return "platform", merchant, TIER_AMBIGUOUS, m.group(0)
    return None, None, None, None

"""Alternative (v2) classifier: missing keywords, false positives, tiers, exclusions."""
import pytest

from subrec.config import TIER_AMBIGUOUS, TIER_HIGH
from subrec.data import lower_text
from subrec.rules_baseline import classify_baseline
from subrec.rules_v2 import classify_v2


def v2(memo):
    return classify_v2(lower_text(memo))


@pytest.mark.parametrize("memo,subtype,merchant", [
    ("NETFLIX.COM 866-579-7172 CA", "streaming", "netflix"),
    ("YOUTUBE PREMIUM 650-253-0000", "streaming", "youtube_premium_tv"),
    ("HBO MAX 855-442-6629", "streaming", "max_hbo"),
    ("SIRIUSXM RADIO 888-635-5144", "streaming", "siriusxm"),
    ("DISNEY PLUS 888-905-7888 CA", "streaming", "disney_plus"),
    ("AMZN PRIME*2K4LM8", "streaming", "amazon_prime"),
    ("AMAZON PRIME*2K4LM8", "streaming", "amazon_prime"),
    ("PARAMOUNT+ MONTHLY FEE", "streaming", "paramount_plus"),
    ("AMC+ SUBSCRIPTION", "streaming", "amc_plus"),
    ("ADOBE *CREATIVE CLD 408-536-6000", "software / apps", "adobe"),
    ("OPENAI *CHATGPT SUBSCR", "software / apps", "openai"),
    ("DROPBOX*7HJ2K", "software / apps", "dropbox"),
    ("XBOX GAME PASS ULTIMATE", "software / apps", "xbox_game_pass"),
    ("CRUNCH FITNESS #221", "gym", "crunch"),
    ("ANYTIME FITNESS 4411", "gym", "anytime_fitness"),
    ("YMCA OF SAN DIEGO", "gym", "ymca"),
    ("PLANET FITNESS CLUB FEES", "gym", "planet_fitness"),
])
def test_high_confidence_merchants(memo, subtype, merchant):
    assert v2(memo)[:3] == (subtype, merchant, TIER_HIGH)


@pytest.mark.parametrize("memo,merchant", [
    ("APPLE.COM/BILL 866-712-7753 CA", "apple_platform"),
    ("GOOGLE *GOOGLE PLAY APP", "google_platform"),
    ("MICROSOFT*STORE PURCHASE", "microsoft_platform"),
    ("XBOX DIGITAL PURCHASE", "xbox_platform"),
    ("AMZN DIGITAL*2K81 WA", "amazon_digital"),
    ("YOUTUBE 650-253-0000", "youtube_generic"),
])
def test_ambiguous_platforms(memo, merchant):
    assert v2(memo)[:3] == ("platform", merchant, TIER_AMBIGUOUS)


@pytest.mark.parametrize("memo,reason", [
    ("DRAFTKINGS SPORTSBOOK", "gambling"),
    ("FANDUEL SPORTSBOOK", "gambling"),
    ("TICKETMASTER 800-653-8000", "events / recreation"),
    ("AMC THEATRES 2231", "events / recreation"),
    ("DISNEYLAND RESORT TICKETS", "events / recreation"),
    ("ROUND1 BOWLING SAN DIEGO", "events / recreation"),
    ("GOOGLE PAY TRANSFER 4821", "transfer_or_p2p"),
    ("APPLE CASH SENT MONEY", "transfer_or_p2p"),
    ("GOOGLE *ADS4821", "advertising"),
    ("FOREIGN TRANSACTION FEE NETFLIX", "bank_fee"),
    ("APPLE STORE R102 UTC", "retail_store"),
])
def test_exclusions(memo, reason):
    sub, merchant, tier, _ = v2(memo)
    assert sub == f"excluded:{reason}" and merchant is None and tier is None


@pytest.mark.parametrize("memo", ["TRADER JOE'S #552", "OFFICE MAX 1123", "MONTHLY MAINTENANCE FEE",
                                  "PANDORA JEWELRY 22", "SHELL OIL 57442", "TIDAL WAVE CAR WASH"])
def test_non_subscriptions_unmatched(memo):
    assert v2(memo) == (None, None, None, None)


def test_baseline_gaps_that_v2_fixes():
    """Keywords listed in pa7's SUBTYPES that never reach the subscription slice in the baseline."""
    for memo in ["YOUTUBE TV", "HBO MAX", "ADOBE *CREATIVE CLD", "CRUNCH FITNESS", "OPENAI *CHATGPT SUBSCR"]:
        assert classify_baseline(lower_text(memo))[0] is None
        assert v2(memo)[2] in (TIER_HIGH, TIER_AMBIGUOUS)


def test_baseline_false_positives_that_v2_excludes():
    for memo in ["GOOGLE PAY TRANSFER 4821", "DISNEYLAND RESORT TICKETS"]:
        assert classify_baseline(lower_text(memo))[0] in {"software / apps", "streaming"}
        assert v2(memo)[0].startswith("excluded:")

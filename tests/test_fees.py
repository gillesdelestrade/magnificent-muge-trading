from datetime import date

import pytest

from mugetrading.engine.fees import (FeeBracket, FeeCalculator, FeeProfile, get_profile,
                                     load_profiles, save_profiles)

PROFILES = load_profiles()
AMOUNTS = [100, 500, 501, 1000, 4400, 10000]

# Expected brokerage for orders of 100, 500, 501, 1 000, 4 400 and 10 000 €
EXPECTED = {
    "Trade Republic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "DEGIRO": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "Bourse Direct": [0.99, 0.99, 1.90, 1.90, 3.80, 9.00],
    "BoursoBank – Découverte": [1.99, 1.99, 3.01, 6.00, 26.40, 60.00],
    "Fortuneo – Starter": [0.35, 1.75, 1.75, 3.50, 15.40, 35.00],  # outside the free order
    "Interactive Brokers – Fixed": [3.0, 3.0, 3.0, 3.0, 3.0, 5.0],
    "Personnalisé": [0.0] * 6,
    "Sans frais": [0.0] * 6,
}


def test_all_shipped_profiles_are_tested():
    assert {p.name for p in PROFILES} == set(EXPECTED)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_profile_brokerage(name):
    profile = get_profile(name, PROFILES)
    assert [profile.brokerage(a) for a in AMOUNTS] == pytest.approx(EXPECTED[name])


def test_bracket_formula_min_and_max():
    b = FeeBracket(fixed=1.0, pct=1.0, minimum=5.0, maximum=20.0)
    assert b.fee(100) == 5.0      # 1 + 1 = 2 -> minimum
    assert b.fee(1000) == 11.0    # 1 + 10
    assert b.fee(5000) == 20.0    # 51 -> maximum


def test_free_orders_counter_resets_every_month():
    calc = FeeCalculator(get_profile("Fortuneo – Starter", PROFILES))
    jan, feb = date(2025, 1, 10), date(2025, 2, 3)
    # Quoting does not consume the free order
    assert calc.compute(400, jan, "buy").brokerage == 0.0
    assert calc.compute(400, jan, "buy", commit=True).brokerage == 0.0
    assert calc.compute(400, jan, "sell", commit=True).brokerage == pytest.approx(1.40)
    assert calc.compute(400, feb, "buy", commit=True).brokerage == 0.0   # new month
    assert calc.compute(400, feb, "buy", commit=True).brokerage == pytest.approx(1.40)


def test_free_order_only_under_max_amount():
    calc = FeeCalculator(get_profile("Fortuneo – Starter", PROFILES))
    day = date(2025, 1, 10)
    assert calc.compute(501, day, "buy", commit=True).brokerage == pytest.approx(1.75)
    assert calc.compute(500, day, "buy", commit=True).brokerage == 0.0  # still available


def test_pea_cap():
    cto = FeeProfile(name="x", brackets=(FeeBracket(fixed=2.0),))
    pea = FeeProfile(name="x", brackets=(FeeBracket(fixed=2.0),), account_type="PEA")
    assert cto.brokerage(100) == 2.0
    assert pea.brokerage(100) == 0.50      # capped at 0.5 % of the order
    assert pea.brokerage(1000) == 2.0      # cap (5 €) not reached


def test_ttf_on_french_buys_only():
    calc = FeeCalculator(get_profile("Trade Republic", PROFILES), ttf_enabled=True, ttf_rate=0.004)
    day = date(2025, 1, 10)
    buy = calc.compute(1000, day, "buy", "MC.PA")
    assert (buy.brokerage, buy.ttf, buy.total) == (1.0, 4.0, 5.0)
    assert calc.compute(1000, day, "sell", "MC.PA").ttf == 0.0
    assert calc.compute(1000, day, "buy", "MT.AS").ttf == 0.0
    assert FeeCalculator(ttf_enabled=False).compute(1000, day, "buy", "MC.PA").ttf == 0.0


def test_profiles_roundtrip(tmp_path):
    path = tmp_path / "fees.yaml"
    save_profiles(PROFILES, path)
    assert load_profiles(path) == PROFILES

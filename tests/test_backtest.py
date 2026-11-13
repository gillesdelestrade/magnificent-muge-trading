"""Engine tests on synthetic prices whose result is known by hand."""
from datetime import date

import pandas as pd
import pytest

from conftest import flat, make_prices
from mugetrading.engine.backtest import (REASON_END, SKIP_MAX_POSITIONS, SKIP_NO_CASH,
                                         run_backtest)
from mugetrading.engine.fees import FeeBracket, FeeProfile, get_profile, load_profiles
from mugetrading.strategy.params import COMBINE_BREAKOUT, COMBINE_MOMENTUM, StrategyParams


def params(**kw):
    base = dict(
        momentum_period=3, breakout_period=3, combination=COMBINE_BREAKOUT,
        stop_loss_enabled=True, stop_loss_pct=0.10, trailing_stop_enabled=False,
        initial_capital=1000.0, position_size_pct=0.5, max_positions=2,
        ttf_enabled=False, slippage=0.0,
    )
    base.update(kw)
    return StrategyParams(**base)


# Days 0-3 flat at 100, day 4 closes at 105 (breakout), day 5 opens at 106.
BREAKOUT = flat(4) + [(100, 105, 100, 105), (106, 108, 104, 107)]


def day(df, i):
    return df.index[i]


def test_breakout_buys_at_next_open():
    df = make_prices(BREAKOUT + [(107, 107, 107, 107)])
    r = run_backtest({"AAA": df}, params())
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["date"] == day(df, 5)            # signal on day 4 -> executed on day 5
    assert (t["side"], t["reason"], t["price"]) == ("buy", "Breakout", 106.0)
    assert t["quantity"] == 4                 # 50 % of 1 000 € = 500 € -> 4 shares at 106
    assert t["cash_after"] == pytest.approx(1000 - 4 * 106)
    assert r.equity["value"].iloc[-1] == pytest.approx(576 + 4 * 107)
    assert r.open_positions == {"AAA": 4}


def test_stop_loss_sells_at_stop_level():
    # Bought at 106 -> stop at 95.40. Day 7: low 94 touches the stop without gap.
    df = make_prices(BREAKOUT + [(107, 107, 100, 101), (101, 102, 94, 96)])
    r = run_backtest({"AAA": df}, params())
    sell = r.trades.iloc[1]
    assert (sell["side"], sell["reason"], sell["date"]) == ("sell", "Stop loss", day(df, 7))
    assert sell["price"] == pytest.approx(95.40)
    assert sell["pnl"] == pytest.approx(4 * (95.40 - 106))
    assert sell["pnl_pct"] == pytest.approx(-0.10)
    assert r.equity["value"].iloc[-1] == pytest.approx(1000 - 42.40)


def test_stop_loss_gap_sells_at_open():
    df = make_prices(BREAKOUT + [(107, 107, 100, 101), (90, 92, 88, 91)])
    r = run_backtest({"AAA": df}, params())
    sell = r.trades.iloc[1]
    assert sell["price"] == 90.0              # opened below the stop: sold at the open
    assert sell["reason"] == "Stop loss"


def test_trailing_stop_rises_and_never_falls():
    rows = BREAKOUT + [
        (107, 120, 107, 118),   # day 6: high 120 -> stop 108 from day 7
        (118, 119, 110, 112),   # day 7: low 110 > 108, no sell; high stays 120
        (112, 113, 109, 110),   # day 8: stop still 108 (never goes down)
        (110, 111, 105, 106),   # day 9: low 105 <= 108 -> sold at 108
    ]
    df = make_prices(rows)
    r = run_backtest({"AAA": df}, params(stop_loss_enabled=False, trailing_stop_enabled=True,
                                         trailing_stop_pct=0.10))
    stops = r.stops.set_index("date")["stop"]
    assert stops.loc[day(df, 6)] == pytest.approx(108 * 0.9)   # from the buy-day high (108)
    assert stops.loc[day(df, 7)] == pytest.approx(108.0)
    assert stops.loc[day(df, 8)] == pytest.approx(108.0)
    assert stops.is_monotonic_increasing
    sell = r.trades.iloc[1]
    assert (sell["date"], sell["price"], sell["reason"]) == (day(df, 9), 108.0, "Stop suiveur")


def test_new_trailing_stop_applies_only_from_next_day():
    # Day 6 makes a high of 130 then falls to 110 the same day: the stop of that day is
    # still the one known the day before (97.2), so no sell on day 6.
    rows = BREAKOUT + [(107, 130, 110, 112), (112, 113, 111, 112)]
    df = make_prices(rows)
    r = run_backtest({"AAA": df}, params(stop_loss_enabled=False, trailing_stop_enabled=True))
    assert len(r.trades) == 2
    assert r.trades.iloc[1]["date"] == day(df, 7)       # 117 = 130 x 0.9 hit by the gap at 112
    assert r.trades.iloc[1]["price"] == 112.0


def test_highest_of_both_stops_is_used():
    rows = BREAKOUT + [(107, 107, 96, 100)]
    df = make_prices(rows)
    # stop loss 5 % -> 100.70 ; trailing 10 % of 108 -> 97.20 : stop loss wins
    r = run_backtest({"AAA": df}, params(stop_loss_pct=0.05, trailing_stop_enabled=True))
    sell = r.trades.iloc[1]
    assert sell["reason"] == "Stop loss" and sell["price"] == pytest.approx(100.70)


def test_fees_ttf_and_slippage():
    df = make_prices(BREAKOUT + [(107, 107, 100, 101), (101, 102, 94, 96)])
    df.index.name = "date"
    profile = FeeProfile(name="t", brackets=(FeeBracket(pct=0.5, minimum=2.0),))
    r = run_backtest({"AAA.PA": df}, params(ttf_enabled=True, ttf_rate=0.004, slippage=0.01),
                     fee_profile=profile)
    buy, sell = r.trades.iloc[0], r.trades.iloc[1]
    assert buy["price"] == pytest.approx(107.06)          # 106 + 1 % slippage
    assert buy["quantity"] == 4
    assert buy["brokerage"] == pytest.approx(2.14)        # 0.5 % of 428.24
    assert buy["ttf"] == pytest.approx(1.71)              # 0.4 % of 428.24
    assert buy["net"] == pytest.approx(428.24 + 3.85)
    stop = 107.06 * 0.9
    assert sell["price"] == pytest.approx(stop * 0.99)    # slippage against us
    assert sell["brokerage"] == 2.0 and sell["ttf"] == 0.0   # minimum fee, no TTF on sells
    assert sell["pnl"] == pytest.approx(sell["net"] - buy["net"])
    assert r.stats["total_fees"] == pytest.approx(3.85 + 2.0)


def test_buy_skipped_when_cash_is_insufficient():
    df = make_prices([(o * 10, h * 10, l * 10, c * 10) for o, h, l, c in BREAKOUT])
    r = run_backtest({"AAA": df}, params(initial_capital=1000, position_size_pct=0.5))
    assert r.trades.empty                      # one share costs 1 060 € > 500 € budget
    assert r.skipped.iloc[0]["cause"] == SKIP_NO_CASH
    assert r.equity["value"].eq(1000).all()


def test_fees_reduce_quantity_and_cash_never_negative():
    df = make_prices(BREAKOUT)
    profile = FeeProfile(name="t", brackets=(FeeBracket(fixed=80.0),))
    r = run_backtest({"AAA": df}, params(), fee_profile=profile)
    assert r.trades.iloc[0]["quantity"] == 3   # 4 x 106 + 80 > 500
    assert (r.equity["cash"] >= 0).all() and (r.trades["cash_after"] >= 0).all()


def test_max_positions_and_momentum_priority():
    strong = flat(4) + [(100, 120, 100, 120), (120, 121, 119, 120)]
    weak = flat(4) + [(100, 110, 100, 110), (110, 111, 109, 110)]
    r = run_backtest({"WEAK": make_prices(weak), "STRONG": make_prices(strong)},
                     params(max_positions=1, combination=COMBINE_MOMENTUM))
    assert r.trades["ticker"].tolist() == ["STRONG"]
    assert r.skipped.iloc[0][["ticker", "cause"]].tolist() == ["WEAK", SKIP_MAX_POSITIONS]


def test_no_rebuy_on_the_day_of_the_sale():
    rows = BREAKOUT + [
        (107, 115, 107, 114),     # day 6: new breakout signal (already held)
        (95, 120, 94, 119),       # day 7: gap under the stop -> sold; signal of day 6 ignored
        (119, 119, 119, 119),
    ]
    df = make_prices(rows)
    r = run_backtest({"AAA": df}, params())
    day7 = r.trades[r.trades["date"] == day(df, 7)]
    assert day7["side"].tolist() == ["sell"]
    # ... but it can be bought again on a later day (signal of day 7 executed on day 8)
    assert r.trades.iloc[-1][["side", "date"]].tolist() == ["buy", day(df, 8)]


def test_sells_before_buys_free_cash_the_same_day():
    a = BREAKOUT + [(90, 91, 89, 90), (90, 90, 90, 90)]                      # sold on day 6 (gap)
    b = flat(5) + [(100, 105, 100, 105), (106, 106, 106, 106), (106, 106, 106, 106)]  # bought day 6
    r = run_backtest({"AAA": make_prices(a), "BBB": make_prices(b)},
                     params(position_size_pct=1.0, max_positions=1))
    assert r.trades[["ticker", "side"]].values.tolist() == [
        ["AAA", "buy"], ["AAA", "sell"], ["BBB", "buy"]]
    assert r.trades.iloc[2]["quantity"] == 8   # 46 + 9 x 90 = 856 € of cash -> 8 x 106 = 848 €


def test_no_future_data_is_used():
    """Changing prices after day J must not change anything decided up to day J."""
    base = BREAKOUT + [(107, 112, 105, 110), (110, 111, 100, 101), (101, 125, 99, 124),
                       (124, 130, 120, 128)]
    full = run_backtest({"AAA": make_prices(base)}, params(trailing_stop_enabled=True))
    for cut in range(5, len(base)):
        altered = base[:cut] + [(1.0, 500.0, 0.5, 300.0)] * (len(base) - cut)
        r = run_backtest({"AAA": make_prices(altered)}, params(trailing_stop_enabled=True))
        limit = make_prices(base).index[cut - 1]
        pd.testing.assert_frame_equal(
            r.trades[r.trades["date"] <= limit].reset_index(drop=True),
            full.trades[full.trades["date"] <= limit].reset_index(drop=True))
        pd.testing.assert_frame_equal(r.equity.loc[:limit], full.equity.loc[:limit])


def test_liquidation_at_end_of_period():
    df = make_prices(BREAKOUT + [(107, 109, 107, 108)])
    r = run_backtest({"AAA": df}, params(liquidate_at_end=True))
    last = r.trades.iloc[-1]
    assert (last["side"], last["reason"], last["price"]) == ("sell", REASON_END, 108.0)
    assert r.open_positions == {} and r.equity["invested"].iloc[-1] == 0


def test_fractions_invest_the_whole_budget():
    df = make_prices(BREAKOUT)
    r = run_backtest({"AAA": df}, params(allow_fractions=True),
                     fee_profile=get_profile("Trade Republic", load_profiles()))
    buy = r.trades.iloc[0]
    assert buy["quantity"] == pytest.approx(499 / 106, abs=1e-4)
    assert buy["net"] <= 500.0 + 1e-9


def test_start_date_and_missing_data():
    # BBB only exists from day 3: simply ignored before, no crash. Warm-up before `start`.
    a = make_prices(BREAKOUT + flat(2, 107))
    b = make_prices(flat(5), start=str(a.index[3].date()))
    r = run_backtest({"AAA": a, "BBB": b}, params(), start=a.index[5].date())
    assert r.equity.index[0] == a.index[5]
    assert r.trades.iloc[0]["date"] == a.index[5]     # signal from the warm-up period


def test_stats():
    df = make_prices(BREAKOUT + [(107, 107, 100, 101), (101, 102, 94, 96)])
    s = run_backtest({"AAA": df}, params()).stats
    assert s["n_operations"] == 2 and s["n_wins"] == 0 and s["n_losses"] == 1
    assert s["win_rate"] == 0.0
    assert s["avg_loss"] == pytest.approx(-42.40)
    assert s["final_value"] == pytest.approx(957.60)
    assert s["total_return_pct"] == pytest.approx(-0.0424)
    assert s["avg_holding_days"] == 2
    assert s["max_drawdown"] < 0

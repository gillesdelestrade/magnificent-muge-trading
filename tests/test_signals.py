import pandas as pd
import pytest

from conftest import flat, make_prices
from mugetrading.strategy.params import (COMBINE_AND, COMBINE_BREAKOUT, COMBINE_MOMENTUM,
                                         COMBINE_OR, StrategyParams)
from mugetrading.strategy.signals import breakout_level, compute_signals, momentum


def test_momentum_value():
    close = pd.Series([100.0, 101, 102, 110])
    assert momentum(close, 3).iloc[3] == pytest.approx(0.10)
    assert momentum(close, 3).iloc[:3].isna().all()


def test_breakout_level_excludes_current_day():
    high = pd.Series([10.0, 12, 11, 50])
    level = breakout_level(high, 3)
    assert level.iloc[3] == 12  # highest of the 3 previous days, the 50 of the day is excluded
    assert level.iloc[:3].isna().all()


def _signals(rows, **kw):
    return compute_signals(make_prices(rows), StrategyParams(momentum_period=3, breakout_period=3, **kw))


def test_breakout_needs_close_strictly_above_previous_high():
    rows = flat(3) + [(100, 100, 100, 100), (100, 103, 100, 102)]
    s = _signals(rows, combination=COMBINE_BREAKOUT)
    assert s["buy_signal"].tolist() == [False, False, False, False, True]
    assert s["reason"].iloc[-1] == "Breakout"


def test_breakout_margin():
    rows = flat(4) + [(100, 103, 100, 102)]
    assert not _signals(rows, combination=COMBINE_BREAKOUT, breakout_margin=0.03)["buy_signal"].any()
    assert _signals(rows, combination=COMBINE_BREAKOUT, breakout_margin=0.01)["buy_signal"].iloc[-1]


def test_momentum_threshold_is_inclusive():
    rows = flat(3) + [(100, 105, 100, 105)]
    s = _signals(rows, combination=COMBINE_MOMENTUM, momentum_threshold=0.05)
    assert s["buy_signal"].iloc[-1] and s["reason"].iloc[-1] == "Momentum"


def test_combinations():
    # Day 4: +6 % momentum but close below the previous high (104 < 110) -> momentum only
    rows = [(98, 98, 98, 98), (100, 110, 100, 100), (100, 100, 100, 100), (104, 104, 104, 104)]
    assert not _signals(rows, combination=COMBINE_AND)["buy_signal"].any()
    s_or = _signals(rows, combination=COMBINE_OR)
    assert s_or["buy_signal"].iloc[-1] and s_or["reason"].iloc[-1] == "Momentum"
    # Both
    rows = flat(3) + [(100, 107, 100, 106)]
    s = _signals(rows, combination=COMBINE_AND)
    assert s["buy_signal"].iloc[-1] and s["reason"].iloc[-1] == "Momentum + Breakout"


def test_signals_do_not_depend_on_future_data():
    rows = flat(5) + [(100, 104, 99, 103), (103, 109, 102, 108), (108, 108, 90, 91), (91, 120, 91, 119)]
    params = StrategyParams(momentum_period=3, breakout_period=3, combination=COMBINE_OR)
    full = compute_signals(make_prices(rows), params)
    for n in range(1, len(rows)):
        partial = compute_signals(make_prices(rows[:n]), params)
        pd.testing.assert_frame_equal(partial, full.iloc[:n])

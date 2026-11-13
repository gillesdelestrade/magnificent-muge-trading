"""Day-by-day simulation loop. Pure function, no Streamlit dependency.

Rules (CLAUDE.md §6):
1. A signal computed on the close of day J is executed at the open of day J+1.
2. Stops are checked every day: open below the stop -> sold at the open (gap);
   otherwise low touching the stop -> sold at the stop level.
3. Each day: sells first, then buys with the cash available after the sells.
4. A stock sold today is not bought back today.
5. At the end, open positions are valued at the last close (or sold, as an option).
6. Daily valuation at the close: cash + sum(quantity x close).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..broker.base import BUY, SELL, Fill, Order
from ..broker.paper import PaperBroker
from ..data.provider import adjusted_ohlc
from ..strategy.params import StrategyParams
from ..strategy.signals import compute_signals
from ..strategy.stops import stop_level
from .fees import NO_FEES, FeeCalculator, FeeProfile
from .stats import compute_stats

REASON_END = "Fin de période"
SKIP_NO_CASH = "Liquidités insuffisantes"
SKIP_MAX_POSITIONS = "Nombre maximum de positions atteint"
SKIP_SOLD_TODAY = "Titre vendu le jour même"

TRADE_COLUMNS = [
    "date", "ticker", "side", "reason", "quantity", "price", "gross", "brokerage", "ttf", "fees",
    "net", "pnl", "pnl_pct", "holding_days", "entry_date", "cash_after", "value_after",
]
SKIPPED_COLUMNS = ["date", "ticker", "signal", "cause"]
STOP_COLUMNS = ["date", "ticker", "stop", "reason"]
FRACTION_DECIMALS = 4


@dataclass
class BacktestResult:
    trades: pd.DataFrame            # journal, one row per operation (TRADE_COLUMNS)
    equity: pd.DataFrame            # daily: value, cash, invested (index = date)
    stats: dict
    skipped: pd.DataFrame           # signals not executed (SKIPPED_COLUMNS)
    stops: pd.DataFrame             # stop level applicable each day of holding (STOP_COLUMNS)
    params: StrategyParams
    fee_profile: FeeProfile
    open_positions: dict[str, float] = field(default_factory=dict)


def _prepare(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker, df in prices.items():
        if df is None or df.empty:
            continue
        out[ticker] = adjusted_ohlc(df) if "adj_close" in df.columns else df
    return out


def _size_order(broker: PaperBroker, ticker: str, price: float, day: date, budget: float,
                allow_fractions: bool) -> float:
    """Largest quantity such that quantity x price + fees <= budget (0 if impossible)."""
    def total(qty: float) -> float:
        order = Order(ticker, BUY, qty, price, day)
        return qty * price + broker.quote_fees(order).total

    if budget <= 0 or price <= 0:
        return 0.0
    if not allow_fractions:
        qty = math.floor(budget / price)
        while qty >= 1 and total(qty) > budget + 1e-9:
            qty -= 1
        return float(max(qty, 0))
    scale = 10 ** FRACTION_DECIMALS
    qty = math.floor(budget / price * scale) / scale
    for _ in range(50):
        if qty <= 0 or total(qty) <= budget + 1e-9:
            break
        fees = total(qty) - qty * price
        qty = min(qty - 1 / scale, math.floor((budget - fees) / price * scale) / scale)
    return max(qty, 0.0)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    params: StrategyParams,
    fee_profile: FeeProfile = NO_FEES,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    """Simulate the strategy on ``prices`` (ticker -> daily bars, warm-up included).

    Bars before ``start`` are only used to compute the indicators.
    """
    frames = _prepare(prices)
    bars: dict[str, dict[pd.Timestamp, tuple[float, float, float, float]]] = {}
    candidates: dict[pd.Timestamp, list[tuple[float, str, str]]] = {}
    calendar = pd.DatetimeIndex([])

    for ticker, df in frames.items():
        bars[ticker] = {
            ts: (o, h, l, c)
            for ts, o, h, l, c in zip(df.index, df["open"], df["high"], df["low"], df["close"])
        }
        calendar = calendar.union(df.index)
        signals = compute_signals(df, params)
        # No look-ahead: the signal of bar J is only visible on the stock's next bar (J+1).
        previous = signals[["buy_signal", "momentum", "reason"]].shift(1)
        for ts, mom, reason in previous.loc[previous["buy_signal"].eq(True),
                                           ["momentum", "reason"]].itertuples():
            mom = -math.inf if pd.isna(mom) else float(mom)
            candidates.setdefault(ts, []).append((mom, ticker, reason))

    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]

    broker = PaperBroker(
        cash=round(params.initial_capital, 2),
        fee_calculator=FeeCalculator(fee_profile, params.ttf_enabled, params.ttf_rate),
    )
    portfolio = broker.portfolio
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    skipped: list[dict] = []
    stops: list[dict] = []
    equity: list[dict] = []

    def mark_prices(day_bars: dict[str, tuple], use_open: bool) -> dict[str, float]:
        out = {}
        for t in portfolio.positions:
            bar = day_bars.get(t)
            out[t] = (bar[0] if use_open else bar[3]) if bar else last_close[t]
        return out

    def record(fill: Fill, marks: dict[str, float]) -> None:
        o = fill.order
        marks = {**marks, o.ticker: o.price}
        trades.append({
            "date": pd.Timestamp(o.day), "ticker": o.ticker, "side": o.side, "reason": o.reason,
            "quantity": o.quantity, "price": o.price, "gross": fill.gross,
            "brokerage": fill.fees.brokerage, "ttf": fill.fees.ttf, "fees": fill.fees.total,
            "net": fill.net, "pnl": fill.pnl, "pnl_pct": fill.pnl_pct,
            "holding_days": fill.holding_days,
            "entry_date": pd.Timestamp(fill.entry_date) if fill.entry_date else pd.NaT,
            "cash_after": portfolio.cash, "value_after": portfolio.value(marks),
        })

    for i, ts in enumerate(calendar):
        day = ts.date()
        day_bars = {t: b[ts] for t, b in bars.items() if ts in b}
        open_marks = mark_prices(day_bars, use_open=True)

        # 1. Sells: stops computed the day before
        sold_today: set[str] = set()
        for ticker in list(portfolio.positions):
            position = portfolio.positions[ticker]
            bar = day_bars.get(ticker)
            if bar is None or position.stop is None:
                continue
            stops.append({"date": ts, "ticker": ticker, "stop": position.stop,
                          "reason": position.stop_reason})
            open_, _, low, _ = bar
            if open_ <= position.stop:
                price = open_          # gap: opened below the stop
            elif low <= position.stop:
                price = position.stop
            else:
                continue
            price *= 1.0 - params.slippage
            fill = broker.place_order(
                Order(ticker, SELL, position.quantity, price, day, position.stop_reason))
            sold_today.add(ticker)
            record(fill, open_marks)

        # 2. Buys: signals of the previous bar, strongest momentum first
        for mom, ticker, reason in sorted(candidates.get(ts, []), key=lambda c: (-c[0], c[1])):
            if ticker in portfolio.positions:
                continue
            base = {"date": ts, "ticker": ticker, "signal": reason}
            if ticker in sold_today:
                skipped.append({**base, "cause": SKIP_SOLD_TODAY})
                continue
            if len(portfolio.positions) >= params.max_positions:
                skipped.append({**base, "cause": SKIP_MAX_POSITIONS})
                continue
            price = day_bars[ticker][0] * (1.0 + params.slippage)
            marks = mark_prices(day_bars, use_open=True)
            budget = min(params.position_size_pct * portfolio.value(marks), portfolio.cash)
            quantity = _size_order(broker, ticker, price, day, budget, params.allow_fractions)
            if quantity <= 0:
                skipped.append({**base, "cause": SKIP_NO_CASH})
                continue
            fill = broker.place_order(Order(ticker, BUY, quantity, price, day, reason))
            record(fill, marks)

        # 3. End of day: update highs; the new stop only applies from tomorrow
        for ticker, position in portfolio.positions.items():
            bar = day_bars.get(ticker)
            if bar is None:
                continue
            last_close[ticker] = bar[3]
            position.highest = max(position.highest, bar[1])
            position.stop, position.stop_reason = stop_level(
                position.entry_price, position.highest, params)
        for ticker in portfolio.positions:
            last_close.setdefault(ticker, portfolio.positions[ticker].entry_price)

        # 4. Optional liquidation on the last day, at the close
        if params.liquidate_at_end and i == len(calendar) - 1:
            for ticker in list(portfolio.positions):
                position = portfolio.positions[ticker]
                price = last_close[ticker] * (1.0 - params.slippage)
                fill = broker.place_order(
                    Order(ticker, SELL, position.quantity, price, day, REASON_END))
                record(fill, mark_prices(day_bars, use_open=False))

        # 5. Valuation at the close
        invested = portfolio.invested({t: last_close[t] for t in portfolio.positions})
        equity.append({"date": ts, "value": portfolio.cash + invested,
                       "cash": portfolio.cash, "invested": invested})

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_df = pd.DataFrame(equity, columns=["date", "value", "cash", "invested"]).set_index("date")
    return BacktestResult(
        trades=trades_df,
        equity=equity_df,
        stats=compute_stats(trades_df, equity_df, params.initial_capital),
        skipped=pd.DataFrame(skipped, columns=SKIPPED_COLUMNS),
        stops=pd.DataFrame(stops, columns=STOP_COLUMNS),
        params=params,
        fee_profile=fee_profile,
        open_positions=broker.get_positions(),
    )

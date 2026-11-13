"""Day-by-day simulation loop. Pure function, no Streamlit dependency.

Rules (CLAUDE.md §6):
1. A signal computed on the close of day J is executed at the open of day J+1.
2. Stops are checked every day: open below the stop -> sold at the open (gap);
   otherwise low touching the stop -> sold at the stop level.
   Take profit: open above the target -> sold at the open; otherwise high touching the
   target -> sold at the target. If the stop and the target are both touched on the same
   day, the stop is assumed to come first (daily bars do not tell the order).
3. Each day: sells first, then buys with the cash available after the sells.
4. A stock sold today is not bought back today.
5. At the end, open positions are valued at the last close (or sold, as an option).
6. Daily valuation at the close: cash + sum(quantity x close).
7. News filter (optional): the articles attached to day J or before decide at the close of
   J; a blocked purchase is logged, a very negative article sells at the open of J+1.
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
from ..strategy.news_filter import REASON_NEGATIVE_NEWS, news_signals
from ..strategy.signals import compute_signals
from ..strategy.stops import REASON_TAKE_PROFIT, stop_level, target_level
from .fees import NO_FEES, FeeCalculator, FeeProfile
from .stats import compute_stats

REASON_END = "Fin de période"
SKIP_NO_CASH = "Liquidités insuffisantes"
SKIP_MAX_POSITIONS = "Nombre maximum de positions atteint"
SKIP_SOLD_TODAY = "Titre vendu le jour même"
SKIP_NEWS = "Actualités défavorables"

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
    news: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simulate the strategy on ``prices`` (ticker -> daily bars, warm-up included).

    Bars before ``start`` are only used to compute the indicators. ``news`` (analysed
    articles: ticker, day, sentiment, pertinence, confiance) is only read when the news
    filter of ``params`` is enabled.
    """
    frames = _prepare(prices)
    bars: dict[str, dict[pd.Timestamp, tuple[float, float, float, float]]] = {}
    candidates: dict[pd.Timestamp, list[tuple[float, str, str, bool]]] = {}
    news_sells: dict[pd.Timestamp, set[str]] = {}
    news_by_ticker = ({t: g for t, g in news.groupby("ticker")}
                      if params.news_filter_enabled and news is not None and not news.empty else {})
    calendar = pd.DatetimeIndex([])

    for ticker, df in frames.items():
        bars[ticker] = {
            ts: (o, h, l, c)
            for ts, o, h, l, c in zip(df.index, df["open"], df["high"], df["low"], df["close"])
        }
        calendar = calendar.union(df.index)
        signals = compute_signals(df, params)
        # No look-ahead: the signal of bar J is only visible on the stock's next bar (J+1).
        # The news of bar J follow the same rule.
        signals[["news_ok", "news_sell"]] = news_signals(
            df.index, news_by_ticker.get(ticker), params)[["buy_ok", "sell"]]
        previous = signals[["buy_signal", "momentum", "reason", "news_ok", "news_sell"]].shift(1)
        for ts, mom, reason, news_ok in previous.loc[
                previous["buy_signal"].eq(True), ["momentum", "reason", "news_ok"]].itertuples():
            mom = -math.inf if pd.isna(mom) else float(mom)
            candidates.setdefault(ts, []).append((mom, ticker, reason, bool(news_ok)))
        for ts in previous.index[previous["news_sell"].eq(True)]:
            news_sells.setdefault(ts, set()).add(ticker)

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

        # 1. Sells: stops and take-profit levels computed the day before
        sold_today: set[str] = set()
        for ticker in list(portfolio.positions):
            position = portfolio.positions[ticker]
            bar = day_bars.get(ticker)
            if bar is None:
                continue
            stop, target = position.stop, position.target
            if stop is not None:
                stops.append({"date": ts, "ticker": ticker, "stop": stop,
                              "reason": position.stop_reason})
            open_, high, low, _ = bar
            if ticker in news_sells.get(ts, ()):
                price, reason = open_, REASON_NEGATIVE_NEWS      # decided yesterday: at the open
            elif stop is not None and open_ <= stop:
                price, reason = open_, position.stop_reason      # gap: opened below the stop
            elif target is not None and open_ >= target:
                price, reason = open_, REASON_TAKE_PROFIT        # gap: opened above the target
            elif stop is not None and low <= stop:
                price, reason = stop, position.stop_reason       # stop first if both are touched
            elif target is not None and high >= target:
                price, reason = target, REASON_TAKE_PROFIT
            else:
                continue
            price *= 1.0 - params.slippage
            fill = broker.place_order(Order(ticker, SELL, position.quantity, price, day, reason))
            sold_today.add(ticker)
            record(fill, open_marks)

        # 2. Buys: signals of the previous bar, strongest momentum first
        for mom, ticker, reason, news_ok in sorted(candidates.get(ts, []),
                                                   key=lambda c: (-c[0], c[1])):
            if ticker in portfolio.positions:
                continue
            base = {"date": ts, "ticker": ticker, "signal": reason}
            if ticker in sold_today:
                skipped.append({**base, "cause": SKIP_SOLD_TODAY})
                continue
            if len(portfolio.positions) >= params.max_positions:
                skipped.append({**base, "cause": SKIP_MAX_POSITIONS})
                continue
            if not news_ok:      # logged only when the news really prevented the purchase
                skipped.append({**base, "cause": SKIP_NEWS})
                continue
            price = day_bars[ticker][0] * (1.0 + params.slippage)
            marks = mark_prices(day_bars, use_open=True)
            budget = min(params.position_budget_pct * portfolio.value(marks), portfolio.cash)
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
            position.target = target_level(position.entry_price, params)
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

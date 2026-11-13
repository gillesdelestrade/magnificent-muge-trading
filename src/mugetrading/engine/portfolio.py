"""Portfolio state: cash, open positions, valuation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Position:
    ticker: str
    quantity: float
    entry_price: float       # execution price (slippage included)
    entry_date: date
    cost: float              # total paid, fees included
    buy_reason: str = ""
    highest: float = 0.0     # highest price reached since the purchase
    stop: float | None = None  # stop level applicable today (computed the day before)
    stop_reason: str = ""
    target: float | None = None  # take-profit level applicable today (set the day before)


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def value(self, prices: dict[str, float]) -> float:
        """cash + sum(quantity x price). ``prices`` must contain every held ticker."""
        return self.cash + self.invested(prices)

    def invested(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices[t] for t, p in self.positions.items())

    def buy(self, position: Position) -> None:
        if position.ticker in self.positions:
            raise ValueError(f"Already holding {position.ticker}")
        if position.cost > self.cash + 1e-9:
            raise ValueError("Insufficient cash")
        self.cash -= position.cost
        self.positions[position.ticker] = position

    def sell(self, ticker: str, net_proceeds: float) -> Position:
        position = self.positions.pop(ticker)
        self.cash += net_proceeds
        return position

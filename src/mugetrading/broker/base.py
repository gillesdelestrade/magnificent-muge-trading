"""Abstract Broker interface.

Only the simulated implementation (PaperBroker) exists. No code may send a real order
or connect to a brokerage account until the "Réel" part is explicitly started.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from ..engine.fees import Fees

BUY = "buy"
SELL = "sell"


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str            # BUY or SELL
    quantity: float
    price: float         # execution price decided by the simulation
    day: date
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    order: Order
    order_id: int
    gross: float         # quantity x price
    fees: Fees
    net: float           # cash debited (buy) or credited (sell)
    pnl: float | None = None       # realised profit/loss, fees included (sells only)
    pnl_pct: float | None = None
    holding_days: int | None = None
    entry_date: date | None = None


class Broker(ABC):
    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """ticker -> quantity"""

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def place_order(self, order: Order) -> Fill: ...

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool: ...

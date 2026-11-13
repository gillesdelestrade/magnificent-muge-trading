"""Simulated broker used by the backtest: orders are filled immediately, on paper only."""
from __future__ import annotations

from ..engine.fees import FeeCalculator, Fees
from ..engine.portfolio import Portfolio, Position
from .base import BUY, SELL, Broker, Fill, Order


class PaperBroker(Broker):
    def __init__(self, cash: float, fee_calculator: FeeCalculator | None = None):
        self.portfolio = Portfolio(cash=cash)
        self.fees = fee_calculator or FeeCalculator()
        self._next_id = 1

    def get_positions(self) -> dict[str, float]:
        return {t: p.quantity for t, p in self.portfolio.positions.items()}

    def get_cash(self) -> float:
        return self.portfolio.cash

    def quote_fees(self, order: Order) -> Fees:
        """Fees the order would cost, without recording anything."""
        return self.fees.compute(order.quantity * order.price, order.day, order.side, order.ticker)

    def place_order(self, order: Order) -> Fill:
        if order.quantity <= 0:
            raise ValueError("Quantity must be positive")
        gross = order.quantity * order.price
        if order.side == BUY:
            fees = self.quote_fees(order)
            net = gross + fees.total
            self.portfolio.buy(Position(
                ticker=order.ticker, quantity=order.quantity, entry_price=order.price,
                entry_date=order.day, cost=net, buy_reason=order.reason, highest=order.price,
            ))
            fill = Fill(order=order, order_id=self._next_id, gross=gross, fees=fees, net=net)
        elif order.side == SELL:
            if order.ticker not in self.portfolio.positions:
                raise ValueError(f"No position on {order.ticker}")
            fees = self.quote_fees(order)
            net = gross - fees.total
            position = self.portfolio.sell(order.ticker, net)
            pnl = net - position.cost
            fill = Fill(
                order=order, order_id=self._next_id, gross=gross, fees=fees, net=net,
                pnl=pnl, pnl_pct=pnl / position.cost if position.cost else 0.0,
                holding_days=(order.day - position.entry_date).days, entry_date=position.entry_date,
            )
        else:
            raise ValueError(f"Unknown side: {order.side}")
        self.fees.compute(gross, order.day, order.side, order.ticker, commit=True)
        self._next_id += 1
        return fill

    def cancel_order(self, order_id: int) -> bool:
        return False  # paper orders are filled immediately: nothing to cancel

"""Strategy parameters (defaults = CLAUDE.md §5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

COMBINE_MOMENTUM = "momentum"
COMBINE_BREAKOUT = "breakout"
COMBINE_AND = "and"
COMBINE_OR = "or"
COMBINATIONS = (COMBINE_MOMENTUM, COMBINE_BREAKOUT, COMBINE_AND, COMBINE_OR)

SIZING_PERCENT = "percent"   # fixed share of the portfolio value
SIZING_RISK = "risk"         # sized so that a stopped trade loses a given share of the portfolio
SIZING_MODES = (SIZING_PERCENT, SIZING_RISK)

MIN_CAPITAL = 100.0


@dataclass(frozen=True)
class StrategyParams:
    # Buy signals. Rates are fractions (0.05 = 5 %).
    momentum_period: int = 20
    momentum_threshold: float = 0.05
    breakout_period: int = 20
    breakout_margin: float = 0.0
    combination: str = COMBINE_AND
    # Sell signals
    stop_loss_enabled: bool = True
    stop_loss_pct: float = 0.08
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.10
    take_profit_enabled: bool = False
    take_profit_pct: float = 0.20          # sell at entry price x (1 + pct)
    # Portfolio
    initial_capital: float = 1000.0
    position_size_pct: float = 0.25
    max_positions: int = 4
    allow_fractions: bool = False
    # Position sizing: with SIZING_RISK, position_size_pct becomes the maximum size
    sizing_mode: str = SIZING_PERCENT
    risk_per_trade_pct: float = 0.01       # share of the portfolio lost if the initial stop is hit
    # Costs independent from the broker
    ttf_enabled: bool = True
    ttf_rate: float = 0.004
    slippage: float = 0.0005
    # End of period: sell open positions (reason "Fin de période") instead of just valuing them
    liquidate_at_end: bool = False

    def __post_init__(self):
        if self.combination not in COMBINATIONS:
            raise ValueError(f"Unknown combination: {self.combination}")
        if self.momentum_period < 1 or self.breakout_period < 1:
            raise ValueError("Periods must be >= 1")
        if self.initial_capital < MIN_CAPITAL:
            raise ValueError(f"Initial capital must be >= {MIN_CAPITAL}")
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("Position size must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if self.take_profit_enabled and self.take_profit_pct <= 0:
            raise ValueError("Take profit must be > 0")
        if self.sizing_mode not in SIZING_MODES:
            raise ValueError(f"Unknown sizing mode: {self.sizing_mode}")
        if self.sizing_mode == SIZING_RISK:
            if not 0 < self.risk_per_trade_pct <= 1:
                raise ValueError("Risk per trade must be in (0, 1]")
            if self.initial_stop_pct is None:
                raise ValueError("Risk-based sizing needs a stop loss or a trailing stop")

    @property
    def uses_momentum(self) -> bool:
        return self.combination in (COMBINE_MOMENTUM, COMBINE_AND, COMBINE_OR)

    @property
    def uses_breakout(self) -> bool:
        return self.combination in (COMBINE_BREAKOUT, COMBINE_AND, COMBINE_OR)

    @property
    def initial_stop_pct(self) -> float | None:
        """Distance between the purchase price and the stop on the day of the purchase
        (the closest of the active stops), None without any stop."""
        active = [pct for enabled, pct in ((self.stop_loss_enabled, self.stop_loss_pct),
                                           (self.trailing_stop_enabled, self.trailing_stop_pct))
                  if enabled and pct > 0]
        return min(active) if active else None

    @property
    def position_budget_pct(self) -> float:
        """Share of the portfolio value invested in one purchase (before the cash limit)."""
        if self.sizing_mode == SIZING_RISK:
            return min(self.position_size_pct, self.risk_per_trade_pct / self.initial_stop_pct)
        return self.position_size_pct

    @property
    def longest_period(self) -> int:
        return max(self.momentum_period, self.breakout_period)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

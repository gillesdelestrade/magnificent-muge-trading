"""Strategy parameters (defaults = CLAUDE.md §5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

COMBINE_MOMENTUM = "momentum"
COMBINE_BREAKOUT = "breakout"
COMBINE_AND = "and"
COMBINE_OR = "or"
COMBINATIONS = (COMBINE_MOMENTUM, COMBINE_BREAKOUT, COMBINE_AND, COMBINE_OR)

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
    # Portfolio
    initial_capital: float = 1000.0
    position_size_pct: float = 0.25
    max_positions: int = 4
    allow_fractions: bool = False
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

    @property
    def uses_momentum(self) -> bool:
        return self.combination in (COMBINE_MOMENTUM, COMBINE_AND, COMBINE_OR)

    @property
    def uses_breakout(self) -> bool:
        return self.combination in (COMBINE_BREAKOUT, COMBINE_AND, COMBINE_OR)

    @property
    def longest_period(self) -> int:
        return max(self.momentum_period, self.breakout_period)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

"""Stop loss and trailing stop levels."""
from __future__ import annotations

from .params import StrategyParams

REASON_STOP_LOSS = "Stop loss"
REASON_TRAILING = "Stop suiveur"


def stop_level(entry_price: float, highest: float, params: StrategyParams) -> tuple[float | None, str]:
    """Stop applicable from the NEXT day, given the highest price reached so far.

    If both stops are active the higher level wins. Returns (None, "") without any stop.
    """
    level, reason = None, ""
    if params.stop_loss_enabled:
        level, reason = entry_price * (1.0 - params.stop_loss_pct), REASON_STOP_LOSS
    if params.trailing_stop_enabled:
        trailing = highest * (1.0 - params.trailing_stop_pct)
        if level is None or trailing > level:
            level, reason = trailing, REASON_TRAILING
    return level, reason

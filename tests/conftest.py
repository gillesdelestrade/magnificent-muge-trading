import pandas as pd
import pytest


def make_prices(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close) on consecutive business days."""
    idx = pd.bdate_range(start, periods=len(rows), name="date")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype="float64")
    df["volume"] = 1000.0
    return df


def flat(n, price=100.0):
    return [(price, price, price, price)] * n


@pytest.fixture
def prices_factory():
    return make_prices

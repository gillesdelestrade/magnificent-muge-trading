"""French formatting of amounts, percentages and dates + CSV export."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from . import APP_NAME

NBSP = " "


def fmt_number(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{float(value):,.{decimals}f}".replace(",", NBSP).replace(".", ",")
    return "0" + text[2:] if text.startswith("-0") and float(text.replace(NBSP, "").replace(",", ".")) == 0 else text


def fmt_eur(value, decimals: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{fmt_number(value, decimals)}{NBSP}€"


def fmt_eur_signed(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return ("+" if value > 0 else "") + fmt_eur(value)


def fmt_pct(value, decimals: int = 2, signed: bool = False) -> str:
    """value is a fraction: 0.05 -> '5,00 %'."""
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{fmt_number(value * 100, decimals)}{NBSP}%"


def fmt_quantity(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return fmt_number(value, 0) if float(value).is_integer() else fmt_number(value, 4)


def fmt_date(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime("%d/%m/%Y")


def to_french_csv(df: pd.DataFrame, title: str = "", today: date | None = None) -> bytes:
    """CSV for Excel in French: ';' separator, decimal comma, UTF-8 with BOM.

    The first line is a comment with the application name and the date.
    """
    header = f"# {APP_NAME} — {title + ' — ' if title else ''}export du {fmt_date(today or date.today())}\n"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%d/%m/%Y")
    body = out.to_csv(sep=";", decimal=",", index=False, float_format="%.4f")
    return (header + body).encode("utf-8-sig")


def fmt_usd(value, decimals: int = 2) -> str:
    """API costs are billed in dollars."""
    return "—" if value is None or pd.isna(value) else f"{fmt_number(value, decimals)}{NBSP}$"

"""Plotly charts, themed like the logo (transparent background, discreet grid)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.universes import display_name
from ..engine.stats import drawdown
from ..formatting import fmt_eur, fmt_eur_signed, fmt_pct, fmt_quantity

BLUE = "#2EA8FF"
GOLD = "#D4A85A"
TEXT = "#E8EDF5"
MUTED = "#8A96AB"
GREEN = "#2ECC8F"
RED = "#E5484D"
SURFACE = "#0B1220"
GRID = "rgba(232,237,245,0.08)"
# Compared simulations: fixed order, each with its own dash so colour is never the only cue.
SERIES = [(BLUE, "solid"), (GOLD, "dash"), ("#B58CFF", "dot"), ("#F07FB0", "dashdot")]

PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), separators=", ",
        hovermode="x unified", hoverlabel=dict(bgcolor="#131C2E", font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickformat="%d/%m/%Y", hoverformat="%d/%m/%Y")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _trade_hover(t) -> str:
    side = "Achat" if t.side == "buy" else "Vente"
    lines = [
        f"<b>{side} — {display_name(t.ticker)}</b>",
        f"Motif : {t.reason}",
        f"{fmt_quantity(t.quantity)} × {fmt_eur(t.price)} = {fmt_eur(t.gross)}",
        f"Frais : {fmt_eur(t.fees)}",
    ]
    if t.side == "sell" and not pd.isna(t.pnl):
        lines.append(f"Plus/moins-value : {fmt_eur_signed(t.pnl)} ({fmt_pct(t.pnl_pct, signed=True)})")
    return "<br>".join(lines)


def _markers(trades: pd.DataFrame, side: str, y, name: str) -> go.Scatter:
    sub = trades[trades["side"] == side]
    is_buy = side == "buy"
    return go.Scatter(
        x=sub["date"], y=y(sub), mode="markers", name=name,
        marker=dict(symbol="triangle-up" if is_buy else "triangle-down", size=11,
                    color=GREEN if is_buy else RED, line=dict(color=SURFACE, width=2)),
        text=[_trade_hover(t) for t in sub.itertuples()],
        hovertemplate="%{text}<extra></extra>",
    )


def equity_chart(equity: pd.DataFrame, benchmark: pd.Series, trades: pd.DataFrame,
                 benchmark_name: str) -> go.Figure:
    """Portfolio value vs benchmark with buy/sell markers, and the drawdown underneath."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)
    if not benchmark.empty:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark, name=f"Indice {benchmark_name}", mode="lines",
            line=dict(color=MUTED, width=2, dash="dash"),
            hovertemplate="%{y:,.2f} €<extra>Indice</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["value"], name="Portefeuille", mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{y:,.2f} €<extra>Portefeuille</extra>"), row=1, col=1)
    if not trades.empty:
        value_at = lambda sub: equity["value"].reindex(sub["date"]).to_numpy()  # noqa: E731
        fig.add_trace(_markers(trades, "buy", value_at, "Achats"), row=1, col=1)
        fig.add_trace(_markers(trades, "sell", value_at, "Ventes"), row=1, col=1)
    dd = drawdown(equity["value"]) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Perte depuis le sommet", mode="lines", fill="tozeroy",
        line=dict(color=RED, width=1), fillcolor="rgba(229,72,77,0.25)",
        hovertemplate="%{y:.2f} %<extra>Perte depuis le sommet</extra>"), row=2, col=1)
    fig.update_yaxes(ticksuffix=" €", row=1, col=1)
    fig.update_yaxes(ticksuffix=" %", title_text="Perte max.", row=2, col=1)
    return _style(fig, 620)


def allocation_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["invested"], name="Investi", mode="lines", stackgroup="one",
        line=dict(color=BLUE, width=1), fillcolor="rgba(46,168,255,0.55)",
        hovertemplate="%{y:,.2f} €<extra>Investi</extra>"))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["cash"], name="Liquidités", mode="lines", stackgroup="one",
        line=dict(color=GOLD, width=1), fillcolor="rgba(212,168,90,0.45)",
        hovertemplate="%{y:,.2f} €<extra>Liquidités</extra>"))
    fig.update_yaxes(ticksuffix=" €")
    return _style(fig, 340)


def stock_chart(prices: pd.DataFrame, trades: pd.DataFrame, stops: pd.DataFrame,
                title: str, candles: bool = True) -> go.Figure:
    """Price of one stock with buy/sell points and the successive stop levels."""
    fig = go.Figure()
    if candles:
        fig.add_trace(go.Candlestick(
            x=prices.index, open=prices["open"], high=prices["high"], low=prices["low"],
            close=prices["close"], name="Cours",
            increasing_line_color=GREEN, decreasing_line_color=RED,
            increasing_fillcolor=GREEN, decreasing_fillcolor=RED))
    else:
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices["close"], name="Cours de clôture", mode="lines",
            line=dict(color=TEXT, width=2), hovertemplate="%{y:,.2f} €<extra>Clôture</extra>"))
    if not stops.empty:
        # One segment per holding period: break the line between two positions.
        xs, ys = [], []
        buys = trades.loc[trades["side"] == "buy", "date"].tolist()
        for i, buy_date in enumerate(buys):
            until = buys[i + 1] if i + 1 < len(buys) else pd.Timestamp.max
            seg = stops[(stops["date"] > buy_date) & (stops["date"] <= until)]
            xs += seg["date"].tolist() + [None]
            ys += seg["stop"].tolist() + [None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="Niveau de stop", mode="lines", connectgaps=False,
            line=dict(color=GOLD, width=2, dash="dot", shape="hv"),
            hovertemplate="%{y:,.2f} €<extra>Stop</extra>"))
    if not trades.empty:
        price_of = lambda sub: sub["price"]  # noqa: E731
        fig.add_trace(_markers(trades, "buy", price_of, "Achats"))
        fig.add_trace(_markers(trades, "sell", price_of, "Ventes"))
    fig.update_layout(title=dict(text=title, font=dict(color=GOLD, size=16)),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(ticksuffix=" €")
    fig = _style(fig, 520)
    fig.update_layout(margin=dict(t=70))
    return fig


def comparison_chart(curves: dict[str, pd.Series]) -> go.Figure:
    """Superimposed portfolio values of 2 to 4 simulations."""
    fig = go.Figure()
    for (name, values), (color, dash) in zip(curves.items(), SERIES):
        fig.add_trace(go.Scatter(
            x=values.index, y=values, name=name, mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{y:,.2f} €<extra>" + name + "</extra>"))
    fig.update_yaxes(ticksuffix=" €")
    return _style(fig, 480)


# -- News & AI ------------------------------------------------------------------------

NEUTRAL = "#3A4458"   # midpoint of the diverging scale (negative red -> neutral -> positive green)
DIVERGING = [[0.0, RED], [0.5, NEUTRAL], [1.0, GREEN]]


def sentiment_chart(prices: pd.DataFrame, daily: pd.DataFrame, hover: pd.Series,
                    title: str) -> go.Figure:
    """Price of a stock with the daily news sentiment underneath (bars).

    ``hover``: text shown for each day (titles and summaries of the day's articles).
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["close"], name="Cours de clôture", mode="lines",
        line=dict(color=BLUE, width=2), hovertemplate="%{y:,.2f} €<extra>Clôture</extra>"),
        row=1, col=1)
    colors = [GREEN if s >= 0 else RED for s in daily["sentiment"]]
    fig.add_trace(go.Bar(
        x=daily["day"], y=daily["sentiment"], name="Sentiment du jour", marker_color=colors,
        marker_line_width=0, text=hover.reindex(daily["day"]).fillna("").to_numpy(),
        textposition="none",
        hovertemplate="Sentiment %{y:.2f}<br>%{text}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(ticksuffix=" €", row=1, col=1)
    fig.update_yaxes(range=[-1.05, 1.05], title_text="Sentiment", row=2, col=1)
    fig.update_layout(title=dict(text=title, font=dict(color=GOLD, size=16)), showlegend=False,
                      bargap=0.2)
    fig = _style(fig, 560)
    fig.update_layout(margin=dict(t=60), hovermode="closest")
    return fig


def scatter_chart(x: pd.Series, y: pd.Series, y_label: str) -> go.Figure:
    """Sentiment vs following return, with the least-squares trend line."""
    import numpy as np

    data = pd.DataFrame({"x": x, "y": y * 100}).dropna()
    fig = go.Figure(go.Scatter(
        x=data["x"], y=data["y"], mode="markers", name="Jours avec actualités",
        marker=dict(color=BLUE, size=8, opacity=0.55, line=dict(color=SURFACE, width=1)),
        hovertemplate="Sentiment %{x:.2f}<br>Rendement %{y:.2f} %<extra></extra>"))
    if len(data) >= 2 and data["x"].nunique() > 1:
        slope, intercept = np.polyfit(data["x"], data["y"], 1)
        xs = np.array([data["x"].min(), data["x"].max()])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines", name="Tendance",
            line=dict(color=GOLD, width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text="Sentiment du jour", tickformat=None, hoverformat=None,
                     zeroline=True, zerolinecolor=GRID)
    fig.update_yaxes(title_text=y_label, ticksuffix=" %", zeroline=True, zerolinecolor=GRID)
    fig = _style(fig, 420)
    fig.update_xaxes(tickformat=".1f", hoverformat=".2f")
    fig.update_layout(hovermode="closest")
    return fig


def event_chart(study: pd.DataFrame, counts: dict[str, int]) -> go.Figure:
    """Mean cumulative abnormal return around strong news (positive green, negative red)."""
    fig = go.Figure()
    for column, name, color, dash in (("positive", "Informations positives", GREEN, "solid"),
                                      ("negative", "Informations négatives", RED, "dash")):
        if study[column].notna().any():
            fig.add_trace(go.Scatter(
                x=study.index, y=study[column] * 100, mode="lines",
                name=f"{name} ({counts.get(column, 0)})",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate="%{y:.2f} %<extra>" + name + "</extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
    fig.update_yaxes(title_text="Rendement anormal cumulé", ticksuffix=" %",
                     zeroline=True, zerolinecolor=GRID)
    fig = _style(fig, 420)
    fig.update_xaxes(title_text="Jours de bourse autour de l'information (J = 0)",
                     tickformat="d", hoverformat="d")
    return fig


def heatmap_chart(corr: pd.DataFrame, counts: pd.DataFrame, row_labels: dict[str, str]) -> go.Figure:
    """Correlation by news category x horizon; cells with too few observations are empty."""
    text = [["n < 30" if pd.isna(corr.iloc[i, j]) else f"{corr.iloc[i, j]:+.2f}".replace(".", ",")
             for j in range(corr.shape[1])] for i in range(corr.shape[0])]
    hover = [[f"{int(counts.iloc[i, j])} articles" for j in range(corr.shape[1])]
             for i in range(corr.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=[row_labels.get(r, r) for r in corr.index],
        zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        text=text, texttemplate="%{text}", textfont=dict(color=TEXT), customdata=hover,
        hovertemplate="%{y} · %{x}<br>Corrélation %{text}<br>%{customdata}<extra></extra>",
        colorbar=dict(title="Corrélation", thickness=12)))
    fig = _style(fig, max(260, 60 + 40 * len(corr)))
    fig.update_xaxes(tickformat=None, hoverformat=None, side="top", showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(hovermode="closest")
    return fig

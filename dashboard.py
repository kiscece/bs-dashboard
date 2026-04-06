"""
Black-Scholes Interactive Dashboard
====================================
Run with:  python dashboard.py
Then open: http://localhost:8050 in your browser

Dependencies:
    pip install dash dash-bootstrap-components plotly yfinance scipy numpy pandas
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

import yfinance as yf

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

from bs_solver import solve_bs
from calibrations import fetch_and_calibrate, bs_price
from greeks import compute_greeks

# ══════════════════════════════════════════════════════════════
#  DASH APP
# ══════════════════════════════════════════════════════════════

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                title="BS Dashboard")
server = app.server  # Expose the Flask server instance

# ── Colour palette ───────────────────────────────────────────
C = dict(bg="#0f1117", card="#1a1d27", border="#2a2d3e",
         accent="#00d4aa", accent2="#7c6af7",
         text="#e8eaf0", muted="#8b90a0",
         call="#4fc3f7", put="#f48fb1", model="#00d4aa")

card_style = dict(background=C["card"], border=f"1px solid {C['border']}",
                  borderRadius="12px", padding="20px", marginBottom="16px")

input_style = dict(background=C["bg"], border=f"1px solid {C['border']}",
                   borderRadius="8px", color=C["text"], padding="8px 12px",
                   width="100%", fontSize="14px")

label_style = dict(color=C["muted"], fontSize="12px",
                   fontWeight="500", marginBottom="6px", display="block",
                   letterSpacing="0.06em", textTransform="uppercase")

btn_style = dict(background=C["accent"], border="none", borderRadius="8px",
                 color="#0f1117", fontWeight="600", padding="12px 28px",
                 fontSize="14px", cursor="pointer", width="100%",
                 letterSpacing="0.04em")

app.layout = html.Div(style=dict(background=C["bg"], minHeight="100vh",
                                  fontFamily="'DM Sans', sans-serif",
                                  color=C["text"], padding="32px"), children=[

    html.Link(rel="stylesheet",
          href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap"),

    # ── Header ───────────────────────────────────────────────
    html.Div(style=dict(marginBottom="32px"), children=[
        html.Div(style=dict(display="flex", alignItems="baseline", gap="12px"), children=[
            html.H1("Black-Scholes", style=dict(margin=0, fontSize="28px",
                    fontWeight="300", color=C["text"], letterSpacing="-0.5px")),
            html.Span("PDE Solver", style=dict(fontSize="28px", fontWeight="600",
                      color=C["accent"])),
        ]),
        html.P("Rannacher scheme · real-world calibration · interactive",
               style=dict(color=C["muted"], fontSize="13px", margin="4px 0 0")),
    ]),

    dbc.Row([

        # ── Left panel : inputs ───────────────────────────────
        dbc.Col(width=3, children=[
            html.Div(style=card_style, children=[
                html.P("Market data", style={**label_style, "marginBottom":"14px",
                       "color": C["accent"], "fontSize":"11px"}),

                html.Label("Ticker", style=label_style),
                dcc.Input(id="ticker", value="AAPL", type="text",
                          style=input_style, debounce=True),

                html.Div(style=dict(marginTop="16px")),
                html.Button("Fetch & Calibrate", id="btn-calibrate", n_clicks=0,
                            style=btn_style),

                html.Div(id="calib-status",
                         style=dict(marginTop="12px", fontSize="12px",
                                    color=C["muted"], minHeight="18px")),
            ]),

            html.Div(style=card_style, children=[
                html.P("PDE parameters", style={**label_style, "marginBottom":"14px",
                       "color": C["accent2"], "fontSize":"11px"}),

                html.Label("Option type", style=label_style),
                dcc.Dropdown(id="option-type",
                             options=[{"label":"Call","value":"call"},
                                      {"label":"Put", "value":"put"}],
                             value="call", clearable=False,
                             style=dict(background=C["bg"], color=C["text"],
                                        border=f"1px solid {C['border']}",
                                        borderRadius="8px")),

                html.Div(style=dict(marginTop="14px")),
                html.Label("Strike K", style=label_style),
                dcc.Input(id="strike", value=150, type="number", style=input_style),

                html.Div(style=dict(marginTop="14px")),
                html.Label("Maturity T (years)", style=label_style),
                dcc.Input(id="maturity", value=1.0, type="number",
                          step=0.1, style=input_style),

                html.Div(style=dict(marginTop="14px")),
                html.Label("Risk-free rate r", style=label_style),
                dcc.Input(id="rate", value=0.05, type="number",
                          step=0.01, style=input_style),

                html.Div(style=dict(marginTop="14px")),
                html.Label("Volatility σ (override calibration)", style=label_style),
                dcc.Input(id="sigma-override", value="", type="text",
                          placeholder="auto from calibration",
                          style={**input_style, "color": C["muted"]}),

                html.Div(style=dict(marginTop="20px"),
                         children=[html.P("Grid resolution", style=label_style)]),
                html.Label(id="ns-label", style={**label_style, "textTransform":"none",
                           "fontSize":"13px"}),
                dcc.Slider(id="ns", min=50, max=400, step=50, value=200,
                           marks={50:"50", 200:"200", 400:"400"},
                           tooltip={"always_visible":False}),

                html.Div(style=dict(marginTop="12px")),
                html.Label(id="nt-label", style={**label_style, "textTransform":"none",
                           "fontSize":"13px"}),
                dcc.Slider(id="nt", min=50, max=400, step=50, value=200,
                           marks={50:"50", 200:"200", 400:"400"},
                           tooltip={"always_visible":False}),

                html.Div(style=dict(marginTop="20px")),
                html.Button("Run PDE Solver", id="btn-solve", n_clicks=0,
                            style={**btn_style, "background": C["accent2"]}),
            ]),
        ]),

        # ── Right panel : plots ───────────────────────────────
        dbc.Col(width=9, children=[

            # Metric cards row
            html.Div(id="metrics-row", style=dict(
                display="grid",
                gridTemplateColumns="repeat(4, 1fr)",
                gap="12px", marginBottom="16px"
            )),

            # Tabs
            dcc.Tabs(id="tabs", value="smile", style=dict(
                background="transparent", border="none"),
                children=[
                    dcc.Tab(label="Vol smile", value="smile",
                            style=dict(background="transparent",
                                       color=C["muted"], border="none",
                                       padding="10px 20px"),
                            selected_style=dict(background=C["card"],
                                                color=C["accent"],
                                                border=f"1px solid {C['border']}",
                                                borderRadius="8px 8px 0 0",
                                                padding="10px 20px")),
                    dcc.Tab(label="PDE price surface", value="surface",
                            style=dict(background="transparent",
                                       color=C["muted"], border="none",
                                       padding="10px 20px"),
                            selected_style=dict(background=C["card"],
                                                color=C["accent"],
                                                border=f"1px solid {C['border']}",
                                                borderRadius="8px 8px 0 0",
                                                padding="10px 20px")),
                    dcc.Tab(label="Price & error", value="price",
                            style=dict(background="transparent",
                                       color=C["muted"], border="none",
                                       padding="10px 20px"),
                            selected_style=dict(background=C["card"],
                                                color=C["accent"],
                                                border=f"1px solid {C['border']}",
                                                borderRadius="8px 8px 0 0",
                                                padding="10px 20px")),
                    dcc.Tab(label="Greeks", value="greeks",
                            style=dict(background="transparent",
                                       color=C["muted"], border="none",
                                       padding="10px 20px"),
                            selected_style=dict(background=C["card"],
                                                color=C["accent"],
                                                border=f"1px solid {C['border']}",
                                                borderRadius="8px 8px 0 0",
                                                padding="10px 20px")),
                ]),

            html.Div(id="tab-content",
                     style={**card_style, "borderRadius":"0 12px 12px 12px",
                            "marginTop":"0", "minHeight":"480px"}),
        ]),
    ]),

    # Hidden stores
    dcc.Store(id="store-calib"),
    dcc.Store(id="store-pde"),
])


# ── Plot theme helper ─────────────────────────────────────────
def fig_layout(fig, title=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text"], size=12),
        title=dict(text=title, font=dict(size=14, color=C["muted"])),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                    borderwidth=1),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    )
    return fig


def metric_card(label, value, color=None):
    return html.Div(style=dict(
        background=C["card"], border=f"1px solid {C['border']}",
        borderRadius="10px", padding="14px 16px"), children=[
        html.P(label, style=dict(color=C["muted"], fontSize="11px",
               fontWeight="500", margin="0 0 6px",
               letterSpacing="0.06em", textTransform="uppercase")),
        html.P(value, style=dict(color=color or C["text"], fontSize="20px",
               fontWeight="500", margin="0", fontFamily="DM Mono")),
    ])


# ── Callback : calibration ────────────────────────────────────
@app.callback(
    Output("store-calib", "data"),
    Output("calib-status", "children"),
    Output("strike", "value"),
    Input("btn-calibrate", "n_clicks"),
    State("ticker", "value"),
    prevent_initial_call=True,
)
def run_calibration(_, ticker):
    try:
        res = fetch_and_calibrate(ticker.upper().strip(), min_days=30)
        status = (f"Calibrated  σ = {res['sigma']*100:.1f}%  |  "
                  f"{res['n_calls']} calls, {res['n_puts']} puts  |  "
                  f"expiry {res['expiry']}")
        res["calls"] = res["calls"].to_dict("records")
        res["puts"]  = res["puts"].to_dict("records")
        return res, status, round(res["S"])
    except Exception as e:
        traceback.print_exc()   # ← prints full traceback to terminal
        return None, f"Error: {e}", dash.no_update

# ── Callback : slider labels ──────────────────────────────────
@app.callback(
    Output("ns-label", "children"),
    Output("nt-label", "children"),
    Input("ns", "value"),
    Input("nt", "value"),
)
def update_labels(ns, nt):
    return f"Price steps Ns = {ns}", f"Time steps Nt = {nt}"


# ── Callback : PDE solve ──────────────────────────────────────
@app.callback(
    Output("store-pde", "data"),
    Input("btn-solve", "n_clicks"),
    State("store-calib", "data"),
    State("option-type", "value"),
    State("strike", "value"),
    State("maturity", "value"),
    State("rate", "value"),
    State("sigma-override", "value"),
    State("ns", "value"),
    State("nt", "value"),
    prevent_initial_call=True,
)
def run_pde(_, calib, option_type, K, T, r, sigma_override, Ns, Nt):
    if calib is None:
        return None
    S_spot = calib["S"]
    sigma  = float(sigma_override) if sigma_override else calib["sigma"]
    K      = float(K);  T = float(T);  r = float(r)
    Smax   = max(4 * K, 4 * S_spot)

    S_grid, t_grid, V_all = solve_bs(Smax, K, T, r, sigma, Ns, Nt, option_type)
    dS = Smax / Ns

    V0     = V_all[0]
    V_anal = bs_price(S_grid, K, T, r, sigma, option_type)
    delta  = np.full_like(V0, np.nan)
    gamma  = np.full_like(V0, np.nan)
    delta[1:-1] = (V0[2:] - V0[:-2]) / (2*dS)
    gamma[1:-1] = (V0[2:] - 2*V0[1:-1] + V0[:-2]) / dS**2

    idx_atm = int(np.argmin(np.abs(S_grid - S_spot)))

    return dict(
        S=S_grid.tolist(), t=t_grid.tolist(),
        V_surface=V_all.tolist(),
        V0=V0.tolist(), V_anal=V_anal.tolist(),
        delta=delta.tolist(), gamma=gamma.tolist(),
        sigma=sigma, K=K, T=T, r=r,
        S_spot=S_spot, idx_atm=idx_atm,
        option_type=option_type,
    )


# ── Callback : render tab ─────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Output("metrics-row", "children"),
    Input("tabs", "value"),
    Input("store-calib", "data"),
    Input("store-pde", "data"),
)
def render_tab(tab, calib, pde):
    metrics = []

    if calib:
        metrics += [
            metric_card("Spot price",  f"${calib['S']:.2f}", C["call"]),
            metric_card("Cal. sigma",  f"{calib['sigma']*100:.1f}%", C["accent"]),
            metric_card("Expiry",       calib["expiry"]),
        ]
    if pde:
        V0 = np.array(pde["V0"]);  S = np.array(pde["S"])
        idx = pde["idx_atm"]
        metrics.append(metric_card("ATM price",
                                   f"${V0[idx]:.4f}", C["accent2"]))

    # ── Vol smile ─────────────────────────────────────────────
    if tab == "smile":
        if calib is None:
            return html.P("Fetch market data first.",
                          style=dict(color=C["muted"], padding="40px")), metrics
        calls = pd.DataFrame(calib["calls"])
        puts  = pd.DataFrame(calib["puts"])
        fig   = go.Figure()
        if not calls.empty:
            fig.add_trace(go.Scatter(
                x=calls["strike"]/calib["S"], y=calls["iv"]*100,
                mode="markers+lines", name="Calls IV",
                marker=dict(color=C["call"], size=7),
                line=dict(color=C["call"], width=1.5)))
        if not puts.empty:
            fig.add_trace(go.Scatter(
                x=puts["strike"]/calib["S"], y=puts["iv"]*100,
                mode="markers+lines", name="Puts IV",
                marker=dict(color=C["put"], size=7),
                line=dict(color=C["put"], width=1.5)))
        fig.add_hline(y=calib["sigma"]*100, line_dash="dash",
                      line_color=C["accent"],
                      annotation_text=f"Flat σ = {calib['sigma']*100:.1f}%",
                      annotation_font_color=C["accent"])
        fig.add_vline(x=1.0, line_dash="dot", line_color=C["muted"])
        fig.update_xaxes(title="Moneyness  K / S")
        fig.update_yaxes(title="Implied volatility (%)")
        return dcc.Graph(figure=fig_layout(fig, "Volatility smile"),
                         config={"displayModeBar": False},
                         style=dict(height="440px")), metrics

    # ── PDE surface ───────────────────────────────────────────
    if tab == "surface":
        if pde is None:
            return html.P("Run the PDE solver first.",
                          style=dict(color=C["muted"], padding="40px")), metrics
        S = np.array(pde["S"]);  t = np.array(pde["t"])
        V = np.array(pde["V_surface"])
        mask = S <= 2 * pde["K"]
        fig = go.Figure(go.Surface(
            x=S[mask], y=t, z=V[:, mask],
            colorscale=[[0,"#1a1d27"],[0.5, C["accent2"]],[1, C["accent"]]],
            showscale=False))
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="S", backgroundcolor=C["bg"],
                           gridcolor=C["border"], color=C["muted"]),
                yaxis=dict(title="t", backgroundcolor=C["bg"],
                           gridcolor=C["border"], color=C["muted"]),
                zaxis=dict(title="V", backgroundcolor=C["bg"],
                           gridcolor=C["border"], color=C["muted"]),
                bgcolor=C["bg"],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=C["text"]),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False},
                         style=dict(height="440px")), metrics

    # ── Price & error ─────────────────────────────────────────
    if tab == "price":
        if pde is None:
            return html.P("Run the PDE solver first.",
                          style=dict(color=C["muted"], padding="40px")), metrics
        S = np.array(pde["S"]);  V0 = np.array(pde["V0"])
        Va = np.array(pde["V_anal"]);  K = pde["K"]
        mask = (S > 0.2*K) & (S < 2.5*K)
        err  = np.abs(V0 - Va)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Price at t=0", "Absolute error"])
        fig.add_trace(go.Scatter(x=S[mask], y=Va[mask], name="Analytical",
                                 line=dict(color=C["muted"], dash="dash", width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=S[mask], y=V0[mask], name="Rannacher",
                                 line=dict(color=C["accent"], width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=S[mask], y=err[mask], name="| error |",
                                 fill="tozeroy",
                                 line=dict(color=C["accent2"], width=1.5),
                                 fillcolor=f"rgba(124,106,247,0.15)"),
                      row=1, col=2)
        fig.update_xaxes(title_text="S", gridcolor=C["border"])
        fig.update_yaxes(title_text="V", gridcolor=C["border"], row=1, col=1)
        fig.update_yaxes(title_text="|error|", gridcolor=C["border"], row=1, col=2)
        return dcc.Graph(figure=fig_layout(fig), config={"displayModeBar": False},
                         style=dict(height="440px")), metrics

    # ── Greeks ────────────────────────────────────────────────
    if tab == "greeks":
        if pde is None:
            return html.P("Run the PDE solver first.",
                          style=dict(color=C["muted"], padding="40px")), metrics
        S = np.array(pde["S"]);  K = pde["K"]
        delta = np.array(pde["delta"]);  gamma = np.array(pde["gamma"])
        mask = (S > 0.3*K) & (S < 2.2*K)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Delta", "Gamma"])
        fig.add_trace(go.Scatter(x=S[mask], y=delta[mask],
                                 line=dict(color=C["call"], width=2), name="Δ"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=S[mask], y=gamma[mask],
                                 line=dict(color=C["put"], width=2), name="Γ"),
                      row=1, col=2)
        for col in [1, 2]:
            fig.add_vline(x=K, line_dash="dot", line_color=C["muted"], col=col, row=1)
        fig.update_xaxes(title_text="S", gridcolor=C["border"])
        fig.update_yaxes(gridcolor=C["border"])
        return dcc.Graph(figure=fig_layout(fig), config={"displayModeBar": False},
                         style=dict(height="440px")), metrics

    return html.Div(), metrics


if __name__ == "__main__":
    print("\n  Black-Scholes Dashboard")
    print("  Open http://localhost:8050 in your browser\n")
    app.run(debug=False)



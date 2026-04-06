import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from datetime import datetime


# ─────────────────────────────────────────────
#  Analytical BS — used inside the optimiser
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"): #Closed-form Black-Scholes price. Vectorised over K.
    """Closed-form Black-Scholes price. Vectorised over K."""
    S, K = np.asarray(S, float), np.asarray(K, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(
            (S > 0) & (T > 0) & (sigma > 0),
            (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)),
            -np.inf,
        )
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return np.where(S > 0, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)
    else:
        return np.where(S > 0, K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), K * np.exp(-r * T))


# ─────────────────────────────────────────────
#  Data fetching & cleaning
# ─────────────────────────────────────────────

def fetch_and_calibrate(ticker, min_days=7):
    stock  = yf.Ticker(ticker)
    S      = stock.fast_info["last_price"]
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    valid = []
    for e in stock.options:
        try:
            days = (datetime.strptime(e, "%Y-%m-%d") - today_midnight).days
            if days >= min_days:
                valid.append(e)
        except (ValueError, TypeError):
            continue   # skip anything that isn't a proper date string

    if not valid:
        raise ValueError("No valid expiries found")

    expiry = valid[0]
    T      = (datetime.strptime(expiry, "%Y-%m-%d") - today_midnight).days / 365.0
    chain  = stock.option_chain(expiry)
    r      = 0.05

    def clean(df, otype):
        df = df.copy()
        for col in ["bid", "ask", "strike", "lastPrice"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Use lastPrice as fallback when bid is zero (market closed)
        df["mid"] = np.where(
            df["bid"] > 0,
            (df["bid"] + df["ask"]) / 2,
            df["lastPrice"]
        )

        df = df[df["mid"] > 0]   # ← replaces df[df["bid"] > 0]
        df = df[(df["ask"] - df["bid"]).abs() / df["mid"].clip(lower=0.01) < 0.5]
        df = df[(df["strike"] > 0.6*S) & (df["strike"] < 1.4*S)]
        intr = np.maximum(S - df["strike"], 0) if otype == "call" else np.maximum(df["strike"] - S, 0)
        df = df[df["mid"] > intr + 0.01]
        df["T"] = T;  df["S"] = S;  df["r"] = r;  df["type"] = otype
        return df[["strike","mid","T","S","r","type"]].reset_index(drop=True)

    calls = clean(chain.calls, "call")
    puts  = clean(chain.puts,  "put")
    all_opts = pd.concat([calls, puts], ignore_index=True)

    # per-strike IV
    def iv_single(mkt, K, T, otype):
        obj = lambda s: (bs_price(float(S), float(K), float(T), float(r), s, otype) - mkt)**2
        res = minimize_scalar(obj, bounds=(1e-4, 5.0), method="bounded")
        return res.x if res.fun < 1e-4 else np.nan

    for df in [calls, puts]:
        df["iv"] = [iv_single(row.mid, row.strike, row["T"], row.type)
            for _, row in df.iterrows()]

    calls = calls.dropna(subset=["iv"])
    puts  = puts.dropna(subset=["iv"])

    # flat sigma calibration
    def mse(sigma):
        p = np.array([
            bs_price(float(row.S), float(row.strike), float(row["T"]),
                    float(row.r), sigma, row.type)
            for _, row in all_opts.iterrows()
            ])
        return np.mean((p - all_opts["mid"].values)**2)

    res = minimize_scalar(mse, bounds=(0.01, 2.0), method="bounded")
    sigma_cal = res.x

    return dict(S=S, T=T, r=r, expiry=expiry,
                sigma=sigma_cal, calls=calls, puts=puts,
                n_calls=len(calls), n_puts=len(puts))

# ─────────────────────────────────────────────
#  Per-strike implied volatility
# ─────────────────────────────────────────────

def implied_vol_single(market_price, S, K, T, r, option_type="call", tol=1e-6):
    """
    Find implied volatility for a single option via Brent's method.
    Returns NaN if no solution found.
    """
    def objective(sigma):
        return (bs_price(S, K, T, r, sigma, option_type) - market_price) ** 2

    try:
        result = minimize_scalar(objective, bounds=(1e-4, 5.0), method="bounded",
                                 options={"xatol": tol})
        if result.fun < 1e-6:   # residual small enough → good fit
            return result.x
        return np.nan
    except Exception:
        return np.nan


def compute_iv_surface(df):
    """Compute implied vol for every row of a calls or puts DataFrame."""
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol_single(
            row["mid"], row["S"], row["strike"], row["T"], row["r"], row["type"]
        )
        ivs.append(iv)
    df = df.copy()
    df["iv"] = ivs
    return df.dropna(subset=["iv"])


# ─────────────────────────────────────────────
#  Flat vol calibration (single sigma for all strikes)
# ─────────────────────────────────────────────

def calibrate_flat_vol(df):
    """
    Find a single sigma that minimises MSE across all market prices.
    This is the simplest calibration — one number for the whole surface.
    """
    S_arr = df["S"].values
    K_arr = df["strike"].values
    T_arr = df["T"].values
    r_arr = df["r"].values
    mkt   = df["mid"].values
    types = df["type"].values

    def mse(sigma):
        prices = np.array([
            bs_price(S_arr[i], K_arr[i], T_arr[i], r_arr[i], sigma, types[i])
            for i in range(len(df))
        ])
        return np.mean((prices - mkt) ** 2)

    result = minimize_scalar(mse, bounds=(0.01, 2.0), method="bounded")
    return result.x, result.fun

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = fetch_and_calibrate("AAPL")
    print(f"Calibrated sigma: {result['sigma']*100:.1f}%")
    print(f"Expiry: {result['expiry']}")
    print(f"Calls: {result['n_calls']}, Puts: {result['n_puts']}")
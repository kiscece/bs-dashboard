import numpy as np
import pandas as pd
from calibrations import bs_price

def compare_markets(calib, pde_S, pde_V0):
    """
    For each traded strike, compute model price vs market price.
    
    Returns a DataFrame with columns:
    strike, market_price, model_price, error, error_pct, option_type
    """
    rows = []
    
    for record in calib["calls"] + calib["puts"]:
        K      = record["strike"]
        mkt    = record["mid"]
        otype  = record["type"]
        S      = record["S"]
        T      = record["T"]
        r      = record["r"]
        sigma  = calib["sigma"]   # flat calibrated vol
        
        # Model price using analytical BS at calibrated sigma
        model = bs_price(S, K, T, r, sigma, otype)
        
        error     = model - mkt
        error_pct = error / mkt * 100 if mkt > 0.01 else np.nan
        
        rows.append(dict(strike=K, market_price=mkt,
                         model_price=float(model),
                         error=float(error),
                         error_pct=float(error_pct) if not np.isnan(error_pct) else None,
                         option_type=otype))
    
    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
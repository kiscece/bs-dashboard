# Black-Scholes PDE Solver 
An interactive dashboard for pricing European options by solving the Black-Scholes PDE numerically, calibrated on live market data. 

## What this does 

Most option pricing tools either use the closed-form Black-Scholes formula directly, or run Monte Carlo simulations. This project takes a third route : it discretises the Black-Scholes PDE on a finite difference grid and solves it backwards in time using the **Rannacher scheme**, a numerical method used in production quant systems to handle the non-smooth payoff at expiry. 

The volatility parameter $\sigma$ is not chosen arbitrarily. It is calibrated from **real market data** : the dashboard downloads live option chains from Yahoo Finance, computes the implied volatility of each traded option, and finds the flat $\sigma$ that best fits observed market prices in the least-square sense. 

---

## The Mathematics 
The Black-Scholes PDE for a European option price V(S,t) is :  
>$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} -rV = 0$  

with terminal condition $V(S,t) = \max{(S-K,0)}$ for a call, or $V(S,t) = \max{(K-S,0)}$ for a put, and Dirichlet boundary conditions at $S=0$ and $S= S_{max}$.  

This is structurally identical to the heat equation from physics - the term $\frac{1}{2}\sigma^2 S^2$ plays the role of a spatially varying diffusion coefficient.  

### Discretisation : the Rannacher scheme 
Crank-Nicolson (CN) achieves second-order accuracy in time, but produces oscillations near the strike K when the payoff has a discontinuous derivative. The **Rannacher** fix is simple : apply 2-4 fully implicit steps first (which work well for oscillations), the switch to CN for the remaining steps. This recovers second-order accuracy while eleminating the oscillations.  

At each time step the fully implicit scheme produces the linear system :  
>$ A \textbf{V}_{new} = \textbf{V}_{old}$,  

where $\textbf{V}_{new}$ if the option price at time $t$ and $\textbf{V}_{old}$ is the option price at the previous time (known). For the following time-steps, the CN scheme produces a tridiagonal linear system :  
>$A_{impl} \textbf{V}_{new} = A_{exp} \textbf{V}_{old} + \text{boundary terms}$,  

which is solved with a hand-implemented **Conjugate Gradient** solver, appropriate because the matrix is symmetric positive definite.


### Calibration 
For each traded option with market price $C_{mkt}$ we find the implied volatility $\sigma_i$ that satisfies :  
> $BS(S, K_i, T, r, \sigma_i) = C_{mkt,i}$    

via Brent's method. The flat calibrated $\sigma$ minimises the mean squared error across all strikes :  
>$\sigma^{*} = \argmin{\left[ \sum (BS(S, K_i, T, r, \sigma) - C_{mkt,i})^2 \right]} $

--- 

## Features 
- **Rannacher PDE solver** : second-order accurate in time and space, oscillation-free  
- **Live market calibration** : fetches real option chains from Yahoo Finance  
- **Interactive dashboard** : adjustable strike, maturity, grid resolution  
- **Volatility smile** : per-strike implied volatility vs moneyness  
- **3D price surface** : V(S,t) rendered as an interactive surface  
- **Greeks** : Delta and Gamma computed by fine differences on the grid  
- **Convergence Analysis** : empirical verification of second-order convergence  
- **Works outside market hours** : falls back to last traded prices automatically  
- **Global markets** : supports any ticker on Yahoo Finance (US, European, Asian) 

---

## Supported tickers

| Market | Examples |
|--------|---------|
| US | AAPL, TSLA, META, NVDA, SPY |
| Europe | MC.PA (LVMH), AIR.PA (Airbus), ASML.AS (ASML) |
| Asia | 9984.T (SoftBank), 0700.HK (Tencent), 005930.KS (Samsung) |

---

## Installation

```bash
git clone https://github.com/your-repo.git
cd your-repo
pip install -r requirements.txt
python dashboard.py 
```

Then open http://localhost:8050 in your browser.

### Dependencies

```shell
dash
dash-bootstrap-components
plotly
yfinance
scipy
numpy
pandas
gunicorn
```
---

## Project Strucutre 

``` bs-dashboard/
├── dashboard.py      # Dash app — layout, callbacks, plots
├── bs_solver.py      # Rannacher PDE solver + CG solver
├── calibrations.py   # Market data fetching + IV calibration
├── greeks.py         # Delta, Gamma, Theta
├── convergence.py    # Convergence analysis vs analytical solution
├── requirements.txt
└── render.yaml       # Render deployment config
```

---

## How to use 

1. Enter a ticker and click **Fetch & Calibrate**, it then downloads live data and calibrates $\sigma$  
2. Adjust strike $K$, maturity $T$, and grid resolution with the sliders  
3. Click **Run PDE Solver**, it solves the PDE with the calibrated $\sigma$  
4. Explore the four tabs : volatility smile, price surface, price vs analytical, Greeks.  

--- 

## Limitations and honest caveats

- **Flat volatility** : the calibration fits a single $\sigma$ across all strikes. Real market exhibit a volatility smile that a flat BS model cannot capture. *Extensions* : local vol (Dupire), stochastic vol (Heston).  
- **European options only** : the PDE approach here assumes no early exercise. American options require a free boundary (linear complementarity) extension. 
- **Constant parameters** : r and $\sigma$ are assumed constant. In practise both are term-structure dependent.  
- **Data quality** : outside market hours, last traded prices are used as a proxy for fair value. These can be stale for illiquid strikes.  

--- 

## References

- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy.
- Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. Numerische Mathematik.
- Wilmott, P., Howison, S. & Dewynne, J. (1995). The Mathematics of Financial Derivatives. Cambridge University Press.


## Author 
Independent project developed outside of coursework.  

Originally built upon numerical methods implemented for solving physical systems, this work extends those techniques to quantitative finance, with a focus on the Black–Scholes PDE and option pricing.

## Notes

AI tools were used as a support for debugging, structuring, and exploring implementation ideas. All numerical methods and final implementations were developed and validated independently.
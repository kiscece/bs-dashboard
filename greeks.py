import numpy as np

def compute_greeks(S, V_all, dS, dt):
    """
    Compute Delta and Gamma at t=0 from the numerical solution.
 
    Delta = ∂V/∂S  ≈ (V_{i+1} - V_{i-1}) / (2 dS)
    Gamma = ∂²V/∂S² ≈ (V_{i+1} - 2V_i + V_{i-1}) / dS²
 
    Returns arrays of same length as S (NaN at boundaries).
    """

    V0 = V_all[0, :]  # option values at t=0, shape (Ns+1,)
    delta = np.full_like(V0, np.nan)
    gamma = np.full_like(V0, np.nan)
 
    #finite difference approximations, along S
    delta[1:-1] = (V0[2:] - V0[:-2]) / (2 * dS)  
    gamma[1:-1] = (V0[2:] - 2 * V0[1:-1] + V0[:-2]) / dS**2

    #finite difference approx along t 
    theta = np.full_like(V0, np.nan)
    theta = ((V_all[1, :] - V_all[0, :]) / dt)/365  # daily
    
 
    return delta, gamma, theta
 
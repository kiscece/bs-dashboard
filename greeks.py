import numpy as np

def compute_greeks(S: np.ndarray, V_all: np.ndarray, dS: float, dt: float):
    """ 
    Compute option Greeks (Delta, Gamma, Theta) from a finite-difference solution.

    Greeks are evaluated at t = 0 using central finite differences:

        Delta ≈ (V_{i+1} - V_{i-1}) / (2 dS)
        Gamma ≈ (V_{i+1} - 2V_i + V_{i-1}) / dS²
        Theta ≈ (V(t+dt) - V(t)) / dt

    Parameters:
        S      : array_like
            Asset price grid
    V_all  : ndarray
            Option values of shape (Nt+1, Ns+1)
        dS     : float
            Spatial step
        dt     : float
            Time step

    Returns:
        delta, gamma, theta : ndarray
            Arrays of same length as S (NaN at boundaries)
    """

    V0 = V_all[0, :] 
    # Boundaries left as NaN due to lack of neighboring points
    delta = np.full_like(V0, np.nan)
    gamma = np.full_like(V0, np.nan)
 
    delta[1:-1] = (V0[2:] - V0[:-2]) / (2 * dS)  
    gamma[1:-1] = (V0[2:] - 2 * V0[1:-1] + V0[:-2]) / dS**2

    # Forward difference in time (theta) 
    theta = np.full_like(V0, np.nan)
    theta = -((V_all[1, :] - V_all[0, :]) / dt)/365   #Negative sign ensures consistency with standard financial definition
    
 
    return delta, gamma, theta
 
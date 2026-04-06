import numpy as np
import scipy
import scipy.sparse as sp
from scipy.stats import norm
from scipy.sparse import diags

def build_impl_matrix(S, dS, dt, r, sigma) -> scipy.sparse.csr_matrix: #Implicit matrix for the fully implicit scheme, used in the first n_rannacher steps
    N = len(S)
    M = N-2
    lower = np.zeros(N-3)
    main  = np.zeros(N-2)
    upper = np.zeros(N-3)

    for k in range(M):
        i = k+1  # interior point
        main[k] = 1 + dt*(r + sigma**2*S[i]**2 / dS**2)
        if k > 0:
            lower[k-1] = -(dt/2)*(sigma**2*S[i]**2 / dS**2 - r*S[i]/dS)
        if k < M-1:
            upper[k] = -(dt/2)*(sigma**2*S[i]**2 / dS**2 + r*S[i]/dS)

    A = diags([lower, main, upper], offsets=[-1,0,1], format='csr')
    return A
    

def build_cn_matrix(S, dS, dt, r, sigma): #Build matrices for the Crank-Nicolson scheme, used in the later steps
    N = len(S)
    M = N - 2
    #implicit diagnonals 
    lower_i = np.zeros(M - 1)
    main_i  = np.zeros(M)
    upper_i = np.zeros(M - 1)
    
    #explicit diagonals 
    lower_e = np.zeros(M - 1)
    main_e  = np.zeros(M)
    upper_e = np.zeros(M - 1)
 
    for k in range(M):
        i = k + 1
        diff = 0.5 * sigma**2 * S[i]**2 / dS**2
        conv = 0.5 * r * S[i] / dS
 
        main_i[k]  = 1.0 + (dt / 2) * (2.0 * diff + r)
        main_e[k]  = 1.0 - (dt / 2) * (2.0 * diff + r)
 
        if k > 0:
            lower_i[k - 1] = -(dt / 2) * (diff - conv)
            lower_e[k - 1] =  (dt / 2) * (diff - conv)
        if k < M - 1:
            upper_i[k] = -(dt / 2) * (diff + conv)
            upper_e[k] =  (dt / 2) * (diff + conv)
 
    A_impl = diags([lower_i, main_i, upper_i], offsets=[-1, 0, 1], format="csr") #create implicit matrix
    A_expl = diags([lower_e, main_e, upper_e], offsets=[-1, 0, 1], format="csr") #create explicit matrix
    return A_impl, A_expl


def boundary_conditions(S, K, T, r, t_now, option_type): #Call the boundary conditions for the system, used in the solver to update the boundaries at each time step, Dirichlet boundary conditions are used for the standard BS model
    tau = T - t_now  # time to maturity
    Smax = S[-1]
 
    if option_type == "call": #if you can buy the option, the value at S=0 is 0, and at Smax is the discounted payoff of exercising immediately, which is max(Smax-K*exp(-r*tau),0)
        v_left  = 0.0
        v_right = max(Smax - K * np.exp(-r * tau), 0.0)
    else: #if you can sell the option, the value at S=0 is the discounted payoff of exercising immediately, which is max(K*exp(-r*tau)-0,0), and at Smax is 0
        v_left  = K * np.exp(-r * tau)
        v_right = 0.0
 
    return v_left, v_right

def add_bc_contribution(b, S, dS, dt, r, sigma, v_left, v_right, theta): # Add boundary condition contributions to the RHS vector, depends on the scheme used 
    diff_left  = 0.5 * sigma**2 * S[1]**2  / dS**2
    conv_left  = 0.5 * r * S[1]  / dS

    diff_right = 0.5 * sigma**2 * S[-2]**2 / dS**2
    conv_right = 0.5 * r * S[-2] / dS
 
    b[0]  += theta * dt * (diff_left  - conv_left)  * v_left
    b[-1] += theta * dt * (diff_right + conv_right) * v_right
 



##### Main solver function ######

def cg_solve(A, b, tol: float, maxiter: int):  #CG solver function, solves x for Ax = b
    n = len(b)
    x = np.zeros(n) #initial guess 
    r = b - A @ x #define the residual
    p = r.copy() # p_0 = r_0
    n_iter = 0 #initialisation of the number of iterations

    if np.allclose(b, 0): #checks if b is zero, "not" is more memory efficient for large arrays$
        return x, n_iter, np.linalg.norm(r) #all 0
    
    else: 
        for i in range(maxiter):
            n_iter += 1
            Ap = A @ p
            alpha = np.dot(r,r) / np.dot(p, Ap) #step size
            x += alpha * p 
            r_new = r- alpha*Ap #update residual
            
            if (np.linalg.norm(r_new) / np.linalg.norm(b)) < tol : #end condition
                return x, n_iter, np.linalg.norm(r_new)

            beta = np.dot(r_new,r_new) / np.dot(r,r)
            p = r_new + beta*p
            r = r_new 
        raise ValueError(f"CG did not converge within {maxiter} iterations") 


def solve_bs(Smax, K, T, r, sigma, Ns, Nt, option_type = "call", n_rannacher  = 2, tol=1e-10, maxiter=1000, store_all= True):
    #  Make sure the parameters are valid 
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if Smax <= K:
        raise ValueError(f"Smax ({Smax}) should be > K ({K}); try Smax = 3–5×K")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    #Create the grids
    dS = Smax / Ns
    dt = T  / Nt
    S  = np.linspace(0.0, Smax, Ns + 1)   # S[0]=0, S[Ns]=Smax
    t_grid = np.linspace(0.0, T, Nt + 1) #Time grid 

    # Boundary conditions at t=T (Create the payoff at maturity (terminal condition))
    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)
 
    V = payoff.copy()
 
    if store_all: #if we want to store the solution at all time steps, we create a 2D array to hold them, otherwise we only keep the current solution 
        V_all = np.zeros((Nt + 1, Ns + 1))
        V_all[Nt, :] = V
 
    #Build the matrices for implicit and CN schemes
    A_impl_fi = build_impl_matrix(S, dS, dt, r, sigma)   # fully implicit
    A_impl_cn, A_expl_cn = build_cn_matrix(S, dS, dt, r, sigma) #Crank-Nicolson scheme 
 
    # Time stepping loop (backward in time)
    for step in reversed(range(Nt)):          # step = n  →  solve for V^n
        t_now  = step * dt                    # current time t^n
        t_next = (step + 1) * dt             # t^{n+1}  (known)
 
        use_implicit = (Nt - 1 - step) < n_rannacher #condition to use the fully implicit scheme 

         #Get boundary conditions for current and next time steps
        v_left_now,  v_right_now  = boundary_conditions(S, K, T, r, t_now,  option_type)
        v_left_next, v_right_next = boundary_conditions(S, K, T, r, t_next, option_type)
 
        V_int = V[1:-1].copy()   # interior values at t^{n+1}
 
        if use_implicit: #solving for fully implicit scheme, first time steps to dampen the initial condition and avoid oscillations
            b = V_int.copy()
            add_bc_contribution(b, S, dS, dt, r, sigma,
                                 v_left_now, v_right_now, theta=1.0)
 
            V_new_int, _, _ = cg_solve(A_impl_fi, b, tol, maxiter)
 
        else: #solving for Crank-Nicolson scheme, later time steps for better accuracy
            b = A_expl_cn @ V_int
            add_bc_contribution(b, S, dS, dt, r, sigma,
                                 v_left_now,  v_right_now,  theta=0.5)
            add_bc_contribution(b, S, dS, dt, r, sigma,
                                 v_left_next, v_right_next, theta=0.5)
 
            V_new_int, _, _ = cg_solve(A_impl_cn, b, tol, maxiter)
 
        # Build full solution, add boundaries 
        V[0]    = v_left_now
        V[-1]   = v_right_now
        V[1:-1] = V_new_int
 
        if store_all:
            V_all[step, :] = V.copy() #copy in the solution at the current time step to the array that stores all solutions
 
    if store_all:
        return S, t_grid, V_all
    else:
        return S, t_grid, V[np.newaxis, :]
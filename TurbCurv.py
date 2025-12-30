import numpy as np
import scipy.linalg as la
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_bvp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu, LinearOperator, gmres, norm
from scipy.interpolate import interp1d

def chebyshev_grid(N, zi):
    """Generate Chebyshev grid and mapping to physical space z."""
    xi = np.cos(np.pi-np.pi * np.arange(N + 1) / N)  # Chebyshev points in [-1,1]
    z = (zi / 2) * (xi + 1)  # Map to [0, zi]
    return z

import numpy as np

def chebyshev_diff_matrix(N, zi):
    """Compute Chebyshev differentiation matrix using Trefethen's formula."""
    
    # Chebyshev nodes (roots of the Chebyshev polynomial of the first kind)
    xi = np.cos(np.pi - np.pi * np.arange(N + 1) / N)
    
    # Compute the differentiation matrix
    D = np.zeros((N + 1, N + 1))
    
    # Compute the weight factor for the differentiation
    C = np.ones(N + 1)
    C[0], C[-1] = 2, 2  # Boundary correction (for the first and last points)
    C = C * (-1) ** np.arange(N + 1)  # Alternating signs
    
    # Construct the differentiation matrix (using Trefethen's formula)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = C[i] / C[j] * (-1) ** (i + j) / (xi[i] - xi[j])
            else:
                D[i, i] = -np.sum(D[i, :])  # Sum of rows should be zero for spectral differentiation
    
    # Scale the differentiation matrix for the physical domain (assuming zi is scalar or vector)
    D = (2 / zi) * D  # This assumes 'zi' is properly defined as a scalar or array
    
    return D


def interpolate_to_chebyshev(z_old, f_old, z_cheb):
    """Interpolate a function defined on original z-grid to Chebyshev grid."""
    interpolator = interp1d(z_old, f_old, kind='cubic', fill_value='extrapolate')
    return interpolator(z_cheb)
    
def D_matrices(N, dz):
    """ Generate finite difference matrix for the second derivative """
    D  = np.zeros((N, N))
    D2 = np.zeros((N, N))
    D3 = np.zeros((N, N))
    for i in range(1, N-1):
        D[i,i-1] = -0.5/dz
        D[i,i+1] = 0.5/dz

        D2[i, i-1] = 1 / dz**2
        D2[i, i] = -2 / dz**2
        D2[i, i+1] = 1 / dz**2

    for i in range(2, N-2):
        D3[i, i-2] = -0.5 / dz**3
        D3[i, i-1] = 1/ dz**3
        D3[i,i]=0.
        D3[i, i+1] = 1 / dz**3
        D3[i, i+2] = 0.5 / dz**3

    return D,D2,D3

def D_matrices_non_uniform(z):
    """ Generate finite difference matrices for the first, second, and third derivatives on a non-uniform grid """
    N = len(z)
    D = np.zeros((N, N))
    D2 = np.zeros((N, N))
    D3 = np.zeros((N, N))
    
    dz = np.diff(z)

    # First derivative matrix D
    for i in range(1, N-1):
        dz_p = z[i+1]-z[i]
        dz_n = z[i]-z[i-1]
        
        D[i, i-1] = -1 / (dz_p+dz_n)
        D[i, i+1] =  1 / (dz_p+dz_n)

    # Second derivative matrix D2
    for i in range(1, N-1):
        dz_p = z[i+1]-z[i]
        dz_n = z[i]-z[i-1]
        
        D2[i, i-1] =  2 / (dz_n * (dz_n + dz_p))
        D2[i, i]   = -2 / (dz_n * dz_p)
        D2[i, i+1] =  2 / (dz_p * (dz_n + dz_p))

    # Third derivative matrix D3
    for i in range(2, N-2):
        dz_p = z[i+1]-z[i]
        dz_n =   z[i]-z[i-1]
        dz_p_2 = z[i+2]-z[i]
        dz_n_2 =   z[i]-z[i-2]
        
    return D, D2, D3


def apply_boundary_conditions(A, b, z_bc,dz):
    """ Apply boundary conditions to the matrix and RHS vector """
    b = b.astype(complex)
    
    print(A.dtype)
    print(b.dtype)
    
    # Apply dirichlet boundary conditions first
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = z_bc[0]

    A[-1, :] = 0
    A[-1,-1] = 1
    b[-1] = z_bc[2]

    # Apply Neumann boundary conditions next
    A[1, :] = 0
    A[1, 0] =-1/dz
    A[1, 1] = 1/dz
    b[1] = z_bc[1]

    A[-2, :] = 0
    A[-2, -2] = -1/dz
    A[-2, -1] =  1/dz
    b[-2] = z_bc[3]

    return A, b

def apply_boundary_conditions_non_uniform(A, b, z_bc, z):
    """ Apply boundary conditions to the matrix and RHS vector for a non-uniform grid """
    b = b.astype(complex)
    
    # Apply Dirichlet boundary conditions (unchanged)
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = z_bc[0]

    A[-1, :] = 0
    A[-1,-1] = 1
    b[-1] = z_bc[2]

    # Apply Neumann boundary conditions with one-sided differences for non-uniform grid
    
    #Bottom boundary
    dz_p=z[1]-z[0]
    A[1, :] = 0
    A[1, 0] =-1/dz_p
    A[1, 1] = 1/dz_p
    b[1] = z_bc[1]
    
    #Top boundary
    dz_n = z[-1]-z[-2]
    A[-2, :] = 0
    A[-2, -2] = -1/dz_n
    A[-2, -1] =  1/dz_n
    b[-2] = z_bc[3]

    return A, b



def orr_sommerfeld(N, z, nu, nuT, nuT_prime, nuT_double_prime, k, U, c, U_double_prime, f,z_bc):
    #z = np.linspace(0, zi, N)
    dz = z[1] - z[0]

    D, D2, D3 = D_matrices(N, dz)
    I = np.eye(N)

    # Left-hand side
    A = -(nu+np.diag(nuT))/(1j*k)*(D2 - k**2 * I)@(D2 - k**2 * I)+(np.diag(U)-c) * (D2 - k**2 * I) - np.diag(U_double_prime) - 2/(1j*k)* np.diag(nuT_prime)*(D2-k**2*I)@D-(np.diag(nuT_double_prime)/(1j*k))*(D2 + k**2 * I)
 

    # Add the inhomogeneous term f(z) to the right-hand side
    b = np.zeros((N,),dtype=complex)
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions(A, b, z_bc,dz)
    #print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return w


def orr_sommerfeld_non_uniform(N, z, nu, nuT, nuT_prime, nuT_double_prime, k, U, c, U_double_prime, f,z_bc):
    #z = np.logspace(log10(1e-6), log10(zi), N)
    #dz = np.diff(z)
    
    D, D2, D3 = D_matrices_non_uniform(z)
    I = np.eye(N)
    nu_matrix = np.diag(nu*np.ones(N))
    
    # Left-hand side
    A = np.diag(U-c) @ (D2 - k**2 * I) - np.diag(U_double_prime) - 2/(1j*k)* np.diag(nuT_prime)@(D2-k**2*I)@D -(nu_matrix+np.diag(nuT))/(1j*k)@(D2 - k**2 * I)@(D2 - k**2 * I)  -(np.diag(nuT_double_prime)/(1j*k))@(D2 + k**2 * I)
     

    # Add the inhomogeneous term f(z) to the right-hand side
    b = np.zeros((N,),dtype=complex)
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions_non_uniform(A, b, z_bc, z)
    #print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return w

def orr_sommerfeld_non_uniform_nuT_ij(N, z, nu, nuT11, nuT33, nuT13, nuT31, k, U, c, U_double_prime, f,z_bc):
    #z = np.logspace(log10(1e-6), log10(zi), N)
    #dz = np.diff(z)
    
    D, D2, D3 = D_matrices_non_uniform(z)
    I = np.eye(N)
    nu_matrix = np.diag(nu*np.ones(N))
    
    #nuT11_prime = np.gradient(nuT11,z,edge_order=2)
    #nuT33_prime = np.gradient(nuT33,z,edge_order=2)
    #nuT13_prime = np.gradient(nuT13,z,edge_order=2)
    #nuT31_prime = np.gradient(nuT31,z,edge_order=2)
    #nuT13_double_prime = np.gradient(nuT13_prime,z,edge_order=2)

    nuT11_spline = CubicSpline(z,nuT11, bc_type='not-a-knot')
    nuT13_spline = CubicSpline(z,nuT13, bc_type='not-a-knot')
    nuT31_spline = CubicSpline(z,nuT31, bc_type='not-a-knot')
    nuT33_spline = CubicSpline(z,nuT33, bc_type='not-a-knot')

    nuT11_prime = nuT11_spline.derivative(nu=1)(z)
    nuT13_prime = nuT13_spline.derivative(nu=1)(z)
    nuT31_prime = nuT31_spline.derivative(nu=1)(z)
    nuT33_prime = nuT33_spline.derivative(nu=1)(z)
    nuT13_double_prime = nuT13_spline.derivative(nu=2)(z)
    
    
    # Left-hand side
    A =   np.diag(U-c) @ (D2 - k**2 * I)
    - np.diag(U_double_prime) 
    - (nu_matrix)/(1j*k)@(D2 - k**2 * I)@(D2 - k**2 * I) 
    - (2/(1j*k)) * np.diag(nuT13_prime)@(D2 + k**2*I)@D
    + 2*k/1j*np.diag(nuT11_prime+nuT33_prime)@D
    - (1/(1j*k)) * np.diag(k**2*nuT31+nuT13_double_prime)@(D2 + k**2 * I)
    - (1/(1j*k)) * np.diag(nuT13)@(D2 + k**2 * I)@D2
    + (2*k/1j)*np.diag(nuT11+nuT33)@D2

    # Add the inhomogeneous term f(z) to the right-hand side
    b = np.zeros((N,),dtype=complex)
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions_non_uniform(A, b, z_bc, z)
    #print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return w

def orr_sommerfeld_non_uniform_nuT_ij_itertive(N, eta_hat, z, zi, nu, nuT11, nuT33, nuT13, nuT31, k, U, c, U_prime, U_double_prime, f,z_bc):
    #z = np.logspace(log10(1e-6), log10(zi), N)
    #dz = np.diff(z)
    
    D, D2, D3 = D_matrices_non_uniform(z)
    I = np.eye(N)
    nu_matrix = np.diag(nu*np.ones(N))
    
    nuT11_spline = CubicSpline(z,nuT11, bc_type='not-a-knot')
    nuT13_spline = CubicSpline(z,nuT13, bc_type='not-a-knot')
    nuT31_spline = CubicSpline(z,nuT31, bc_type='not-a-knot')
    nuT33_spline = CubicSpline(z,nuT33, bc_type='not-a-knot') 
    
    # Left-hand side
    A =   np.diag(U-c) @ (D2 - k**2 * I)
    - np.diag(U_double_prime) 
    - (nu_matrix)/(1j*k)@(D2 - k**2 * I)@(D2 - k**2 * I) 
    
    # Add the inhomogeneous term f(z) to the right-hand side
    b  = np.zeros((N,),dtype=complex)
    w0 = np.zeros((N,),dtype=complex)
    u0 = np.zeros((N,),dtype=complex)
    w  = np.zeros((N,),dtype=complex)
    u  = np.zeros((N,),dtype=complex)

    w_guess = z_bc[0]*np.exp(-k*z)
    w0      = w_guess
    dwdz    = np.gradient(w_guess,z,edge_order=2)
        
    err_tol = 1e-4
    err     = 1
    g       = z/zi-1
    g_zeta  = 1/zi
    u0_s    = -(1/(1j*k))*(-z_bc[1]-g*(1j*k*eta_hat)*U_prime[0])
    u0      = u0_s*np.exp(-k*z)
    
    while err>err_tol:
        dudz = np.gradient(u0,z,edge_order=2)
        dwdz = np.gradient(w0,z,edge_order=2)
        
        tau11 =  2*nuT11*dwdz
        tau13 = -2*nuT13*(dudz+1j*k*w0)-nuT13*U_prime*eta_hat
        tau31 = -2*nuT31*(dudz+1j*k*w0)
        tau33 =    nuT33*(1j*k*u0-dwdz)        
        
        d2tau13dz2 = np.gradient(np.gradient(tau13,z,edge_order=2),z,edge_order=2)
        
        b = f + 1j*k*np.gradient(tau11-tau33,z,edge_order=2) + d2tau13dz2 + k**2*tau31
        # Apply boundary conditions
        A, b = apply_boundary_conditions_non_uniform(A, b, z_bc, z)
        w = la.solve(A, b)
        dwdz = np.gradient(w,z,edge_order=2)
        u = -(1/(1j*k))*(-dwdz-g*(1j*k*eta_hat)*U_prime)

        err=np.max(np.abs((w-w0)/z_bc[0]*100))
        if err>err_tol:
            w0=w
            u0=u
            
    return w

def orr_sommerfeld_non_uniform_nuT_ij_iterative_debug(N, eta_hat, z, zi, nu, nuT11, nuT33, nuT13, nuT31, k, U, c, U_prime, U_double_prime, f, z_bc, relaxation_factor):
    D, D2, D3 = D_matrices_non_uniform(z)
    I = np.eye(N)
    nu_matrix = nu*I

    # Spline interpolations
    nuT11_spline = CubicSpline(z, nuT11, bc_type='not-a-knot')
    nuT13_spline = CubicSpline(z, nuT13, bc_type='not-a-knot')
    nuT31_spline = CubicSpline(z, nuT31, bc_type='not-a-knot')
    nuT33_spline = CubicSpline(z, nuT33, bc_type='not-a-knot')

    # Right-hand side
    b  = np.zeros((N,), dtype=complex)
    w0 = np.zeros((N,), dtype=complex)
    u0 = np.zeros((N,), dtype=complex)
    
    w_guess = z_bc[0] * np.exp(-k * z)
    w0 = w_guess
    
    err_tol = 1e-4
    err     = 1
    g       = z / zi - 1
    g_zeta  = 1 / zi
    u0_s    = -(1 / (1j * k)) * (-z_bc[1] - g[0] * (1j * k * eta_hat) * U_prime[0])
    u0      = u0_s * np.exp(-k * z)
    
    while err > err_tol:
        dudz = np.gradient(u0, z, edge_order=2)
        dwdz = np.gradient(w0, z, edge_order=2)
        
        tau11 = 2 * nuT11 * dwdz
        tau13 = -2 * nuT13 * (dudz + 1j * k * w0) - nuT13 * U_prime * eta_hat
        tau31_u = -2 * nuT31 * (dudz + 0*1j * k * w0)
        
        tau33 = nuT33 * (1j * k * u0 - dwdz)
    
        d2tau13dz2 = np.gradient(np.gradient(tau13, z, edge_order=2), z, edge_order=2)
    
        # Check for NaNs or Infs in stress terms
        for name, var in zip(["tau11", "tau13", "tau31_u", "tau33", "d2tau13dz2"], 
                              [tau11, tau13, tau31_u, tau33, d2tau13dz2]):
            if np.isnan(var).any() or np.isinf(var).any():
                print(f"Error: {name} contains NaN or Inf values")
                return None
    
        b = f + 1j * k * np.gradient(tau11 - tau33, z, edge_order=2) + d2tau13dz2 + k**2 * tau31_u
        
        if np.isnan(b).any() or np.isinf(b).any():
            print("Error: b contains NaN or Inf values")
            return None


        # Left-hand side matrix
        A = (np.diag(U - c) @ (D2 - k**2 * I) - np.diag(U_double_prime) 
        - (nu_matrix) / (1j * k) @ (D2 - k**2 * I) @ (D2 - k**2 * I)) 
        + k**2 * 2 * np.diag(nuT31) * (1j * k)
        
        epsilon = 1e-5
        A += epsilon * I  # Add small perturbation to improve conditioning
    
        # Apply boundary conditions
        A, b = apply_boundary_conditions_non_uniform(A, b, z_bc, z)
    
        
        # Check the condition number of A
        cond_A = np.linalg.cond(A)
        print(f"Condition number of A: {cond_A:.2e}")
        if cond_A > 1e12:
            print("Warning: A is ill-conditioned, which may cause numerical instability.")
        
        if np.isnan(A).any() or np.isinf(A).any():
            print("Error: A contains NaN or Inf values")
            return None
        
        w = la.solve(A, b)
        # Check for NaNs or Infs in solution
        if np.isnan(w).any() or np.isinf(w).any():
            print("Error: w contains NaN or Inf values")
            return None
    
        # Update w and u
        dwdz = np.gradient(w, z, edge_order=2)
        u = -(1 / (1j * k)) * (-dwdz - g * (1j * k * eta_hat) * U_prime)
        
        # Calculate error and update guesses
        err = np.max(np.abs((w - w0) / z_bc[0] * 100))
        print(f"Iteration error: {err:.6f}")
    
        if err > err_tol:
            relaxation_factor = max(0.1, relaxation_factor * 0.9)
            w_update = w - w0
            w0 += np.sign(w_update) * np.minimum(np.abs(w_update), relaxation_factor * np.max(np.abs(w0)))
    
            u_update = u - u0
            u0 += np.sign(u_update) * np.minimum(np.abs(u_update), relaxation_factor * np.max(np.abs(u0)))
        else:
            return w


def solve_bvp_model(N, z_grid, nu_molec, nuT11, nuT33, nuT13, nuT31, k, U, c, U_double_prime, RHS_func, z_bc):
    
    # --- Define coefficient functions ---
    def A(U,U_double_prime,c,nu_molec,k,nuT31,nuT13_double_prime):
        return (-U_double_prime - k**2*(U-c)
                - (nu_molec/(1j))*k**3
                - (1/(1j))*(k*nuT31 + nuT13_double_prime*k))
    
    def B(nuT11_prime,nuT33,nuT13_prime,k):
        return (nuT11_prime + nuT33 - nuT13_prime)*(2*k/1j)
    
    def C(U,nu_molec,nuT11,nuT13_double_prime,nuT33,nuT13,nuT31,k):
        return ((U-c)
                - (1/(1j*k))*nuT13_double_prime
                + (2*nu_molec + 2*nuT11 + 2*nuT33 - nuT13 - nuT31)*(k/1j))
    
    def D(nuT13_prime,k):
        return -2*nuT13_prime/(1j*k)
    
    def E(nu_molec,nuT13,k):
        return -(nu_molec+nuT13)/(1j*k)
    
    # --- Define the system of ODEs ---
    def odes(z,y):
        nuT11_spline = CubicSpline(z_grid,nuT11, bc_type='not-a-knot')
        nuT13_spline = CubicSpline(z_grid,nuT13, bc_type='not-a-knot')
        nuT31_spline = CubicSpline(z_grid,nuT31, bc_type='not-a-knot')
        nuT33_spline = CubicSpline(z_grid,nuT33, bc_type='not-a-knot')

        U_spline = CubicSpline(z_grid,U, bc_type='not-a-knot')

        RHS_func_spline = CubicSpline(z_grid,RHS_func, bc_type='not-a-knot')

        U_profile = U_spline(z)
        U_double_prime_profile = U_spline.derivative(nu=2)(z)
        nuT11_prime = nuT11_spline.derivative(nu=1)(z)
        nuT13_prime = nuT13_spline.derivative(nu=1)(z)
        nuT31_prime = nuT31_spline.derivative(nu=1)(z)
        nuT33_prime = nuT33_spline.derivative(nu=1)(z)
        nuT13_double_prime = nuT13_spline.derivative(nu=2)(z)
        
        """
        Converts the fourth-order ODE into a system of four first-order ODEs.
        y[0] = w, y[1] = w', y[2] = w'', y[3] = w'''.
        """
        dydz = np.zeros_like(y, dtype=complex)
        dydz[0] = y[1]
        dydz[1] = y[2]
        dydz[2] = y[3]
        
        # Evaluate coefficients elementwise
        Ez = E(nu_molec,nuT13_spline(z),k)
        Az = A(U_profile,U_double_prime_profile,c,nu_molec,k,nuT31_spline(z),nuT13_double_prime)
        Bz = B(nuT11_prime,nuT33_spline(z),nuT13_prime,k)
        Cz = C(U_profile,nu_molec,nuT11_spline(z),nuT13_double_prime,nuT33_spline(z),nuT13_spline(z),nuT31_spline(z),k)
        Dz = D(nuT13_prime,k)
        RHSz = RHS_func_spline(z)  # RHS is passed as a function
        # Fourth derivative:
        dydz[3] = (RHSz - Az*y[0] - Bz*y[1] - Cz*y[2] - Dz*y[3]) / Ez
        return dydz
    
    # --- Define the boundary conditions ---
    def bc(ya, yb):
        # At z = 0:
        bc1 = ya[0] - z_bc[0]
        bc3 = ya[1] - z_bc[1]
        # At z = z[-1] (approximation to infinity):
        bc2 = yb[0] - z_bc[2]    # w(z_max) = 0
        bc4 = yb[1] - z_bc[3]    # w'(z_max) = 0
        return np.array([bc1, bc2, bc3, bc4])
    
    # --- Initial Guess ---
    y_guess = np.zeros((4, N), dtype=complex)
    # Provide a simple linear guess for w (y1), zeros for derivatives.
    y_guess[0] = z_bc[0] * np.exp(-k*z_grid)
    
    # --- Solve the BVP ---
    sol = solve_bvp(odes, bc, z_grid, y_guess)
    return sol

def orr_sommerfeld_non_uniform_nuT_ij_Chebyshev(N, a, z, zi, nu, nuT11, nuT33, nuT13, nuT31, k, U, c, U_prime, U_double_prime, f, z_bc, relaxation_factor):
    import numpy as np
    from scipy.sparse.linalg import gmres
    import scipy.linalg as la
    
    print("Initializing Chebyshev grid and differentiation matrices...")
    z_cheb = chebyshev_grid(N, zi)
    z_cheb = z_cheb + 1.5e-4
    D  = chebyshev_diff_matrix(N, zi)
    D2 = np.dot(D, D)
    I  = np.eye(N+1)
    g = z_cheb / zi - 1
    eta_hat = a/2
    
    print("Interpolating inputs to the Chebyshev grid...")
    nuT11_cheb = interpolate_to_chebyshev(z, nuT11, z_cheb)
    nuT33_cheb = interpolate_to_chebyshev(z, nuT33, z_cheb)
    nuT13_cheb = interpolate_to_chebyshev(z, nuT13, z_cheb)
    nuT31_cheb = interpolate_to_chebyshev(z, nuT31, z_cheb)
    U_cheb = interpolate_to_chebyshev(z, U, z_cheb)
    f_cheb = interpolate_to_chebyshev(z, f, z_cheb)
    
    U_prime_cheb = D @ U_cheb
    U_double_prime_cheb = D2 @ U_cheb
    
    print("Checking for NaNs or Infs in interpolated values...")
    for name, arr in zip(["nuT11_cheb", "nuT33_cheb", "nuT13_cheb", "nuT31_cheb", "U_cheb", "U_prime_cheb", "U_double_prime_cheb", "f_cheb"],
                          [nuT11_cheb, nuT33_cheb, nuT13_cheb, nuT31_cheb, U_cheb, U_prime_cheb, U_double_prime_cheb, f_cheb]):
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"Error: {name} contains NaN or Inf values!")
    
    print("Initializing w and u...")
    w = np.zeros(N + 1, dtype=complex)
    u = np.zeros(N + 1, dtype=complex)
    w = z_bc[0] * np.exp(-k * z_cheb)
    
    for iter_num in range(1000):
        w_old = w.copy()
        u_old = u.copy()
        
        w[0] = z_bc[0]
        w[N] = z_bc[2]
        dwdz = D @ w
        dwdz[0] = z_bc[1]
        dwdz[N] = z_bc[3]
        u = - (dwdz + g * 1j * k * eta_hat * U_prime_cheb) / (1j * k)
        dudz = D @ u
        
        print(f"Iteration {iter_num}: Checking tau stress terms for NaNs or Infs...")
        tau11_cheb =  2 * nuT11_cheb * dwdz
        tau13_cheb = -2 * nuT13_cheb * (dudz + 1j * k * w) - nuT13_cheb * U_prime_cheb * eta_hat
        tau31_cheb = -2 * nuT31_cheb * (dudz + 1j * k * w)
        tau33_cheb = nuT33_cheb * (1j * k * u - dwdz)
        
        # Debug checks for tau stresses
        for name, arr in zip(["tau11_cheb", "tau13_cheb", "tau31_cheb", "tau33_cheb"], 
                             [tau11_cheb, tau13_cheb, tau31_cheb, tau33_cheb]):
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Error: {name} contains NaNs or Infs!")
                print(f"{name}:", arr)
        
        print(f"Iteration {iter_num}: Constructing LHS matrix L and RHS vector...")
        L = (U_cheb - c) * (D2 - k**2 * I) - U_double_prime_cheb - (nu / (1j * k)) * ((D2 - k**2 * I) @ (D2 - k**2 * I))
        RHS = f_cheb + 1j * k * D @ (tau11_cheb - tau33_cheb) + D2 @ tau13_cheb + k**2 * tau31_cheb
        
        print("Checking for NaNs or Infs in L and RHS...")
        if np.isnan(L).any() or np.isinf(L).any():
            print("Error: L matrix contains NaNs or Infs!")
        if np.isnan(RHS).any() or np.isinf(RHS).any():
            print("Error: RHS contains NaNs or Infs!")
            print("RHS:", RHS)

            # Check each term in RHS separately
            if np.isnan(f_cheb).any() or np.isinf(f_cheb).any():
                print("Error: f_cheb contains NaNs or Infs!")

            if np.isnan(D @ (tau11_cheb - tau33_cheb)).any() or np.isinf(D @ (tau11_cheb - tau33_cheb)).any():
                print("Error: D @ (tau11_cheb - tau33_cheb) contains NaNs or Infs!")

            if np.isnan(D2 @ tau13_cheb).any() or np.isinf(D2 @ tau13_cheb).any():
                print("Error: D2 @ tau13_cheb contains NaNs or Infs!")

            if np.isnan(k**2 * tau31_cheb).any() or np.isinf(k**2 * tau31_cheb).any():
                print("Error: k**2 * tau31_cheb contains NaNs or Infs!")

            # Print values to debug
            print("f_cheb:", f_cheb)
            print("D @ (tau11_cheb - tau33_cheb):", D @ (tau11_cheb - tau33_cheb))
            print("D2 @ tau13_cheb:", D2 @ tau13_cheb)
            print("k**2 * tau31_cheb:", k**2 * tau31_cheb)
        
        print("Checking if L is singular...")
        try:
            det_L = np.linalg.det(L)
            print(f"Det(L) = {det_L}")
        except:
            print("Warning: Could not compute determinant of L (may be singular).")
        
        print("Solving for w...")
        try:
            w = la.solve(L, RHS)
        except la.LinAlgError as e:
            print(f"Linear solve failed: {e}")
            break
        
        w = relaxation_factor * w + (1 - relaxation_factor) * w_old
        u = relaxation_factor * u + (1 - relaxation_factor) * u_old
        
        if np.linalg.norm(w - w_old) < 1e-6:
            print(f"Converged after {iter_num} iterations.")
            break
        
    w_z = interpolate_to_chebyshev(z_cheb, w, z)
    return w_z

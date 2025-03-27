import numpy as np
from numpy.linalg import inv

from build_matrices import buildFEM_matrices
from build_matrices import apply_dirichlet_bc
from build_load import compute_load_vector


def solve_pde_dirichlet_IE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    We'll build M,A,B,C for all vertices => E*nx + #all_vertices dof.
    Then at each time step, we do row modifications for boundary dofs => y(b)= e^-t.
    Meanwhile, the *initial condition* at each vertex dof is the exact solution
    at t=0 for that vertex, determined by which edge sub-interval the vertex is on.
    """
    E= len(edges)
    #dx= 1.0/(nx+1)
    a, b, p, f, y0, g  = problem_data
    # Gather all vertices in the graph
    allv= set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices= sorted(list(allv))

    # 1) Build global matrices
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof= M_H.shape[0]

    # 3) Time stepping
    dt= T/ nt
    times= np.linspace(0, T, nt)
    Y= np.zeros((nt, total_dof))
    Y[0,:]= y0

    for n in range(nt-1):
        t_n_plus_1= times[n+1]
        # build load
        F= compute_load_vector(f, edges, nx, all_vertices, t_n_plus_1)
        A_sys= M_H + dt*( A_H + B_H + C_H )
        RHS= M_H @ Y[n,:] + dt* F

        # impose Dirichlet => y(b, t)= e^-t on boundary vertices
        A_sys, RHS= apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_n_plus_1, E, nx, g)

        # solve
        try:
            y_next= inv(A_sys)@ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system for Dirichlet BC!")
        Y[n+1,:]= y_next

    return Y


def solve_pde_dirichlet_CN(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the PDE with Dirichlet BC using the Crank–Nicolson scheme.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather all vertices in the graph.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    # Use provided initial condition.
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Precompute constant matrices.
    LHS_matrix = M_H + (dt/2) * (A_H + B_H + C_H)
    RHS_matrix = M_H - (dt/2) * (A_H + B_H + C_H)
    
    for n in range(nt-1):
        t_n = times[n]
        t_np1 = times[n+1]
        # Compute load vectors.
        F_n = compute_load_vector(f, edges, nx, all_vertices, t_n)
        F_np1 = compute_load_vector(f, edges, nx, all_vertices, t_np1)
        
        # Assemble RHS.
        RHS = RHS_matrix @ Y[n, :] + (dt/2) * (F_n + F_np1)
        
        # Impose Dirichlet BCs: y(boundary,t_{n+1}) = g(t_{n+1}).
        A_sys, RHS = apply_dirichlet_bc(LHS_matrix, RHS,
                                        boundary_vertices,
                                        all_vertices,
                                        t_np1, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Crank-Nicolson!")
        Y[n+1, :] = y_next
        
    return Y


from scipy.linalg import expm

def check_CFL_EE(M, A, B, C, dt):
    r"""
    For the Explicit Euler scheme applied to
    \[
    M \dot{y} = - (A+B+C) y,
    \]
    stability for the scalar test problem \(y' = -\lambda y\) requires
    \[
    \Delta t < \frac{2}{\lambda}.
    \]
    In the matrix case we demand
    \[
    \Delta t < \frac{2}{\lambda_{\max}(M^{-1}(A+B+C))}.
    \]
    """
    L = np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_theta(M, A, B, C, dt, theta):
    r"""
    For the \(\theta\)-method applied to
    \[
    M \dot{y} = - (A+B+C) y,
    \]
    when \(\theta < \frac{1}{2}\) one obtains the conditional stability requirement
    \[
    \Delta t < \frac{2}{(1-2\theta)\,\lambda_{\max}(M^{-1}(A+B+C))}.
    \]
    For \(\theta\ge\frac{1}{2}\) (A-stable) no CFL restriction applies.
    """
    if theta >= 0.5:
        return dt, True  # A-stable: no restriction.
    L = np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / ((1 - 2*theta) * lambda_max)
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_SIE(M, A, B, C, dt):
    r"""
    For the Semi-Implicit Euler method, where the implicit part is
    \(A\) and the explicit part is \(B+C\), the explicit update is stable if
    \[
    \Delta t < \frac{2}{\lambda_{\max}(M^{-1}(B+C))}.
    \]
    """
    L_explicit = np.linalg.inv(M) @ (B+C)
    eigenvalues = np.linalg.eigvals(L_explicit)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_EXPE(M, A, B, C, dt):
    r"""
    Although the Exponential Euler method is unconditionally stable,
    accuracy requires \(\Delta t\,\lambda_{\max}(L) < 1\) with
    \[
    L = - M^{-1}(A+B+C).
    \]
    Thus, one recommends
    \[
    \Delta t < \frac{1}{\lambda_{\max}(L)}.
    \]
    """
    L = -np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.abs(np.real(eigenvalues)))
    dt_bound = 1.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

# ----------------------------------------------------------------------
# 1. Explicit Euler Method
# ----------------------------------------------------------------------

def solve_pde_dirichlet_EE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Explicit Euler method.
    
    Update:
      \[
      y^{n+1} = y^n + \Delta t\,M^{-1}\Bigl(F - (A+B+C)y^n\Bigr).
      \]
    A CFL check is performed.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for Explicit Euler.
    dt_check, is_stable = check_CFL_EE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        raise ValueError(f"Explicit Euler: dt = {dt:.3e} exceeds CFL limit; recommend dt = {dt_check:.3e}.")
    else:
        print(f"CFL condition satisfied. Maximum possible dt = {dt_check:.3e}.")
        
    for n in range(nt-1):
        t_next = times[n+1]
        # Compute load vector.
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        RHS = M_H @ Y[n, :] + dt * (F - (A_H + B_H + C_H) @ Y[n, :])
        A_sys = M_H.copy()  # For explicit update.
        
        # Impose Dirichlet BCs.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Explicit Euler!")
        Y[n+1, :] = y_next
        
    return times, Y, all_vertices

def solve_pde_dirichlet_theta(edges, boundary_vertices, nx, T, nt, problem_data, theta):
    """
    Solve the parabolic PDE with Dirichlet BCs using the \(\theta\)-method.
    
    Update:
      \[
      [M + \theta\,\Delta t\,(A+B+C)]\,y^{n+1} = M\,y^n + \Delta t\Bigl[(1-\theta)(F-(A+B+C)y^n) + \theta\,F\Bigr].
      \]
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for the theta-method (if theta < 0.5).
    dt_check, is_stable = check_CFL_theta(M_H, A_H, B_H, C_H, dt, theta)
    if not is_stable:
        print(f"Theta-method: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")
        
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        A_sys = M_H + dt * theta * (A_H + B_H + C_H)
        RHS = M_H @ Y[n, :] + dt * ((1 - theta) * (F - (A_H+B_H+C_H) @ Y[n, :]) + theta * F)
        
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Theta-method!")
        Y[n+1, :] = y_next
        
    return Y

def solve_pde_dirichlet_SIEM(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Semi-Implicit Euler method.
    
    Here, the operator \(A\) is treated implicitly while \(B+C\) is explicit:
      \[
      [M + \Delta t\,A]\,y^{n+1} = M\,y^n + \Delta t\,(F - (B+C)y^n).
      \]
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for the explicit part.
    dt_check, is_stable = check_CFL_SIE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Semi-Implicit Euler: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")
        
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        A_sys = M_H + dt * A_H
        RHS = M_H @ Y[n, :] + dt * (F - (B_H + C_H) @ Y[n, :])
        
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Semi-Implicit Euler!")
        Y[n+1, :] = y_next
        
    return Y

def solve_pde_dirichlet_EXPE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Exponential Euler method.
    
    The update is given by
      \[
      y^{n+1} = e^{\Delta t\,L}\,y^n + \Delta t\,\varphi_1(\Delta t\,L)\,M^{-1}F,
      \]
    with
      \[
      L = -M^{-1}(A+B+C) \quad \text{and} \quad \varphi_1(z)=\frac{e^z-1}{z}.
      \]
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check accuracy condition.
    dt_check, is_stable = check_CFL_EXPE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Exponential Euler: dt = {dt:.3e} exceeds accuracy limit; recommend dt = {dt_check:.3e}.")
        
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        L = -inv(M_H) @ (A_H+B_H+C_H)
        exp_dtL = expm(dt * L)
        # Compute \(\varphi_1(\Delta t\,L) = (e^{\Delta t\,L} - I)/(\Delta t\,L)\).
        phi1_dtL = np.linalg.solve(dt * L, (exp_dtL - np.eye(L.shape[0])))
        y_next = exp_dtL @ Y[n, :] + dt * phi1_dtL @ (inv(M_H) @ F)
        
        # Enforce Dirichlet conditions a posteriori.
        A_dummy = np.eye(M_H.shape[0])
        RHS_dummy = y_next.copy()
        A_dummy, RHS_dummy = apply_dirichlet_bc(A_dummy, RHS_dummy, boundary_vertices, all_vertices, t_next, E, nx, g)
        y_next = RHS_dummy
        
        Y[n+1, :] = y_next
        
    return Y
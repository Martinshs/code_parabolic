import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

from build_matrices import buildFEM_matrices, apply_dirichlet_bc
from build_load import compute_load_vector

# =============================================================================
# Implicit Euler (IE) Scheme
# =============================================================================

def solve_pde_dirichlet_IE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet boundary conditions using the Implicit Euler scheme.
    
    This function assembles the global FEM matrices and, at each time step, builds the load
    vector and enforces Dirichlet conditions by modifying the system matrix and right-hand side.
    
    The update follows:
      M_H (y^{n+1} - y^n) / dt + (A_H+B_H+C_H) y^{n+1} = F^{n+1},
    where F^{n+1} is evaluated at time t_{n+1}.
    
    Parameters:
    -----------
    edges : list of tuples
        List of edge connectivity tuples.
    boundary_vertices : list
        List of vertices where Dirichlet conditions are imposed.
    nx : int
        Number of interior degrees of freedom per edge.
    T : float
        Final simulation time.
    nt : int
        Number of time steps.
    problem_data : list
        Contains [a, b, p, f, y0, g]:
          a, b, p: Coefficient functions,
          f: PDE source function,
          y0: Initial condition vector,
          g: Dirichlet boundary condition function.
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) containing the solution at each time step.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Collect all vertices from the edges.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))

    # 1) Build global matrices: M_H, A_H, B_H, and C_H.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]

    # 2) Set time stepping parameters.
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0

    # 3) Time stepping loop.
    for n in range(nt - 1):
        t_n_plus_1 = times[n + 1]
        # Compute the load vector at time t_{n+1}.
        F = compute_load_vector(f, edges, nx, all_vertices, t_n_plus_1)
        # Build the system: M_H + dt*(A_H+B_H+C_H) is used for the implicit update.
        A_sys = M_H + dt * (A_H + B_H + C_H)
        RHS = M_H @ Y[n, :] + dt * F

        # Impose Dirichlet boundary conditions on the system.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_n_plus_1, E, nx, g)
        
        # Solve the linear system; raise an error if the system is singular.
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system for Dirichlet BC in Implicit Euler!")
        Y[n + 1, :] = y_next

    return Y

# =============================================================================
# Crank-Nicolson (CN) Scheme
# =============================================================================

def solve_pde_dirichlet_CN(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet boundary conditions using the Crank–Nicolson scheme.
    
    The update is performed with a trapezoidal rule in time:
      [M_H + (dt/2)(A_H+B_H+C_H)] y^{n+1} = [M_H - (dt/2)(A_H+B_H+C_H)] y^n + (dt/2)(F^n+F^{n+1}).
    
    Parameters:
    -----------
    edges : list of tuples
        List of edge connectivity tuples.
    boundary_vertices : list
        List of vertices with Dirichlet conditions.
    nx : int
        Number of interior degrees of freedom per edge.
    T : float
        Final simulation time.
    nt : int
        Number of time steps.
    problem_data : list
        [a, b, p, f, y0, g] as described above.
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) containing the solution at each time step.
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

    # Time discretization.
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0

    # Precompute constant matrices for the CN update.
    LHS_matrix = M_H + (dt/2) * (A_H + B_H + C_H)
    RHS_matrix = M_H - (dt/2) * (A_H + B_H + C_H)

    # Time stepping loop.
    for n in range(nt - 1):
        t_n = times[n]
        t_np1 = times[n + 1]
        # Compute load vectors at current and next time steps.
        F_n = compute_load_vector(f, edges, nx, all_vertices, t_n)
        F_np1 = compute_load_vector(f, edges, nx, all_vertices, t_np1)

        # Assemble the right-hand side.
        RHS = RHS_matrix @ Y[n, :] + (dt/2) * (F_n + F_np1)

        # Impose Dirichlet conditions for time t_{n+1}.
        A_sys, RHS = apply_dirichlet_bc(LHS_matrix, RHS, boundary_vertices, all_vertices, t_np1, E, nx, g)
        
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Crank-Nicolson!")
        Y[n + 1, :] = y_next

    return Y

# =============================================================================
# CFL Condition Checks for Various Schemes
# =============================================================================

def check_CFL_EE(M, A, B, C, dt):
    r"""
    Check the CFL condition for the Explicit Euler scheme applied to
      M y' = - (A+B+C) y.
    
    For the scalar test problem y' = -λ y, stability requires Δt < 2/λ.
    In the matrix case, we require
      Δt < 2 / λ_max(M^{-1}(A+B+C)).
    
    Parameters:
    -----------
    M, A, B, C : np.ndarray
        Global FEM matrices.
    dt : float
        Proposed time step.
    
    Returns:
    --------
    tuple
        (dt_used, is_stable), where dt_used is the maximum stable time step if dt is too large,
        and is_stable is a Boolean indicating whether the given dt satisfies the CFL condition.
    """
    L = np.linalg.inv(M) @ (A + B + C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_theta(M, A, B, C, dt, theta):
    r"""
    Check the CFL condition for the θ-method applied to
      M y' = - (A+B+C) y.
    
    For θ < 1/2, stability requires
      Δt < 2 / ((1 - 2θ) * λ_max(M^{-1}(A+B+C))).
    For θ ≥ 1/2 (A-stable methods), no CFL restriction applies.
    
    Parameters:
    -----------
    M, A, B, C : np.ndarray
        Global FEM matrices.
    dt : float
        Proposed time step.
    theta : float
        Parameter of the θ-method.
    
    Returns:
    --------
    tuple
        (dt_used, is_stable) as in check_CFL_EE.
    """
    if theta >= 0.5:
        return dt, True  # A-stable: no CFL restriction.
    L = np.linalg.inv(M) @ (A + B + C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / ((1 - 2*theta) * lambda_max)
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_SIE(M, A, B, C, dt):
    r"""
    Check the CFL condition for the Semi-Implicit Euler method, where the implicit part is A
    and the explicit part is B+C. The explicit update is stable if
      Δt < 2 / λ_max(M^{-1}(B+C)).
    
    Parameters:
    -----------
    M, A, B, C : np.ndarray
        Global FEM matrices.
    dt : float
        Proposed time step.
    
    Returns:
    --------
    tuple
        (dt_used, is_stable).
    """
    L_explicit = np.linalg.inv(M) @ (B + C)
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
    accuracy requires Δt * λ_max(L) < 1, where
      L = -M^{-1}(A+B+C).
    Therefore, one recommends
      Δt < 1 / λ_max(L).
    
    Parameters:
    -----------
    M, A, B, C : np.ndarray
        Global FEM matrices.
    dt : float
        Proposed time step.
    
    Returns:
    --------
    tuple
        (dt_used, is_stable) indicating if the accuracy condition is met.
    """
    L = -np.linalg.inv(M) @ (A + B + C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.abs(np.real(eigenvalues)))
    dt_bound = 1.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

# =============================================================================
# Explicit Euler (EE) Scheme
# =============================================================================

def solve_pde_dirichlet_EE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Explicit Euler method.
    
    The update is performed as:
      y^{n+1} = y^n + Δt * M^{-1} (F - (A+B+C) y^n),
    where a CFL condition is checked prior to time stepping.
    
    Parameters:
    -----------
    edges : list of tuples
        Edge connectivity.
    boundary_vertices : list
        Vertices with Dirichlet conditions.
    nx : int
        Interior dofs per edge.
    T : float
        Final time.
    nt : int
        Number of time steps.
    problem_data : list
        [a, b, p, f, y0, g].
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) with the solution at each time step.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather all vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check the CFL condition for stability.
    dt_check, is_stable = check_CFL_EE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        raise ValueError(f"Explicit Euler: dt = {dt:.3e} exceeds CFL limit; recommend dt = {dt_check:.3e}.")
    else:
        print(f"CFL condition satisfied. Maximum possible dt = {dt_check:.3e}.")

    # Time stepping.
    for n in range(nt - 1):
        t_next = times[n + 1]
        # Compute the load vector at time t_next.
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        # Explicit update.
        RHS = M_H @ Y[n, :] + dt * (F - (A_H + B_H + C_H) @ Y[n, :])
        A_sys = M_H.copy()  # For explicit Euler, we use M_H as the system matrix.
        
        # Enforce Dirichlet boundary conditions.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Explicit Euler!")
        Y[n + 1, :] = y_next

    return Y

# =============================================================================
# Theta-Method Scheme
# =============================================================================

def solve_pde_dirichlet_theta(edges, boundary_vertices, nx, T, nt, problem_data, theta):
    """
    Solve the parabolic PDE with Dirichlet BCs using the θ-method.
    
    The update scheme is:
      [M + θ Δt (A+B+C)] y^{n+1} = M y^n + Δt [ (1-θ)(F - (A+B+C)y^n) + θ F ].
    
    Parameters:
    -----------
    edges : list of tuples
        Edge connectivity.
    boundary_vertices : list
        Vertices with Dirichlet conditions.
    nx : int
        Number of interior dofs per edge.
    T : float
        Final simulation time.
    nt : int
        Number of time steps.
    problem_data : list
        [a, b, p, f, y0, g].
    theta : float
        Parameter for the θ-method (θ=0: explicit, θ=1: implicit, θ=1/2: Crank-Nicolson).
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) with the solution at each time step.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check the CFL condition for the theta-method if theta < 0.5.
    dt_check, is_stable = check_CFL_theta(M_H, A_H, B_H, C_H, dt, theta)
    if not is_stable:
        print(f"Theta-method: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")

    # Time stepping.
    for n in range(nt - 1):
        t_next = times[n + 1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        A_sys = M_H + dt * theta * (A_H + B_H + C_H)
        RHS = M_H @ Y[n, :] + dt * ((1 - theta) * (F - (A_H + B_H + C_H) @ Y[n, :]) + theta * F)
        
        # Impose Dirichlet BCs.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Theta-method!")
        Y[n + 1, :] = y_next
        
    return Y

# =============================================================================
# Semi-Implicit Euler (SIEM) Scheme
# =============================================================================

def solve_pde_dirichlet_SIEM(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Semi-Implicit Euler method.
    
    In this method, the operator A is treated implicitly and the operators B+C explicitly:
      [M + Δt A] y^{n+1} = M y^n + Δt (F - (B+C)y^n).
    
    Parameters:
    -----------
    edges : list of tuples
        Edge connectivity.
    boundary_vertices : list
        Vertices with Dirichlet conditions.
    nx : int
        Number of interior dofs per edge.
    T : float
        Final simulation time.
    nt : int
        Number of time steps.
    problem_data : list
        [a, b, p, f, y0, g].
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) with the solution at each time step.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for the explicit part of SIEM.
    dt_check, is_stable = check_CFL_SIE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Semi-Implicit Euler: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")

    # Time stepping.
    for n in range(nt - 1):
        t_next = times[n + 1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        A_sys = M_H + dt * A_H
        RHS = M_H @ Y[n, :] + dt * (F - (B_H + C_H) @ Y[n, :])
        
        # Enforce Dirichlet conditions.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx, g)
        try:
            y_next = inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Semi-Implicit Euler!")
        Y[n + 1, :] = y_next
        
    return Y

# =============================================================================
# Exponential Euler (EXPE) Scheme
# =============================================================================

def solve_pde_dirichlet_EXPE(edges, boundary_vertices, nx, T, nt, problem_data):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Exponential Euler method.
    
    The update is given by:
      y^{n+1} = exp(dt*L) y^n + dt * φ_1(dt*L) M^{-1} F,
    where L = - M^{-1}(A+B+C) and φ_1(z) = (exp(z)-1)/z.
    Although unconditionally stable, accuracy requires dt*λ_max(L) < 1.
    
    Parameters:
    -----------
    edges : list of tuples
        Edge connectivity.
    boundary_vertices : list
        Vertices with Dirichlet conditions.
    nx : int
        Number of interior dofs per edge.
    T : float
        Final simulation time.
    nt : int
        Number of time steps.
    problem_data : list
        [a, b, p, f, y0, g].
    
    Returns:
    --------
    np.ndarray
        Array Y of shape (nt, total_dof) with the solution at each time step.
    """
    E = len(edges)
    a, b, p, f, y0, g = problem_data

    # Gather vertices.
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global matrices.
    M_H, A_H, B_H, C_H = buildFEM_matrices(edges, nx, all_vertices, a, b, p)
    total_dof = M_H.shape[0]
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check the accuracy condition for Exponential Euler.
    dt_check, is_stable = check_CFL_EXPE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Exponential Euler: dt = {dt:.3e} exceeds accuracy limit; recommend dt = {dt_check:.3e}.")

    # Time stepping.
    for n in range(nt - 1):
        t_next = times[n + 1]
        F = compute_load_vector(f, edges, nx, all_vertices, t_next)
        # Compute L = - M^{-1}(A+B+C).
        L = -inv(M_H) @ (A_H + B_H + C_H)
        exp_dtL = expm(dt * L)
        # Compute φ₁(dt L) = (exp(dt L) - I) / (dt L).
        phi1_dtL = np.linalg.solve(dt * L, (exp_dtL - np.eye(L.shape[0])))
        y_next = exp_dtL @ Y[n, :] + dt * phi1_dtL @ (inv(M_H) @ F)
        
        # Enforce Dirichlet BCs a posteriori.
        # Here, A_dummy is the identity, so applying Dirichlet BC simply replaces
        # the corresponding entries in y_next.
        A_dummy = np.eye(M_H.shape[0])
        RHS_dummy = y_next.copy()
        A_dummy, RHS_dummy = apply_dirichlet_bc(A_dummy, RHS_dummy, boundary_vertices, all_vertices, t_next, E, nx, g)
        y_next = RHS_dummy
        
        Y[n + 1, :] = y_next
        
    return Y
